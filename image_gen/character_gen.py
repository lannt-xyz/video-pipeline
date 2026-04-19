import hashlib
import json
import unicodedata
from pathlib import Path

from loguru import logger

from config.settings import settings
from image_gen.comfyui_client import comfyui_client
from llm.character_extractor import load_all_characters
from models.schemas import Character


_VN_SPECIAL = str.maketrans({"đ": "d", "Đ": "D"})


def _slugify(name: str) -> str:
    # Handle đ/Đ (does not decompose via NFD), then strip remaining diacritics
    name = name.translate(_VN_SPECIAL)
    normalized = unicodedata.normalize("NFD", name.lower())
    ascii_name = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    return ascii_name.replace(" ", "_").replace("-", "_")

_ANCHOR_WORKFLOW = "image_gen/workflows/anchor_gen.json"

_ANCHOR_ETHNICITY_POSITIVE = (
    "east asian facial features, chinese facial features, han chinese aesthetics, "
    "asian eyes, black hair"
)

_NEGATIVE = (
    # Hard NSFW block
    "nsfw, nudity, naked, nude, nipples, pussy, penis, genitals, "
    "underwear, lingerie, bikini, swimsuit, cleavage, navel, bare skin, "
    "undressing, topless, bottomless, lewd, ecchi, explicit, uncensored, "
    "alluring, seductive, suggestive, provocative, "
    # Monster / non-human anatomy block
    "(horn:1.5), (horns:1.5), (armor:1.5), (wings:1.5), (monster:1.5), "
    "(tail:1.3), (claws:1.3), (fangs:1.3), demon, beast, creature, "
    # Quality anti-tags
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, normal quality, "
    "jpeg artifacts, signature, watermark, username, blurry, score_1, score_2, "
    # Reduce western-looking anchors
    "western facial features, caucasian, european face, deep eye sockets, high nose bridge"
)


def _build_anchor_scene_prompt(character: Character, angle_tags: str) -> str:
    """Build a stable anchor prompt with explicit East Asian facial bias."""
    return (
        f"{_ANCHOR_ETHNICITY_POSITIVE}, {character.description}, "
        f"close-up portrait, face focus, head and shoulders only, {angle_tags}, "
        "anime style, manhua art style, plain background, "
        "detailed face, high quality, masterpiece, best quality, ultra detailed"
    )


def generate_character_anchors(force: bool = False) -> None:
    """Generate anchor reference images for all known characters.
    Run once before any episode image generation.
    Skips characters that already have an anchor unless force=True.
    """
    characters = load_all_characters()

    if not characters:
        logger.warning("No characters found — run LLM phase first to extract characters")
        return

    for character in characters:
        _generate_single_anchor(character, force=force)


def _generate_single_anchor(character: Character, force: bool = False) -> Path:
    """Generate 3 anchor views (front, 3/4, slight turn) per character.
    IPAdapter works best with multiple reference angles to build stable embeddings.
    The main anchor.png is the front view; views are stored alongside it.
    """
    char_slug = _slugify(character.name)
    anchor_dir = Path(settings.data_dir) / "characters" / char_slug
    anchor_path = anchor_dir / "anchor.png"

    if anchor_path.exists() and not force:
        logger.debug(
            "Anchor already exists, skipping | character={}", character.name
        )
        return anchor_path

    anchor_dir.mkdir(parents=True, exist_ok=True)

    base_seed = int(hashlib.md5(character.name.encode()).hexdigest(), 16) % (2**32)

    # 3 views with different angles — all close-up face focus
    _VIEWS = [
        ("anchor.png", "looking at viewer, front view"),
        ("anchor_3q.png", "looking slightly to the side, three-quarter view"),
        ("anchor_side.png", "looking to the side, profile view, side angle"),
    ]

    for filename, angle_tags in _VIEWS:
        view_path = anchor_dir / filename
        if view_path.exists() and not force:
            continue

        scene_prompt = _build_anchor_scene_prompt(character, angle_tags)

        comfyui_client.generate_image(
            workflow_path=_ANCHOR_WORKFLOW,
            replacements={
                "SCENE_PROMPT": scene_prompt,
                "NEGATIVE_PROMPT": _NEGATIVE,
                "WIDTH": 768,
                "HEIGHT": 768,
                "SEED": base_seed + _VIEWS.index((filename, angle_tags)),
            },
            output_path=view_path,
        )
        logger.info(
            "Generated anchor view | character={} view={}", character.name, filename
        )

    # Update character JSON with anchor_path
    char_json = (
        Path(settings.data_dir) / "characters" / char_slug / "profile.json"
    )
    if char_json.exists():
        data = json.loads(char_json.read_text(encoding="utf-8"))
        data["anchor_path"] = str(anchor_path)
        char_json.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    logger.info(
        "Generated anchor set | character={} path={}", character.name, anchor_dir
    )
    return anchor_path


def get_anchor_paths(character_name: str) -> list[Path]:
    """Return all anchor view paths for a character (for IPAdapter batch)."""
    char_slug = _slugify(character_name)
    anchor_dir = Path(settings.data_dir) / "characters" / char_slug
    views = [anchor_dir / f for f in ("anchor.png", "anchor_3q.png", "anchor_side.png")]
    return [p for p in views if p.exists()]


def get_anchor_path(character_name: str) -> Path:
    """Return expected anchor image path for a character name."""
    char_slug = _slugify(character_name)
    return Path(settings.data_dir) / "characters" / char_slug / "anchor.png"
