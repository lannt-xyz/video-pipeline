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

_SCENE_WORKFLOW = "image_gen/workflows/txt2img_scene.json"

_NEGATIVE = (
    # Hard NSFW block
    "nsfw, nudity, naked, nude, nipples, pussy, penis, genitals, "
    "underwear, lingerie, bikini, swimsuit, cleavage, navel, bare skin, "
    "undressing, topless, bottomless, lewd, ecchi, explicit, uncensored, "
    "alluring, seductive, suggestive, provocative, "
    # Quality anti-tags
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, normal quality, "
    "jpeg artifacts, signature, watermark, username, blurry, score_1, score_2"
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
    """Generate anchor image for one character."""
    char_slug = _slugify(character.name)
    anchor_path = (
        Path(settings.data_dir) / "characters" / char_slug / "anchor.png"
    )

    if anchor_path.exists() and not force:
        logger.debug(
            "Anchor already exists, skipping | character={}", character.name
        )
        return anchor_path

    anchor_path.parent.mkdir(parents=True, exist_ok=True)

    scene_prompt = (
        f"{character.description}, "
        "full body portrait, anime style, manhua art style, 9:16 portrait, "
        "detailed face, high quality, masterpiece, best quality, ultra detailed, "
        "score_9, score_8_up, score_7_up"
    )

    comfyui_client.generate_image(
        workflow_path=_SCENE_WORKFLOW,
        replacements={
            "SCENE_PROMPT": scene_prompt,
            "NEGATIVE_PROMPT": _NEGATIVE,
            "WIDTH": settings.image_width,
            "HEIGHT": settings.image_height,
            "SEED": int(hashlib.md5(character.name.encode()).hexdigest(), 16) % (2**32),
        },
        output_path=anchor_path,
    )

    # Update character JSON with anchor_path
    char_json = (
        Path(settings.data_dir) / "characters" / f"{char_slug}.json"
    )
    if char_json.exists():
        data = json.loads(char_json.read_text(encoding="utf-8"))
        data["anchor_path"] = str(anchor_path)
        char_json.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    logger.info(
        "Generated anchor | character={} path={}", character.name, anchor_path
    )
    return anchor_path


def get_anchor_path(character_name: str) -> Path:
    """Return expected anchor image path for a character name."""
    char_slug = _slugify(character_name)
    return Path(settings.data_dir) / "characters" / char_slug / "anchor.png"
