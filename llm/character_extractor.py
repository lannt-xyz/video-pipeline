import json
from pathlib import Path
from typing import List

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from llm.client import ollama_client
from models.schemas import Character

_EXTRACTOR_SYSTEM = """You are a character analyst for Vietnamese fantasy novels.
Given story summaries, extract all named characters with their appearance descriptions suitable for AI image generation.
Return a JSON array of character objects with EXACTLY this schema:
[
  {
    "name": "string — full character name",
    "alias": ["string — alternative names or nicknames"],
    "description": "string — detailed appearance description in English for ComfyUI (hair color, eye color, clothing style, notable features)",
    "relationships": {"other_character_name": "relationship description"}
  }
]
Focus on physical appearance details useful for image generation. Include at least 5-10 characters."""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
def _extract_raw(summaries_text: str) -> List[dict]:
    prompt = f"Extract characters from these story summaries:\n\n{summaries_text}"
    result = ollama_client.generate_json(
        prompt=prompt, system=_EXTRACTOR_SYSTEM, temperature=0.2
    )
    if isinstance(result, list):
        return result
    # Some LLMs wrap in a key
    if isinstance(result, dict):
        for key in ("characters", "data", "result"):
            if key in result and isinstance(result[key], list):
                return result[key]
    return []


def extract_all_characters() -> List[Character]:
    """Extract characters from all arc summaries. Meant to run once globally."""
    summaries_dir = Path(settings.data_dir) / "summaries"
    arc_files = sorted(summaries_dir.glob("*-arc.json"))

    if not arc_files:
        logger.warning("No arc summary files found for character extraction")
        # Fall back to settings characters list
        return [
            Character(
                name=name,
                description=(
                    f"{name}, xianxia fantasy character, detailed anime style, "
                    "manhua art style, full body"
                ),
            )
            for name in settings.characters
        ]

    # Use first 10 arcs for extraction (reasonable context size)
    all_summaries = []
    for arc_file in arc_files[:10]:
        data = json.loads(arc_file.read_text(encoding="utf-8"))
        all_summaries.append(data.get("arc_summary", ""))

    combined = "\n\n---\n\n".join(all_summaries)
    logger.info("Extracting characters from {} arc summaries", len(all_summaries))

    raw_list = _extract_raw(combined)
    characters = [Character(**c) for c in raw_list]

    # Save each character to its own JSON file
    chars_dir = Path(settings.data_dir) / "characters"
    chars_dir.mkdir(parents=True, exist_ok=True)

    for char in characters:
        char_path = chars_dir / f"{_slugify(char.name)}.json"
        char_path.write_text(char.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved character | name={}", char.name)

    return characters


def load_character(name: str) -> Character:
    char_path = (
        Path(settings.data_dir) / "characters" / f"{_slugify(name)}.json"
    )
    if not char_path.exists():
        raise FileNotFoundError(f"Character not found: {name}")
    return Character(**json.loads(char_path.read_text(encoding="utf-8")))


def load_all_characters() -> List[Character]:
    chars_dir = Path(settings.data_dir) / "characters"
    if not chars_dir.exists():
        return []

    characters = []
    for f in chars_dir.glob("*.json"):
        try:
            characters.append(
                Character(**json.loads(f.read_text(encoding="utf-8")))
            )
        except Exception as exc:
            logger.warning("Failed to load character from {} | error={}", f, exc)
    return characters


def _slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")
