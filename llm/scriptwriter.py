import json
from pathlib import Path
from typing import List

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from llm.client import ollama_client
from llm.summarizer import load_arc_overview
from models.schemas import EpisodeScript, ShotScript

_SCRIPTWRITER_SYSTEM = """You are a Vietnamese short video scriptwriter for TikTok/YouTube Shorts.
Given an arc overview of a fantasy story episode, write a video script with 8-10 shots for a 60-second video.

STRICT RULES:
1. scene_prompt: English description for ComfyUI image generation. Include: "anime style, 9:16 portrait, dramatic lighting, manhua art style". Be vivid and specific.
2. narration_text: Vietnamese narration for text-to-speech. Natural storytelling voice, engaging, first-person narrator.
3. duration_sec: 6 for standard shots, 7 for climactic moments.
4. is_key_shot: Mark EXACTLY 2-3 shots as true — the most emotionally intense or action-packed moments.
5. characters: list character name strings present in shot. Use [] for scene-only (no characters visible).

Return a JSON object with EXACTLY this schema:
{
  "title": "string — episode title in Vietnamese",
  "shots": [
    {
      "scene_prompt": "string",
      "narration_text": "string",
      "duration_sec": 6,
      "is_key_shot": false,
      "characters": []
    }
  ]
}
The shots array MUST have 8-10 elements. EXACTLY 2-3 shots must have is_key_shot=true."""


@retry(
    stop=stop_after_attempt(settings.llm_max_retries),
    wait=wait_exponential(min=2, max=15),
)
def _write_raw(arc_text: str, episode_num: int) -> dict:
    prompt = (
        f"Write a video script for Episode {episode_num}.\n\n"
        f"Arc Overview:\n{arc_text}\n\n"
        "Remember: 8-10 shots, EXACTLY 2-3 with is_key_shot=true, "
        "scene_prompt in English, narration_text in Vietnamese."
    )
    return ollama_client.generate_json(
        prompt=prompt, system=_SCRIPTWRITER_SYSTEM, temperature=0.7
    )


def write_episode_script(episode_num: int) -> EpisodeScript:
    """Generate shot script from arc overview. Saves JSON to data/scripts/."""
    arc = load_arc_overview(episode_num)
    arc_text = (
        f"Summary: {arc.arc_summary}\n\n"
        f"Key Events:\n"
        + "\n".join(f"- {e}" for e in arc.key_events)
        + f"\n\nCharacters: {', '.join(arc.characters_in_episode)}"
    )

    logger.info("Writing script | episode={}", episode_num)
    raw = _write_raw(arc_text, episode_num)

    shots = [ShotScript(**s) for s in raw["shots"]]
    shots = _normalize_key_shots(shots, episode_num)

    script = EpisodeScript(
        episode_num=episode_num,
        title=raw.get("title", f"Tập {episode_num}"),
        shots=shots,
    )

    script_path = (
        Path(settings.data_dir)
        / "scripts"
        / f"episode-{episode_num:03d}-script.json"
    )
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script.model_dump_json(indent=2), encoding="utf-8")

    logger.info("Script written | episode={} shots={}", episode_num, len(script.shots))
    return script


def load_episode_script(episode_num: int) -> EpisodeScript:
    script_path = (
        Path(settings.data_dir)
        / "scripts"
        / f"episode-{episode_num:03d}-script.json"
    )
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found for episode {episode_num}")
    return EpisodeScript(**json.loads(script_path.read_text(encoding="utf-8")))


def _normalize_key_shots(shots: List[ShotScript], episode_num: int) -> List[ShotScript]:
    """Ensure exactly 2-3 shots are marked as key shots."""
    key_count = sum(1 for s in shots if s.is_key_shot)

    if key_count < 2:
        logger.warning(
            "Only {} key shots from LLM, promoting first 2 | episode={}",
            key_count,
            episode_num,
        )
        promoted = 0
        shots = list(shots)
        for i in range(len(shots)):
            if not shots[i].is_key_shot and promoted < (2 - key_count):
                shots[i] = shots[i].model_copy(update={"is_key_shot": True})
                promoted += 1

    elif key_count > 3:
        logger.warning(
            "{} key shots from LLM, capping to 3 | episode={}", key_count, episode_num
        )
        count = 0
        shots = list(shots)
        for i in range(len(shots)):
            if shots[i].is_key_shot:
                if count >= 3:
                    shots[i] = shots[i].model_copy(update={"is_key_shot": False})
                else:
                    count += 1

    return shots
