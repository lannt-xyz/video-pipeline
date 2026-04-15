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
You will receive ORDERED SCENES — story events in exact chronological order.

YOUR TASK: Write EXACTLY 8 shots that cover the story from start to finish in order.
- Distribute the scenes evenly across 8 shots: if there are 7 scenes, some shots cover 1 scene, others cover 2 adjacent scenes.
- Shot 1 = beginning of story, Shot 8 = end of story. No scene may come before an earlier scene.

NARRATIVE RULES (most critical):
- Each shot's narration_text tells what SPECIFICALLY happens — name the action, the character, the location.
- Shots must connect: the last sentence of shot N sets up shot N+1.
- Voice: first-person narrator ("Tôi..."), present-tense tension.
- FORBIDDEN phrases: "mọi chuyện leo thang", "những bí ẩn được hé lộ", "cuộc chiến tiếp tục", "các nhân vật xuất hiện".

SCENE PROMPT RULES — CRITICAL: ComfyUI uses Stable Diffusion, which requires comma-separated tags, NOT sentences.
scene_prompt must be a SHORT TAG LIST only. Structure:
  [location tags], [action/pose tags], [atmosphere/lighting tags], anime style, manhua art style, dramatic lighting, detailed background, no text, no watermarks

FORBIDDEN in scene_prompt:
- English sentences or clauses ("He walks into...", "Fifteen years later...")
- Vietnamese words anywhere
- Character names (Diệp Thiếu Dương, Tiểu Mã, etc.) — character appearance is handled separately
- Adverbs or qualifiers ("mysteriously", "fiercely", "inadvertently")

REQUIRED in scene_prompt:
- At least 1 specific LOCATION tag: "university corridor", "abandoned building interior", "mountain temple courtyard", "city street at night", "coffin shop interior", "rooftop night"
- At least 1 ACTION or ATMOSPHERE tag
- End ALWAYS with: "anime style, manhua art style, dramatic lighting, detailed background, no text, no watermarks"

EXAMPLES:
WRONG: "A mysterious man excavates a newly discovered tomb in the sacred Diệp gia village at night."
RIGHT: "outdoor gravesite, ancient tomb excavation, night scene, eerie moonlight, dark soil, stone ruins, anime style, manhua art style, dramatic lighting, detailed background, no text, no watermarks"

WRONG: "Fifteen years later, Diệp Thiếu Dương returns to Mao Sơn to continue learning Daoist arts from Thanh Vân Tử."
RIGHT: "mountain monastery courtyard, daoist training, morning mist, stone staircase, bamboo grove, anime style, manhua art style, dramatic lighting, detailed background, no text, no watermarks"

WRONG: "At the coffin shop, Diệp Thiếu Dương fiercely fights a white zombie using the Willow Wood Sword."
RIGHT: "coffin shop interior, intense fight scene, wooden shelves, dim lantern light, dynamic action pose, anime style, manhua art style, dramatic lighting, detailed background, no text, no watermarks"

OTHER RULES:
- duration_sec: 6 for standard shots, 8 for climactic action shots.
- is_key_shot: Mark EXACTLY 2-3 shots as true — the most action-packed.
- characters: list character names visible. Use [] for scenery-only.

Return JSON:
{
  "title": "string — episode title in Vietnamese",
  "shots": [ { "scene_prompt": "string", "narration_text": "string", "duration_sec": 6, "is_key_shot": false, "characters": [] } ]
}
shots MUST have EXACTLY 8 elements. EXACTLY 2-3 must have is_key_shot=true."""


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


_MAX_SHOTS_PER_EPISODE = 8  # shots that fit in one video; excess flows to next episode


def _load_carryover(episode_num: int) -> List[ShotScript]:
    """Load leftover shots carried over from episode_num."""
    path = Path(settings.data_dir) / "scripts" / f"episode-{episode_num:03d}-carryover.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ShotScript(**s) for s in data]


def _save_carryover(episode_num: int, shots: List[ShotScript]) -> None:
    """Persist leftover shots so episode_num+1 can pick them up."""
    path = Path(settings.data_dir) / "scripts" / f"episode-{episode_num:03d}-carryover.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([s.model_dump() for s in shots], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_chunk_summaries(episode_num: int) -> list:
    """Load ordered chunk summaries from disk, sorted by chunk_index."""
    path = Path(settings.data_dir) / "summaries" / f"episode-{episode_num:03d}-chunks.json"
    if not path.exists():
        return []
    chunks = json.loads(path.read_text(encoding="utf-8"))
    return sorted(chunks, key=lambda c: c["chunk_index"])


def write_episode_script(episode_num: int) -> EpisodeScript:
    """Generate shot script. Caps at _MAX_SHOTS_PER_EPISODE; excess saved as carry-over.
    First checks carry-over from previous episode before calling LLM.
    """
    # 1. Load carry-over from previous episode
    carryover = _load_carryover(episode_num - 1)
    all_shots: List[ShotScript] = list(carryover)
    raw_title = f"Tập {episode_num}"

    # 2. If carry-over doesn't fill the episode, call LLM for new shots
    if len(all_shots) < _MAX_SHOTS_PER_EPISODE:
        arc = load_arc_overview(episode_num)

        chunks = _load_chunk_summaries(episode_num)
        if chunks:
            # Truncate each chunk to 400 chars so prompt fits in context
            scenes_text = "\n\n".join(
                f"Scene {i + 1} (chương {c['chapter_start']}-{c['chapter_end']}):\n{c['summary'][:400]}"
                for i, c in enumerate(chunks)
            )
            arc_text = (
                f"Characters: {', '.join(arc.characters_in_episode)}\n\n"
                f"ORDERED SCENES:\n\n{scenes_text}"
            )
        else:
            arc_text = (
                f"Summary: {arc.arc_summary}\n\n"
                f"ORDERED SCENES:\n"
                + "\n".join(f"Scene {i+1}: {e}" for i, e in enumerate(arc.key_events))
                + f"\n\nCharacters: {', '.join(arc.characters_in_episode)}"
            )

        logger.info("Writing script | episode={}", episode_num)
        raw = _write_raw(arc_text, episode_num)
        raw_title = raw.get("title", raw_title)

        new_shots = [ShotScript(**s) for s in raw["shots"]]
        new_shots = _normalize_key_shots(new_shots, episode_num)
        all_shots.extend(new_shots)
    else:
        logger.info("Using carry-over shots | episode={} shots={}", episode_num, len(all_shots))

    # 3. Take first _MAX_SHOTS_PER_EPISODE; save remainder as carry-over for next episode
    episode_shots = all_shots[:_MAX_SHOTS_PER_EPISODE]
    leftover = all_shots[_MAX_SHOTS_PER_EPISODE:]

    if leftover:
        _save_carryover(episode_num, leftover)
        logger.info(
            "Carry-over saved | episode={} leftover={} → episode={}",
            episode_num, len(leftover), episode_num + 1,
        )

    script = EpisodeScript(
        episode_num=episode_num,
        title=raw_title,
        shots=episode_shots,
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


def _normalize_duration(shots: List[ShotScript], episode_num: int) -> List[ShotScript]:
    """Scale shot durations so total equals settings.target_duration_sec (±2s).
    Distributes extra/missing seconds evenly; each shot is clamped to [4, 10].
    Applied after LLM generation so the validator range [58, 62] is always met.
    """
    target = settings.target_duration_sec
    total = sum(s.duration_sec for s in shots)
    if abs(total - target) <= 2:
        return shots  # already within tolerance

    logger.warning(
        "Script total {}s != target {}s, normalizing | episode={}",
        total,
        target,
        episode_num,
    )

    shots = list(shots)
    delta = target - total  # positive = need to add, negative = need to remove
    # Distribute delta round-robin; clamp each shot to [4, 10]
    for i in range(abs(delta)):
        idx = i % len(shots)
        current = shots[idx].duration_sec
        adjusted = current + (1 if delta > 0 else -1)
        adjusted = max(4, min(10, adjusted))
        shots[idx] = shots[idx].model_copy(update={"duration_sec": adjusted})

    return shots


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
