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

HOOK RULE — Shot 1 CRITICAL:
- Shot 1 MUST open with a SPECIFIC ACTION or SHOCKING DIALOGUE/NARRATION — never scene-setting or backstory.
- Keep chronological order — do NOT flash-forward to a later scene. Reframe the first scene from a shocking angle.
- narration_text for Shot 1 MUST be 10 words or fewer (must fit 2–3 seconds of TTS).
- WRONG: "Diệp Thiếu Dương đứng trước ngôi miếu cổ tại làng Diệp gia, bắt đầu hành trình của mình."
- RIGHT: "Hắn mở nắp quan tài... và thứ bên trong không phải xác người."
- WRONG: "Tôi bắt đầu hành trình tại Diệp gia thôn linh thiêng, nơi một người đàn ông bí ẩn đang khai quật ngôi mộ."
- RIGHT: "Một tiếng thét xé lòng vang lên từ phía sau ngôi mộ cổ."

CLIFFHANGER RULE — Shot 8 CRITICAL:
- Shot 8 MUST end at the CLIMAX OF CURIOSITY — an unresolved question, an action cut mid-way, or a revelation stopped just before the reveal.
- FORBIDDEN endings: resolved conclusions, CTA phrases ("theo dõi tiếp", "hãy chú ý"), full explanation of what happened.
- RIGHT pattern: sentence cut before answer ("Và thứ hắn nhìn thấy trong quan tài... chính là—"), or open question ("Nhưng tại sao cô gái đó lại mỉm cười?").

PACING RULE:
- Shot 1 and Shot 2: duration_sec MUST be 2 or 3.
- Shot 1 and Shot 2: narration_text MUST be 12 words or fewer — TTS must fit within 2–3 seconds.
- Shots 3–8: duration_sec 6 for standard, 8 for climactic action.
- Shots 3–8: narration_text MUST be 20–30 words (2–3 sentences) — TTS must fill 7–10 seconds each.
- TOTAL narration_text across all 8 shots must produce at least 60 seconds of TTS. At ~3 words/second Vietnamese TTS (edge-tts vi-VN-HoaiMyNeural), that means at least 180 words total across all shots.

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
- Character appearance descriptors (age, hair, eyes, physique, clothing of specific characters) — e.g., "old daoist", "black-haired girl", "young man in white robes". Use role/action tags instead: "daoist figure", "warrior silhouette", "female protagonist"
- Adverbs or qualifiers ("mysteriously", "fiercely", "inadvertently")
- NSFW or suggestive tags: alluring, seductive, suggestive, provocative, cleavage, navel, bare skin, skinny, undressing, erotic, sensual, mysterious aura, bedroom eyes

CLOTHING SAFETY RULE — CRITICAL for all scene_prompt:
- Every scene_prompt with a human figure MUST include at least 2 of: fully clothed, high collar, long sleeves, covered body, modest clothing, traditional attire, armored, formal wear.
- NEVER use tags that expose skin: bare shoulders, bare midriff, open shirt, low cut, tight clothing.
- Prepend every scene_prompt with: "sfw, fully clothed, "

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
- duration_sec: 2 or 3 for shots 1–2; 6 for standard shots, 8 for climactic action shots.
- is_key_shot: Mark EXACTLY 2-3 shots as true — the most action-packed.
- characters: list character names visible. Use [] for scenery-only.

Return JSON:
{
  "title": "string — episode title in Vietnamese",
  "shots": [ { "scene_prompt": "string", "narration_text": "string", "duration_sec": 6, "is_key_shot": false, "characters": [] } ]
}
shots MUST have EXACTLY 8 elements. EXACTLY 2-3 must have is_key_shot=true."""


_HOOK_SYSTEM = """You are a Vietnamese short video scriptwriter for TikTok/YouTube Shorts.
Write EXACTLY 1 hook shot that opens an episode.

RULES:
- Must open with a SPECIFIC ACTION or SHOCKING NARRATION — no backstory, no scene-setting.
- Keep chronological order — do NOT flash-forward. Reframe the very first scene from a shocking angle.
- narration_text MUST be 10 words or fewer.
- scene_prompt: comma-separated tags for Stable Diffusion (English only, no character names, no sentences).
- scene_prompt MUST end with: "anime style, manhua art style, dramatic lighting, detailed background, no text, no watermarks"

Return JSON:
{ "scene_prompt": "string", "narration_text": "string", "duration_sec": 3, "is_key_shot": false, "characters": [] }"""


@retry(
    stop=stop_after_attempt(settings.llm_max_retries),
    wait=wait_exponential(min=2, max=15),
)
@retry(
    stop=stop_after_attempt(settings.llm_max_retries),
    wait=wait_exponential(min=2, max=15),
)
def _generate_hook_shot(arc_text: str, episode_num: int) -> ShotScript:
    """Ask LLM to write exactly 1 hook shot for the episode opening."""
    prompt = (
        f"Write 1 hook shot for Episode {episode_num} opening.\n\n"
        f"Story context:\n{arc_text}"
    )
    raw = ollama_client.generate_json(
        prompt=prompt, system=_HOOK_SYSTEM, temperature=0.8
    )
    return ShotScript(**raw)


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
    Always generates a fresh hook shot (shot 0) regardless of carry-over state.
    """
    # Build arc_text upfront — needed for both LLM path and carryover hook regeneration
    arc = load_arc_overview(episode_num)
    chunks = _load_chunk_summaries(episode_num)
    if chunks:
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

    # 1. Load carry-over from previous episode
    carryover = _load_carryover(episode_num - 1)
    all_shots: List[ShotScript] = list(carryover)
    raw_title = f"Tập {episode_num}"

    # 2. If carry-over doesn't fill the episode, call LLM for new shots
    if len(all_shots) < _MAX_SHOTS_PER_EPISODE:
        logger.info("Writing script | episode={}", episode_num)
        raw = _write_raw(arc_text, episode_num)
        raw_title = raw.get("title", raw_title)
        new_shots = [ShotScript(**s) for s in raw["shots"]]
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

    # 4. Always force-regenerate shot 0 as a fresh hook for this episode
    logger.info("Generating hook shot | episode={}", episode_num)
    try:
        hook_shot = _generate_hook_shot(arc_text, episode_num)
        episode_shots[0] = hook_shot
        logger.debug("Hook shot injected | episode={} narration={!r}", episode_num, hook_shot.narration_text)
    except Exception:
        logger.warning("Hook shot generation failed, keeping carry-over shot 0 | episode={}", episode_num)

    # 5. Normalize in correct order — all on episode_shots (not new_shots)
    episode_shots = _normalize_duration(episode_shots, episode_num)
    episode_shots = _normalize_hook_durations(episode_shots, episode_num)
    episode_shots = _normalize_key_shots(episode_shots, episode_num)

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
    Skip shots 0 and 1 when redistributing — hook durations are enforced separately.
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
    delta = int(round(target - total))  # cast to int — duration_sec is float
    # Distribute delta round-robin on shots[2:] only — leave hook shots untouched
    adjustable = list(range(2, len(shots)))
    if not adjustable:
        adjustable = list(range(len(shots)))
    for i in range(abs(delta)):
        idx = adjustable[i % len(adjustable)]
        current = shots[idx].duration_sec
        adjusted = current + (1 if delta > 0 else -1)
        adjusted = max(4, min(10, adjusted))
        shots[idx] = shots[idx].model_copy(update={"duration_sec": adjusted})

    return shots


def _normalize_hook_durations(shots: List[ShotScript], episode_num: int) -> List[ShotScript]:
    """Enforce shots 0 and 1 duration to max 3s for fast opening pacing."""
    shots = list(shots)
    changed = False
    for i in (0, 1):
        if i < len(shots) and shots[i].duration_sec > 3:
            shots[i] = shots[i].model_copy(update={"duration_sec": 3})
            changed = True
    if changed:
        logger.debug(
            "Hook norm | shot0={}s shot1={}s | episode={}",
            shots[0].duration_sec,
            shots[1].duration_sec if len(shots) > 1 else "n/a",
            episode_num,
        )
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
