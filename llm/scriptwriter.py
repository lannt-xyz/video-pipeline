import json
from pathlib import Path
from typing import List

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from llm.client import ollama_client
from llm.summarizer import load_arc_overview
from models.schemas import CameraFlow, EpisodeScript, ShotScript

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
- Use the character names provided in the Characters list when they are clearly the character in the scene.
- Shots must connect: the last sentence of shot N sets up shot N+1.
- Voice: first-person narrator ("Tôi..."), present-tense tension.
- FORBIDDEN phrases: "mọi chuyện leo thang", "những bí ẩn được hé lộ", "cuộc chiến tiếp tục", "các nhân vật xuất hiện".

CAMERA FLOW — Choose the right camera movement for each shot:
- "wide_to_close": Camera starts wide, zooms to close-up. Use for dialogue, narrative exposition, showing environment then focusing on character.
- "close_to_wide": Camera starts close, pulls back to reveal wider scene. Use for twists, revelations, surprise reveals.
- "pan_across": Camera pans horizontally across the scene. Use for action/fight sequences, chases, showing multiple characters engaging.
- "detail_reveal": Extreme close-up on a detail, then pulls to medium shot. Use for horror, mystery, discovering clues, creepy objects.
- "static_close": Single close-up frame, no movement needed. Use ONLY for hook shots (shots 1-2, duration ≤3s).
- "static_wide": Single wide frame. Use ONLY for brief establishing scene-only shots with no characters.

CAMERA FLOW GUIDELINES:
- Shot 1 (hook): MUST be "static_close" or "detail_reveal"
- Shot 2 (hook): MUST be "static_close" or "wide_to_close"
- Action/fight shots: prefer "pan_across"
- Twist/revelation shots: prefer "close_to_wide"
- Horror/discovery shots: prefer "detail_reveal"
- Standard narrative/dialogue: prefer "wide_to_close"
- Shot 8 (cliffhanger): prefer "detail_reveal" or "close_to_wide"

SCENE PROMPT RULES — CRITICAL: ComfyUI uses Stable Diffusion, which requires comma-separated tags, NOT sentences.
scene_prompt must be a SHORT TAG LIST only. Structure:
  [specific location], [specific action/pose], [foreground element], [background element], [specific lighting], anime style, manhua art style, no text, no watermarks

SCENE PROMPT QUALITY — Each scene_prompt MUST contain ALL of the following:
1. At least 1 SPECIFIC LOCATION tag (NOT generic): "dimly lit coffin shop with wooden shelves", NOT just "coffin shop interior"
2. At least 1 SPECIFIC ACTION or POSE tag: "figure lunging forward with wooden staff", NOT just "action pose" or "fighting"
3. FOREGROUND LAYER (close to camera, 2+ elements): "cracked stone tablet and scattered ritual candles", "glowing talisman in hand near weathered pillar"
4. MIDGROUND LAYER: where character stands — "stone staircase of ruined temple", "muddy excavation pit floor"
5. BACKGROUND LAYER (depth, 2+ elements): "ancient crumbling gateway half-buried in fog, distant pine-covered mountain ridges" — always TWO background entities
6. SPECIFIC LIGHTING description: "dim oil lantern casting long shadows on walls", NOT just "dramatic lighting"

SCENE PROMPT EXAMPLES:
WRONG: "coffin shop interior, intense fight scene, wooden shelves, dim lantern light, dynamic action pose, anime style, manhua art style, dramatic lighting, detailed background, no text, no watermarks"
→ Problems: "intense fight scene" is vague (WHO doing WHAT?), "dynamic action pose" is generic, "dramatic lighting" has no specifics

RIGHT: "dimly lit coffin shop with hanging red paper, figure lunging forward with wooden sword, shattered pottery on floor, rows of dark coffins receding into shadow, flickering oil lamp casting long orange shadows, anime style, manhua art style, no text, no watermarks"

WRONG: "mountain monastery courtyard, daoist training, morning mist, stone staircase, bamboo grove, anime style, manhua art style, dramatic lighting, detailed background, no text, no watermarks"
→ Problems: "daoist training" is too vague, no foreground object, "dramatic lighting" not specific

RIGHT: "stone courtyard at mountain temple summit, figure in meditation stance with hands raised, crumbling stone incense burner in foreground, bamboo forest and mist-covered peaks in background, pale golden dawn light filtering through clouds, anime style, manhua art style, no text, no watermarks"

WRONG: "outdoor gravesite, ancient tomb excavation, night scene, eerie moonlight, dark soil, stone ruins, anime style, manhua art style, dramatic lighting, detailed background, no text, no watermarks"
RIGHT: "muddy excavation pit with exposed stone sarcophagus, figure kneeling and prying open stone lid, scattered ritual candles on wet earth, ancient crumbling gateway half-buried behind, cold blue moonlight with drifting fog wisps, anime style, manhua art style, no text, no watermarks"

FORBIDDEN in scene_prompt:
- English sentences or clauses ("He walks into...", "Fifteen years later...")
- Vietnamese words anywhere
- Character names (Diệp Thiếu Dương, Tiểu Mã, etc.) — character appearance is handled separately
- Character appearance descriptors (age, hair, eyes, physique, clothing of specific characters) — e.g., "old daoist", "black-haired girl", "young man in white robes". Use role/action tags instead: "daoist figure", "warrior silhouette", "female protagonist"
- Adverbs or qualifiers ("mysteriously", "fiercely", "inadvertently")
- NSFW or suggestive tags: alluring, seductive, suggestive, provocative, cleavage, navel, bare skin, skinny, undressing, erotic, sensual, mysterious aura, bedroom eyes
- Generic placeholder tags: "dramatic lighting", "detailed background", "action pose", "fight scene" — be SPECIFIC
- Extreme close-up framing tags: "extreme close-up", "extreme close-up detail", "face close-up", "macro shot" — these erase background context entirely. Use "medium close-up", "medium shot", or "wide shot" instead

CLOTHING SAFETY RULE — CRITICAL for all scene_prompt:
- Every scene_prompt with a human figure MUST include at least 2 of: fully clothed, high collar, long sleeves, covered body, modest clothing, traditional attire, armored, formal wear.
- NEVER use tags that expose skin: bare shoulders, bare midriff, open shirt, low cut, tight clothing.
- Prepend every scene_prompt with: "sfw, fully clothed, "

REQUIRED in scene_prompt:
- At least 1 specific LOCATION tag with visual detail
- At least 1 specific ACTION or POSE tag with object/weapon/gesture
- At least 1 foreground element (close to camera)
- At least 1 background element (depth/environment)
- At least 1 specific lighting description
- End ALWAYS with: "anime style, manhua art style, no text, no watermarks"

OTHER RULES:
- duration_sec: 2 or 3 for shots 1–2; 6 for standard shots, 8 for climactic action shots.
- is_key_shot: Mark EXACTLY 2-3 shots as true — the most action-packed.
- characters: CRITICAL — list the EXACT character names (from the provided Characters list) whose body, face, or silhouette is PHYSICALLY VISIBLE in the shot. If the scene_prompt describes only environment, objects, or atmosphere with NO human figure present, use []. DO NOT add a character just because they are the narrator or implied. MAXIMUM 2 characters per shot — never list 3 or more.

Return JSON:
{
  "title": "string — episode title in Vietnamese",
  "shots": [ { "scene_prompt": "string", "narration_text": "string", "duration_sec": 6, "is_key_shot": false, "characters": ["Tên Nhân Vật"], "camera_flow": "wide_to_close" } ]
}
shots MUST have EXACTLY 8 elements. EXACTLY 2-3 must have is_key_shot=true.
camera_flow MUST be one of: "wide_to_close", "close_to_wide", "pan_across", "detail_reveal", "static_close", "static_wide"."""


_HOOK_SYSTEM = """You are a Vietnamese short video scriptwriter for TikTok/YouTube Shorts.
Write EXACTLY 1 hook shot that opens an episode.

RULES:
- Must open with a SPECIFIC ACTION or SHOCKING NARRATION — no backstory, no scene-setting.
- Keep chronological order — do NOT flash-forward. Reframe the very first scene from a shocking angle.
- narration_text MUST be 10 words or fewer.
- scene_prompt: comma-separated tags for Stable Diffusion (English only, no character names, no sentences).
- scene_prompt MUST contain: 1 specific location with detail, 1 specific action/pose, 1 foreground element, 1 background element, 1 specific lighting.
- scene_prompt MUST end with: "anime style, manhua art style, no text, no watermarks"
- camera_flow: MUST be "static_close" or "detail_reveal" for hook shots.

Return JSON:
{ "scene_prompt": "string", "narration_text": "string", "duration_sec": 3, "is_key_shot": false, "characters": ["Tên Nhân Vật"], "camera_flow": "static_close" }"""


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


def _build_characters_ref(arc_char_names: List[str]) -> str:
    """Build a canonical character reference string for LLM prompts.

    Enriches arc character names with their aliases so the LLM knows exactly
    which string to use in the `characters` list of each shot.
    Format: "Canonical Name (also: alias1, alias2), ..."
    Falls back to the arc name if no matching character JSON is found.
    """
    from llm.character_extractor import load_all_characters

    all_chars = load_all_characters()
    # Build a lookup: any known name/alias → Character
    lookup: dict = {}
    for c in all_chars:
        lookup[c.name] = c
        for a in c.alias:
            lookup[a] = c

    parts: List[str] = []
    seen_canonical: set[str] = set()
    for arc_name in arc_char_names:
        char = lookup.get(arc_name)
        if char is None:
            parts.append(arc_name)
        elif char.name not in seen_canonical:
            seen_canonical.add(char.name)
            if char.alias:
                parts.append(f"{char.name} (also: {', '.join(char.alias)})")
            else:
                parts.append(char.name)
    return ", ".join(parts)


def write_episode_script(episode_num: int) -> EpisodeScript:
    """Generate shot script. Caps at _MAX_SHOTS_PER_EPISODE; excess saved as carry-over.
    First checks carry-over from previous episode before calling LLM.
    Always generates a fresh hook shot (shot 0) regardless of carry-over state.
    """
    # Build arc_text upfront — needed for both LLM path and carryover hook regeneration
    arc = load_arc_overview(episode_num)
    chunks = _load_chunk_summaries(episode_num)
    # Enrich character names with aliases so LLM uses canonical names in shot.characters
    chars_ref = _build_characters_ref(arc.characters_in_episode)

    # Always include arc-level context (key events give LLM the full episode arc)
    arc_event_text = "\n".join(
        f"Event {i+1}: {e}" for i, e in enumerate(arc.key_events)
    )

    if chunks:
        # No character limit on chunk summary — truncating loses the climax
        scenes_text = "\n\n".join(
            f"Scene {i + 1} (chương {c['chapter_start']}-{c['chapter_end']}):\n{c['summary']}"
            for i, c in enumerate(chunks)
        )
        arc_text = (
            f"Episode arc (key events in order):\n{arc_event_text}\n\n"
            f"Detailed scenes:\n\n{scenes_text}\n\n"
            f"Characters: {chars_ref}"
        )
    else:
        arc_text = (
            f"Summary: {arc.arc_summary}\n\n"
            f"ORDERED SCENES:\n{arc_event_text}"
            + f"\n\nCharacters: {chars_ref}"
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
    episode_shots = _normalize_camera_flow(episode_shots, episode_num)
    episode_shots = _backfill_characters(episode_shots, arc.characters_in_episode, episode_num)

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

    Target is inflated by (n-1) × shot_transition_duration to compensate for
    time lost to xfade overlaps in the final video assembly.
    """
    n_transitions = max(0, len(shots) - 1)
    target = settings.target_duration_sec + n_transitions * settings.shot_transition_duration
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


def _normalize_camera_flow(shots: List[ShotScript], episode_num: int) -> List[ShotScript]:
    """Ensure camera_flow is set correctly based on shot position and type.
    - Shots 0-1 (hooks, ≤3s): force static_close if not already static_close or detail_reveal
    - Last shot (cliffhanger): prefer close_to_wide for reveal effect
    """
    shots = list(shots)
    for i in range(len(shots)):
        shot = shots[i]
        # Hook shots should be static or detail_reveal
        if i <= 1 and shot.duration_sec <= 3:
            if shot.camera_flow not in (CameraFlow.STATIC_CLOSE, CameraFlow.DETAIL_REVEAL):
                shots[i] = shot.model_copy(update={"camera_flow": CameraFlow.STATIC_CLOSE})
        # Last shot (cliffhanger) benefits from reveal
        elif i == len(shots) - 1:
            if shot.camera_flow == CameraFlow.WIDE_TO_CLOSE:
                shots[i] = shot.model_copy(update={"camera_flow": CameraFlow.CLOSE_TO_WIDE})
    return shots


def _backfill_characters(
    shots: List[ShotScript],
    arc_characters: List[str],
    episode_num: int,
) -> List[ShotScript]:
    """If LLM left characters empty on most shots, assign from arc_characters.

    Strategy: shots 0-1 (hooks) get the first character; remaining shots cycle
    through arc_characters round-robin.  Pure scenery shots (wide establishing,
    no action words) keep [].
    """
    if not arc_characters:
        return shots

    non_empty = sum(1 for s in shots if s.characters)
    if non_empty >= len(shots) // 2:
        return shots  # LLM already filled enough

    logger.warning(
        "Backfilling characters — LLM left {}/{} shots empty | episode={}",
        len(shots) - non_empty, len(shots), episode_num,
    )

    shots = list(shots)
    main_char = arc_characters[0]
    for i in range(len(shots)):
        if shots[i].characters:
            continue  # keep LLM choice
        # static_wide = intentionally scene-only shot — never backfill characters
        if shots[i].camera_flow == CameraFlow.STATIC_WIDE:
            continue
        if i <= 1:
            # Hook shots: main character
            shots[i] = shots[i].model_copy(update={"characters": [main_char]})
        else:
            # Rotate through available characters, assign 1-2
            char_idx = (i - 2) % len(arc_characters)
            assigned = [arc_characters[char_idx]]
            if len(arc_characters) > 1 and i % 3 == 0:
                second = arc_characters[(char_idx + 1) % len(arc_characters)]
                assigned.append(second)
            shots[i] = shots[i].model_copy(update={"characters": assigned})

    return shots
