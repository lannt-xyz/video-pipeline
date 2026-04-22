import json
import re
from pathlib import Path
from typing import List

from loguru import logger
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import settings
from llm.client import script_client as ollama_client
from llm.client import scene_prompt_client
from llm.summarizer import load_arc_overview
from models.schemas import CameraFlow, EpisodeScript, ShotScript

_SCRIPTWRITER_SYSTEM = """You are a Vietnamese short video scriptwriter for TikTok/YouTube Shorts.
GENRE: HORROR / SUPERNATURAL / MYSTERY — this is a Maoshan exorcism story. Every shot must evoke dread, curiosity, or supernatural tension.
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

PACING RULE — STRICTLY ENFORCED:
- Shot 1 and Shot 2: duration_sec MUST be 2 or 3.
- Shot 1 and Shot 2: narration_text MUST be 10 words or fewer — TTS must fit within 2–3 seconds.
- Shots 3–8: duration_sec = 8 for standard shots, 10 for climactic action shots.
- Shots 3–8: narration_text MUST be 25–35 words (2–4 sentences) — TTS must fill 8–11 seconds each.
- TOTAL narration_text across ALL 8 shots MUST be at least 180 words. At ~3 words/second Vietnamese TTS (edge-tts vi-VN-HoaiMyNeural), that means at least 60 seconds of TTS.
- COUNT YOUR WORDS before outputting each shot. If narration_text for shot 3–8 is fewer than 25 words, REWRITE IT.

NARRATION LENGTH EXAMPLES:
WRONG (too short for a 8s shot): "Lão đạo sĩ này sẽ mang Diệp Thiếu Dương về Mao Sơn để dạy nó đạo pháp." — 15 words, only ~5s TTS, leaves 3s of silence.
RIGHT (correct length for a 8s shot): "Thanh Vân Tử nhìn thẳng vào mắt Diệp Đại Công, giọng trầm xuống: Đứa trẻ này có căn cơ không bình thường. Ta sẽ đưa hắn lên Mao Sơn, dạy đạo pháp, rèn chân thân. Nhưng đây là con đường không thể quay đầu." — 42 words, ~14s TTS. ✓
WRONG (too short for a 9s shot): "Thi Du Cao là một loại độc thi được sử dụng trong cổ thuật." — 13 words, only ~4s TTS.
RIGHT (correct length for a 9s shot): "Thanh Vân Tử chậm rãi giải thích: Thi Du Cao không phải là bệnh, mà là một loại thi độc từ cổ thuật. Nó xâm nhập vào thi thể người chết, khiến xác không thể phân hủy, và dần biến thành một thứ nguy hiểm hơn bất kỳ con quỷ nào ta từng gặp." — 47 words, ~16s TTS. ✓

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

HORROR ATMOSPHERE — CRITICAL for this story genre:
Every scene_prompt MUST include at least 1 HORROR/SUPERNATURAL ATMOSPHERE element from this palette:
- LIGHTING: "sickly green glow from below", "blood-red candle flame", "cold blue moonlight piercing through cracks", "flickering torch casting distorted shadows on walls", "pale ghostly luminescence", "dim amber candlelight barely reaching corners"
- ENVIRONMENT: "crumbling moss-covered walls", "cobwebs hanging from ceiling beams", "mist creeping along the ground", "twisted dead tree silhouettes", "rusted iron chains on stone wall", "dark stain spreading on floor"
- SUPERNATURAL: "faint ghostly silhouette in background", "glowing ritual symbols on ground", "unnatural fog swirling around ankles", "floating dust particles in shaft of pale light", "eerie green spirit wisps"
- TENSION OBJECTS: "scattered yellowed talisman papers", "overturned ritual candles dripping wax", "cracked ancient mirror reflecting distorted image", "half-open coffin lid with darkness inside", "bloody handprint on wall"
Choose elements appropriate to the specific scene — graveyard scenes get fog/moonlight, indoor scenes get candles/shadows, ritual scenes get glowing symbols/talismans.
Do NOT use the word "mysterious" — use specific visual descriptors instead.

FORBIDDEN in scene_prompt:
- English sentences or clauses ("He walks into...", "Fifteen years later...")
- Vietnamese words anywhere
- Character names (Diệp Thiếu Dương, Tiểu Mã, etc.) — character appearance is handled separately
- Character appearance descriptors (age, hair, eyes, physique, clothing of specific characters) — e.g., "old daoist", "black-haired girl", "young man in white robes". Use role/action tags instead: "daoist figure", "warrior silhouette", "female protagonist"
- Adverbs or qualifiers ("mysteriously", "fiercely", "inadvertently")
- NSFW or suggestive tags: alluring, seductive, suggestive, provocative, cleavage, navel, bare skin, skinny, undressing, erotic, sensual, bedroom eyes
- Generic placeholder tags: "dramatic lighting", "detailed background", "action pose", "fight scene" — be SPECIFIC
- Extreme close-up framing tags: "extreme close-up", "extreme close-up detail", "face close-up", "macro shot" — these erase background context entirely. Use "medium close-up", "medium shot", or "wide shot" instead

CLOTHING SAFETY:
- FORBIDDEN: bare skin, exposed midriff, cleavage, tight clothing, suggestive poses.
- Do NOT add clothing/style/safety tags (e.g., sfw, fully clothed, anime style, no watermarks) — they are injected automatically downstream.

REQUIRED in scene_prompt — USE ALL TAG POSITIONS FOR ACTUAL CONTENT:
- At least 1 specific LOCATION tag with visual detail
- At least 1 specific ACTION or POSE tag with object/weapon/gesture
- At least 1 foreground element (close to camera)
- At least 1 background element (depth/environment)
- At least 1 specific lighting description
- Do NOT include "anime style", "manhua art style", "no text", "no watermarks", "sfw", "fully clothed" — these are added automatically

OTHER RULES:
- duration_sec: 2 or 3 for shots 1–2; 8 for standard shots 3-8, 10 for climactic action shots.
- is_key_shot: Mark EXACTLY 2-3 shots as true — the most action-packed.
- characters: CRITICAL — list the EXACT character names (from the provided Characters list) whose body, face, or silhouette is PHYSICALLY VISIBLE in the shot. If the scene_prompt describes only environment, objects, or atmosphere with NO human figure present, use []. DO NOT add a character just because they are the narrator or implied. MAXIMUM 2 characters per shot — never list 3 or more.

SCENE_ID RULES — CRITICAL for visual consistency:
- scene_id is a short snake_case English label for the physical location (e.g. "coffin_shop", "temple_gate", "dark_forest", "excavation_pit").
- Assign the SAME scene_id to ALL shots that take place in the SAME physical location within this episode.
- When the story moves to a new location, assign a NEW scene_id.
- scene_id MUST be consistent with narration: if shots 3, 4, 5 all happen inside the coffin shop, all three get scene_id="coffin_shop".
- Shots with the same scene_id will share base environment tags at image-generation time — so accuracy matters.

Return JSON:
{
  "title": "string — episode title in Vietnamese",
  "shots": [ { "scene_prompt": "string", "narration_text": "string", "duration_sec": 6, "is_key_shot": false, "characters": ["Tên Nhân Vật"], "camera_flow": "wide_to_close", "scene_id": "location_slug" } ]
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


def _coerce_shot_item(item: object, episode_num: int, index: int) -> dict | None:
    """Coerce a single shot item from Ollama output into a plain dict.

    Ollama occasionally returns shots as JSON strings instead of objects.
    Returns None when the item cannot be interpreted as a shot dict.
    """
    if isinstance(item, dict):
        return item

    if isinstance(item, str):
        text = item.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                logger.debug(
                    "Shot {} was a JSON string — parsed as dict | episode={}",
                    index, episode_num,
                )
                return parsed
        except json.JSONDecodeError:
            pass

        logger.warning(
            "Shot {} is a plain string (not JSON object) — skipping | episode={} value={!r}",
            index, episode_num, text[:120],
        )
        return None

    logger.warning(
        "Shot {} has unexpected type {}; skipping | episode={}",
        index, type(item).__name__, episode_num,
    )
    return None


@retry(
    stop=stop_after_attempt(settings.llm_max_retries),
    wait=wait_exponential(min=2, max=15),
    retry=retry_if_exception_type((json.JSONDecodeError, ValueError)),
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
    shot_data = raw if isinstance(raw, dict) else _coerce_shot_item(raw, episode_num, 0)
    if shot_data is None:
        raise ValueError(f"Hook shot response is not a valid dict | episode={episode_num}")
    return ShotScript(**shot_data)


@retry(
    stop=stop_after_attempt(settings.llm_max_retries),
    wait=wait_exponential(min=2, max=15),
    retry=retry_if_exception_type((json.JSONDecodeError, ValueError)),
)
def _write_raw(arc_text: str, episode_num: int) -> dict:
    prompt = (
        f"Write a video script for Episode {episode_num}.\n\n"
        f"Arc Overview:\n{arc_text}\n\n"
        "Remember: 8-10 shots, EXACTLY 2-3 with is_key_shot=true, "
        "scene_prompt in English, narration_text in Vietnamese.\n"
        "CRITICAL: shots 3-8 MUST each have 25-35 words in narration_text. "
        "Total narration_text across all shots MUST be at least 180 words."
    )
    result = ollama_client.generate_json(
        prompt=prompt, system=_SCRIPTWRITER_SYSTEM, temperature=0.7
    )
    # Reject and retry if total narration is too short (LLM ignored the word-count rules).
    shots = result.get("shots", [])
    if isinstance(shots, list) and shots:
        total_words = sum(
            len(str(s.get("narration_text", "")).split())
            for s in shots
            if isinstance(s, dict)
        )
        if total_words < 150:
            logger.warning(
                "Script rejected: total_words={} < 150, retrying | episode={}",
                total_words, episode_num,
            )
            raise ValueError(f"narration too short: {total_words} words")
    return result


_MAX_SHOTS_PER_EPISODE = 8  # shots that fit in one video; excess flows to next episode
_ID_LIKE_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)+$")

# --------------------------------------------------------------------------- #
#  Narration-to-scene-prompt alignment (LLM rewrite pass)                     #
# --------------------------------------------------------------------------- #
_NARRATION_ALIGN_SYSTEM = """You are a ComfyUI Stable Diffusion prompt engineer for a HORROR/SUPERNATURAL story.
You receive a list of video shot objects. Each shot has:
  - shot_index (int)
  - narration_text (Vietnamese sentence — what the narrator says)
  - scene_prompt (existing English tag list for ComfyUI)

YOUR TASK: Rewrite each "scene_prompt" so it VISUALLY DEPICTS the exact action/character/object/location described in "narration_text", while MAXIMIZING horror/supernatural atmosphere.

EXTRACTION RULES — read narration and extract these 5 elements:
1. WHO: which character role is physically visible (use generic role tags: daoist figure, elder figure, young warrior, female figure, hooded figure — NEVER character names)
2. ACTION: the specific physical action/pose being performed (must be concrete: "figure prying open stone lid", "figure slamming fist on table", "figure running through fog" — NOT "action pose", "performing ritual", "fighting")
3. OBJECT/PROP: key objects mentioned in narration (coffin, dagger, talisman, candles, corpse, compass)
4. LOCATION: specific place described in narration (abandoned temple courtyard, dark excavation pit, candlelit coffin shop interior)
5. HORROR MOOD: choose 1-2 atmosphere tags that match the narration's tension level:
   - Discovery/reveal: "eerie green glow emanating from below", "cold mist creeping along ground"
   - Confrontation: "blood-red candlelight", "distorted shadows stretching on walls"
   - Ritual/supernatural: "glowing ritual symbols on ground", "spirit wisps floating in air"
   - Dread/suspense: "oppressive darkness pressing in", "pale ghostly luminescence from above"

REWRITE RULES:
- Structure: [LOCATION with visual detail], [ACTION/POSE — must match narration action], [foreground object from narration], [background depth 2 elements], [HORROR ATMOSPHERE lighting/mood]
- Do NOT add style/safety metadata tags (sfw, fully clothed, anime style, manhua art style, no text, no watermarks) — they are injected automatically downstream. Use ALL tag positions for visual content.
- ACTION tag MUST reflect what narration_text says is happening — not a standing portrait
- HORROR ATMOSPHERE is MANDATORY: every prompt must have at least 1 eerie/dark/supernatural lighting or mood tag — NEVER use plain "bright daylight" or "warm sunlight" unless the narration explicitly describes a safe daytime scene
- If narration says "prying open coffin lid" → scene_prompt must contain prying/opening action tags
- If narration says "shouting accusation at someone" → scene_prompt must contain accusatory gesture tags
- If narration describes discovery/reveal → scene_prompt must show discovery moment with eerie reveal lighting
- Keep comma-separated tags — NO English sentences, NO Vietnamese words
- FORBIDDEN tags: "action pose", "dynamic pose", "performing ritual", "figure standing", "fight scene", "dramatic lighting", "detailed background"
- FORBIDDEN content: bare skin, exposed midriff, cleavage, tight clothing, suggestive poses

Return a JSON ARRAY (same length as input, same order):
[{"shot_index": 0, "scene_prompt": "rewritten tags..."}, ...]
CRITICAL: Return ONLY the JSON array, no markdown, no explanation."""
# Object tags — appended at the END of scene_prompt (noun/prop hints).
_SCENE_ALIGN_OBJECT_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(dù|ô)\b", re.IGNORECASE), "red umbrella"),
    (re.compile(r"\b(nhang|hương)\b", re.IGNORECASE), "burning incense sticks"),
    (re.compile(r"quan tài|nắp quan", re.IGNORECASE), "red lacquered coffin"),
    (re.compile(r"dao găm|\bdao\b", re.IGNORECASE), "dagger in hand"),
    (re.compile(r"\bđinh\b", re.IGNORECASE), "rusted coffin nails"),
    (re.compile(r"chậu đồng", re.IGNORECASE), "bronze ritual basin"),
    (re.compile(r"bùa|\bphù\b|chu sa", re.IGNORECASE), "yellow talisman paper"),
    (re.compile(r"la bàn", re.IGNORECASE), "feng shui compass"),
    (re.compile(r"xé( toạc)? áo|rách áo", re.IGNORECASE), "torn clothing"),
    (re.compile(r"vết cắn|dấu răng", re.IGNORECASE), "bite mark on shoulder"),
    (re.compile(r"cười điên|cười man|cười điên loạn|cười quỷ|cười dị|cười ác", re.IGNORECASE), "maniacal grin expression with wide unnatural smile"),
    (re.compile(r"gào|thét|\bkhóc\b", re.IGNORECASE), "screaming expression with open mouth"),
    (re.compile(r"kiếm|đao trấn|long tuyền", re.IGNORECASE), "ornate daoist sword in hand"),
    (re.compile(r"thiên sư bài|bài thiên sư", re.IGNORECASE), "ritual celestial master badge in hand"),
    (re.compile(r"hồ lô|bầu hồ lô", re.IGNORECASE), "daoist gourd flask"),
    # Horror/supernatural object tags
    (re.compile(r"hồn|vong|linh hồn|oan hồn", re.IGNORECASE), "faint ghostly silhouette in shadow"),
    (re.compile(r"máu|vết máu|đẫm máu", re.IGNORECASE), "dark blood stain on surface"),
    (re.compile(r"xương|bộ xương|sọ người", re.IGNORECASE), "scattered bones on ground"),
    (re.compile(r"thi thể|xác chết|\bxác\b|thây|tử thi|nữ tử thi", re.IGNORECASE), "pale lifeless corpse with rigid limbs"),
    (re.compile(r"mộ|ngôi mộ|mồ", re.IGNORECASE), "ancient moss-covered grave mound"),
    (re.compile(r"nến|đèn cầy", re.IGNORECASE), "dripping wax candle with flickering flame"),
    (re.compile(r"trắng bệch|trắng nhợt|xanh xao|tái mét", re.IGNORECASE), "deathly pale white skin with blue-grey veins"),
    (re.compile(r"trợn trừng|mắt mở to|mắt trợn", re.IGNORECASE), "wide staring dead eyes with dilated pupils"),
    (re.compile(r"khí âm|âm khí|khí lạnh|hàn khí", re.IGNORECASE), "cold dark mist emanating from body"),
    (re.compile(r"đứa trẻ|đứa bé|hài nhi|trẻ con", re.IGNORECASE), "small child body wrapped in cloth"),
]

# Action/pose tags — inserted EARLY in scene_prompt and replace weak generic poses.
# These ensure ComfyUI draws the specific action, not just a standing portrait.
_SCENE_ALIGN_ACTION_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bxẻng\b|khai quật|\bđào mộ\b|đào đất", re.IGNORECASE),
     "figure crouching and digging with long-handled shovel into dark wet earth"),
    (re.compile(r"\bnạy\b|\bbẩy\b|\bcậy\b|mở nắp|nhấc nắp quan", re.IGNORECASE),
     "figure straining to pry open heavy coffin lid with iron crowbar"),
    (re.compile(r"niệm chú|đọc chú|niệm kinh", re.IGNORECASE),
     "figure kneeling with both arms raised in solemn ritual chant"),
    (re.compile(r"ngã xuống|ngã lăn|đổ người|ngất xỉu", re.IGNORECASE),
     "figure collapsing forward onto muddy ground"),
    (re.compile(r"chỉ thẳng|buộc tội|vạch tội", re.IGNORECASE),
     "figure pointing accusingly finger outstretched at another person"),
    (re.compile(r"kéo áo|giật áo|lột áo", re.IGNORECASE),
     "figure grabbing and pulling shirt of another person forcefully"),
    (re.compile(r"vừa khóc vừa đánh|xông lên.*đánh|đánh.*khóc", re.IGNORECASE),
     "female figure striking while crying in confrontation"),
    (re.compile(r"sờ đỉnh đầu|sờ đầu|đặt tay lên đầu", re.IGNORECASE),
     "elder figure carefully placing palm on child head in examination"),
    (re.compile(r"cắm cờ|chiêu hồn|\bcây cờ\b", re.IGNORECASE),
     "figure planting tall ritual banner staff into wet muddy ground"),
    (re.compile(r"đặt thi thể|đặt xác|đặt lên thi", re.IGNORECASE),
     "figure carefully placing shrouded corpse into open coffin"),
    (re.compile(r"đóng đinh|cắm đinh", re.IGNORECASE),
     "figure hammering ritual iron nails into wooden coffin surface"),
    (re.compile(r"ôm cổ|nuốt.*vật|nhai.*vật", re.IGNORECASE),
     "figure clutching own throat with convulsing hands"),
    (re.compile(r"cắt chỉ đỏ|cắt.*chỉ|dao.*chỉ", re.IGNORECASE),
     "figure slicing red thread with knife in focused stance"),
    (re.compile(r"phát hiện ra|nhìn thấy lần đầu", re.IGNORECASE),
     "dramatic close-up reveal of discovered object in foreground"),
    # Horror/supernatural action rules
    (re.compile(r"bị nhập|nhập vào|chiếm hữu|nhập hồn", re.IGNORECASE),
     "figure convulsing with head thrown back and arms rigid in supernatural possession"),
    (re.compile(r"trừ tà|trừ quỷ|trục quỷ|bắt quỷ", re.IGNORECASE),
     "figure thrusting talisman forward with outstretched arm in exorcism stance"),
    (re.compile(r"run rẩy|rùng mình|sợ hãi|kinh hãi", re.IGNORECASE),
     "figure recoiling in terror with wide eyes and raised defensive hands"),
    (re.compile(r"hiện hình|hiện nguyên|biến mất|tan biến", re.IGNORECASE),
     "translucent spectral form materializing from swirling dark mist"),
    (re.compile(r"nắm chặt|ôm chặt|giữ chặt", re.IGNORECASE),
     "figure clutching tightly with rigid white-knuckled hands"),
    (re.compile(r"nhảy ra|bật dậy|ngồi bật|vùng dậy", re.IGNORECASE),
     "figure lunging out of coffin with arms outstretched"),
    (re.compile(r"cười quỷ|cười dị|cười ác|cười ma|nhe răng", re.IGNORECASE),
     "figure with wide eerie grin showing teeth in unsettling expression"),
]

# Generic pose tags that get REPLACED when a specific action rule fires.
_WEAK_POSE_TAGS: frozenset[str] = frozenset([
    "figure standing",
    "figure performing ritual",
    "standing figure",
    "ritual figure",
    "action pose",
    "figure in ceremonial stance",
    "dynamic pose",
    "action scene",
    "figure and ritual",
    "performing ritual",
])

# Tags that are purely structural/style — skip when extracting environment anchors.
_PROMPT_SKIP_TAGS: frozenset[str] = frozenset([
    "sfw", "fully clothed", "high collar", "long sleeves", "covered body",
    "modest clothing", "traditional attire", "armored", "formal wear",
    "anime style", "manhua art style", "no text", "no watermarks",
    "wide establishing shot", "full scene view", "environment focus",
    "no characters in foreground",
])

# --------------------------------------------------------------------------- #
#  Visual Brief Enrichment — extraction prompt                                #
# --------------------------------------------------------------------------- #
_VISUAL_BRIEF_SYSTEM = """You are a visual director for a Vietnamese horror/supernatural short video.
You receive a batch of shot descriptions. For each shot you will extract structured visual information
from the narration_text, which is in Vietnamese.

YOUR TASK: For each shot, extract a structured visual brief that a Python script will use to
build a ComfyUI tag list. You are NOT writing tags — you are extracting semantic meaning.

FIELDS TO EXTRACT:
- subjects: List of role tags (max 2). Must be generic roles, NOT character names.
  subjects[0] MUST be the PRIMARY subject (the one performing the action).
  Example: ["hooded daoist figure", "kneeling young warrior"]
- action: ONE specific, observable physical action. Must contain a concrete verb + direction/result.
  GOOD: "figure prying open stone coffin lid with iron crowbar"
  GOOD: "figure turning head sharply toward distant sound"
  GOOD: "elder figure slamming fist onto wooden table"
  BAD: "performing ritual", "looking around", "standing", "mysterious gesture", "conducting ceremony"
  BLACKLIST — NEVER use these words in action: ritual, ceremony, pose, scene, performing, conducting
  If narration describes DISCOVERY ("nhìn thấy", "phát hiện") → action = the revealing moment
  If narration describes ACCUSATION ("chỉ mặt", "buộc tội") → action = accusatory gesture with outstretched arm
  If narration is PURELY environment/atmosphere with NO clear character action → use static pose or
    environment motion (e.g. "figure crouching motionless behind stone pillar", "leaves drifting through empty courtyard")
- setting: Physical location with ONE visual detail. E.g. "dimly lit coffin shop with rows of dark wooden coffins"
  MUST be a descriptive English phrase. NEVER use an identifier or slug (e.g. "dark_forest", "mountain_road" are FORBIDDEN — write "moonlit forest path with ancient stone graves" instead).
- key_objects: List of specific props visible in the scene (max 4). Concrete nouns only.
  Example: ["glowing talisman paper", "ritual candles", "iron chains on wall"]
- mood_lighting: MUST use format "light source + color palette + effect".
  GOOD: "dim amber candle light, teal shadow palette, volumetric fog drifting along floor"
  GOOD: "cold blue moonlight, deep violet shadows, mist creeping along stone floor"
  BAD: "spooky lighting", "dark atmosphere", "dramatic light"
  This MUST be horror-appropriate. No warm sunlight unless narration explicitly describes daytime safety.
- composition: Camera framing tag if obvious from narration. Otherwise leave empty string "".
  Examples: "medium close-up", "wide establishing shot", "medium shot"

INPUT FORMAT: JSON array of shots, each with: shot_index, narration_text, scene_id, characters

OUTPUT FORMAT: JSON array, same length as input, same order:
[{"shot_index": 0, "subjects": [...], "action": "...", "setting": "...", "key_objects": [...], "mood_lighting": "...", "composition": ""}]

CRITICAL: Return ONLY the JSON array, no markdown, no explanation."""

# Map camera_flow values to composition tags when brief.composition is empty.
_CAMERA_FLOW_TO_COMPOSITION: dict[str, str] = {
    "static_close": "medium close-up",
    "detail_reveal": "medium close-up",
    "static_wide": "wide establishing shot",
    "wide_to_close": "",
    "close_to_wide": "",
    "pan_across": "",
}

# Tags in synthesized prompts are capped at this count to leave buffer for
# rule-based _align_scene_prompt_with_narration() pass.
_SYNTHESIS_MAX_TAGS = 16


def _subject_already_in_action(subject: str, action: str) -> bool:
    """Check if the exact subject phrase is a substring of the action string."""
    return subject.lower().strip() in action.lower()


def _synthesize_scene_prompt(brief: "ShotVisualBrief", shot: ShotScript) -> str:
    """Deterministic Python synthesis of a ComfyUI tag list from a visual brief.

    Tag order: [composition+setting] → [action] → [key_objects] → [mood_lighting] → [subjects]
    Caps at _SYNTHESIS_MAX_TAGS with priority-based dropping.
    No LLM involved.
    """
    from models.schemas import ShotVisualBrief  # local import avoids circular at module-level

    # Resolve composition from brief or map from camera_flow.
    composition = brief.composition.strip()
    if not composition:
        composition = _CAMERA_FLOW_TO_COMPOSITION.get(shot.camera_flow.value, "")

    # Build ordered candidate tag groups (by priority).
    # Priority: action > setting > subjects[0] > mood_lighting > key_objects > subjects[1]
    # These first four are NEVER dropped.
    never_drop: list[str] = []
    if composition:
        never_drop.append(composition)
    never_drop.append(brief.setting)
    never_drop.append(brief.action)

    # Primary subject — never drop.
    primary_subject = brief.subjects[0] if brief.subjects else ""
    if primary_subject and not _subject_already_in_action(primary_subject, brief.action):
        never_drop.append(primary_subject)

    never_drop.append(brief.mood_lighting)

    # Key objects — deduplicated against setting, wrapped with weight.
    setting_lower = brief.setting.lower()
    key_object_tags: list[str] = []
    for obj in brief.key_objects[:4]:
        obj = obj.strip()
        if not obj:
            continue
        if obj.lower() in setting_lower:
            continue  # already represented in setting
        key_object_tags.append(f"({obj}:1.15)")

    # Secondary subject — can be dropped when budget exhausted.
    secondary_subject = ""
    if len(brief.subjects) > 1:
        sub = brief.subjects[1].strip()
        if sub and not _subject_already_in_action(sub, brief.action):
            secondary_subject = sub

    # Fill up to _SYNTHESIS_MAX_TAGS.
    tags: list[str] = list(never_drop)
    remaining = _SYNTHESIS_MAX_TAGS - len(tags)

    for obj_tag in key_object_tags:
        if remaining <= 0:
            break
        tags.append(obj_tag)
        remaining -= 1

    if secondary_subject and remaining > 0:
        tags.append(secondary_subject)

    return ", ".join(t for t in tags if t)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=5),
    retry=retry_if_exception_type((json.JSONDecodeError, ValueError)),
)
def _extract_visual_briefs(
    shots: List[ShotScript],
    episode_num: int,
) -> List[ShotScript]:
    """LLM extraction pass: convert narration_text to ShotVisualBrief for each shot.

    Shots with duration_sec <= 2 are skipped — their visual_brief stays None.
    3-second hook shots ARE processed so they receive proper English scene_prompts.
    scene_id is NOT sent in the payload to prevent the LLM from copying the slug
    into setting/composition fields.
    On full failure, returns shots unchanged so the caller can fallback.
    """
    from models.schemas import ShotVisualBrief

    # Build payload — skip only very-short shots (<=2s) that have too little narration
    # to yield a useful brief.  3s hook shots ARE processed so they get an English
    # ComfyUI prompt instead of whatever the LLM scriptwriter happened to produce.
    # scene_id is intentionally NOT sent to prevent the LLM from cargo-culting the
    # slug (e.g. "dark_forest") as the setting or composition value.
    payload = []
    for i, shot in enumerate(shots):
        if shot.duration_sec <= 2:
            continue  # visual_brief stays None for very-short shots
        payload.append({
            "shot_index": i,
            "narration_text": shot.narration_text,
            "characters": shot.characters,
        })

    if not payload:
        logger.debug("No eligible shots for visual brief extraction | episode={}", episode_num)
        return shots

    prompt = (
        f"Extract visual briefs for Episode {episode_num} shots.\n\n"
        f"Shots:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    try:
        raw = scene_prompt_client.generate_json(
            prompt=prompt, system=_VISUAL_BRIEF_SYSTEM, temperature=0.3
        )
    except Exception as exc:
        logger.warning(
            "Visual brief extraction LLM call failed ({}) | episode={}", exc, episode_num
        )
        return shots

    if not isinstance(raw, list):
        if isinstance(raw, dict):
            for key in ("shots", "briefs", "result", "items"):
                if isinstance(raw.get(key), list):
                    raw = raw[key]
                    break
        if not isinstance(raw, list):
            logger.warning(
                "Visual brief extraction returned non-list (type={}) | episode={}",
                type(raw).__name__, episode_num,
            )
            return shots

    shots = list(shots)
    populated = 0
    empty_action = 0
    empty_objects = 0
    empty_mood = 0

    for item in raw:
        if not isinstance(item, dict):
            continue
        idx = item.get("shot_index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(shots):
            continue
        try:
            setting_val = item.get("setting") or ""
            composition_val = item.get("composition") or ""

            # Guard: reject settings that look like identifier slugs (e.g. "dark_forest").
            # The LLM must produce a descriptive phrase, not an internal ID.
            if re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)+", setting_val.strip().lower()):
                logger.warning(
                    "Visual brief setting looks like a slug (%r) — skipping brief | episode={} shot={}",
                    setting_val, episode_num, idx,
                )
                continue

            # Guard: clear composition if it also looks like a slug.
            if re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)+", composition_val.strip().lower()):
                logger.warning(
                    "Visual brief composition looks like a slug (%r) — clearing | episode={} shot={}",
                    composition_val, episode_num, idx,
                )
                composition_val = ""

            brief = ShotVisualBrief(**{
                "subjects": item.get("subjects") or [],
                "action": item.get("action") or "",
                "setting": setting_val,
                "key_objects": item.get("key_objects") or [],
                "mood_lighting": item.get("mood_lighting") or "",
                "composition": composition_val,
            })
            shots[idx] = shots[idx].model_copy(update={"visual_brief": brief})
            populated += 1

            # Track quality metrics.
            if not brief.action:
                empty_action += 1
            if not brief.key_objects:
                empty_objects += 1
            if not brief.mood_lighting:
                empty_mood += 1

            logger.debug(
                "Visual brief extracted | episode={} shot={} action={!r} objects={} mood={!r}",
                episode_num, idx, brief.action[:60], brief.key_objects, brief.mood_lighting[:60],
            )

            if not brief.action:
                logger.warning(
                    "Visual brief has empty action | episode={} shot={}", episode_num, idx
                )
            if not brief.key_objects:
                logger.warning(
                    "Visual brief has empty key_objects | episode={} shot={}", episode_num, idx
                )
            if not brief.mood_lighting:
                logger.warning(
                    "Visual brief has empty mood_lighting | episode={} shot={}", episode_num, idx
                )

        except Exception as exc:
            logger.warning(
                "Failed to parse ShotVisualBrief for shot={} ({}) — skipping | episode={}",
                idx, exc, episode_num,
            )

    logger.info(
        "Visual brief quality | episode={} populated={} empty_action={} empty_objects={} empty_mood={}",
        episode_num, populated, empty_action, empty_objects, empty_mood,
    )
    return shots


def _synthesize_scene_prompts_from_briefs(
    shots: List[ShotScript],
    episode_num: int,
) -> List[ShotScript]:
    """Iterate all shots; synthesize scene_prompt from visual_brief when available.

    Shots with visual_brief=None keep their existing scene_prompt unchanged.
    """
    shots = list(shots)
    synthesized = 0
    for i, shot in enumerate(shots):
        if shot.visual_brief is None:
            continue
        old_prompt = shot.scene_prompt
        new_prompt = _synthesize_scene_prompt(shot.visual_brief, shot)
        shots[i] = shot.model_copy(update={"scene_prompt": new_prompt})
        synthesized += 1
        logger.debug(
            "scene_prompt synthesized | episode={} shot={} old={!r} new={!r}",
            episode_num, i, old_prompt[:80], new_prompt[:80],
        )

    logger.info(
        "Visual brief synthesis done | episode={} synthesized={}/{}",
        episode_num, synthesized, len(shots),
    )
    return shots


def _build_scene_prompts_from_narration(
    shots: List[ShotScript],
    episode_num: int,
) -> List[ShotScript]:
    """Two-pass replacement for _rewrite_scene_prompts_from_narration.

    Pass 1: LLM extraction — narration_text → ShotVisualBrief (structured semantics)
    Pass 2: Python synthesis — ShotVisualBrief → ComfyUI tag list (no LLM)

    Falls back to _rewrite_scene_prompts_from_narration() if extraction yields 0 briefs.
    """
    shots = _extract_visual_briefs(shots, episode_num)
    populated = sum(1 for s in shots if s.visual_brief is not None)
    if populated == 0:
        logger.warning(
            "Visual brief extraction yielded 0 results — falling back to rewrite pass | episode={}",
            episode_num,
        )
        return _rewrite_scene_prompts_from_narration(shots, episode_num)
    return _synthesize_scene_prompts_from_briefs(shots, episode_num)


# Location markers — used to identify the single "place" tag in a prompt.
_LOCATION_MARKERS: tuple[str, ...] = (
    "shop", "temple", "pit", "courtyard", "cave", "room", "forest",
    "mountain", "street", "house", "inn", "tomb", "coffin", "village",
    "shrine", "hall", "chamber", "corridor", "roof", "bridge", "cliff",
    "staircase", "gate", "field", "alley", "market", "dock", "river", "lake",
    "exterior", "interior", "ruins", "building", "camp", "altar", "path",
    "clearing", "valley", "hill", "cemetery", "graveyard", "warehouse",
)

# Lighting markers — used to identify the single "light" tag in a prompt.
_LIGHTING_MARKERS: tuple[str, ...] = (
    "lamp", "lantern", "candle", "torch", "moonlight", "sunlight", "shadow",
    "glow", "casting", "illumin", "flame", "fire", "light", "dawn", "dusk",
    "noon", "dark", "dim", "bright", "haze", "mist", "fog",
)


def _extract_env_anchors(scene_prompt: str) -> tuple[str, str]:
    """Extract (location_tag, lighting_tag) from a scene_prompt.

    Returns the first tag that matches each category.
    """
    tags = [t.strip() for t in scene_prompt.split(",") if t.strip()]
    location = ""
    lighting = ""
    for tag in tags:
        lower = tag.lower()
        if lower in _PROMPT_SKIP_TAGS:
            continue
        if not location and any(m in lower for m in _LOCATION_MARKERS):
            location = tag
        if not lighting and any(m in lower for m in _LIGHTING_MARKERS):
            lighting = tag
        if location and lighting:
            break
    return location, lighting


def _enforce_scene_continuity(
    shots: List[ShotScript],
    episode_num: int,
) -> List[ShotScript]:
    """For each scene_id group, propagate the FIRST shot's location+lighting tags
    to all subsequent shots in the same group.

    This is purely deterministic — no LLM involvement. It runs AFTER the LLM
    rewrite pass so the base content is already action-aligned. It only touches
    the location and lighting tags.
    """
    if not any(s.scene_id for s in shots):
        # LLM didn't assign any scene_ids — skip silently.
        logger.debug("No scene_ids assigned by LLM — skipping continuity pass | episode={}", episode_num)
        return shots

    # Build anchor map: scene_id → (location_tag, lighting_tag) from first shot.
    anchor_map: dict[str, tuple[str, str]] = {}
    for shot in shots:
        sid = (shot.scene_id or "").strip()
        if sid and sid not in anchor_map:
            loc, light = _extract_env_anchors(shot.scene_prompt)
            if loc or light:
                anchor_map[sid] = (loc, light)

    result = list(shots)
    fixed = 0
    for i, shot in enumerate(result):
        sid = (shot.scene_id or "").strip()
        if not sid or sid not in anchor_map:
            continue

        canon_loc, canon_light = anchor_map[sid]
        curr_loc, curr_light = _extract_env_anchors(shot.scene_prompt)

        loc_ok = not canon_loc or canon_loc.lower() == curr_loc.lower()
        light_ok = not canon_light or canon_light.lower() == curr_light.lower()
        if loc_ok and light_ok:
            continue  # Already consistent.

        # Rebuild: safety prefix → canon anchors → body (minus old anchors).
        tags = [t.strip() for t in shot.scene_prompt.split(",") if t.strip()]
        prefix = [t for t in tags if t.lower() in _PROMPT_SKIP_TAGS]
        body = [t for t in tags if t.lower() not in _PROMPT_SKIP_TAGS]

        # Drop stale location/lighting from body.
        if not loc_ok and curr_loc:
            body = [t for t in body if t.lower() != curr_loc.lower()]
        if not light_ok and curr_light:
            body = [t for t in body if t.lower() != curr_light.lower()]

        # Inject canon anchors at the front of body (right after safety prefix).
        injected: list[str] = []
        if not loc_ok and canon_loc:
            injected.append(canon_loc)
        if not light_ok and canon_light:
            injected.append(canon_light)

        new_prompt = ", ".join(prefix + injected + body)
        result[i] = shot.model_copy(update={"scene_prompt": new_prompt})
        fixed += 1
        logger.debug(
            "scene_id continuity fix | episode={} shot={} scene_id={} loc={!r} light={!r}",
            episode_num, i, sid, canon_loc, canon_light,
        )

    if fixed:
        logger.info(
            "Scene continuity enforced | episode={} fixed={}/{} shots",
            episode_num, fixed, len(shots),
        )
    return result


def _rewrite_scene_prompts_from_narration(
    shots: List[ShotScript],
    episode_num: int,
) -> List[ShotScript]:
    """LLM rewrite pass: align scene_prompt with the exact action in narration_text.

    Sends all shots in a single call. On failure, returns originals unchanged.
    """
    payload = [
        {
            "shot_index": i,
            "narration_text": s.narration_text,
            "scene_prompt": s.scene_prompt,
        }
        for i, s in enumerate(shots)
    ]
    prompt = (
        f"Rewrite scene_prompts for Episode {episode_num} shots so each one "
        f"visually matches its narration_text.\n\n"
        f"Shots:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    try:
        raw = scene_prompt_client.generate_json(
            prompt=prompt, system=_NARRATION_ALIGN_SYSTEM, temperature=0.3
        )
    except Exception as exc:
        logger.warning(
            "scene_prompt narration-rewrite failed ({}) — keeping originals | episode={}",
            exc, episode_num,
        )
        return shots

    if not isinstance(raw, list):
        # LLM may wrap array in {"shots": [...]} or similar
        if isinstance(raw, dict):
            for key in ("shots", "scene_prompts", "result", "items"):
                if isinstance(raw.get(key), list):
                    raw = raw[key]
                    break
        if not isinstance(raw, list):
            logger.warning(
                "scene_prompt narration-rewrite returned non-list (type={}) — keeping originals | episode={}",
                type(raw).__name__, episode_num,
            )
            return shots

    shots = list(shots)
    rewrote = 0
    for item in raw:
        if not isinstance(item, dict):
            continue
        idx = item.get("shot_index")
        new_prompt = (item.get("scene_prompt") or "").strip()
        if not isinstance(idx, int) or not new_prompt or idx < 0 or idx >= len(shots):
            continue
        shots[idx] = shots[idx].model_copy(update={"scene_prompt": new_prompt})
        rewrote += 1

    logger.info(
        "scene_prompt narration-rewrite done | episode={} rewrote={}/{}",
        episode_num, rewrote, len(shots),
    )
    return shots


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
            if _ID_LIKE_RE.match(arc_name.lower()):
                logger.debug("Skipping unresolved id-like character token | token={}", arc_name)
                continue
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
        raw_shots = raw.get("shots", [])
        if not isinstance(raw_shots, list):
            logger.warning(
                "shots field is not a list (type={}); resetting to empty | episode={}",
                type(raw_shots).__name__, episode_num,
            )
            raw_shots = []
        coerced = [
            _coerce_shot_item(s, episode_num, i) for i, s in enumerate(raw_shots)
        ]
        new_shots = [ShotScript(**d) for d in coerced if d is not None]
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
    episode_shots = _validate_narration_length(episode_shots, episode_num)
    episode_shots = _normalize_key_shots(episode_shots, episode_num)
    episode_shots = _normalize_camera_flow(episode_shots, episode_num)
    episode_shots = _backfill_characters(episode_shots, arc.characters_in_episode, episode_num)
    # Two-pass: LLM extraction (narration → brief) + Python synthesis (brief → tags)
    episode_shots = _build_scene_prompts_from_narration(episode_shots, episode_num)
    # Rule-based layer: inject specific artifact/object tags that LLM might miss
    episode_shots = _align_scene_prompt_with_narration(episode_shots, episode_num)
    # Deterministic continuity: enforce shared location+lighting within each scene_id group
    episode_shots = _enforce_scene_continuity(episode_shots, episode_num)

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


def _validate_narration_length(shots: List[ShotScript], episode_num: int) -> List[ShotScript]:
    """Log warnings for shots 2+ whose narration is too short relative to duration_sec.

    Rule: non-hook shots should have >= (duration_sec * 2.5) words.
    At ~3 words/sec Vietnamese TTS, 2.5 gives ~83% audio fill — avoids dead-air silence.
    Does NOT modify narration — this is a diagnostic/audit pass only.
    """
    total_words = sum(len(s.narration_text.split()) for s in shots)
    logger.info(
        "Narration word count | episode={} total_words={} (~{:.0f}s TTS)",
        episode_num, total_words, total_words / 3.0,
    )
    for i, shot in enumerate(shots):
        if i < 2:
            continue  # Hook shots are exempt
        words = len(shot.narration_text.split())
        min_words = int(shot.duration_sec * 2.5)
        if words < min_words:
            logger.warning(
                "Narration too short | episode={} shot={} words={} min={} duration={}s text={!r}",
                episode_num, i, words, min_words, shot.duration_sec,
                shot.narration_text[:80],
            )
    return shots


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


def _align_scene_prompt_with_narration(
    shots: List[ShotScript],
    episode_num: int,
) -> List[ShotScript]:
    """Inject missing visual tags into scene_prompt based on narration_text.

    Two-tier pass:
    1. ACTION rules — insert at position 2 (right after location).
    2. OBJECT rules — append at end (all positions are content now, no metadata suffix).
    """
    aligned = 0
    updated: List[ShotScript] = []

    for shot in shots:
        scene_tags = [t.strip() for t in shot.scene_prompt.split(",") if t.strip()]
        scene_lower = {t.lower() for t in scene_tags}
        narration = shot.narration_text

        action_tags: List[str] = []
        object_tags: List[str] = []

        for pattern, tag in _SCENE_ALIGN_ACTION_RULES:
            if pattern.search(narration) and tag.lower() not in scene_lower:
                action_tags.append(tag)

        for pattern, tag in _SCENE_ALIGN_OBJECT_RULES:
            if pattern.search(narration) and tag.lower() not in scene_lower:
                object_tags.append(tag)

        if not action_tags and not object_tags:
            updated.append(shot)
            continue

        if action_tags:
            # Remove weak/generic pose tags that the specific action makes redundant.
            scene_tags = [
                t for t in scene_tags
                if t.lower() not in _WEAK_POSE_TAGS
            ]
            # Insert action tags at position 2 (after location tag).
            insert_at = min(2, len(scene_tags))
            for i, tag in enumerate(action_tags[:2]):
                scene_tags.insert(insert_at + i, f"({tag}:1.3)")

        # Append object tags at end — no metadata suffix to worry about anymore.
        scene_lower_now = {t.lower() for t in scene_tags}
        for tag in object_tags[:6]:
            if tag.lower() not in scene_lower_now:
                scene_tags.append(f"({tag}:1.2)")
                scene_lower_now.add(tag.lower())

        updated.append(shot.model_copy(update={"scene_prompt": ", ".join(scene_tags)}))
        aligned += 1

    if aligned:
        logger.info(
            "Scene-prompt alignment applied | episode={} shots={}",
            episode_num,
            aligned,
        )
    return updated
