import json
import re
from pathlib import Path
from typing import List, Optional

from loguru import logger
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import settings
from llm.client import script_client as ollama_client
from llm.client import scene_prompt_client
from llm.summarizer import load_arc_overview
from models.schemas import CameraFlow, EpisodeScript, ShotScript, ViralMoment

# Prompt version tag persisted on every EpisodeScript for DB / regression tracing.
_PROMPT_VERSION_LEGACY = "v1"
_PROMPT_VERSION_V2 = "v2-constraint"

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
- Shots 3–8: narration_text MUST be 28–40 words (3–4 full sentences) — TTS must fill 8–11 seconds each.
- TOTAL narration_text across ALL 8 shots MUST be AT LEAST 210 words. At ~3.5 words/second Vietnamese TTS (edge-tts vi-VN-HoaiMyNeural), that yields ≥60 seconds of TTS.
- COUNT YOUR WORDS before outputting each shot. If narration_text for shot 3–8 is fewer than 28 words, REWRITE IT by adding descriptive detail (expand dialogue, add sensory detail, add character reaction).
- HARD FAILURE: if the total is below 210 words your output WILL be rejected and you will be asked again.

NARRATION LENGTH EXAMPLES:
WRONG (too short for a 8s shot): "Lão đạo sĩ này sẽ mang Diệp Thiếu Dương về Mao Sơn để dạy nó đạo pháp." — 15 words, only ~5s TTS, leaves 3s of silence.
RIGHT (correct length for a 8s shot): "Thanh Vân Tử nhìn thẳng vào mắt Diệp Đại Công, giọng trầm xuống: Đứa trẻ này có căn cơ không bình thường. Ta sẽ đưa hắn lên Mao Sơn, dạy đạo pháp, rèn chân thân. Nhưng đây là con đường không thể quay đầu." — 42 words, ~14s TTS. ✓
WRONG (too short for a 9s shot): "Thi Du Cao là một loại độc thi được sử dụng trong cổ thuật." — 13 words, only ~4s TTS.
RIGHT (correct length for a 9s shot): "Thanh Vân Tử chậm rãi giải thích: Thi Du Cao không phải là bệnh, mà là một loại thi độc từ cổ thuật. Nó xâm nhập vào thi thể người chết, khiến xác không thể phân hủy, và dần biến thành một thứ nguy hiểm hơn bất kỳ con quỷ nào ta từng gặp." — 47 words, ~16s TTS. ✓

NARRATIVE RULES (most critical):
- Each shot's narration_text tells what SPECIFICALLY happens — name the action, the character, the location.
- Use the character names provided in the Characters list when they are clearly the character in the scene.
- Shots must connect: the last sentence of shot N sets up shot N+1.
- LANGUAGE: narration_text MUST be written entirely in Vietnamese — no English words, phrases, or sentences anywhere, including hook shots.
  WRONG hook (English): "A face from the abyss" — FORBIDDEN even as a stylistic choice
  RIGHT hook (Vietnamese): "Hắn mở nắp quan tài... và thứ bên trong nhìn tôi." or "Khuôn mặt ấy không phải của người sống."
- Voice: first-person narrator ("Tôi..."), present-tense tension. The narrator is ALWAYS speaking — never describe what "Thanh Vân Tử" does from the outside.
  WRONG (third-person): "Thanh Vân Tử phát hiện ra rằng Diệp Thiếu Dương đã bị trúng độc thi. Hắn quyết định điều tra nguyên nhân."
  RIGHT (first-person): "Tôi nhìn vào vết thương trên cổ Thiếu Dương — đây không phải bệnh thông thường. Đây là độc thi. Kẻ nào đó đã cố tình gieo mầm tử thần vào cơ thể cậu ta."
  WRONG (third-person): "Thanh Vân Tử yêu cầu Diệp Đại Công dẫn hắn đến mộ."
  RIGHT (first-person): "Tôi yêu cầu lão dẫn tôi đến ngôi mộ đó — ngôi mộ của người đàn bà chết khi đang mang thai."
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

SCENE PROMPT RULES — CRITICAL: ComfyUI uses **Flux Dev**, which reads prompts as natural descriptive language, NOT Stable Diffusion attention-weighted tags.
scene_prompt must be a SHORT DESCRIPTIVE PHRASE LIST. Structure:
  [specific location with visual detail], [specific action/pose], [foreground element], [background element], [specific lighting]

SCENE PROMPT QUALITY — Each scene_prompt MUST contain ALL of the following:
1. At least 1 SPECIFIC LOCATION tag (NOT generic): "dimly lit coffin shop with wooden shelves", NOT just "coffin shop interior"
2. At least 1 SPECIFIC ACTION or POSE tag: "figure lunging forward with wooden staff", NOT just "action pose" or "fighting"
3. FOREGROUND LAYER (close to camera, 2+ elements): "cracked stone tablet and scattered ritual candles", "glowing talisman in hand near weathered pillar"
4. MIDGROUND LAYER: where character stands — "stone staircase of ruined temple", "muddy excavation pit floor"
5. BACKGROUND LAYER (depth, 2+ elements): "ancient crumbling gateway half-buried in fog, distant pine-covered mountain ridges" — always TWO background entities
6. SPECIFIC LIGHTING description: "dim oil lantern casting long shadows on walls", NOT just "dramatic lighting"

SCENE PROMPT EXAMPLES:
WRONG: "coffin shop interior, intense fight scene, wooden shelves, dim lantern light, dynamic action pose, anime style, dramatic lighting, detailed background, no text, no watermarks"
→ Problems: "intense fight scene" is vague (WHO doing WHAT?), "dynamic action pose" is generic, "dramatic lighting" has no specifics, weight syntax like (tag:1.15) is NOT valid for Flux

RIGHT: "dimly lit coffin shop with hanging red paper lanterns, figure lunging forward with wooden sword, shattered pottery on floor, rows of dark coffins receding into shadow, flickering oil lamp casting long orange shadows"

WRONG: "mountain monastery courtyard, daoist training, morning mist, stone staircase, bamboo grove, anime style, dramatic lighting, detailed background, no text, no watermarks"
→ Problems: "daoist training" is too vague, no foreground object, "dramatic lighting" not specific

RIGHT: "stone courtyard at mountain temple summit, figure in meditation stance with hands raised, crumbling stone incense burner in foreground, bamboo forest and mist-covered peaks in background, pale golden dawn light filtering through clouds"

WRONG: "outdoor gravesite, ancient tomb excavation, night scene, eerie moonlight, dark soil, stone ruins, (stone lid:1.3), (ritual candles:1.2)"
RIGHT: "muddy excavation pit with exposed stone sarcophagus, figure kneeling and prying open stone lid with iron crowbar, scattered ritual candles on wet earth, ancient crumbling gateway half-buried behind, cold blue moonlight with drifting fog wisps"

FLUX-SPECIFIC RULE: NEVER use `(tag:weight)` or `(tag:1.2)` syntax — these are Stable Diffusion/SDXL features that Flux does not support. Flux reads prompt text literally; weighted parentheses will be passed as literal characters. Write plain descriptive phrases instead.

HORROR ATMOSPHERE — CRITICAL for this story genre:
Every scene_prompt MUST include at least 1 HORROR/SUPERNATURAL ATMOSPHERE element from this palette:
- LIGHTING: "sickly green glow from below", "blood-red candle flame", "cold blue moonlight piercing through cracks", "flickering torch casting distorted shadows on walls", "pale ghostly luminescence", "dim amber candlelight barely reaching corners"
- ENVIRONMENT: "crumbling moss-covered walls", "cobwebs hanging from ceiling beams", "mist creeping along the ground", "twisted dead tree silhouettes", "rusted iron chains on stone wall", "dark stain spreading on floor"
- SUPERNATURAL: "faint ghostly silhouette in background", "glowing ritual symbols on ground", "unnatural fog swirling around ankles", "floating dust particles in shaft of pale light", "eerie green spirit wisps"
- TENSION OBJECTS: "scattered yellowed talisman papers", "overturned ritual candles dripping wax", "cracked ancient mirror reflecting distorted image", "half-open coffin lid with darkness inside", "bloody handprint on wall"
Choose elements appropriate to the specific scene — graveyard scenes get fog/moonlight, indoor scenes get candles/shadows, ritual scenes get glowing symbols/talismans.
Do NOT use the word "mysterious" — use specific visual descriptors instead.

HOODED / MASKED CHARACTER EXPRESSIONS:
- If a character wears a hood ("hooded daoist figure", "cloaked figure") and the shot requires an emotional expression (worried, accusatory, shocked, angry), you MUST add "face partially visible under hood" to the scene_prompt so the expression can render.
- WRONG: "hooded daoist figure, worried expression" (hood blocks the face — expression cannot render)
- RIGHT: "hooded daoist figure, face partially visible under hood, worried expression visible in the shadow"

WET / RAIN ENVIRONMENT:
- Rain or wet-ground scenes MUST include depth cues: "reflections on wet stone path", "rain streaks on wooden surfaces", "puddles reflecting candlelight". Without these, Flux renders rain as flat gray blur.

FORBIDDEN in scene_prompt:
- English sentences or clauses ("He walks into...", "Fifteen years later...")
- Vietnamese words anywhere
- Character names (Diệp Thiếu Dương, Tiểu Mã, etc.) — character appearance is handled separately
- Character appearance descriptors (age, hair, eyes, physique, clothing of specific characters) — e.g., "old daoist", "black-haired girl", "young man in white robes". Use role/action tags instead: "daoist figure", "warrior silhouette", "female protagonist"
- Adverbs or qualifiers ("mysteriously", "fiercely", "inadvertently")
- NSFW or suggestive tags: alluring, seductive, suggestive, provocative, cleavage, navel, bare skin, skinny, undressing, erotic, sensual, bedroom eyes
- Generic placeholder tags: "dramatic lighting", "detailed background", "action pose", "fight scene" — be SPECIFIC
- Flat/boring composition tags: "figure standing next to coffin", "character posing", "centered portrait" — these produce stock-photo dead frames. Choose a dramatic angle.

SHOT_SUBJECT — CRITICAL for retention (shock-first visual hero):
Each shot MUST declare a shot_subject that tells the image generator WHAT THE CAMERA FOCUSES ON.
Allowed values: "person_action" | "corpse_face" | "wound" | "bloody_object" | "supernatural_entity" | "ritual_object" | "environment"

MANDATORY RULES:
- If narration contains ANY of: xác, thi thể, tử thi, thây, xác chết, mắt trợn, mắt mở to, trắng bệch, trắng nhợt, tái mét — shot_subject = "corpse_face" and scene_prompt opens with "extreme close-up of pale dead face" or "macro shot of wide staring dead eyes".
- If narration contains: vết cắn, vết thương, máu chảy, rách da, dấu răng, cào xé — shot_subject = "wound" and scene_prompt opens with "macro shot of [wound] on [body part]".
- If narration centers on: dao đẫm máu, kiếm dính máu, bùa cháy, talisman burning, object covered in blood — shot_subject = "bloody_object" and scene_prompt opens with "extreme close-up of [object]".
- If narration describes: ma, quỷ, hồn, vong, thi biến, bị nhập, hiện hình, possession, spirit manifest — shot_subject = "supernatural_entity" and scene_prompt opens with "low angle shot of ghostly figure emerging from mist" or similar.
- If narration describes a pure atmosphere/location with no human and no shock object — shot_subject = "environment".
- Otherwise — shot_subject = "person_action".

FRAMING BY shot_subject — scene_prompt structure:
- corpse_face / wound: MUST use "extreme close-up" or "macro shot" at the start. Characters list MUST be []. NO "figure standing" — the face/wound IS the subject.
- bloody_object / ritual_object: MUST open with "extreme close-up" or "close-up detail shot". Characters list MUST be []. Hands holding the object are allowed but not a full person.
- supernatural_entity: MUST open with "low angle shot" or "silhouette through fog" — no standard medium close-up. Characters list MUST be [].
- environment: wide shot, no human. Characters list MUST be [].
- person_action: existing rules apply (medium/wide with figure performing action).

These close-up framings (extreme close-up, macro shot, detail shot) are REQUIRED for shock shots — they are the whole point of horror retention. Do NOT default to "medium close-up" or "wide shot" for corpse/wound/blood shots.

CLOTHING SAFETY:
- FORBIDDEN: bare skin, exposed midriff, cleavage, tight clothing, suggestive poses.

PHYSICAL MOTION VOCABULARY — MANDATORY:
NEVER describe mental/emotional states in scene_prompt — describe the PHYSICAL BODY STATE instead.
Flux renders pixels, not feelings. Use precise physical descriptors:
- shock/surprise → "frozen rigid posture, widened eyes with dilated pupils, jaw slightly open"
- fear/dread → "hunched shoulders pulled inward, eyes darting sideways, hands raised defensively"
- struggle/effort → "strained tendons visible on neck, white knuckles gripping object, jaw clenched"
- grief/sorrow → "head bowed low, shoulders shaking, both hands pressed to face"
- anger/confrontation → "chest forward, chin raised, finger pointing accusatorially"
- concentration/ritual → "eyes half-closed, lips moving in silent chant, hands positioned precisely"
- dying/weakened → "body slumped sideways, eyelids half-closed, lips faintly parted"
FORBIDDEN action words: "reacting", "feeling", "shocked", "afraid", "emotional", "disturbed", "stunned" — describe the body, not the mind.

CINEMATIC QUALITY LAYER — each scene_prompt MUST include:
- LENS: choose one based on shot type:
  - Hook/horror close-up → "35mm anamorphic lens, anamorphic lens flares"
  - Establishing/wide graveyard → "14mm wide angle lens" (creates imposing vastness)
  - Ritual detail/corpse macro → "85mm macro lens, shallow depth of field, bokeh background"
  - Standard narrative → "50mm lens"
- LIGHTING FORMULA: combine temperature + effect, NOT generic adjectives:
  - Candle/indoor scenes → "2700K amber candlelight" + one of: "chiaroscuro high-contrast shadows", "rim light separating figure from dark background", "Tyndall effect light shaft through incense smoke"
  - Graveyard/night → "9000K cold blue moonlight" + one of: "volumetric god rays through mist", "rim light on hooded figure edges"
  - Ghost/supernatural → "desaturated cyan light" + "subsurface scattering on translucent skin"
  - Corpse close-up → "subsurface scattering on pale parchment skin" (essential for dead flesh realism)
- FILM STOCK: choose one based on scene mood:
  - Night/graveyard/horror → "bleach bypass color grade"
  - Indoor candle/amber → "Kodak Vision3 5219 warm film grain"
  - Ghost/supernatural → "desaturated cold cyan color grade"

TEXTURE VOCABULARY — use these specific surface descriptors for key props:
- Coffin → "weathered red lacquer with chipped edges and dark wood grain showing through"
- Ritual nails/crowbar/iron tools → "corroded iron with rust bloom and pitted surface"
- Corpse skin → "parchment-dry skin, blue-grey veins visible through translucent dermis"
- Funeral shroud → "coarse undyed hemp fabric with loose thread ends"
- Talisman paper → "yellowed aged rice paper with brushed ink strokes"
- Stone grave marker → "moss-covered weathered granite with hairline cracks"
Do NOT use generic "old" or "ancient" — pick from the texture vocabulary above.

COMPOSITION RULE: when shot is not a static close-up hook, specify a compositional anchor:
- "rule of thirds, subject on left third" or "leading lines toward coffin" or "low angle Dutch tilt"
- Do NOT use "centered portrait" or "symmetrical composition" for horror shots.
- Do NOT add clothing/style/safety tags (e.g., sfw, fully clothed, anime style, no watermarks) — they are injected automatically downstream.

REQUIRED in scene_prompt — USE ALL TAG POSITIONS FOR ACTUAL CONTENT:
- At least 1 specific LOCATION tag with visual detail
- At least 1 specific ACTION or POSE tag with object/weapon/gesture
- At least 1 foreground element (close to camera)
- At least 1 background element (depth/environment)
- At least 1 specific lighting description
- Do NOT include "anime style", "no text", "no watermarks", "sfw", "fully clothed" — these are added automatically

HOOK SHOT_SUBJECT PRIORITY:
- Shot 1 and Shot 2 (hook) SHOULD prefer a shock-forward shot_subject (corpse_face, wound, bloody_object, supernatural_entity) whenever narration allows it. A hook of "person_action" is permitted only when the first scene has NO corpse/wound/blood/spirit element.
- This is the SINGLE biggest retention lever — never open with "figure standing next to coffin"; open with the thing INSIDE the coffin.

TIME-OF-DAY CONSISTENCY — CRITICAL:
- Establish the time-of-day in shot 1. If shot 1 uses moonlight/night/storm, ALL subsequent shots MUST use night lighting. Never introduce sunlight or daytime in later shots unless the story explicitly jumps to a new day.
  WRONG: shots 1-4 are moonlit graveyard, shot 5 narration says "dưới ánh mặt trời" (sunlight) — CONTRADICTION, Flux will break the night atmosphere.
  RIGHT: if the story says "ánh sáng lạnh lẽo", render it as cold moonlight or lantern glow, NOT sunlight.
- If the source narration contains a time-of-day contradiction (e.g., says "ánh mặt trời" but story is at night), silently correct it in narration_text to match the established time-of-day.

CHARACTER DESCRIPTOR ACCURACY:
- When a character is physically visible and has a known role/age, use a role-accurate descriptor in scene_prompt:
  WRONG: "young man" when the character is middle-aged → mismatches any IPAdapter reference
  RIGHT: "middle-aged Chinese man", "elderly daoist priest", "hooded daoist figure"
- The `characters` list MUST include a character if their face, body, or silhouette is visible — even for action shots. Only use [] if NO human is visible (pure environment/object shots).
  WRONG: shot 7 shows Diệp Đại Công standing at coffin → characters=[]
  RIGHT: shot 7 shows Diệp Đại Công standing at coffin → characters=["Diệp Đại Công"]

CORPSE / GHOST PROMPT QUALITY:
- For corpse_face shots showing teeth or fangs: MUST include texture detail: "pale grey decomposed skin", "sharp serrated teeth" or "elongated fangs", "cold light catching white enamel" — vague "sharp teeth" alone is insufficient.
- For supernatural_entity shots (ghost/spirit): MUST include "translucent silhouette", "ethereal wisps of light", "semi-transparent floating form" — Flux needs transparency cues to render a ghost, not just "faint ghostly silhouette in shadow".

OTHER RULES:
- duration_sec: 2 or 3 for shots 1–2; 8 for standard shots 3-8, 10 for climactic action shots.
- is_key_shot: Mark EXACTLY 2-3 shots as true — the most action-packed.
- characters: CRITICAL — list the EXACT character names (from the provided Characters list) whose body, face, or silhouette is PHYSICALLY VISIBLE in the shot. If the scene_prompt describes only environment, objects, or atmosphere with NO human figure present, use []. DO NOT add a character just because they are the narrator or implied. MAXIMUM 2 characters per shot — never list 3 or more.
- When shot_subject is corpse_face / wound / bloody_object / ritual_object / supernatural_entity / environment → characters MUST be [] (the subject is the thing, not the person).

SCENE_ID RULES — CRITICAL for visual consistency:
- scene_id is a short snake_case English label for the PHYSICAL VISUAL SETTING described in scene_prompt (e.g. "outdoor_graveyard", "coffin_shop_interior", "temple_gate", "dark_forest", "excavation_pit").
- Derive scene_id from WHAT IS VISIBLE IN THE FRAME — not from chapter labels or story arc names.
  WRONG: scene_prompt describes a graveyard → scene_id="coffin_shop_int" (you thought of the story arc, not the frame)
  RIGHT: scene_prompt describes a graveyard → scene_id="outdoor_graveyard"
  WRONG: scene_prompt describes a street → scene_id="ruined_temple_int"
  RIGHT: scene_prompt describes a street → scene_id="village_street_night"
- Assign the SAME scene_id to ALL shots that take place in the SAME physical location within this episode.
- When the story moves to a new location, assign a NEW scene_id.
- Shots with the same scene_id will share base environment tags at image-generation time — so accuracy matters.

Return JSON:
{
  "title": "string — episode title in Vietnamese",
  "shots": [ { "scene_prompt": "string", "narration_text": "string", "duration_sec": 6, "is_key_shot": false, "characters": ["Tên Nhân Vật"], "camera_flow": "wide_to_close", "scene_id": "location_slug", "shot_subject": "person_action" } ]
}
shots MUST have EXACTLY 8 elements. EXACTLY 2-3 must have is_key_shot=true.
camera_flow MUST be one of: "wide_to_close", "close_to_wide", "pan_across", "detail_reveal", "static_close", "static_wide".
shot_subject MUST be one of: "person_action", "corpse_face", "wound", "bloody_object", "supernatural_entity", "ritual_object", "environment"."""


# V2 prompt — rule-list format, paired with `constraint_validator.py` checks.
# Behind `retention.use_constraint_system`. Legacy V1 (above) stays in code as
# rollback path. Anti-pattern examples derive from episode-001-script.json
# shots 0/3/4 (English hook, lore-dump exposition wall).
_SCRIPTWRITER_SYSTEM_V2 = """You are a Vietnamese short-video scriptwriter for TikTok/YouTube Shorts.
GENRE: Vietnamese supernatural-horror (Maoshan exorcism). Every shot must evoke dread, curiosity, or supernatural tension.

You will receive ORDERED SCENES + (optionally) VIRAL MOMENT CANDIDATES.
- ORDERED SCENES = the narrative spine. Cover them in chronological order.
- VIRAL MOMENT CANDIDATES (if present) = bias for HOOK ONLY (shot 0) and at most 1-2 other shots. The rest stays faithful to ORDERED SCENES.

Write EXACTLY 8 shots. Output JSON matching [OUTPUT CONTRACT] at the end.

[NARRATION RULES] — hard limits, validated programmatically:
1. LANGUAGE: Vietnamese only. No English words anywhere — including the hook. (Rule: hook_language)
2. HOOK SHOT 0: ≤10 words. Open with a SPECIFIC visual action or shocking line. No setup, no character introduction.
   ❌ "The stench of death filled the air..."  (English — REJECTED)
   ❌ "Diệp Thiếu Dương bắt đầu hành trình tại Diệp gia thôn."  (setup — REJECTED)
   ✅ "Hắn mở nắp quan tài… và thứ bên trong đang nhìn lại."
3. PER SENTENCE: max 15 words. Split long sentences.
4. EXPOSITION: max 2 expository sentences in a row across the entire script. (Rule: exposition_density)
   Expository = definition style ("X là một loại Y..."), abstract noun + general statement, "Theo cổ thuật...", "Đó là...".
   ❌ Two consecutive shots both lore-dumping — see anti-pattern below.
5. BANNED OPENINGS for ANY sentence: "X là một loại", "Theo ", "Vì ", "Bởi ", "Do ", "Thật ra ", "Vốn dĩ ".
6. SHOT 1 must contain at least 1 tension verb (nhìn, hét, chạy, đập, mở, lao, vung, đâm, chạm, ngã, rít, xé, túm, đẩy, kéo).
7. CLIFFHANGER SHOT 7: cut mid-revelation or open question. No CTA, no "theo dõi tiếp", no resolution.

[STRUCTURE RULES]:
8. Exactly 8 shots. 2-3 must have `is_key_shot=true`.
9. Shot 0-1 duration: 2 or 3 seconds. Shots 2-7: 6-10 seconds. No shot >12s.
10. Episode total narration ≥ 200 words (after the hook is finalized).

[LORE-CURIOSITY RULES]:
11. Lore terms (Thi Du Cao, Mao Sơn pháp thuật, cổ thuật names — anything proper-noun that is NOT a character) may appear ONLY after at least 2 tension shots have set up curiosity.
12. When a lore term DOES appear, frame it as CONSEQUENCE or MYSTERY, not definition.
   ❌ "Thi Du Cao là một loại thi độc cổ xưa được dùng trong các nghi thức huyền bí."  (definition + abstract dump)
   ✅ "Vết bầm trên cổ Thiếu Dương không phải bệnh — đó là dấu của Thi Du Cao."  (consequence, plants question)

[VISUAL-NARRATION ALIGNMENT]:
13. `scene_prompt` must visually depict the same action the narration describes. If narration says "hắn rút kiếm", scene_prompt cannot show a serene landscape.
14. Foreground/midground/background layers required (see existing scene_prompt rules — those still apply).

[CAMERA + SUBJECT]:
15. `camera_flow` ∈ {"wide_to_close", "close_to_wide", "pan_across", "detail_reveal", "static_close", "static_wide"}.
    - Hook (shot 0): "static_close" or "detail_reveal"
    - Twist/revelation: "close_to_wide"
    - Horror discovery / clue: "detail_reveal"
    - Action/fight: "pan_across"
16. `shot_subject` ∈ {"person_action", "corpse_face", "wound", "bloody_object", "supernatural_entity", "ritual_object", "environment"}.
    Choose deliberately — non-person subjects (corpse_face, wound, bloody_object) are SHOCK levers; use them on at least 1-2 shots.

[OUTPUT CONTRACT] — return ONLY this JSON, nothing else:
{
  "title": "string — episode title in Vietnamese",
  "shots": [
    {
      "scene_prompt": "string (English — descriptive phrase list with foreground/midground/background)",
      "narration_text": "string (Vietnamese)",
      "duration_sec": 6,
      "is_key_shot": false,
      "characters": ["Tên Nhân Vật"],
      "camera_flow": "wide_to_close",
      "scene_id": "location_slug",
      "shot_subject": "person_action"
    }
  ]
}
shots length MUST be 8.

[ANTI-PATTERN — DO NOT EMIT]:
❌ Shot A: "Thi Du Cao là một loại thi độc cổ xưa..."
❌ Shot A+1: "Theo cổ thuật, nó được dùng trong các nghi thức huyền bí..."
   (Both expository, back-to-back — violates rule 4 + 5 + 12.)
✅ Shot A: "Hắn lùi lại — vết bầm trên cổ thằng bé chuyển sang đen."
✅ Shot A+1: "Tôi nhận ra đó là dấu của Thi Du Cao. Đã quá muộn."
   (Action → consequence-framed lore. Curiosity preserved.)"""


_HOOK_SYSTEM = """You are a Vietnamese short video scriptwriter for TikTok/YouTube Shorts.
Write EXACTLY 1 hook shot that opens an episode.

RULES:
- Must open with a SPECIFIC ACTION or SHOCKING NARRATION — no backstory, no scene-setting.
- Keep chronological order — do NOT flash-forward. Reframe the very first scene from a shocking angle.
- narration_text MUST be 10 words or fewer.
- scene_prompt: comma-separated tags for Stable Diffusion (English only, no character names, no sentences).
- scene_prompt MUST contain: 1 specific location with detail, 1 specific action/pose, 1 foreground element, 1 background element, 1 specific lighting.
- DO NOT include style/safety tags (anime style, no text, no watermarks, sfw) — they are injected automatically downstream.
- camera_flow: MUST be "static_close" or "detail_reveal" for hook shots.

SHOT_SUBJECT — hook MUST prefer shock:
- Choose shot_subject from: "corpse_face" | "wound" | "bloody_object" | "supernatural_entity" | "ritual_object" | "person_action" | "environment".
- If the opening story has a corpse/wound/blood/ghost element → shot_subject MUST be the corresponding non-person value, characters = [], and scene_prompt MUST open with "extreme close-up" or "macro shot" of that thing — NOT a figure standing next to it.
- Only use "person_action" when the opening scene has no shock element available.

Return JSON:
{ "scene_prompt": "string", "narration_text": "string", "duration_sec": 3, "is_key_shot": false, "characters": ["Tên Nhân Vật"], "camera_flow": "static_close", "shot_subject": "corpse_face" }"""


# Hook narration is capped at 10 words by `_HOOK_SYSTEM`. Used by `_write_raw`
# to validate the post-hook total — original shot 0 word count is replaced by
# this budget when computing whether the final assembled script will pass the
# 200-word floor.
_HOOK_WORD_BUDGET = 10


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
        "CRITICAL: shots 3-8 MUST each have 28-40 words in narration_text. "
        "Total narration_text across SHOTS 2..N (excluding shot 1, which will be "
        "replaced by a 10-word hook downstream) MUST be AT LEAST 200 words. "
        "Aim for 220+ total words across all shots so the script survives the hook swap."
    )
    # Phase 3: switch system prompt based on feature flag.
    system_prompt = (
        _SCRIPTWRITER_SYSTEM_V2
        if settings.retention.use_constraint_system
        else _SCRIPTWRITER_SYSTEM
    )
    result = ollama_client.generate_json(
        prompt=prompt, system=system_prompt, temperature=0.7
    )
    # Reject and retry if total narration is severely short (LLM output is broken/empty).
    # Threshold 80 catches truly degenerate outputs while allowing the model to produce
    # somewhat shorter scripts when the arc content is genuinely thin.
    # Outputs between 80–150 words are accepted with a WARNING for visibility.
    shots = result.get("shots", [])
    if not isinstance(shots, list) or len(shots) < 6:
        # LLM returned nothing usable (empty/missing/truncated). Retry.
        logger.warning(
            "Script rejected: shots missing or too few (got {}), retrying | episode={}",
            len(shots) if isinstance(shots, list) else "non-list", episode_num,
        )
        raise ValueError(
            f"shots missing or too few: got {len(shots) if isinstance(shots, list) else 'non-list'}"
        )
    if isinstance(shots, list) and shots:
        # Ollama occasionally returns shots as JSON-encoded strings instead of
        # dicts. Parse those before counting so the validator doesn't spuriously
        # reject a valid response (total_words would otherwise be 0).
        def _narration_of(s: object) -> str:
            if isinstance(s, dict):
                return str(s.get("narration_text", ""))
            if isinstance(s, str):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, dict):
                        return str(parsed.get("narration_text", ""))
                except json.JSONDecodeError:
                    return ""
            return ""

        total_words = sum(len(_narration_of(s).split()) for s in shots)
        # Shot 0 will be REPLACED by _generate_hook_shot() (≤10 words by design).
        # Validate the post-hook total instead of raw total, otherwise scripts
        # whose original shot 0 was long enough to mask thin shots 1+ pass here
        # but fail the downstream hard gate after the hook swap.
        shot0_words = len(_narration_of(shots[0]).split()) if shots else 0
        effective_total = total_words - shot0_words + _HOOK_WORD_BUDGET
        # Vietnamese edge-tts (vi-VN-HoaiMyNeural) clocks ~3.5 wps empirically.
        # Validator requires final video ≥57s, so we need ≥57*3.5 ≈ 200 words.
        # Use 200 as hard floor; below this TTS undershoots and validation fails.
        min_total = 200
        if effective_total < min_total:
            # Identify short shots for diagnostics
            short_shots = []
            for idx, s in enumerate(shots):
                w = len(_narration_of(s).split())
                if idx >= 2 and w < 20:
                    short_shots.append(f"shot{idx+1}={w}w")
            logger.warning(
                "Script rejected: post-hook total_words={} (raw={}) < {}, retrying | episode={} short_shots={}",
                effective_total, total_words, min_total, episode_num, short_shots or "none",
            )
            raise ValueError(
                f"narration too short post-hook: {effective_total} words < {min_total}; "
                f"raw={total_words}, shot0={shot0_words}, short_shots={short_shots}"
            )
        if effective_total < 210:
            logger.warning(
                "Script accepted but narration is tight: post-hook total_words={} < 210 | episode={}",
                effective_total, episode_num,
            )
    return result


_MAX_SHOTS_PER_EPISODE = 8  # shots that fit in one video; excess flows to next episode
_ID_LIKE_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)+$")

# --------------------------------------------------------------------------- #
#  Narration-to-scene-prompt alignment (LLM rewrite pass)                     #
# --------------------------------------------------------------------------- #
_NARRATION_ALIGN_SYSTEM = """You are a ComfyUI Flux Dev prompt engineer for a HORROR/SUPERNATURAL story.
You receive a list of video shot objects. Each shot has:
  - shot_index (int)
  - narration_text (Vietnamese sentence — what the narrator says)
  - scene_prompt (existing English phrase list for ComfyUI)

YOUR TASK: Rewrite each "scene_prompt" so it VISUALLY DEPICTS the exact action/character/object/location described in "narration_text", while MAXIMIZING horror/supernatural atmosphere.

EXTRACTION RULES — read narration and extract these 5 elements:
1. WHO: which character role is physically visible in THIS shot (use generic role tags matching this shot's character, NOT the protagonist's role by default: daoist figure, elder figure, middle-aged man, young man, female figure, hooded figure — NEVER character names). Match the role to WHO is described in narration_text — if narration describes a villager digging, use "villager" not "daoist figure".
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
- FLUX SYNTAX: Write plain descriptive English phrases. NEVER use `(tag:weight)` or `(tag:1.2)` syntax — Flux reads prompts as natural language; weighted parentheses are passed literally and will corrupt the prompt.
- Do NOT add style/safety metadata tags (sfw, fully clothed, anime style, no text, no watermarks) — they are injected automatically downstream. Use ALL tag positions for visual content.
- ACTION tag MUST reflect what narration_text says is happening — not a standing portrait
- HORROR ATMOSPHERE is MANDATORY: every prompt must have at least 1 eerie/dark/supernatural lighting or mood tag — NEVER use plain "bright daylight" or "warm sunlight" unless the narration explicitly describes a safe daytime scene
- If narration says "prying open coffin lid" → scene_prompt must contain prying/opening action tags
- If narration says "shouting accusation at someone" → scene_prompt must contain accusatory gesture tags
- If narration describes discovery/reveal → scene_prompt must show discovery moment with eerie reveal lighting
- FORBIDDEN tags: "action pose", "dynamic pose", "performing ritual", "figure standing", "fight scene", "dramatic lighting", "detailed background"
- FORBIDDEN content: bare skin, exposed midriff, cleavage, tight clothing, suggestive poses
- PHYSICAL MOTION RULE: NEVER use emotion or mental-state words in ACTION descriptions. Translate to observable body states:
  "reacting in shock" → "frozen rigid posture, jaw open, widened eyes with dilated pupils"
  "in fear" → "hunched shoulders, hands raised defensively, weight shifted back on heels"
  "struggling" → "strained tendons on neck, white knuckles clutching iron bar, jaw clenched tight"
  "grieving" → "head bowed low, shoulders shaking, hands pressed to face"
  "surprised" → "body recoiling backward, chin tucked, hands instinctively raised"
  FORBIDDEN in action tags: "shocked", "afraid", "emotional", "reacting", "feeling", "disturbed"

Return a JSON ARRAY (same length as input, same order):
[{"shot_index": 0, "scene_prompt": "rewritten phrases..."}, ...]
CRITICAL: Return ONLY the JSON array, no markdown, no explanation."""
# Object tags — appended at the END of scene_prompt (noun/prop hints).
_SCENE_ALIGN_OBJECT_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(dù|ô)\b", re.IGNORECASE), "red umbrella"),
    (re.compile(r"\b(nhang|hương)\b", re.IGNORECASE), "burning incense sticks"),
    (re.compile(r"quan tài|nắp quan", re.IGNORECASE), "weathered red lacquer coffin with chipped edges and dark wood grain"),
    (re.compile(r"dao găm|\bdao\b", re.IGNORECASE), "dagger in hand"),
    (re.compile(r"\bđinh\b", re.IGNORECASE), "corroded iron nails with rust bloom on coffin lid"),
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
    # NOTE: do NOT use bare `mồ` — it matches "mồ hôi" (sweat), "mồ côi" (orphan).
    # Require explicit grave-related compounds only.
    (re.compile(r"\bmộ\b|ngôi mộ|phần mộ|nấm mồ|mồ mả|mộ phần|đào mộ|khai quật mộ", re.IGNORECASE), "chinese earthen grave mound with carved stone stele"),
    (re.compile(r"nến|đèn cầy", re.IGNORECASE), "dripping wax candle with flickering flame"),
    (re.compile(r"trắng bệch|trắng nhợt|xanh xao|tái mét", re.IGNORECASE), "parchment-dry pale skin with blue-grey veins visible through translucent dermis"),
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
    # Physical motion corrections — catch English emotion words the LLM might still use
    # and replace with observable body-state descriptions
    (re.compile(r"\breacting in shock\b|\bin shock\b|\bshocked expression\b", re.IGNORECASE),
     "frozen rigid posture, jaw open, widened eyes with dilated pupils"),
    (re.compile(r"\bin fear\b|\bfear expression\b|\bterrified expression\b", re.IGNORECASE),
     "hunched shoulders pulled inward, eyes darting sideways, hands raised defensively"),
    (re.compile(r"\bstruggling\b|\bstraining to\b", re.IGNORECASE),
     "strained neck tendons, white knuckles clutching object, jaw clenched tight"),
    (re.compile(r"\bin grief\b|\bgrieving\b|\bsorrow expression\b", re.IGNORECASE),
     "head bowed low, shoulders shaking, hands pressed to face"),
    (re.compile(r"\bin anger\b|\bangry expression\b|\brage expression\b", re.IGNORECASE),
     "chest thrust forward, chin raised, finger pointing accusatorially"),
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

# mood_lighting phrases that violate the horror/supernatural genre constraints.
# Briefs containing any of these terms are rejected unless narration explicitly
# describes a safe daytime scene (detected by _DAYTIME_SAFE_KEYWORDS).
_HORROR_MOOD_VIOLATIONS: frozenset[str] = frozenset([
    "bright white light",
    "clear visibility",
    "sunny daylight",
    "warm sunlight",
    "bright daylight",
    "cheerful sunlight",
    "golden sunlight",
    "morning sunlight",
    "soft natural light",
    "bright natural light",
])

# Vietnamese keywords indicating a narration is set in a safe daytime scene;
# when present, horror mood violations are NOT flagged.
_DAYTIME_SAFE_KEYWORDS: tuple[str, ...] = (
    "ban ngày", "ban mai", "bình minh", "giữa trưa", "buổi sáng", "ánh nắng",
)

# Generic action strings that the rule-based _align_scene_prompt_with_narration
# should still override (pass-3 action injection IS needed for these).
_GENERIC_ACTIONS: frozenset[str] = frozenset([
    "looking around",
    "standing",
    "figure standing",
    "figure looking",
    "figure watching",
    "figure waiting",
    "figure contemplating",
    "figure thinking",
    "performing ritual",
    "conducting ceremony",
    "ritual",
    "ceremony",
    "figure posing",
    "figure in stance",
])

# Tags that are purely structural/style — skip when extracting environment anchors.
_PROMPT_SKIP_TAGS: frozenset[str] = frozenset([
    "sfw", "fully clothed", "high collar", "long sleeves", "covered body",
    "modest clothing", "traditional attire", "armored", "formal wear",
    "anime style", "no text", "no watermarks",
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
- actions: List of 1-3 specific, observable physical actions in chronological order.
  EACH action must contain a concrete verb + direction/result.
  HOW TO SPLIT: Read narration_text sentence by sentence. If 1 sentence → 1 action.
  If 2-3 sentences describe different physical events → 1 action per event (max 3 total).
  actions[0] = FIRST event (most visually dominant — used as primary frame).
  actions[1] = SECOND event (if exists — rendered in frame 2).
  actions[2] = THIRD event (if exists — rendered in frame 3).
  GOOD (single action): ["figure prying open stone coffin lid with iron crowbar"]
  GOOD (multi-event): ["figure running downhill through rain", "figure kneeling at gravesite with hands on wet earth"]
  BAD: ["performing ritual", "looking around", "standing", "mysterious gesture", "conducting ceremony"]
  BLACKLIST — NEVER use these words in any action: ritual, ceremony, pose, scene, performing, conducting
  If narration describes DISCOVERY ("nhìn thấy", "phát hiện") → action = the revealing moment
  If narration is PURELY environment/atmosphere → ["figure crouching motionless behind stone pillar"]
- setting: Physical location with ONE visual detail. E.g. "dimly lit coffin shop with rows of dark wooden coffins"
  MUST be a descriptive English phrase. NEVER use an identifier or slug (e.g. "dark_forest", "mountain_road" are FORBIDDEN — write "moonlit forest path with ancient stone graves" instead).
  AESTHETIC: This story is set in rural CHINA. All settings MUST reflect Chinese/East Asian architecture and culture by default.
  - Graves → "earthen chinese grave mound with stone stele", NOT western tombstone/cross
  - Houses → "traditional chinese village house with wooden beams", NOT western cottage
  - Coffins → "red lacquered chinese wooden coffin", NOT plain box or western casket
  - Only use western elements if the narration EXPLICITLY mentions them.
- key_objects: List of specific props visible in the scene (max 4). Concrete nouns with a TEXTURE/MATERIAL qualifier.
  Flux renders surfaces convincingly only when material is explicit. Each prop MUST have a texture or condition word.
  GOOD: ["weathered red lacquer coffin", "corroded iron crowbar", "yellow paper talisman with visible fibers", "coarse hemp burial cloth"]
  BAD (no texture):  ["red coffin", "iron crowbar", "talisman", "white cloth"]
  Texture vocabulary to draw from: weathered, corroded, oxidized, charred, parchment-dry, viscous, blood-soaked, lacquered, rotted, splintered, moss-covered, dust-caked, frost-rimed, soot-stained.
- mood_lighting: MUST follow format "[light_source phrase], [palette/contrast phrase], [atmospheric effect phrase]".
  CRITICAL — DO NOT copy any example verbatim. Combine ONE item from each category below into a UNIQUE phrase per shot.
  When the same physical location appears across multiple shots, vary at least one of the three components (light intensity / palette / effect) so no two shots have identical mood_lighting.
  LIGHT_SOURCE vocabulary (pick 1, adapt wording): single candle flicker, dying oil lantern, paper lantern glow, hanging brazier embers, cold blue moonlight, pre-dawn pale grey, dying torch sputter, faint ember crack, lightning flash through window, ritual fire pit blaze, sickly green spirit glow, glowing talisman radiance.
  PALETTE/CONTRAST vocabulary (pick 1): chiaroscuro high-contrast, deep violet shadows, teal-orange split tone, sickly green tint with black shadows, blood-orange rim with black void, parchment yellow + ink black, desaturated cyan ghost light, rim-lit silhouette against pitch black, hard vertical shadow bars.
  ATMOSPHERIC_EFFECT vocabulary (pick 1): volumetric god rays through dust, drifting mist along floor, fog tendrils crawling up walls, embers floating in still air, heat shimmer above flame, swirling smoke from incense, dust motes in narrow light shaft, condensation steaming off cold stone.
  BAD: "spooky lighting", "dark atmosphere", "dramatic light", "warm sunlight" (forbidden — story is horror at night).
  CINEMATIC LIGHTING TERMS (use when applicable, but do not exhaustively list all in every shot):
    - "chiaroscuro" → ideal for single-source candle/lantern shots
    - "rim lighting" → separates subject from dark background; pair with moonlight or doorway backlight
    - "subsurface scattering" → REQUIRED whenever shot_subject == "corpse_face" or close-up of skin (makes pale dead skin read as flesh, not plastic)
    - "volumetric fog" / "volumetric god rays" → for shots with smoke, mist, dust, or light beams
  MANDATORY LIGHT-SOURCE LOGIC:
    - If setting/key_objects mention candle, lantern, oil lamp, torch → light_source MUST be one of the candle/lantern/torch options above.
    - If narration mentions moon/moonlight → light_source MUST be "cold blue moonlight" and palette MUST include "rim".
    - If shot_subject == "corpse_face" → mood_lighting MUST contain "subsurface scattering".
- composition: Camera framing tag if obvious from narration. Otherwise leave empty string "".
  Examples: "medium close-up", "wide establishing shot", "medium shot", "extreme close-up", "macro shot", "low angle shot"
- shot_subject: What the camera focuses on. One of: "person_action" | "corpse_face" | "wound" | "bloody_object" | "supernatural_entity" | "ritual_object" | "environment".
  MANDATORY mapping (Vietnamese keywords in narration → shot_subject):
    xác / thi thể / tử thi / thây / mắt trợn / trắng bệch / tái mét → "corpse_face"
    vết cắn / vết thương / dấu răng / máu chảy / rách da → "wound"
    dao dính máu / kiếm máu / bùa cháy / object covered in blood as center of frame → "bloody_object"
    ma / quỷ / hồn / vong / bị nhập / hiện hình → "supernatural_entity"
    pure atmosphere, no human, no shock object → "environment"
    glowing talisman / altar / candles as center (no person) → "ritual_object"
    otherwise → "person_action"
  When shot_subject is NOT "person_action":
    - composition MUST be "extreme close-up" / "macro shot" / "low angle shot" / "detail shot" (NEVER "medium close-up" or "wide shot").
    - subjects MUST be [] — the thing is the subject, not a person.
    - actions[0] MUST describe the THING (e.g. "pale lifeless female face with wide bloodshot dead eyes, dark blood trickling from mouth"), NOT a person looking at it.

INPUT FORMAT: JSON array of shots, each with: shot_index, narration_text, characters

OUTPUT FORMAT: JSON array, same length as input, same order:
[{"shot_index": 0, "subjects": [...], "actions": ["primary action", "optional second action"], "setting": "...", "key_objects": [...], "mood_lighting": "...", "composition": "", "shot_subject": "person_action"}]

CRITICAL: Return ONLY the JSON array, no markdown, no explanation."""

# --------------------------------------------------------------------------- #
#  Character Resolution — Phase 1                                              #
# --------------------------------------------------------------------------- #
_CHARACTER_RESOLVE_SYSTEM = """You are a character resolution assistant for a Vietnamese horror video pipeline.
You receive a batch of video shots. Each shot has:
  - shot_index (int)
  - narration_text (Vietnamese — what the narrator says)
  - llm_characters (the initial character list the LLM assigned; may be wrong)
  - arc_characters (all known characters in this arc — use for cross-reference)

YOUR TASK: For each shot, determine which characters are PHYSICALLY VISIBLE in the image.

RULES:
1. SUBJECT: the character who is PERFORMING the main action described in narration_text.
   - Vietnamese patterns for subject: "X đi đến...", "X sử dụng...", "X phát hiện...", "X đặt tay...", "X tiêu diệt..."
   - The GRAMMATICAL SUBJECT of the main verb = primary visible character.
2. OBJECT_VISIBLE: a character who is PHYSICALLY PRESENT and visible (e.g. being examined, confronted, standing nearby).
   - Vietnamese patterns: "kiểm tra cơ thể của Y", "đối mặt với Y", "nhìn thấy Y"
   - DO NOT include a character just because they are MENTIONED or implied — they must be PHYSICALLY VISIBLE.
3. If narration mentions only environment/atmosphere with NO named character acting → return [].
4. MAXIMUM 2 characters total (subject + 1 object_visible). If 3+ would be visible, keep subject + most prominent object_visible.
5. Character names must come from arc_characters list EXACTLY (exact Vietnamese spelling).

IMPORTANT DISTINCTION:
  "Thanh Vân Tử phát hiện ra rằng nam tử là Diệp Đại Bảo"
  → subject = Thanh Vân Tử (person DOING the discovering)
  → object_visible = Diệp Đại Bảo (person being discovered — physically present)

  "Thanh Vân Tử đi đến nhà Diệp Đại Công"
  → subject = Thanh Vân Tử
  → object_visible = Diệp Đại Công (at his own home — physically present in scene)

OUTPUT FORMAT: JSON array, same length as input, same order:
[{"shot_index": 0, "characters": ["PrimarySubject", "OptionalObjectVisible"]}]

CRITICAL: Return ONLY the JSON array, no markdown, no explanation.
characters must be [] if no named character is physically visible."""

# --------------------------------------------------------------------------- #
#  Map camera_flow values to composition tags when brief.composition is empty.
_CAMERA_FLOW_TO_COMPOSITION: dict[str, str] = {
    "static_close": "medium close-up",
    "detail_reveal": "medium close-up",
    "static_wide": "wide establishing shot",
    "wide_to_close": "",
    "close_to_wide": "",
    "pan_across": "",
}

# Shot-subject-driven framing overrides. When brief.shot_subject is non-person,
# these tags take precedence over camera_flow mapping AND over any composition
# field that isn't already a close-up variant — because horror retention depends
# on the shock subject filling the frame.
_SHOT_SUBJECT_FRAMING: dict[str, tuple[str, str]] = {
    # (opening composition tag, subject emphasis prefix)
    "corpse_face": ("extreme close-up", "pale lifeless face with wide bloodshot dead eyes"),
    "wound": ("macro shot", "deep bleeding wound on pale skin"),
    "bloody_object": ("extreme close-up", "blood-soaked object in center of frame"),
    "supernatural_entity": ("low angle shot", "ghostly translucent figure emerging from dark mist"),
    "ritual_object": ("close-up detail shot", "glowing ritual object centered in frame"),
    "environment": ("wide establishing shot", ""),
    "person_action": ("", ""),
}

_CLOSEUP_COMPOSITION_TOKENS: frozenset[str] = frozenset([
    "extreme close-up", "macro shot", "close-up detail shot", "low angle shot",
    "detail shot", "close-up",
])

# Tags in synthesized prompts are capped at this count to leave buffer for
# rule-based _align_scene_prompt_with_narration() pass.
_SYNTHESIS_MAX_TAGS = 16


def _subject_already_in_action(subject: str, action: str) -> bool:
    """Check if the exact subject phrase is a substring of the action string."""
    return subject.lower().strip() in action.lower()


def _synthesize_scene_prompt(brief: "ShotVisualBrief", shot: ShotScript) -> str:
    """Deterministic Python synthesis of a ComfyUI tag list from a visual brief.

    Uses brief.actions[0] (primary action) as the scene_prompt action tag.
    Secondary actions (actions[1+]) are injected by frame_decomposer per-frame.

    Tag order: [composition+setting] → [primary_action] → [key_objects] → [mood_lighting] → [subjects]
    Caps at _SYNTHESIS_MAX_TAGS with priority-based dropping.
    No LLM involved.

    shot_subject override: when brief.shot_subject is non-person, composition is
    forced to an extreme close-up / macro / low-angle variant, subjects are
    suppressed, and the subject-emphasis prefix replaces the primary action as
    the visual hero. This is the retention lever for horror shots.
    """
    from models.schemas import ShotVisualBrief, ShotSubject  # noqa: F401  local import avoids circular at module-level

    subject_key = brief.shot_subject.value if brief.shot_subject else "person_action"
    forced_composition, subject_prefix = _SHOT_SUBJECT_FRAMING.get(
        subject_key, ("", "")
    )
    non_person_subject = subject_key not in ("person_action", "environment")

    # Phase 3: focal-point cap (constraint system). Keep subjects + key_objects
    # ≤ 2 total. Drop key_objects first (subjects carry the character anchor
    # used by character_gen IPAdapter). If subjects alone > 2, keep first 2.
    # Mutates a local copy so the caller's brief is untouched.
    if settings.retention.use_constraint_system:
        capped_subjects = list(brief.subjects[:2])
        keep_objects = max(0, 2 - len(capped_subjects))
        capped_objects = list(brief.key_objects[:keep_objects])
        if capped_subjects != list(brief.subjects) or capped_objects != list(brief.key_objects):
            brief = brief.model_copy(
                update={"subjects": capped_subjects, "key_objects": capped_objects}
            )

    # Resolve composition: subject-forced > brief.composition > camera_flow mapping.
    composition = forced_composition or brief.composition.strip()
    if not composition:
        composition = _CAMERA_FLOW_TO_COMPOSITION.get(shot.camera_flow.value, "")

    # If shot_subject is non-person but composition is still a neutral framing
    # (e.g. the LLM ignored the rule), upgrade to the subject's forced framing.
    if non_person_subject and composition.lower() not in _CLOSEUP_COMPOSITION_TOKENS:
        composition = forced_composition or composition

    # Primary action: for non-person subjects, the subject_prefix IS the action
    # (describes the shock thing, not a person performing).
    primary_action = brief.actions[0].strip() if brief.actions else ""
    if non_person_subject:
        primary_action = subject_prefix or primary_action

    # Build ordered candidate tag groups (by priority).
    # Priority: action > setting > subjects[0] > mood_lighting > key_objects > subjects[1]
    # These first four are NEVER dropped.
    never_drop: list[str] = []
    if composition:
        never_drop.append(composition)
    never_drop.append(brief.setting)
    if primary_action:
        never_drop.append(primary_action)

    # Primary subject — never drop. Suppressed entirely for non-person shots.
    primary_subject = ""
    if not non_person_subject and brief.subjects:
        primary_subject = brief.subjects[0]
    if primary_subject and not _subject_already_in_action(primary_subject, primary_action):
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
        key_object_tags.append(obj)

    # Secondary subject — can be dropped when budget exhausted.
    # Suppressed entirely for non-person shots.
    secondary_subject = ""
    if not non_person_subject and len(brief.subjects) > 1:
        sub = brief.subjects[1].strip()
        if sub and not _subject_already_in_action(sub, primary_action):
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
    first_event_context: str | None = None,
) -> List[ShotScript]:
    """LLM extraction pass: convert narration_text to ShotVisualBrief for each shot.

    Shots with duration_sec <= 2 are skipped — their visual_brief stays None.
    3-second hook shots ARE processed so they receive proper English scene_prompts.
    scene_id is NOT sent in the payload to prevent the LLM from copying the slug
    into setting/composition fields.

    first_event_context: when provided, hook shots (index 0-1, duration <=3s) receive
    an extra context field in the payload so the LLM anchors the image to the
    first story event rather than the short meta narration text.
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
        entry: dict = {
            "shot_index": i,
            "narration_text": shot.narration_text,
            "characters": shot.characters,
        }
        # Phase 3: Hook shots have ≤10 word narration (meta/dialogue, not visual).
        # Inject first_event_context so the LLM generates a visual for the story's
        # actual opening moment rather than a generic "looking around" image.
        if first_event_context and i <= 1 and shot.duration_sec <= 3:
            entry["visual_context"] = first_event_context
            entry["note"] = (
                "narration_text is short meta-dialogue. "
                "Extract visual from visual_context instead, not narration_text."
            )
        payload.append(entry)

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
                "actions": item.get("actions") or item.get("action") or [],
                "setting": setting_val,
                "key_objects": item.get("key_objects") or [],
                "mood_lighting": item.get("mood_lighting") or "",
                "composition": composition_val,
            })

            # 4a: Reject briefs whose mood_lighting violates horror tone,
            # unless narration explicitly describes a safe daytime scene.
            mood_lower = brief.mood_lighting.lower()
            narration_lower = shots[idx].narration_text.lower()
            if any(v in mood_lower for v in _HORROR_MOOD_VIOLATIONS):
                if not any(kw in narration_lower for kw in _DAYTIME_SAFE_KEYWORDS):
                    logger.warning(
                        "Visual brief mood_lighting violates horror tone (%r) — skipping brief "
                        "and keeping existing scene_prompt | episode={} shot={}",
                        brief.mood_lighting[:80], episode_num, idx,
                    )
                    continue  # brief NOT saved; shot keeps previous scene_prompt

            shots[idx] = shots[idx].model_copy(update={"visual_brief": brief})
            populated += 1

            # Track quality metrics.
            primary_action = brief.actions[0] if brief.actions else ""
            if not primary_action:
                empty_action += 1
            if not brief.key_objects:
                empty_objects += 1
            if not brief.mood_lighting:
                empty_mood += 1

            logger.debug(
                "Visual brief extracted | episode={} shot={} actions={} objects={} mood={!r}",
                episode_num, idx, brief.actions, brief.key_objects, brief.mood_lighting[:60],
            )

            if not primary_action:
                logger.warning(
                    "Visual brief has empty actions | episode={} shot={}", episode_num, idx
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
    first_event_context: str | None = None,
) -> List[ShotScript]:
    """Two-pass replacement for _rewrite_scene_prompts_from_narration.

    Pass 1: LLM extraction — narration_text → ShotVisualBrief (structured semantics)
    Pass 2: Python synthesis — ShotVisualBrief → ComfyUI tag list (no LLM)

    Falls back to _rewrite_scene_prompts_from_narration() if extraction yields 0 briefs.
    first_event_context is forwarded to _extract_visual_briefs for Phase 3 hook anchoring.
    """
    shots = _extract_visual_briefs(shots, episode_num, first_event_context=first_event_context)
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


def _validate_scene_id_consistency(
    shots: List[ShotScript],
    episode_num: int,
) -> None:
    """4b: Warn when a shot's visual_brief.setting has no keyword overlap with its scene_id.

    This is a diagnostic pass — it does NOT modify shots. The warning surfaces
    LLM scene_id / setting mismatches early so they can be corrected in the prompt.
    """
    for i, shot in enumerate(shots):
        sid = (shot.scene_id or "").strip()
        if not sid or shot.visual_brief is None:
            continue
        setting_lower = shot.visual_brief.setting.lower()
        # Expand underscores in scene_id to individual keywords.
        sid_keywords = [kw for kw in sid.replace("_", " ").split() if len(kw) >= 3]
        if not sid_keywords:
            continue
        if not any(kw in setting_lower for kw in sid_keywords):
            logger.warning(
                "scene_id/setting mismatch | episode={} shot={} scene_id={!r} setting={!r}",
                episode_num, i, sid, shot.visual_brief.setting[:80],
            )


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
        # Guard: curr_loc and curr_light may be the same tag when a single tag
        # contains both a location marker and a lighting marker (e.g.
        # "dimly lit village..."). In that case remove it once, not twice.
        stale_tags: set[str] = set()
        if not loc_ok and curr_loc:
            stale_tags.add(curr_loc.lower())
        if not light_ok and curr_light and curr_light.lower() != curr_loc.lower():
            stale_tags.add(curr_light.lower())
        body = [t for t in body if t.lower() not in stale_tags]

        # Inject canon anchors at the front of body (right after safety prefix).
        # Guard: canon_loc and canon_light may resolve to the same tag — inject once.
        injected: list[str] = []
        if not loc_ok and canon_loc:
            injected.append(canon_loc)
        if not light_ok and canon_light and canon_light.lower() != canon_loc.lower():
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


def write_episode_script(
    episode_num: int,
    viral_moments: Optional[List[ViralMoment]] = None,
) -> EpisodeScript:
    """Generate shot script. Caps at _MAX_SHOTS_PER_EPISODE; excess saved as carry-over.
    First checks carry-over from previous episode before calling LLM.
    Always generates a fresh hook shot (shot 0) regardless of carry-over state.

    `viral_moments` (optional, behind `retention.use_constraint_system` flag) are
    appended to the arc context as hook bias. When None, behavior is identical to
    legacy (faithful summary-driven script).
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

    # Phase 2: append viral-moment hook bias when constraint system is enabled.
    # Narrative spine still comes from arc_text above; this is hook + key-shot bias only.
    if viral_moments:
        moments_text = "\n".join(
            f"- [shock] {m.description}  (mystery: {m.mystery_seed})"
            for m in viral_moments[:5]
        )
        arc_text = (
            arc_text
            + "\n\nVIRAL MOMENT CANDIDATES (use ONLY for hook shot 0 + at most 1-2 key shots; "
            "keep the rest of the script faithful to the ordered scenes above):\n"
            + moments_text
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

    # 4. Always force-regenerate shot 0 as a fresh hook for this episode.
    # Phase 4: when constraint system is on, run competitive selection (3 candidates
    # → judge → pick best) instead of a single LLM call. Falls back to legacy on
    # hard failure so we never lose the hook entirely.
    logger.info("Generating hook shot | episode={}", episode_num)
    hook_shot: Optional[ShotScript] = None
    selected_hook_strength: Optional[float] = None
    if settings.retention.use_constraint_system:
        try:
            from llm.hook_judge import select_hook
            result = select_hook(arc_text, episode_num)
            if result is not None:
                winner, _all = result
                hook_shot = ShotScript(
                    scene_prompt=winner.visual_seed or "close-up shot",
                    narration_text=winner.text,
                    duration_sec=3.0,
                    is_key_shot=False,
                    characters=[],
                    camera_flow=CameraFlow.STATIC_CLOSE,
                )
                selected_hook_strength = winner.total_score
                logger.info(
                    "Competitive hook selected | episode={} score={:.3f} text={!r}",
                    episode_num, winner.total_score, winner.text,
                )
            else:
                logger.warning(
                    "Competitive hook returned no winner; falling back to legacy | episode={}",
                    episode_num,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Competitive hook failed, falling back to legacy | episode={} err={}",
                episode_num, e,
            )

    if hook_shot is None:
        try:
            hook_shot = _generate_hook_shot(arc_text, episode_num)
        except Exception:
            logger.warning(
                "Hook shot generation failed, keeping carry-over shot 0 | episode={}",
                episode_num,
            )

    if hook_shot is not None:
        episode_shots[0] = hook_shot
        logger.debug(
            "Hook shot injected | episode={} narration={!r}",
            episode_num, hook_shot.narration_text,
        )

    # 5. Normalize in correct order — all on episode_shots (not new_shots)
    episode_shots = _normalize_duration(episode_shots, episode_num)
    episode_shots = _normalize_hook_durations(episode_shots, episode_num)
    episode_shots = _validate_narration_length(episode_shots, episode_num)
    episode_shots = _normalize_key_shots(episode_shots, episode_num)
    episode_shots = _normalize_camera_flow(episode_shots, episode_num)
    episode_shots = _backfill_characters(episode_shots, arc.characters_in_episode, episode_num)

    # 5b. Hard total-words gate — final assembled shots (includes carry-over).
    # Prevents running expensive image/tts/video phases on scripts that will
    # fail the downstream duration validator. 200 words ≈ 57s TTS @ 3.5 wps.
    _total_words = sum(len(s.narration_text.split()) for s in episode_shots)
    _min_total = 200
    if _total_words < _min_total:
        per_shot = [
            f"shot{i+1}={len(s.narration_text.split())}w"
            for i, s in enumerate(episode_shots)
        ]
        raise ValueError(
            f"Episode {episode_num} script too short: total_words={_total_words} "
            f"< {_min_total} (≈ {_total_words/3.5:.1f}s TTS, need ≥57s). "
            f"Per-shot: {per_shot}. "
            f"Rerun `--from-phase llm` to regenerate; if this persists, carry-over "
            f"from previous episode may be too short and LLM retries exhausted."
        )
    # Phase 1: LLM character resolution — distinguish subject vs object_visible per shot
    episode_shots = _resolve_shot_characters(episode_shots, arc.characters_in_episode, episode_num)
    # Phase 3: Build first-event context from arc key_events for hook shot anchoring.
    # Hook narration is often short meta-dialogue (≤10 words) — the LLM needs the
    # actual first story event to produce a visually meaningful image.
    first_event_context: str | None = None
    if arc.key_events:
        first_event_context = arc.key_events[0]
        logger.debug(
            "Hook visual context set from arc.key_events[0] | episode={} context={!r}",
            episode_num, first_event_context[:120],
        )

    # Two-pass: LLM extraction (narration → brief) + Python synthesis (brief → tags)
    episode_shots = _build_scene_prompts_from_narration(
        episode_shots, episode_num, first_event_context=first_event_context
    )
    # 4b: Warn when scene_id and brief.setting have no keyword overlap (diagnostic only)
    _validate_scene_id_consistency(episode_shots, episode_num)
    # Rule-based layer: inject specific artifact/object tags that LLM might miss
    episode_shots = _align_scene_prompt_with_narration(episode_shots, episode_num)
    # Deterministic continuity: enforce shared location+lighting within each scene_id group
    episode_shots = _enforce_scene_continuity(episode_shots, episode_num)

    script = EpisodeScript(
        episode_num=episode_num,
        title=raw_title,
        shots=episode_shots,
    )

    # Phase 3: populate constraint signals (energy_level, exposition_ratio,
    # proper_nouns) when feature flag is on. Validator runs at script_review
    # phase; here we just compute the per-shot signals so they persist to JSON.
    if settings.retention.use_constraint_system:
        from llm.constraint_validator import populate_shot_signals
        known_chars = list(arc.characters_in_episode or [])
        for shot in script.shots:
            populate_shot_signals(shot, known_chars)
        script.prompt_version = _PROMPT_VERSION_V2
        if selected_hook_strength is not None:
            script.hook_strength = selected_hook_strength
    else:
        script.prompt_version = _PROMPT_VERSION_LEGACY

    script_path = (
        Path(settings.data_dir)
        / "scripts"
        / f"episode-{episode_num:03d}-script.json"
    )
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script.model_dump_json(indent=2), encoding="utf-8")

    logger.info("Script written | episode={} shots={}", episode_num, len(script.shots))

    # Phase 5: Write alignment report for manual review of narration ↔ image mapping.
    try:
        from pipeline.alignment_report import write_alignment_report
        write_alignment_report(script, episode_num)
    except Exception as exc:
        logger.warning("Alignment report failed (non-fatal): {} | episode={}", exc, episode_num)

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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=5),
    retry=retry_if_exception_type((json.JSONDecodeError, ValueError)),
)
def _resolve_shot_characters(
    shots: List[ShotScript],
    arc_characters: List[str],
    episode_num: int,
) -> List[ShotScript]:
    """Phase 1: LLM pass that resolves which characters are physically visible per shot.

    Distinguishes subject (performs action) vs object_visible (present in scene).
    Saves raw LLM characters into characters_raw for debug, then overrides characters.
    Falls back to existing characters list on any failure.
    """
    if not arc_characters:
        logger.debug("No arc characters — skipping character resolution | episode={}", episode_num)
        return shots

    payload = []
    for i, shot in enumerate(shots):
        payload.append({
            "shot_index": i,
            "narration_text": shot.narration_text,
            "llm_characters": shot.characters,
        })

    prompt = (
        f"Resolve visible characters for Episode {episode_num} shots.\n\n"
        f"Known characters in this arc: {arc_characters}\n\n"
        f"Shots:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    try:
        raw = scene_prompt_client.generate_json(
            prompt=prompt, system=_CHARACTER_RESOLVE_SYSTEM, temperature=0.1
        )
    except Exception as exc:
        logger.warning(
            "Character resolution LLM call failed ({}) — keeping existing | episode={}",
            exc, episode_num,
        )
        return shots

    if not isinstance(raw, list):
        logger.warning(
            "Character resolution returned non-list (type={}) — keeping existing | episode={}",
            type(raw).__name__, episode_num,
        )
        return shots

    shots = list(shots)
    resolved = 0
    changed = 0

    for item in raw:
        if not isinstance(item, dict):
            continue
        idx = item.get("shot_index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(shots):
            continue

        new_chars = item.get("characters")
        if not isinstance(new_chars, list):
            continue

        # Validate: all returned names must be in arc_characters (exact match).
        # Unknown names are logged and dropped to prevent hallucinations.
        # Also strip Vietnamese honorifics + drop placeholders so "Lão đạo sĩ X"
        # resolves to "X" and "Người đàn ông bí ẩn" is filtered out.
        from llm.summarizer import _strip_vn_honorifics, _is_placeholder_character
        arc_lower = {n.lower(): n for n in arc_characters}
        valid_chars: List[str] = []
        for name in new_chars:
            if not isinstance(name, str) or not name.strip():
                continue
            low = name.strip().lower()
            if _is_placeholder_character(low):
                logger.debug(
                    "Character resolution dropped placeholder {!r} | episode={} shot={}",
                    name, episode_num, idx,
                )
                continue
            # Exact match first
            canonical = arc_lower.get(low)
            # Honorific-stripped match (handles "Lão đạo sĩ Thanh Vân Tử" → "Thanh Vân Tử")
            if canonical is None:
                stripped = _strip_vn_honorifics(name).lower()
                canonical = arc_lower.get(stripped)
            if canonical is None:
                logger.warning(
                    "Character resolution returned unknown name {!r} — dropped | episode={} shot={}",
                    name, episode_num, idx,
                )
                continue
            if canonical not in valid_chars:
                valid_chars.append(canonical)
        # Enforce cap of 2 (IPAdapter dual workflow limit)
        valid_chars = valid_chars[:2]

        old_chars = shots[idx].characters
        # Save raw LLM list before override (for debug / alignment report)
        shots[idx] = shots[idx].model_copy(update={
            "characters_raw": list(old_chars),
            "characters": valid_chars,
        })
        resolved += 1
        if set(valid_chars) != set(old_chars):
            changed += 1
            logger.debug(
                "Characters updated | episode={} shot={} old={} new={}",
                episode_num, idx, old_chars, valid_chars,
            )

    logger.info(
        "Character resolution done | episode={} resolved={}/{} changed={}",
        episode_num, resolved, len(shots), changed,
    )
    return shots


def _has_concrete_action(shot: ShotScript) -> bool:
    """Return True if the shot's visual_brief has a non-generic concrete action.

    When True, the rule-based action-injection pass is skipped to prevent tag
    leaking from previous shots or overriding an already-specific action.
    Object-rule injection still runs regardless.
    """
    brief = shot.visual_brief
    if brief is None:
        return False
    primary_action = brief.actions[0].strip().lower() if brief.actions else ""
    if not primary_action:
        return False
    # If the action is in the generic set, it is NOT concrete enough
    return not any(generic in primary_action for generic in _GENERIC_ACTIONS)


def _align_scene_prompt_with_narration(
    shots: List[ShotScript],
    episode_num: int,
) -> List[ShotScript]:
    """Inject missing visual tags into scene_prompt based on narration_text.

    Two-tier pass:
    1. ACTION rules — insert at position 2 (right after location).
       SKIPPED when the shot already has a concrete (non-generic) visual_brief
       action, to prevent tag leaking across shots.
    2. OBJECT rules — append at end (always runs).
    """
    aligned = 0
    updated: List[ShotScript] = []

    for shot in shots:
        scene_tags = [t.strip() for t in shot.scene_prompt.split(",") if t.strip()]
        scene_lower = {t.lower() for t in scene_tags}
        narration = shot.narration_text

        action_tags: List[str] = []
        object_tags: List[str] = []

        # Action rules always run: narration keywords (e.g. "khai quật", "nạy nắp")
        # produce more specific visual tags than the LLM brief.
        # Deduplication is handled by the `tag.lower() not in scene_lower` guard.
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
                scene_tags.insert(insert_at + i, tag)

        # Append object tags at end — no metadata suffix to worry about anymore.
        scene_lower_now = {t.lower() for t in scene_tags}
        for tag in object_tags[:6]:
            if tag.lower() not in scene_lower_now:
                scene_tags.append(tag)
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


# ── Phase 5: Targeted shot regeneration for gatekeeper retry ────────────────

_REGEN_SYSTEM = """You are a Vietnamese supernatural-horror Shorts scriptwriter rewriting failing shots.

You will receive a list of shots that violated retention constraints. For EACH shot:
- Rewrite `narration_text` (Vietnamese, first-person "Tôi...") to be MORE TENSE and MORE MYSTERIOUS — do not just "fix" the violation, IMPROVE the storytelling.
- Keep the same key event / character context.
- AVOID repeating the violation pattern (described in `violations`).

Specific anti-patterns by violation code:
- hook_language: hook MUST be Vietnamese with diacritics. Maximum 10 words.
- exposition_density: too much backstory/lore. Replace with concrete sensory detail (sound, sight, touch).
- lore_before_curiosity: this shot dumps lore before the audience asks. Plant a question instead.

OUTPUT — JSON array, one object per input shot, same order:
[{"shot_index": <int>, "narration_text": "<Vietnamese>", "scene_prompt": "<English tags>"}]
Return ONLY the JSON array."""


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(min=2, max=10),
    retry=retry_if_exception_type((json.JSONDecodeError, ValueError)),
)
def regenerate_failed_shots(
    script: EpisodeScript,
    blocking_violations: List,  # List[Violation] — avoid circular import
    episode_num: int,
) -> EpisodeScript:
    """Rewrite shots flagged by the gatekeeper. Returns a new EpisodeScript.

    Only shots with at least one BLOCKING violation are sent to the LLM.
    Other shots are preserved verbatim.
    """
    # Group violations by shot index. Episode-level (-1) violations are skipped here.
    by_shot: dict = {}
    for v in blocking_violations:
        if getattr(v, "shot_index", -1) >= 0:
            by_shot.setdefault(v.shot_index, []).append(v)

    if not by_shot:
        return script

    payload = []
    for idx in sorted(by_shot.keys()):
        if idx >= len(script.shots):
            continue
        shot = script.shots[idx]
        payload.append({
            "shot_index": idx,
            "violations": [
                {"rule": v.rule, "msg": v.message} for v in by_shot[idx]
            ],
            "narration_text": shot.narration_text,
            "scene_prompt": shot.scene_prompt,
        })

    if not payload:
        return script

    prompt = (
        f"Episode {episode_num}: rewrite the following {len(payload)} failing shot(s).\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    logger.info(
        "Regenerating {} shot(s) via gatekeeper retry | episode={} indices={}",
        len(payload), episode_num, sorted(by_shot.keys()),
    )

    result = ollama_client.generate_json(prompt=prompt, system=_REGEN_SYSTEM, temperature=0.7)
    if not isinstance(result, list):
        raise ValueError(f"regen LLM returned non-list: {type(result)}")

    fixes_by_index: dict = {}
    for item in result:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("shot_index", -1))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(script.shots):
            continue
        narration = str(item.get("narration_text", "") or "").strip()
        scene_prompt = str(item.get("scene_prompt", "") or "").strip()
        if not narration:
            continue
        fixes_by_index[idx] = {
            "narration_text": narration,
            "scene_prompt": scene_prompt or script.shots[idx].scene_prompt,
        }

    if not fixes_by_index:
        raise ValueError("regen LLM returned no usable fixes")

    new_shots = []
    for i, shot in enumerate(script.shots):
        if i in fixes_by_index:
            new_shots.append(shot.model_copy(update=fixes_by_index[i]))
        else:
            new_shots.append(shot)

    return script.model_copy(update={"shots": new_shots})


def _save_script_after_review(script: EpisodeScript, episode_num: int) -> None:
    """Persist a script after gatekeeper review (with constraint_violations attached)."""
    script_path = (
        Path(settings.data_dir)
        / "scripts"
        / f"episode-{episode_num:03d}-script.json"
    )
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "Script persisted post-review | episode={} violations={}",
        episode_num, len(script.constraint_violations),
    )
