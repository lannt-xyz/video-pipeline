import re
from typing import List

from config.settings import settings
from models.schemas import CameraFlow, FrameScript, MotionDirection, ShotScript

# Minimum shot duration (seconds) to warrant multi-frame crossfade.
# Hook shots (≤3s) are too short — crossfade would eat most of the clip.
_MIN_DURATION_FOR_MULTI_FRAME = 4.0

# Max visual concepts SDXL can compose faithfully in a single image.
# Above this, the model drops or merges concepts randomly.
_MAX_CONCEPTS_PER_FRAME = 5

# Camera flow → (frame_configs) mapping.
# Each entry: (camera_tag_prefix, motion_direction)
_FLOW_FRAMES = {
    CameraFlow.WIDE_TO_CLOSE: [
        # Establish environment → medium subject → tighter story beat.
        ("wide establishing shot, ", MotionDirection.ZOOM_IN),
        ("wide shot, ", MotionDirection.ZOOM_IN),
        ("medium shot, ", MotionDirection.ZOOM_IN),
        ("medium close-up, ", MotionDirection.ZOOM_IN),
    ],
    CameraFlow.CLOSE_TO_WIDE: [
        # Reveal pattern: detail first, then pull out to space context.
        ("medium close-up, ", MotionDirection.ZOOM_OUT),
        ("medium shot, ", MotionDirection.ZOOM_OUT),
        ("wide shot, ", MotionDirection.ZOOM_OUT),
        ("wide shot, ", MotionDirection.ZOOM_OUT),
    ],
    CameraFlow.PAN_ACROSS: [
        ("left side wide shot, ", MotionDirection.PAN_RIGHT),
        ("left-center wide shot, ", MotionDirection.PAN_RIGHT),
        ("right-center wide shot, ", MotionDirection.PAN_LEFT),
        ("right side wide shot, ", MotionDirection.PAN_LEFT),
    ],
    CameraFlow.DETAIL_REVEAL: [
        # Was "extreme close-up detail" — caused background to disappear entirely.
        # "close-up detail" still focuses on object but retains surrounding context.
        ("close-up detail, ", MotionDirection.ZOOM_OUT),
        ("medium close-up, ", MotionDirection.ZOOM_OUT),
        ("medium shot, ", MotionDirection.ZOOM_OUT),
        ("medium wide shot, ", MotionDirection.ZOOM_OUT),
    ],
    CameraFlow.STATIC_CLOSE: [
        ("medium close-up, ", MotionDirection.ZOOM_IN),
    ],
    CameraFlow.STATIC_WIDE: [
        ("wide establishing shot, ", MotionDirection.PAN_RIGHT),
    ],
}


def _split_tags(scene_prompt: str) -> tuple[list[str], list[str], list[str]]:
    """Split scene_prompt tags into (environment, action, object) groups.

    - environment: location, lighting, atmosphere tags (always shared across frames)
    - action: tags containing figure/person action verbs
    - object: weighted prop/detail tags like (coffin:1.2)
    """
    tags = [t.strip() for t in scene_prompt.split(",") if t.strip()]
    env: list[str] = []
    action: list[str] = []
    obj: list[str] = []

    _LOCATION_WORDS = frozenset([
        "at night", "interior", "lit ", "dimly", "village", "graveyard",
        "forest", "temple", "house", "tomb", "shrine", "courtyard",
        "moonlight", "room", "hall", "cave", "mountain", "river",
        "bridge", "road", "path", "alley", "market",
    ])

    _ACTION_WORDS = frozenset([
        "figure", "kneeling", "lunging", "striking", "crouching", "chanting",
        "recoiling", "pointing", "examining", "pulling", "prying", "running",
        "standing before", "questioning", "confrontation", "clutching",
        "wielding", "raising", "slashing", "diving", "reaching",
        "woman", "man ", "girl ", "boy ",
    ])

    _OBJECT_WORDS = frozenset([
        "scattered", "talisman", "candle", "coffin", "corpse", "incense",
        "bones", "grave", "ritual", "blood", "chain", "lantern", "mirror",
        "dagger", "sword", "staff", "banner", "flag", "child body",
        "compass", "basin", "gateway", "tablet",
    ])

    for tag in tags:
        lower = tag.lower()
        # Weighted tags like (coffin:1.2) are specific concepts
        if re.search(r"\(.*:\d+\.\d+\)", tag):
            if any(w in lower for w in _ACTION_WORDS):
                action.append(tag)
            else:
                obj.append(tag)
        # Location/atmosphere tags are always environment (shared)
        elif any(w in lower for w in _LOCATION_WORDS):
            env.append(tag)
        elif any(w in lower for w in _ACTION_WORDS):
            action.append(tag)
        elif any(w in lower for w in _OBJECT_WORDS):
            obj.append(tag)
        else:
            env.append(tag)

    return env, action, obj


def _build_frame_prompts(
    scene_prompt: str,
    flow_configs: list[tuple[str, MotionDirection]],
    num_frames: int,
) -> list[FrameScript]:
    """Build frame prompts with content-aware tag distribution.

    When a scene_prompt has many concepts (>MAX_CONCEPTS_PER_FRAME total
    action+object tags), distribute them across frames so each frame has
    ≤MAX_CONCEPTS_PER_FRAME concepts. Environment tags are always shared.

    When the prompt is simple enough, all frames share the same content
    (original behavior — only camera angle differs).
    """
    env, actions, objects = _split_tags(scene_prompt)

    total_concepts = len(actions) + len(objects)

    # Simple prompt: every frame gets everything (original behavior)
    if total_concepts <= _MAX_CONCEPTS_PER_FRAME:
        frames = []
        for tag, motion in flow_configs[:num_frames]:
            frames.append(
                FrameScript(
                    scene_prompt=f"{tag}{scene_prompt}",
                    camera_tag=tag.rstrip(", "),
                    motion=motion,
                )
            )
        return frames

    # Complex prompt: distribute action+object tags across frames.
    # Each frame gets: env + ≤1 action + its share of objects.
    # This ensures SDXL focuses on one clear action per frame.

    # Step 1: Distribute actions — max 1 per frame, in order
    per_frame_actions: list[list[str]] = [[] for _ in range(num_frames)]
    for i, act in enumerate(actions[:num_frames]):
        per_frame_actions[i].append(act)

    # Step 2: Distribute objects round-robin across frames
    per_frame_objects: list[list[str]] = [[] for _ in range(num_frames)]
    for i, obj_tag in enumerate(objects):
        target = i % num_frames
        per_frame_objects[target].append(obj_tag)

    # Step 3: Build per-frame concept lists, respecting MAX_CONCEPTS_PER_FRAME
    per_frame: list[list[str]] = []
    for fidx in range(num_frames):
        concepts = per_frame_actions[fidx] + per_frame_objects[fidx]
        per_frame.append(concepts[:_MAX_CONCEPTS_PER_FRAME])

    frames = []
    for fidx, (cam_config, concepts) in enumerate(
        zip(flow_configs[:num_frames], per_frame)
    ):
        tag, motion = cam_config
        # Build frame prompt: camera_tag + environment + this frame's concepts
        frame_tags = [tag.rstrip(", ")] + env + concepts
        frames.append(
            FrameScript(
                scene_prompt=", ".join(frame_tags),
                camera_tag=tag.rstrip(", "),
                motion=motion,
            )
        )

    return frames


def decompose_shot(shot: ShotScript) -> List[FrameScript]:
    """Generate frame prompts from a shot's scene_prompt and camera_flow.

    Returns 1 frame for short shots (<4s) or static flows,
    up to settings.frames_per_shot frames for longer shots with dynamic camera flows.
    Deterministic — no LLM calls.

    When visual_brief.actions has multiple entries, each action is injected into
    the corresponding frame so the image visually matches the narration timeline.
    """
    flow_configs = _FLOW_FRAMES.get(shot.camera_flow, _FLOW_FRAMES[CameraFlow.WIDE_TO_CLOSE])

    # Short shots or single-frame flows: only use the first frame config
    if shot.duration_sec < _MIN_DURATION_FOR_MULTI_FRAME or len(flow_configs) == 1:
        return _build_frame_prompts(shot.scene_prompt, flow_configs, num_frames=1)

    # Multi-frame: content-aware distribution
    max_frames = min(settings.frames_per_shot, len(flow_configs))
    frames = _build_frame_prompts(shot.scene_prompt, flow_configs, num_frames=max_frames)

    # Per-frame action override from visual_brief.actions (Phase 2).
    # If brief has 2+ actions, replace the action tag in frame[i] with actions[i].
    brief_actions = (
        shot.visual_brief.actions
        if shot.visual_brief and len(shot.visual_brief.actions) > 1
        else None
    )
    if not brief_actions:
        return frames

    updated_frames: list[FrameScript] = []
    for i, frame in enumerate(frames):
        if i >= len(brief_actions):
            updated_frames.append(frame)
            continue
        action_tag = brief_actions[i].strip()
        if not action_tag:
            updated_frames.append(frame)
            continue
        # Extract scene_prompt minus old primary action (actions[0]) and inject new per-frame action.
        primary_action = brief_actions[0].strip()
        old_tags = [t.strip() for t in frame.scene_prompt.split(",") if t.strip()]
        # Remove the primary action tag from all frames (each frame gets its own action now)
        filtered = [t for t in old_tags if t != primary_action]
        # Find camera_tag position (first tag) and inject action immediately after it
        if filtered:
            cam_tag = filtered[0]
            rest = filtered[1:]
            new_tags = [cam_tag, action_tag] + rest
        else:
            new_tags = [action_tag]
        updated_frames.append(
            frame.model_copy(update={"scene_prompt": ", ".join(new_tags)})
        )

    return updated_frames


def decompose_all_shots(shots: List[ShotScript]) -> List[ShotScript]:
    """Populate frames for all shots in-place-style (returns new list with frames set)."""
    result = []
    for shot in shots:
        frames = decompose_shot(shot)
        result.append(shot.model_copy(update={"frames": frames}))
    return result
