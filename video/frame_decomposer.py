from typing import List

from config.settings import settings
from models.schemas import CameraFlow, EnergyLevel, FrameScript, MotionDirection, ShotScript

# Minimum shot duration (seconds) to warrant multi-frame crossfade.
# Hook shots (≤3s) are too short — crossfade would eat most of the clip.
_MIN_DURATION_FOR_MULTI_FRAME = 4.0

# Max visual concepts SDXL can compose faithfully in a single image.
# Above this, the model drops or merges concepts randomly.
_MAX_CONCEPTS_PER_FRAME = 5

# Camera flow → 2-frame configs: (opening_camera_tag, motion), (closing_camera_tag, motion)
# Frame 0 = opening state of the action (where it begins)
# Frame 1 = closing state of the action (where it ends / result)
_FLOW_FRAMES = {
    CameraFlow.WIDE_TO_CLOSE: [
        # Start wide to establish the scene, end close to show the story beat.
        ("wide establishing shot, ", MotionDirection.ZOOM_IN),
        ("medium close-up, ", MotionDirection.ZOOM_IN),
    ],
    CameraFlow.CLOSE_TO_WIDE: [
        # Start on the close detail, pull out to reveal full context.
        ("close-up detail, ", MotionDirection.ZOOM_OUT),
        ("wide shot, ", MotionDirection.ZOOM_OUT),
    ],
    CameraFlow.PAN_ACROSS: [
        # Start from one side of the action, end at the other.
        ("left side wide shot, ", MotionDirection.PAN_RIGHT),
        ("right side wide shot, ", MotionDirection.PAN_LEFT),
    ],
    CameraFlow.DETAIL_REVEAL: [
        # Start extreme close-up on the horror object, end medium to show reaction/context.
        ("close-up detail, ", MotionDirection.ZOOM_OUT),
        ("medium shot, ", MotionDirection.ZOOM_OUT),
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
        # Location/atmosphere tags are always environment (shared across frames)
        if any(w in lower for w in _LOCATION_WORDS):
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

    # Phase 3: SHOCK shots must lead with a close-up regardless of camera_flow.
    # Override frame 0 to a close-up tag while preserving frame N-1 / motion. Behind
    # feature flag so legacy behavior is untouched when constraint system is off.
    if (
        settings.retention.use_constraint_system
        and shot.energy_level == EnergyLevel.SHOCK
        and flow_configs
    ):
        first = flow_configs[0]
        first_tag = first[0] if isinstance(first, tuple) else ""
        if "close" not in first_tag.lower():
            shock_tag = ("extreme close-up, ", MotionDirection.ZOOM_IN)
            flow_configs = [shock_tag] + list(flow_configs[1:])

    # Short shots or single-frame flows: only use the first frame config
    if shot.duration_sec < _MIN_DURATION_FOR_MULTI_FRAME or len(flow_configs) == 1:
        return _build_frame_prompts(shot.scene_prompt, flow_configs, num_frames=1)

    # Multi-frame: content-aware distribution
    max_frames = min(settings.frames_per_shot, len(flow_configs))
    frames = _build_frame_prompts(shot.scene_prompt, flow_configs, num_frames=max_frames)

    # Per-frame action override from visual_brief.actions.
    # Frame 0 gets actions[0] (opening action), frame 1 gets actions[-1] (closing action).
    # This ensures the 2 frames depict different moments in the scene narrative.
    brief_actions = (
        shot.visual_brief.actions
        if shot.visual_brief and shot.visual_brief.actions
        else None
    )
    if not brief_actions:
        return frames

    # Map frame index to action index: first frame → first action, last frame → last action
    num_frames = len(frames)
    frame_action_map: list[str] = []
    for i in range(num_frames):
        if num_frames == 1 or len(brief_actions) == 1:
            frame_action_map.append(brief_actions[0])
        elif i == 0:
            frame_action_map.append(brief_actions[0])
        else:
            frame_action_map.append(brief_actions[-1])

    # Skip injection if all frames would get the same action (no meaningful difference)
    if len(set(frame_action_map)) <= 1 and num_frames > 1:
        return frames

    updated_frames: list[FrameScript] = []
    for i, frame in enumerate(frames):
        action_tag = frame_action_map[i].strip()
        if not action_tag:
            updated_frames.append(frame)
            continue
        old_tags = [t.strip() for t in frame.scene_prompt.split(",") if t.strip()]
        # Remove any existing action tags from brief_actions to avoid duplication
        all_brief_lower = {a.strip().lower() for a in brief_actions}
        filtered = [t for t in old_tags if t.lower() not in all_brief_lower]
        # Inject this frame's action immediately after the camera tag (first tag)
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
