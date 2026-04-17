from typing import List

from config.settings import settings
from models.schemas import CameraFlow, FrameScript, MotionDirection, ShotScript

# Minimum shot duration (seconds) to warrant 2 frames with crossfade.
# Hook shots (≤3s) are too short — crossfade would eat most of the clip.
_MIN_DURATION_FOR_MULTI_FRAME = 4.0

# Camera flow → (frame_configs) mapping.
# Each entry: (camera_tag_prefix, motion_direction)
_FLOW_FRAMES = {
    CameraFlow.WIDE_TO_CLOSE: [
        # Frame 0: wide — establish environment; Frame 1: medium — focus on character
        ("wide establishing shot, ", MotionDirection.ZOOM_IN),
        ("medium shot, ", MotionDirection.ZOOM_IN),
    ],
    CameraFlow.CLOSE_TO_WIDE: [
        # Frame 0: medium close-up — enough face + near background visible
        ("medium close-up, ", MotionDirection.ZOOM_OUT),
        ("wide shot, ", MotionDirection.ZOOM_OUT),
    ],
    CameraFlow.PAN_ACROSS: [
        ("left side wide shot, ", MotionDirection.PAN_RIGHT),
        ("right side wide shot, ", MotionDirection.PAN_LEFT),
    ],
    CameraFlow.DETAIL_REVEAL: [
        # Was "extreme close-up detail" — caused background to disappear entirely.
        # "close-up detail" still focuses on object but retains surrounding context.
        ("close-up detail, ", MotionDirection.ZOOM_OUT),
        ("medium wide shot, ", MotionDirection.ZOOM_OUT),
    ],
    CameraFlow.STATIC_CLOSE: [
        ("medium close-up, ", MotionDirection.ZOOM_IN),
    ],
    CameraFlow.STATIC_WIDE: [
        ("wide establishing shot, ", MotionDirection.PAN_RIGHT),
    ],
}


def decompose_shot(shot: ShotScript) -> List[FrameScript]:
    """Generate frame prompts from a shot's scene_prompt and camera_flow.

    Returns 1 frame for short shots (<4s) or static flows,
    2 frames for longer shots with dynamic camera flows.
    Deterministic — no LLM calls.
    """
    flow_configs = _FLOW_FRAMES.get(shot.camera_flow, _FLOW_FRAMES[CameraFlow.WIDE_TO_CLOSE])

    # Short shots or single-frame flows: only use the first frame config
    if shot.duration_sec < _MIN_DURATION_FOR_MULTI_FRAME or len(flow_configs) == 1:
        tag, motion = flow_configs[0]
        return [
            FrameScript(
                scene_prompt=f"{tag}{shot.scene_prompt}",
                camera_tag=tag.rstrip(", "),
                motion=motion,
            )
        ]

    # Multi-frame: cap at settings.frames_per_shot (default 2)
    max_frames = min(settings.frames_per_shot, len(flow_configs))
    frames = []
    for tag, motion in flow_configs[:max_frames]:
        frames.append(
            FrameScript(
                scene_prompt=f"{tag}{shot.scene_prompt}",
                camera_tag=tag.rstrip(", "),
                motion=motion,
            )
        )
    return frames


def decompose_all_shots(shots: List[ShotScript]) -> List[ShotScript]:
    """Populate frames for all shots in-place-style (returns new list with frames set)."""
    result = []
    for shot in shots:
        frames = decompose_shot(shot)
        result.append(shot.model_copy(update={"frames": frames}))
    return result
