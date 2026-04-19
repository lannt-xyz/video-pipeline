from models.schemas import CameraFlow, ShotScript
from video.frame_decomposer import decompose_shot


def test_dynamic_shot_uses_four_frames_when_enabled(monkeypatch):
    from video import frame_decomposer as fd

    monkeypatch.setattr(fd.settings, "frames_per_shot", 4)
    shot = ShotScript(
        scene_prompt="ruined shrine, daoist figure",
        narration_text="Test",
        duration_sec=6,
        camera_flow=CameraFlow.WIDE_TO_CLOSE,
    )

    frames = decompose_shot(shot)
    assert len(frames) == 4


def test_short_hook_shot_keeps_single_frame(monkeypatch):
    from video import frame_decomposer as fd

    monkeypatch.setattr(fd.settings, "frames_per_shot", 4)
    shot = ShotScript(
        scene_prompt="hook scene",
        narration_text="Hook",
        duration_sec=3,
        camera_flow=CameraFlow.STATIC_CLOSE,
    )

    frames = decompose_shot(shot)
    assert len(frames) == 1
