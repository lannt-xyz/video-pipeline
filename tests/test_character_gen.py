from models.schemas import Character
from image_gen.character_gen import _build_anchor_scene_prompt


class TestCharacterAnchorPrompt:
    def test_anchor_prompt_has_east_asian_bias(self):
        char = Character(
            name="Diep Thieu Duong",
            gender="male",
            description="1boy, solo, short black hair, sharp eyes, dark jacket",
        )
        prompt = _build_anchor_scene_prompt(char, "looking at viewer, front view")

        assert "east asian facial features" in prompt
        assert "chinese facial features" in prompt
        assert "looking at viewer, front view" in prompt
        assert "manhua art style" in prompt
