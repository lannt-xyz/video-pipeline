Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/.github/instructions/pipeline.instructions.md#1-1), lines 1 to 100

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/.github/instructions/python.instructions.md#1-1), lines 1 to 100

Read skill [code-reviewer](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/.github/skills/code-reviewer/SKILL.md?vscodeLinkType%3Dskill#1-1), lines 1 to 100

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/image_gen/character_gen.py)

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/image_gen/comfyui_client.py)

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/models/schemas.py)

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/llm/scriptwriter.py#1-1), lines 1 to 300

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/llm/character_extractor.py#1-1), lines 1 to 200

Now let me read the workflow files and the orchestrator to understand how anchors are used during scene image generation:    

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/image_gen/workflows/txt2img_ipadapter.json)

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/image_gen/workflows/anchor_gen.json)

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/image_gen/workflows/txt2img_scene.json)

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/pipeline/orchestrator.py#1-1), lines 1 to 200

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/pipeline/orchestrator.py#200-200), lines 200 to 350

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/data/characters/diep_dai_bao.json)

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/image_gen/workflows/txt2img_ipadapter_dual.json)

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/data/scripts/episode-001-script.json#1-1), lines 1 to 30

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/data/scripts/episode-001-script.json#30-30), lines 30 to 70

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/data/characters/thanh_van_tu.json)

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/data/characters/xao_van.json)

Now let me check the existing generated images and look at the full flow more carefully:    

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/data/images/episode-001)

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/data/characters/diep_dai_bao)

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/data/characters/thanh_van_tu)

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/data/characters/xao_van)

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/config/settings.yaml#1-1), lines 1 to 20

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/config/settings.py#1-1), lines 1 to 80

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/llm/character_extractor.py#170-170), lines 170 to 260

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/llm/character_extractor.py#260-260), lines 260 to 320

Read [](file:///media/lanntxyz/E0E2753BE27516CE1/Lanntxyz/projects/video-pipiline/llm/character_extractor.py#320-320), lines 320 to 373

Tôi đã đọc toàn bộ pipeline từ character extraction → anchor generation → scene image generation. Đây là phân tích chi tiết các vấn đề gây ra **character anchor sinh ra khác với hình video**.

---

## Issues Found

### [CRITICAL] 1. Anchor image là full-body portrait — FaceID extract face embedding cực yếu
- File: character_gen.py
- **Problem**: Anchor prompt chứa `"full body portrait"` ở resolution 720x1280. Face chỉ chiếm ~100-150px trong ảnh. `InsightFace buffalo_l` cần face lớn để extract embeddings chất lượng cao. IPAdapter FaceID Plus V2 hoạt động tốt nhất với **close-up face reference**, không phải full body.
- **Fix**: Đổi anchor prompt thành close-up portrait (`close-up portrait, face focus, upper body`) thay vì `full body portrait`.

### [CRITICAL] 2. `upload_image()` ghi đè cùng 1 filename "anchor.png" cho tất cả characters
- File: comfyui_client.py
- **Problem**: Mọi character đều có anchor tại `data/characters/{slug}/anchor.png`. Khi upload lên ComfyUI, `path.name` luôn = `"anchor.png"`. Dù serial execution tạm an toàn do overwrite, nhưng ComfyUI có internal caching — nếu 2 shots liên tiếp reference 2 characters khác nhau, LoadImage node có thể load cached version của character trước đó thay vì character mới upload.
- **Fix**: Upload với tên unique: `{char_slug}_anchor.png` thay vì `anchor.png`.

### [HIGH] 3. "solo" tag trong DNA conflicts với multi-character scenes  
- File: orchestrator.py
- **Problem**: `_extract_dna_tags()` giữ lại `"solo"` từ character description. Trong dual-IPAdapter workflow, prompt thành: `"1boy, solo, ..., 1girl, solo, ..."`. Tag `solo` = Danbooru tag nghĩa "chỉ 1 người" → **trực tiếp contradict** mục đích hiển thị 2 characters.
- **Fix**: Filter `"solo"`, `"1boy"`, `"1girl"` khỏi DNA tags khi dùng dual-character workflow. Thay bằng `"2boys"`, `"1boy 1girl"`, etc. tùy gender combo.

### [HIGH] 4. txt2img_scene.json chỉ có 2 LoRAs vs anchor/ipadapter có 4 LoRAs
- File: txt2img_scene.json
- **Problem**: Scene-only workflow dùng `ink_wash_painting + xianxia_fantasy` (2 LoRAs). Anchor + IPAdapter workflows dùng thêm `manhua_style_v2 + dark_horror_atmosphere` (4 LoRAs). → **Aesthetic domain khác nhau** giữa scene-only shots và character shots trong cùng 1 episode = visual inconsistency.
- **Fix**: Thêm 2 LoRAs còn thiếu vào txt2img_scene.json để đồng bộ style.

### [HIGH] 5. `_extract_dna_tags()` không filter role/profession/semantic tags
- File: orchestrator.py
- **Problem**: `_OUTFIT_KEYWORDS` chỉ filter outfit/accessory. Tags như `"craftsman"`, `"carpenter"`, `"daoist master"`, `"intense expression"`, `"determined"` ở trong character description KHÔNG bị filter. Đây là semantic/profession tags, không phải visual identity — chúng **conflict với scene context** và dilute CLIP attention away from face features.
- **Fix**: Mở rộng filter list hoặc dùng whitelist approach — chỉ giữ hair/eyes/face/body/gender tags.

### [MEDIUM] 6. InsightFace `buffalo_l` không designed cho anime faces
- File: txt2img_ipadapter.json
- **Problem**: IPAdapter FaceID dựa vào InsightFace (trained trên real human faces) để extract face embeddings. Với PonyDiffusion anime output, face recognition reliability thấp → face embeddings yếu hoặc inconsistent → character identity không transfer tốt từ anchor sang scene.
- **Recommendation**: Cân nhắc switch sang `IPAdapterStyleComposition` hoặc `IPAdapter Plus` (CLIP-vision based, không dùng InsightFace) — hoạt động tốt hơn nhiều với anime faces. FaceID chỉ hiệu quả với realistic style.

### [MEDIUM] 7. Không có face-crop preprocessing cho anchor trước khi feed IPAdapter
- File: orchestrator.py 
- **Problem**: Anchor full-body image (720x1280) được upload thẳng cho IPAdapter. Không có bước crop/resize face region. IPAdapter FaceID extract từ ảnh gốc — face nhỏ, background noise lớn → embedding bị dilute.
- **Fix**: Thêm bước crop face region từ anchor image trước khi upload (dùng InsightFace detect → crop → resize lên 512x512).

---

## Tóm tắt nguyên nhân gốc

Pipeline hiện tại có **2 vấn đề cốt lõi** khiến character trong video khác hoàn toàn với anchor:

1. **FaceID + anime = unreliable**: Dùng InsightFace (real face) cho PonyDiffusion (anime) → face embeddings cực yếu. Đây là architectural mismatch, không fix được bằng tuning weight.

2. **Anchor image quality kém cho reference**: Full body portrait ở 720x1280 → face quá nhỏ. Upload filename collisions có thể load sai anchor.

**Sếp muốn tôi fix tất cả issues trên không, hay chỉ focus vào các CRITICAL/HIGH trước?**