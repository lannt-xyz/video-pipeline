## Gốc rễ: prompt ép phải có người làm chủ thể

1. **Template cứng buộc mọi shot đều center một PERSON.** Structure hiện tại:
   ```
   [location], [action/pose], [foreground], [background], [lighting]
   ```
   → Mỗi tag action đều là "figure doing X". Không có slot cho *"chủ thể = cái xác"*, *"chủ thể = vết cắn"*, *"chủ thể = bàn tay trồi lên đất"*.
   Khi narration nói "**xác chết mắt mở trừng trừng**", LLM vẫn render "figure standing near corpse" — sai focus.

2. **Cấm close-up cực cận** (scriptwriter.py dòng ~118):
   ```
   FORBIDDEN: "extreme close-up", "face close-up", "macro shot"
   ```
   Đây là thứ KHIẾN ảnh shock nhất bị cấm. Close-up mắt trợn, close-up bàn tay tím tái, macro vết cắn → đều bị chặn. Cần **đảo ngược**: với hook shot và key shot, BẮT BUỘC extreme close-up vào chủ thể sốc.

3. **`_SCENE_ALIGN_ACTION_RULES` chỉ có pose của người.** Ví dụ "đặt thi thể" map thành `figure carefully placing shrouded corpse into open coffin` → lại quay về người đặt xác, không phải *khuôn mặt xác lộ ra khi khăn phủ trượt xuống*. Object rule có `pale lifeless corpse` nhưng chỉ chèn cuối prompt — SDXL không center được.

4. **`anime style` mặc định quá soft.** SDXL/Flux với style tag mỏng → ảnh bet, không có atmosphere. Cần upgrade style suffix thành:
   ```
   cinematic dark anime horror, volumetric fog, desaturated palette, 
   high contrast rim lighting, film grain, shallow depth of field, masterpiece
   ```

5. **Không có negative prompt đủ mạnh** chống "cảnh trống rỗng". Cần thêm: `flat composition, boring pose, empty scene, generic stock pose, neutral expression, centered standing figure`.

6. **Camera flow không có "subject focus"** — chỉ có `wide_to_close`, `static_close`, v.v. Thiếu flow kiểu *"macro subject, no camera movement, tight crop on gore/supernatural object"*.

---

## Fix plan cụ thể (ship được trong 1 buổi)

### A. Thêm khái niệm `shot_subject` vào schema + prompt (scriptwriter.py)
Mỗi shot thêm field:
```json
"shot_subject": "corpse_face" | "wound" | "bloody_object" | "supernatural_entity" | "person_action" | "environment"
```
Rule:
- **Hook shot (1, 2) và key shots**: `shot_subject` PHẢI là non-person (`corpse_face`, `wound`, `bloody_object`, `supernatural_entity`) nếu narration có từ sốc (`xác, máu, thi thể, sát thủ, vết cắn, bàn tay, con mắt, hồn...`).
- Khi `shot_subject != person_action` → `characters=[]`, prompt mở đầu bằng `extreme close-up of [subject]`.

### B. Đảo rule close-up trong `_SCRIPTWRITER_SYSTEM`
- Bỏ dòng cấm `extreme close-up`.
- Thêm rule mới:
  ```
  SHOT FRAMING BY SUBJECT:
  - shot_subject=corpse_face → "extreme close-up of pale dead face, wide bloodshot eyes, ..."
  - shot_subject=wound → "macro shot of deep bleeding bite mark on neck, ..."
  - shot_subject=bloody_object → "close-up of blood-soaked dagger on wooden floor, ..."
  - shot_subject=supernatural_entity → "low angle shot of ghostly figure emerging from mist, ..."
  - shot_subject=person_action → existing figure-pose template
  ```

### C. Thêm narration→subject detector trong `_NARRATION_ALIGN_SYSTEM`
Extract rule thứ 6:
```
6. SUBJECT: if narration focuses on corpse/blood/wound/supernatural entity — subject = that thing, 
   NOT the person observing it. Rewrite scene_prompt to frame that thing as SUBJECT 
   with extreme close-up or macro shot, characters=[].
```

### D. Upgrade style suffix (nơi inject tự động)
Tìm chỗ append `"anime style, no text, no watermarks"` — thay bằng horror style stack:
```
cinematic dark anime horror style, volumetric fog, desaturated cold palette, 
high contrast chiaroscuro lighting, film grain, shallow depth of field, 
intricate details, masterpiece, no text, no watermarks
```

### E. Strengthen negative prompt (comfyui_client.py hoặc workflow JSON)
Thêm: `flat composition, boring centered standing pose, generic stock photo, neutral expression, empty scene, low detail, blurry, bright daylight, cheerful atmosphere`.

### F. Composition variety
Thêm 1 tag random per shot từ palette: `low angle shot from below, dutch angle, over the shoulder POV, bird eye view, extreme wide with tiny figure, through broken mirror reflection`.

---

## Ví dụ trước/sau

**Narration:** "Nắp quan tài bật mở. Bên trong — một xác chết mắt trợn trừng, miệng vẫn còn máu."

**TRƯỚC (bland):**
```
dimly lit coffin shop, figure kneeling and prying open coffin lid, 
wooden coffin in foreground, shelves in background, flickering candle, anime style
```

**SAU (shock):**
```
extreme close-up of pale lifeless female corpse face inside open coffin, 
wide bloodshot dead eyes staring upward, dark blood trickling from corner of mouth, 
blue-grey veins on deathly pale skin, red lacquered coffin interior edges blurred foreground, 
cold blue moonlight cutting through darkness from above, wisps of cold mist rising, 
cinematic dark anime horror style, volumetric fog, desaturated palette, 
high contrast chiaroscuro, film grain, macro shot, masterpiece
```
`characters=[]`, `shot_subject="corpse_face"`, `camera_flow="detail_reveal"`.

---

**ROI**: Fix A+B+D impact cao nhất. Làm trước, regen 1 episode để verify. Sau đó mới tune C, E, F.

Muốn tôi implement luôn phần A+B+D vào scriptwriter.py?