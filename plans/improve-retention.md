Optimizing tool selection...

Read [](file:///media/lanntxyz/ex-data10/Lanntxyz/projects/video-pipiline/plans/retention-optimization.md#1-1), lines 1 to 80

Dưới đây là các đòn bẩy retention thực chiến cho dự án (60s, 9:16, truyện online, AI-gen), xếp theo impact/cost:

## 1. Hook 3 giây đầu — quyết định 70% retention
- Shot 1 mở bằng **hành động/shock**, cấm mô tả bối cảnh. Pattern: "Nó mở nắp quan tài thì thấy…", "Máu phun ra từ…", "Hắn vừa niệm chú thì…"
- Shot 1–2: duration ≤ 3s, narration ≤ 12 từ (TTS phải kịp).
- Text overlay lớn câu hook ở frame 1 (đọc nhanh hơn nghe).
- Plan đã có (retention-optimization.md) — ưu tiên ship ngay.

## 2. Cliffhanger + series hóa
- Shot cuối (8) dừng ở peak curiosity, KHÔNG resolve. Cấm câu CTA rỗng "nhớ theo dõi nhé".
- Title/caption đánh số tập **"Tập 3/100 — [Sự thật về X]"** → viewer bị buộc xem tập trước/sau, tăng watch-time trên toàn channel.
- Pin comment tập tiếp theo ngay khi upload.

## 3. Pacing cực nhanh
- 8 shots / 60s = avg 7.5s/shot. **Quá chậm cho Shorts.** Nên 10–12 shots, avg 5s. Cut càng dày, retention càng cao.
- Mỗi shot đổi góc máy/scale (close-up → wide → POV) — AI-gen rất dễ bị "lặp 1 góc" gây chán.

## 4. Subtitle động (đã có trong plan)
- Karaoke highlight từng từ đang nói.
- Keyword đỏ (máu/chết/quỷ) và vàng (thật ra/hóa ra) — mắt người bắt màu trước khi đọc, giữ attention.
- Font to, stroke dày, đặt 1/3 trên (tránh UI TikTok che).

## 5. Audio layer — thường bị bỏ quên
- **BGM mood-match** (horror/tu tiên có library riêng, duck -18dB khi có voice). Video không nhạc nền = retention drop 20–30%.
- **SFX điểm nhấn**: whoosh khi cut, impact khi reveal, riser trước cliffhanger. Chi phí: vài sample free, tác động lớn.
- Check mixer.py — nếu chưa có BGM/SFX layer thì đây là quick win lớn nhất sau hook.

## 6. Consistency visual = brand recognition
- Character anchor system bạn đang làm là đúng hướng. Viewer nhận ra nhân vật → quay lại xem tập mới.
- Thêm **intro sting 1s** (logo + sound signature) — NHƯNG chỉ từ tập 2 trở đi khi viewer đã quen. Tập đầu hook luôn.

## 7. Metadata / thuật toán
- **Hashtag**: 1 niche cụ thể (#truyenkinhdi #tutien) + 1 broad (#fyp) + 1 series (#maoson). Đừng spam.
- **Title** max 100 char, frontload keyword + cliffhanger question. Đừng để title AI-generic.
- **Thumbnail** (cho YouTube Shorts): mặt nhân vật close-up + emotion cực đoan + 3-5 chữ. Workflow `thumbnail.json` bạn có rồi, cần tune prompt theo emotion tập đó.

## 8. A/B test systematically
- Mỗi 10 tập, đổi 1 biến (hook style / BGM genre / pacing) và track retention curve TikTok analytics. Không guess.
- Log `episode_num → avg_watch_time` vào SQLite, tự vẽ chart.

---

**Nếu chọn 3 việc làm tuần này** (ROI cao nhất):
1. Ship plan retention-optimization.md (hook + cliffhanger + subtitle màu).
2. Thêm BGM + SFX layer vào mixer.py.
3. Tăng shot count 8 → 10–12, giảm avg duration xuống 5s.

Muốn tôi đi sâu vào cái nào — implement BGM/SFX layer, hay tune prompt hook/cliffhanger trong scriptwriter?
