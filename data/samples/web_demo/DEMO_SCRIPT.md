# Web Demo Script

Muc tieu demo: cho thay he thong chay end-to-end, co dieu khien mode, co formatter de output de doc hon, va co limitation trung thuc.

## Case 1: Meeting -> Action Items

File: `01_meeting_demo_release.txt`

Chay:

- Mode: `action_items`
- Length: `medium`
- Sau do bat `Compare all modes`

Noi khi demo:

- Input co nguoi phu trach va han chot ro.
- Mode `action_items` nen bien meeting note thanh viec can lam.
- Neu output co `Chua ro`, day la han che hop ly khi deadline/owner khong duoc noi truc tiep.

## Case 2: Lecture -> Study Notes

File: `03_lecture_transformer_attention.txt`

Chay:

- Mode: `study_notes`
- Length: `medium` hoac `long`
- Co the so sanh voi mode `bullet`

Noi khi demo:

- Input la noi dung bai hoc ve Transformer.
- Mode `study_notes` nen co 4 truong: Khai niem chinh, Can nho, Vi du, Loi de nham.
- Diem manh cua he thong la controllable format; diem yeu la noi dung van can human review.

## Case 3: Article -> Concise/Bullet

File: `07_article_urban_transport.txt`

Chay:

- Mode: `concise`, length `medium`
- Mode: `bullet`, length `medium`

Noi khi demo:

- Article phu hop de chung minh summarization tong quat.
- `concise` nen la doan ngan tu nhien.
- `bullet` nen tach cac y chinh.

## Limitation Case

File: `05_project_update_summarizer.txt`

Chay:

- Mode: `bullet`
- Mode: `action_items`

Noi khi demo:

- Day la case ve chinh project, giup giai thich Phase 1, Phase 2, LoRA va formatter.
- Neu output action item chua hoan hao, dung no de noi limitation: model/formatter giup cau truc tot hon, nhung factuality va owner extraction van can du lieu/giam sat tot hon.

## Cau noi nen dung ve quality

`Heuristic Quality Estimate` khong phai xac suat dung sai. No la proxy dua tren do dai summary, muc do lap tu, keyword coverage va tin hieu generation. Diem nay chi giup debug/demo, khong thay the ROUGE hay human evaluation.
