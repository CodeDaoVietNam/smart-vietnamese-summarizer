# Synthetic Prompts For Phase 2

Use these prompts with ChatGPT or Gemini to generate reviewed synthetic data for Phase 2.

Output target schema:

```json
{
  "document": "Nội dung đầu vào tiếng Việt...",
  "summary": "Đầu ra mong muốn...",
  "mode": "bullet",
  "domain": "lecture_notes"
}
```

Supported `mode`:

- `concise`
- `bullet`
- `action_items`
- `study_notes`

Recommended `domain`:

- `meeting_notes`
- `lecture_notes`
- `project_updates`
- `study_materials`

## General Prompt

```text
Bạn là trợ lý tạo dữ liệu huấn luyện cho hệ thống tóm tắt tiếng Việt.

Hãy tạo 10 mẫu dữ liệu ở dạng JSON array.

Mỗi mẫu phải có 4 field:
- "document": văn bản tiếng Việt dài 120-400 từ
- "summary": bản tóm tắt đúng với mode yêu cầu
- "mode": một trong 4 giá trị: concise, bullet, action_items, study_notes
- "domain": một trong 4 giá trị: meeting_notes, lecture_notes, project_updates, study_materials

Yêu cầu chất lượng:
- Tiếng Việt tự nhiên, không máy móc
- Nội dung cụ thể, có hành động, quyết định, deadline nếu là meeting
- Nếu là lecture thì có khái niệm, ý chính, điểm cần nhớ
- Summary phải ngắn hơn document rõ rệt
- Summary phải đúng format theo mode
- Không lặp cấu trúc giống nhau giữa các mẫu
- Không thêm bất kỳ lời giải thích nào ngoài JSON
```

## Meeting Notes -> Action Items

```text
Tạo 10 mẫu JSON cho bài toán controllable summarization tiếng Việt.

Mode bắt buộc: "action_items"
Domain bắt buộc: "meeting_notes"

Mỗi document là biên bản họp dự án hoặc ghi chú họp nhóm, nên có:
- mục tiêu buổi họp
- các đầu việc cụ thể
- người phụ trách
- deadline

Mỗi summary phải là danh sách việc cần làm, dạng bullet, rõ hành động.

Trả về JSON array với field:
- document
- summary
- mode
- domain
```

## Lecture Notes -> Study Notes

```text
Tạo 10 mẫu JSON cho bài toán controllable summarization tiếng Việt.

Mode bắt buộc: "study_notes"
Domain bắt buộc: "lecture_notes"

Mỗi document là ghi chú bài giảng hoặc phần giải thích kiến thức.
Mỗi summary phải giống ghi chú ôn tập:
- ngắn
- chia ý rõ
- nhấn mạnh khái niệm chính

Trả về JSON array với field:
- document
- summary
- mode
- domain
```

## Mixed Bullet Mode

```text
Tạo 10 mẫu JSON cho bài toán tóm tắt tiếng Việt.

Mode bắt buộc: "bullet"
Domain có thể là meeting_notes, lecture_notes, project_updates hoặc study_materials.

Mỗi document là văn bản dài 120-400 từ.
Mỗi summary phải là các ý chính dạng bullet, không viết thành đoạn văn.

Trả về JSON array với field:
- document
- summary
- mode
- domain
```

## Concise Mode

```text
Tạo 10 mẫu JSON cho bài toán tóm tắt tiếng Việt.

Mode bắt buộc: "concise"
Domain có thể là meeting_notes hoặc project_updates.

Mỗi document là văn bản dài 120-400 từ.
Mỗi summary là 1 đoạn ngắn 1-3 câu, giọng văn tự nhiên, cô đọng.

Trả về JSON array với field:
- document
- summary
- mode
- domain
```
