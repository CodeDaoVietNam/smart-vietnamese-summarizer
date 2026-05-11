# Phase 2 Improvement Plan

Tài liệu này là kế hoạch rất cụ thể để cải thiện Phase 2 sau khi đã chạy xong notebook 02.

Mục tiêu của Phase 2 không phải là đánh bại Phase 1 trên ROUGE news summarization, mà là:

1. Giúp model hiểu rõ hơn các `mode`
2. Giảm domain gap giữa `VietNews` và `meeting / lecture notes`
3. Làm demo controllable summarization đáng tin hơn

---

## 1. Kết Luận Từ Kết Quả Hiện Tại

Hiện trạng:

- Phase 1 cho metric tốt hơn
- Phase 2 chạy đúng pipeline nhưng metric giảm nhẹ
- Nguyên nhân hợp lý nhất: synthetic data quá ít và quá hẹp

Điều này có nghĩa:

- Model không hỏng
- Code không sai
- Nút thắt lớn nhất là **synthetic dataset quality + size**

Vì vậy, nếu muốn cải thiện Phase 2, nên ưu tiên:

```text
data > prompt > review > re-train
```

không nên ưu tiên:

```text
tweak learning rate / tweak beam size / đổi model
```

---

## 2. Nên Tạo Bao Nhiêu Sample

Mức tối thiểu để Phase 2 bắt đầu có ý nghĩa:

```text
80 samples train
20 samples validation
```

Mức tốt hơn cho đồ án:

```text
160 samples train
40 samples validation
```

Mức đề xuất:

| Mode | Train | Validation | Total |
|---|---:|---:|---:|
| `concise` | 20 | 5 | 25 |
| `bullet` | 20 | 5 | 25 |
| `action_items` | 20 | 5 | 25 |
| `study_notes` | 20 | 5 | 25 |
| **Tổng tối thiểu** | **80** | **20** | **100** |

Nếu có thêm thời gian:

| Mode | Train | Validation | Total |
|---|---:|---:|---:|
| `concise` | 40 | 10 | 50 |
| `bullet` | 40 | 10 | 50 |
| `action_items` | 40 | 10 | 50 |
| `study_notes` | 40 | 10 | 50 |
| **Tổng tốt hơn** | **160** | **40** | **200** |

---

## 3. Phân Bổ Theo Domain

Không chỉ chia theo mode, còn nên chia theo domain:

| Domain | Mục tiêu |
|---|---|
| `meeting_notes` | giúp `action_items` và `concise` thực tế hơn |
| `lecture_notes` | giúp `study_notes` và `bullet` tự nhiên hơn |
| `project_updates` | giúp output gần use case đi làm / đồ án |
| `study_materials` | giúp note-style tốt hơn |

Với bộ `100` mẫu, có thể chia:

| Domain | Total |
|---|---:|
| `meeting_notes` | 35 |
| `lecture_notes` | 35 |
| `project_updates` | 15 |
| `study_materials` | 15 |

---

## 4. Format Chuẩn Của Mỗi Sample

Schema Phase 2 nên giữ đơn giản:

```json
{
  "document": "Nội dung đầu vào tiếng Việt...",
  "summary": "Đầu ra mong muốn...",
  "mode": "bullet"
}
```

### Quy tắc chung

- `document` nên dài khoảng `120-400` từ
- `summary` nên ngắn hơn rõ rệt và đúng format theo mode
- Không để sample quá ngắn kiểu 1-2 câu nghèo thông tin
- Không để output bị machine-like hoặc quá template

### Quy tắc theo mode

#### `concise`

- Output là 1 đoạn ngắn
- 1-3 câu
- Giọng văn tự nhiên
- Không cần bullet

Ví dụ:

```json
{
  "document": "Cuộc họp nhóm đồ án thảo luận tiến độ backend, frontend và chuẩn bị demo.",
  "summary": "Nhóm thống nhất hoàn thiện backend, rà soát giao diện và chuẩn bị demo cho buổi báo cáo tới.",
  "mode": "concise"
}
```

#### `bullet`

- Output là nhiều dòng bullet
- Mỗi bullet là 1 ý chính
- Không cần diễn đạt quá dài

Ví dụ:

```json
{
  "document": "Buổi học hôm nay giải thích self-attention, encoder-decoder và beam search trong Transformer.",
  "summary": "- Self-attention là thành phần cốt lõi.\n- Transformer gồm encoder và decoder.\n- Beam search hỗ trợ sinh văn bản tốt hơn.",
  "mode": "bullet"
}
```

#### `action_items`

- Chỉ phù hợp khi document có task, deadline, người phụ trách
- Output nên rất hành động
- Ưu tiên động từ: hoàn thành, gửi, kiểm tra, chuẩn bị

Ví dụ:

```json
{
  "document": "Nhóm thống nhất An làm báo cáo, Bình sửa API, Chi kiểm tra giao diện trước thứ Sáu.",
  "summary": "- An hoàn thành báo cáo.\n- Bình sửa API.\n- Chi kiểm tra giao diện trước thứ Sáu.",
  "mode": "action_items"
}
```

#### `study_notes`

- Dành cho lecture / học tập
- Output nên giống ghi chú học bài
- Có thể bullet hoặc note lines

Ví dụ:

```json
{
  "document": "Giảng viên nhấn mạnh positional encoding giúp mô hình giữ thông tin thứ tự trong chuỗi.",
  "summary": "- Positional encoding giúp giữ thông tin thứ tự.\n- Đây là thành phần quan trọng trong Transformer.",
  "mode": "study_notes"
}
```

---

## 5. Prompt Để Sinh Synthetic Data

### Prompt tổng quát

Dùng prompt này với ChatGPT hoặc Gemini:

```text
Bạn là trợ lý tạo dữ liệu huấn luyện cho hệ thống tóm tắt tiếng Việt.

Hãy tạo 10 mẫu dữ liệu ở dạng JSON array.

Mỗi mẫu phải có 3 field:
- "document": văn bản tiếng Việt dài 120-400 từ
- "summary": bản tóm tắt đúng với mode yêu cầu
- "mode": một trong 4 giá trị: concise, bullet, action_items, study_notes

Yêu cầu chất lượng:
- Tiếng Việt tự nhiên, không máy móc
- Nội dung cụ thể, có tên vai trò, nhiệm vụ, quyết định, deadline nếu là meeting
- Nếu là lecture thì phải có khái niệm, ý chính, điểm cần nhớ
- summary phải ngắn hơn document rõ rệt
- summary phải đúng format theo mode
- Không dùng placeholder như A, B, C quá nhiều
- Không lặp cấu trúc y hệt giữa các mẫu

Trả về JSON array hợp lệ, không thêm giải thích ngoài JSON.
```

### Prompt riêng cho `meeting_notes -> action_items`

```text
Tạo 10 mẫu JSON cho bài toán controllable summarization tiếng Việt.

Mode bắt buộc: "action_items"

Mỗi "document" là biên bản cuộc họp ngắn hoặc ghi chú họp dự án, nên có:
- mục tiêu cuộc họp
- các đầu việc
- người phụ trách
- deadline

Mỗi "summary" phải là danh sách các việc cần làm, dạng bullet, ngắn gọn, rõ hành động.

Trả về JSON array với field:
- document
- summary
- mode
```

### Prompt riêng cho `lecture_notes -> study_notes`

```text
Tạo 10 mẫu JSON cho bài toán controllable summarization tiếng Việt.

Mode bắt buộc: "study_notes"

Mỗi "document" là ghi chú bài giảng, nội dung học, hoặc phần giải thích một khái niệm.
Mỗi "summary" phải giống ghi chú ôn tập:
- ngắn
- chia ý rõ
- nhấn mạnh khái niệm chính

Trả về JSON array với field:
- document
- summary
- mode
```

### Prompt riêng cho `meeting/lecture -> bullet`

```text
Tạo 10 mẫu JSON cho bài toán tóm tắt tiếng Việt.

Mode bắt buộc: "bullet"

Mỗi "document" là văn bản dài 120-400 từ.
Mỗi "summary" phải là các ý chính dạng bullet, không viết thành đoạn văn.

Trả về JSON array với field:
- document
- summary
- mode
```

---

## 6. Quy Trình Review Synthetic Data

Không nên đem toàn bộ output AI đi train ngay. Cần review nhanh.

Checklist review cho từng sample:

- `document` có tự nhiên không?
- `summary` có thật sự ngắn hơn không?
- `summary` có đúng mode không?
- Có lỗi logic rõ ràng không?
- Có tiếng Việt gượng hoặc máy móc quá không?
- Có bị lặp y chang nhiều mẫu khác không?

Nên loại bỏ sample nếu:

- thông tin nghèo
- output mode sai
- tiếng Việt kỳ
- deadline / action vô lý
- lecture note quá chung chung

Mục tiêu:

```text
100 mẫu sinh ra -> giữ lại khoảng 60-80 mẫu tốt
```

rồi sinh thêm cho đủ.

---

## 7. Cấu Trúc File Đề Xuất

Trong `data/synthetic/`, nên tổ chức thế này:

```text
data/synthetic/
  raw_meeting_action_items.json
  raw_meeting_concise.json
  raw_lecture_bullet.json
  raw_lecture_study_notes.json
  reviewed_all.json
  train.jsonl
  validation.jsonl
```

Nếu muốn sạch hơn:

```text
data/synthetic/raw/
data/synthetic/reviewed/
data/synthetic/final/
```

Nhưng với đồ án, structure hiện tại đã đủ.

---

## 8. Các Bước Train Lại Phase 2

### Bước 1: Tạo thêm synthetic data

Mục tiêu đầu tiên:

```text
100 mẫu tổng
```

### Bước 2: Review thủ công

- bỏ mẫu dở
- sửa mẫu chưa tự nhiên
- cân bằng lại 4 mode

### Bước 3: Chuyển sang JSONL

Mỗi dòng:

```json
{"document": "...", "summary": "...", "mode": "bullet"}
```

### Bước 4: Kiểm tra phân bố

Ít nhất cần biết:

- mỗi mode có bao nhiêu mẫu
- train/validation có cân bằng không

Bạn có thể kiểm tra nhanh:

```bash
python - <<'PY'
import json
from collections import Counter

for path in ['data/synthetic/train.jsonl', 'data/synthetic/validation.jsonl']:
    counter = Counter()
    with open(path, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            counter[row['mode']] += 1
    print(path, dict(counter))
PY
```

### Bước 5: Train lại Phase 2

```bash
python scripts/train_phase2.py --config configs/train_phase2.yaml
```

### Bước 6: Evaluate lại

```bash
python scripts/evaluate.py --config configs/eval.yaml
```

### Bước 7: So sánh output mode

Đây là bước quan trọng nhất của Phase 2.

Kiểm tra cùng 1 input:

- `concise`
- `bullet`
- `action_items`
- `study_notes`

Nếu output khác nhau hợp lý, thì phase 2 đã có ích về mặt sản phẩm.

---

## 9. Hyperparameter Có Cần Đổi Không

Hiện tại **chưa cần đổi nhiều**.

Config phase 2 hiện tại đã hợp lý:

- `epochs: 2`
- `learning_rate: 1e-5`
- `gradient_accumulation_steps: 4`

Chỉ cân nhắc sửa nếu:

- data tăng lên đáng kể
- hoặc output bị overfit

Gợi ý:

| Trường hợp | Gợi ý |
|---|---|
| 100 samples | giữ nguyên config |
| 200+ samples | có thể tăng `logging_steps`, giữ `epochs=2` |
| overfit mạnh | giảm `epochs` xuống `1` |
| học chưa rõ mode | tăng data trước, chưa vội tăng epoch |

---

## 10. Có Cần Update Artifact Lên GitHub Không

### Nên push lên GitHub

- code mới
- notebook đã hoàn thiện
- docs
- config
- synthetic guide / plan
- metric JSON nhỏ nếu muốn giữ cho report
- figures nhỏ nếu muốn giữ cho report

### Không nên push lên GitHub

- checkpoint model lớn trong `models/`
- processed data lớn trong `data/processed/`
- cache của Hugging Face / Colab
- file tạm lớn

### Áp dụng cho project hiện tại

`.gitignore` của repo đã chặn:

```text
data/processed/*
models/*
reports/metrics/*
reports/examples/*
reports/figures/*
checkpoint-*/
*.bin
*.safetensors
```

Nghĩa là mặc định:

- model checkpoint: không push
- processed dataset: không push
- report artifacts: cũng đang không push

### Có nên đổi `.gitignore` không?

Có thể, nhưng chỉ nếu bạn thật sự muốn lưu một số artifact nhỏ.

Khuyến nghị:

- giữ nguyên chặn `models/` và `data/processed/`
- nếu cần, chỉ unignore một số file nhỏ như:
  - `reports/metrics/eval_phase1.json`
  - `reports/metrics/eval_phase2.json`
  - `reports/figures/phase1_training_loss.png`

Nếu không muốn chỉnh `.gitignore`, bạn vẫn có thể:

- để artifact ở Google Drive
- trích số liệu vào report bằng tay

---

## 11. Kết Luận Hành Động

Nếu muốn cải thiện Phase 2 theo cách đáng công nhất, làm theo đúng thứ tự này:

1. Tạo thêm synthetic data đến khoảng `100` mẫu
2. Review thủ công và cân bằng 4 mode
3. Train lại Phase 2
4. Evaluate lại
5. So sánh output giữa các mode
6. Đưa phần controllability vào demo và report

Kết luận trung thực nên là:

```text
Phase 2 hiện tại đã hoạt động và xác thực được hướng controllable summarization.
Để Phase 2 mạnh hơn, yếu tố cần cải thiện nhất là synthetic dataset,
không phải model architecture.
```

