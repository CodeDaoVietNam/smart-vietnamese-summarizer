# Project Excellence Roadmap

Muc tieu: dua project tu demo chay duoc len mot do an AI engineering thuyet phuc, co danh gia trung thuc va co trai nghiem demo dep.

## Current Strengths

- Pipeline end-to-end da hoan chinh: data, training, evaluation, API, frontend.
- Co 3 huong model de so sanh: Phase 1, Phase 2 full fine-tune, LoRA mixed-data.
- Kien truc app gon: Streamlit frontend, FastAPI inference backend, offline training jobs.
- LoRA la huong dung khi data controllability con it.
- Hybrid neuro-symbolic decoder giup output on dinh hon trong demo bang extractor, critic va renderer.
- Test coverage da tot hon, dac biet voi formatter/extractor, API, data builder va UI helpers.

## Highest-Impact Improvements

1. Evaluation thuyet phuc hon.
   Tao bang so sanh Phase 1, Phase 2, LoRA bang ROUGE tren cung subset va rubric holdout thu cong.

2. Controllability rubric.
   Cham 20 holdout documents x 4 modes theo mode adherence, factuality, usefulness, format correctness, conciseness.

3. Demo case selection.
   Dung 3 case chinh: meeting action items, lecture study notes, article concise/bullet. Chuan bi 1 limitation case.

4. Report limitation trung thuc.
   Noi ro hybrid decoder la inference-time reliability layer, khong phai bang chung model hieu hoan toan mode.

5. UX polish.
   Giu UI ro mode, ro length, co backend status, co Heuristic Quality Estimate va note human review.

## Medium-Term Improvements

- Batch inference cho `scripts/evaluate.py` de danh gia nhanh hon.
- Refactor training scripts de giam duplicate code.
- Them integration test cho `SmartSummarizer` voi stub model/tokenizer.
- Cai thien action-item extraction bang supervised data hoac rule parser/NER rieng cho Vietnamese meeting notes.
- Cai thien study-notes target bang data chat luong hon, khong chi format 4 dong.

## What Not To Overclaim

- Khong noi Heuristic Quality Estimate la xac suat dung sai.
- Khong noi Phase 2 bat buoc tang ROUGE hon Phase 1.
- Khong noi extractor/formatter thay the factuality evaluation.
- Khong goi he thong la production microservices neu chua co deploy/scale/doc lap service that.

## Final Claim Nen Dung

He thong xay dung pipeline Vietnamese controllable summarization hoan chinh. Phase 1 hoc nang luc tom tat tong quat, Phase 2/LoRA them kha nang dieu khien output style, va hybrid neuro-symbolic decoder giup output on dinh hon khi serving bang extractor, critic va renderer. Ket qua cho thay LoRA giu chat luong tong quat gan Phase 1 hon full fine-tune khi data mode con it, nhung action-item extraction va study-notes van can them data va human review.
