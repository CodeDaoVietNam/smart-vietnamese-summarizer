# Project Progress

Project: **Smart Meeting & Study Notes Summarization System**  
Goal: fine-tune a Transformer model for Vietnamese controllable summarization and build a FastAPI + Streamlit web demo.  
Team size: **2 members**

## Current Status

The current repository is a professional scaffold, not the final trained project yet.

Already done:
- Codebase structure for data, training, evaluation, inference, and web app.
- YAML configs for training, evaluation, and app.
- Sample Vietnamese inputs.
- Unit test skeleton.
- README with setup and usage commands.

Still to do:
- Install dependencies and validate environment.
- Download and inspect dataset.
- Run real preprocessing on VietNews.
- Fine-tune ViT5 in 2 phases.
- Evaluate with real ROUGE scores.
- Test FastAPI backend and Streamlit frontend with a trained checkpoint.
- Write report, slides, and demo script.

## Team Roles

### Member A: NLP / Model / Evaluation Owner

Main responsibility:
- Dataset preparation.
- Fine-tuning.
- Evaluation.
- Error analysis.
- Model artifacts.

Primary files:
- `configs/train_phase1.yaml`
- `configs/train_phase2.yaml`
- `configs/eval.yaml`
- `scripts/prepare_data.py`
- `scripts/train_phase2.py`
- `scripts/generate_synthetic.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `src/smart_summarizer/data/`
- `src/smart_summarizer/modeling/`
- `src/smart_summarizer/evaluation/`
- `reports/metrics/`
- `reports/examples/`

### Member B: Product / App / Report Owner

Main responsibility:
- Streamlit app.
- FastAPI backend.
- Inference UX.
- Sample inputs.
- README/report/slides.
- Demo flow.

Primary files:
- `configs/app.yaml`
- `api/`
- `app/`
- `scripts/predict.py`
- `src/smart_summarizer/product/`
- `data/samples/`
- `README.md`
- `PROGRESS.md`
- final report and presentation files.

## Stage 0: Project Setup

Target duration: **0.5 day**  
Owner: **Both**

Tasks:
- [ ] Create Python environment.
- [ ] Install dependencies with `pip install -e ".[dev]"`.
- [ ] Run `python -m compileall src scripts app tests`.
- [ ] Run `pytest`.
- [ ] Confirm `uvicorn api.main:app --reload` starts without syntax errors.
- [ ] Confirm `streamlit run app/streamlit_app.py` starts without syntax errors.
- [ ] Create shared Drive folder for model weights, dataset copy, report, and slides.

Deliverables:
- [ ] Local environment works for both members.
- [ ] README commands verified.
- [ ] Any setup issue is documented in this file.

Definition of Done:
- Both members can run tests and open the app locally or in Colab.

## Stage 1: Dataset Preparation And Exploration

Target duration: **2 days**  
Owner: **Member A**  
Support: **Member B**

Tasks:
- [ ] Run `python scripts/prepare_data.py --config configs/train_phase1.yaml`.
- [ ] Confirm these files exist:
  - [ ] `data/processed/train.jsonl`
  - [ ] `data/processed/validation.jsonl`
  - [ ] `data/processed/test.jsonl`
- [ ] Count samples in each split.
- [ ] Inspect at least 20 random examples.
- [ ] Record dataset statistics:
  - [ ] number of train/validation/test samples
  - [ ] average document length
  - [ ] average summary length
  - [ ] min/max document length
  - [ ] min/max summary length
- [ ] Identify low-quality examples if any.
- [ ] Confirm VietNews quality is good enough as the main dataset.
- [ ] Record the domain gap between VietNews and real meeting or lecture notes.

Deliverables:
- [ ] Dataset split files.
- [ ] Dataset statistics table for report.
- [ ] 3 good input/output examples for report.
- [ ] Dataset decision note in `reports/examples/dataset_notes.md`.
- [ ] Short note explaining why VietNews is chosen over WikiLingua.

Definition of Done:
- The team can clearly explain what the dataset contains, why it fits the task, and how it is split.

## Stage 2: Baseline Inference

Target duration: **1 day**  
Owner: **Member A**  
Support: **Member B**

Tasks:
- [ ] Run baseline prediction using pretrained/base model before fine-tuning.
- [ ] Test with:
  - [ ] meeting note sample
  - [ ] lecture note sample
  - [ ] article sample
- [ ] Save outputs in `reports/examples/baseline_outputs.md`.
- [ ] Record obvious weaknesses:
  - [ ] wrong language
  - [ ] too generic
  - [ ] missing key points
  - [ ] repetition
  - [ ] hallucination

Deliverables:
- [ ] Baseline examples.
- [ ] Short paragraph explaining why fine-tuning is needed.

Definition of Done:
- There is a before-fine-tuning comparison point for the final report.

## Stage 3: Fine-Tuning

Target duration: **3-5 days**  
Owner: **Member A**

Tasks:
- [ ] Upload project to Colab or mount Google Drive.
- [ ] Enable GPU runtime and confirm T4 is available.
- [ ] Run a tiny training smoke test with limited samples.
- [ ] If smoke test passes, run full Phase 1 fine-tuning.
- [ ] Save checkpoint to `models/vit5-summarizer-v1`.
- [ ] Generate and review synthetic data for Phase 2.
- [ ] Run Phase 2 fine-tuning on synthetic multi-mode data.
- [ ] Save checkpoint to `models/vit5-summarizer-v2`.
- [ ] Save backup checkpoint to Google Drive.
- [ ] Save training log to `reports/metrics/training_log.json`.
- [ ] Record hyperparameters:
  - [ ] model name
  - [ ] epochs
  - [ ] learning rate
  - [ ] batch size
  - [ ] gradient accumulation
  - [ ] max source length
  - [ ] max target length
  - [ ] GPU type
  - [ ] training time

Risk handling:
- [ ] If CUDA out of memory, set train batch size to `1`.
- [ ] If still out of memory, reduce `max_source_length` from `512` to `384`.
- [ ] If training is too slow, train on a subset and document the limitation.
- [ ] If Phase 2 hurts summary quality badly, keep Phase 1 checkpoint and compare both.

Deliverables:
- [ ] Phase 1 model checkpoint.
- [ ] Phase 2 model checkpoint.
- [ ] Training log.
- [ ] Hyperparameter table for report.

Definition of Done:
- The final model can generate summaries through `scripts/predict.py` and API serving.

## Stage 4: Evaluation

Target duration: **2 days**  
Owner: **Member A**

Tasks:
- [ ] Run `python scripts/evaluate.py --config configs/eval.yaml`.
- [ ] Confirm `reports/metrics/eval_results.json` exists.
- [ ] Confirm `reports/examples/test_predictions.jsonl` exists.
- [ ] Report ROUGE-1, ROUGE-2, ROUGE-L.
- [ ] Compare Phase 1 vs Phase 2 qualitatively.
- [ ] Compare baseline vs fine-tuned outputs qualitatively.
- [ ] Select at least 5 good predictions.
- [ ] Select at least 5 failure cases.

Deliverables:
- [ ] ROUGE score table.
- [ ] Prediction examples.
- [ ] Evaluation paragraph for report.

Definition of Done:
- The report can show both metric-based and example-based evaluation.

## Stage 5: Error Analysis

Target duration: **1-2 days**  
Owner: **Member A**  
Support: **Member B**

Tasks:
- [ ] Analyze failure cases from `test_predictions.jsonl`.
- [ ] Categorize errors:
  - [ ] hallucination
  - [ ] missing key points
  - [ ] entity/date mistakes
  - [ ] repetition
  - [ ] long input truncation
  - [ ] unnatural output format
- [ ] Write `reports/examples/error_analysis.md`.
- [ ] Add screenshots or examples to final presentation.

Deliverables:
- [ ] Error analysis document.
- [ ] Limitations section draft.
- [ ] Future work section draft.

Definition of Done:
- The team can honestly explain where the model works, where it fails, and why.

## Stage 6: Web App Productization

Target duration: **2-3 days**  
Owner: **Member B**  
Support: **Member A**

Tasks:
- [ ] Point `configs/app.yaml` to the trained Phase 2 checkpoint.
- [ ] Run `uvicorn api.main:app --reload`.
- [ ] Run `streamlit run app/streamlit_app.py`.
- [ ] Test all output modes:
  - [ ] concise
  - [ ] bullet
  - [ ] action_items
  - [ ] study_notes
- [ ] Test all lengths:
  - [ ] short
  - [ ] medium
  - [ ] long
- [ ] Test compare-all-modes view.
- [ ] Improve sample inputs if needed.
- [ ] Verify app displays:
  - [ ] generated summary
  - [ ] keywords
  - [ ] quality estimate
  - [ ] input token count
  - [ ] inference latency
- [ ] Take screenshots for report and slides.

Deliverables:
- [ ] Working FastAPI backend.
- [ ] Working Streamlit app.
- [ ] Screenshots for report.
- [ ] Demo-ready sample inputs.

Definition of Done:
- A user can paste Vietnamese text, choose a mode, and get a useful output through the web UI through the backend API.

## Stage 7: Report Writing

Target duration: **3 days**  
Owner: **Member B**  
Support: **Member A**

Recommended chapters:
- [ ] Introduction and motivation.
- [ ] Problem definition: input, output, task type.
- [ ] Transformer and ViT5 background.
- [ ] Dataset description and preprocessing.
- [ ] Model and training setup.
- [ ] Evaluation metrics and results.
- [ ] Error analysis.
- [ ] Web application design.
- [ ] Limitations and future work.
- [ ] Conclusion.

Required content:
- [ ] Explain why this problem is useful.
- [ ] Explain why Transformer is suitable.
- [ ] Explain why ViT5 is selected.
- [ ] Explain dataset split.
- [ ] Explain the 2-phase training strategy.
- [ ] Explain ROUGE metrics.
- [ ] Include training hyperparameters.
- [ ] Include screenshots of web app.
- [ ] Include example input/output.
- [ ] Include limitations honestly.

Deliverables:
- [ ] Final report PDF.
- [ ] Source report file if required.

Definition of Done:
- The report answers all assignment requirements and can be understood without reading the code.

## Stage 8: Presentation And Demo

Target duration: **1-2 days**  
Owner: **Both**

Tasks:
- [ ] Create 8-12 slides.
- [ ] Prepare 5-7 minute demo script.
- [ ] Choose 2 live demo examples:
  - [ ] one meeting note
  - [ ] one study/lecture note
- [ ] Prepare backup screenshots in case live demo fails.
- [ ] Prepare answers for likely questions:
  - [ ] What is the input and output?
  - [ ] Why choose this task?
  - [ ] Why use Transformer?
  - [ ] Why use ViT5?
  - [ ] What dataset was used?
  - [ ] How was the model evaluated?
  - [ ] What are the limitations?
  - [ ] What is creative about the project?

Deliverables:
- [ ] Slide deck.
- [ ] Demo script.
- [ ] Backup screenshots.

Definition of Done:
- Both members can present the project smoothly and answer core technical questions.

## Final Submission Checklist

- [ ] Source code.
- [ ] Fine-tuned model weights or Google Drive link.
- [ ] Dataset files or Google Drive link.
- [ ] Synthetic data files or Google Drive link.
- [ ] Final report PDF.
- [ ] README with running instructions.
- [ ] Evaluation results.
- [ ] Web app screenshots.
- [ ] API docs screenshots if useful.
- [ ] Presentation slides if required.

## Suggested Timeline

| Week | Main Focus | Owner |
|---|---|---|
| Week 1 | Setup, VietNews preparation, baseline, synthetic planning | Both, Member A leads |
| Week 2 | Phase 1 and Phase 2 fine-tuning | Member A |
| Week 3 | Error analysis, FastAPI, Streamlit, product polish | Member B leads |
| Week 4 | Report, slides, demo rehearsal, final packaging | Both |

## Daily Tracking

Use this section to record short updates.

| Date | Member | Work Done | Blockers | Next Step |
|---|---|---|---|---|
| YYYY-MM-DD | Member A |  |  |  |
| YYYY-MM-DD | Member B |  |  |  |

## Open Issues

- [ ] Confirm VietNews quality is sufficient for the main summarization base.
- [ ] Confirm Colab T4 can fine-tune `VietAI/vit5-base` with both phase configs.
- [ ] Confirm final model size is acceptable for submission or Drive upload.
- [ ] Decide final report language and slide language.

## Notes

- `action_items` and `study_notes` are controllable output modes, not separate supervised tasks in v1.
- Quality Estimate is a product heuristic, not a calibrated probability.
- Very long input will be truncated by the tokenizer, so this must be mentioned in limitations.
