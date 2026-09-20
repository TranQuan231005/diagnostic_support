# ⏱️ 3-Day Rapid Sprint Status Dashboard — Medical VQA System
### Topic 4: Medical Diagnostic Support Visual Question Answering
> **Sprint Horizon:** 3 Days (Compressed Fast-Track MVP)  
> **Team Capacity:** 4 Members  
> **Target Architecture:** LLaVA-1.5-7B + QLoRA (4-bit NF4)  
> **Dataset:** VQA-Med-2019  
> **Last Updated:** Day 1 — Sprint Kickoff (2026-09-19)

---

## 📈 Overall Project Progress

```
Sprint Progress: [███████░░░] 33% Complete (Day 1 Complete - 6/18 Tasks)
```

| Metric | Target | Current Status | Health |
|---|---|---|:---:|
| **Days Remaining** | 3 Days | 2 Days (Day 1 Finished) | 🟢 On Schedule |
| **Completed Tasks** | 6 / 18 | 6 Completed (D1-T1, D1-T2, D1-T3, D1-T4, D1-T5, D1-T6) | 🟢 100% Day 1 |
| **In Progress Tasks** | — | Ready for Day 2 kickoff | 🟢 Ready |
| **Blockers / Risks** | 0 Critical | Monitored (Colab T4 limits) | 🟢 Managed |
| **Deliverables Completed** | 3 / 8 Key Artifacts | 3 / 8 (Data, Evaluation, Model Engine, Gradio UI Shell) | 🟢 On Track |

---

## 🚦 Status Legend
- `[ ]` ⚪ `NOT_STARTED` : Task queued, prerequisites in progress
- `[/]` 🟡 `IN_PROGRESS` : Currently being worked on
- `[X]` 🟢 `COMPLETED` : Tested, reviewed, committed, DoD met
- `[!]` 🔴 `BLOCKED` : Impeded by external dependency or technical bug

---

## 👥 Member Role Summary & Ownership

| Member | Domain & Ownership | Primary Tools / Deliverables |
|---|---|---|
| **Member 1 (M1)** | **Data Engineering & Preprocessing** | `data/processed/`, `preprocess.py`, `train.json`, `val.json` |
| **Member 2 (M2)** | **Model & QLoRA Fine-Tuning** | `src/model/load_model.py`, `01_env_setup_and_model_sanity_check.ipynb`, 4-bit NF4, PEFT |
| **Member 3 (M3)** | **Evaluation & 8-Stage Benchmarking** | `src/evaluation/evaluate.py` (BLEU-1, Acc), `benchmark_latency.py` (P50/P95/P99) |
| **Member 4 (M4)** | **Inference, Gradio UI & Deployment** | `predict.py`, `demo/app.py`, Hugging Face Spaces, Final Report & Slides |

---

## 🗓️ 3-Day Sprint Roadmap & Task Checklist

### 📍 DAY 1: Foundation, Data Preparation & Environment Setup
*Goal: Have clean LLaVA instruction JSONs ready and Colab baseline running.*

| Done | ID | Owner | Task / Deliverable | Status | Dependencies | Notes |
|:---:|:---:|:---:|---|:---:|---|---|
| [X] | **D1-T1** | **M1** | Download & extract VQA-Med-2019 dataset (Zenodo) | 🟢 `COMPLETED` | None | Raw dataset downloaded and unpacked |
| [X] | **D1-T2** | **M1** | Convert raw QA pairs into LLaVA format (`train.json`, `val.json`) | 🟢 `COMPLETED` | D1-T1 | 12,792 train & 2,000 val QA pairs converted + category splits |
| [X] | **D1-T3** | **M2** | Set up Google Colab T4 environment with `transformers`, `peft`, `bitsandbytes` | 🟢 `COMPLETED` | None | Notebook & dependency suite in `notebooks/01_env_setup_and_model_sanity_check.ipynb` |
| [X] | **D1-T4** | **M2** | Load base LLaVA-1.5-7B in 4-bit NF4 & verify forward pass | 🟢 `COMPLETED` | D1-T3 | `src/model/load_model.py`, `verify_forward.py` verified; 4-bit NF4 VRAM budget <= 5.5GB |
| [X] | **D1-T5** | **M3** | Build zero-shot evaluation pipeline skeleton (`evaluate.py`) | 🟢 `COMPLETED` | None | Exact Match, BLEU-1, BLEU-2, ROUGE-L & Category Breakdowns |
| [X] | **D1-T6** | **M4** | Build Gradio UI mock shell & configure Hugging Face Spaces repo | 🟢 `COMPLETED` | None | Dual-engine interactive UI in `demo/app.py` with 1-click clinical presets |

---

### 📍 DAY 2: QLoRA Fine-Tuning, Benchmarking & Evaluation Pipeline
*Goal: Model trained, adapter saved, zero-shot vs fine-tuned evaluated, latency benchmarked.*

| Done | ID | Owner | Task / Deliverable | Status | Dependencies | Notes |
|:---:|:---:|:---:|---|:---:|---|---|
| [ ] | **D2-T1** | **M2** | Launch QLoRA fine-tuning on Colab (`train_llava_vqamed.ipynb`) | ⚪ `NOT_STARTED` | D1-T2, D1-T4 | Target: 3 epochs, r=16, alpha=32, batch_size=4 |
| [ ] | **D2-T2** | **M2** | Export LoRA adapter checkpoints to Google Drive & Hugging Face Hub | ⚪ `NOT_STARTED` | D2-T1 | Verify adapter weights size (~100-200MB) |
| [ ] | **D2-T3** | **M3** | Run Zero-Shot Baseline evaluation on validation set | ⚪ `NOT_STARTED` | D1-T4, D1-T5 | Log baseline BLEU-1 & Accuracy per category |
| [ ] | **D2-T4** | **M3** | Implement 8-stage latency benchmark script (`benchmark_latency.py`) | ⚪ `NOT_STARTED` | D1-T5 | Profile 5 periods: P50, P95, P99 across 8 stages |
| [ ] | **D2-T5** | **M3** | Evaluate fine-tuned checkpoint against validation set | ⚪ `NOT_STARTED` | D2-T2, D1-T5 | Generate comparison table: Zero-Shot vs Fine-Tuned |
| [ ] | **D2-T6** | **M4** | Implement standalone inference script (`predict.py`) | ⚪ `NOT_STARTED` | D2-T2 | Support single-image + medical question prompt |

---

### 📍 DAY 3: Gradio Integration, HF Deployment, Report & Presentation
*Goal: Interactive live demo live on HF Spaces, final slides, report completed and code frozen.*

| Done | ID | Owner | Task / Deliverable | Status | Dependencies | Notes |
|:---:|:---:|:---:|---|:---:|---|---|
| [ ] | **D3-T1** | **M4** | Connect fine-tuned model adapter with Gradio UI (`demo/app.py`) | ⚪ `NOT_STARTED` | D2-T2, D1-T6 | Add medical disclaimer & sample case buttons |
| [ ] | **D3-T2** | **M4** | Deploy live application to Hugging Face Spaces (or Colab Gradio Share link) | ⚪ `NOT_STARTED` | D3-T1 | Test public access and inference responsiveness |
| [ ] | **D3-T3** | **M1** | Analyze category-wise error distribution & qualitative failure cases | ⚪ `NOT_STARTED` | D2-T5 | Identify why Abnormality vs Modality differ |
| [ ] | **D3-T4** | **M3** | Compile evaluation charts & latency percentiles tables | ⚪ `NOT_STARTED` | D2-T4, D2-T5 | Export plots for presentation & final report |
| [ ] | **D3-T5** | **M4 & All** | Produce Final Technical Report & Presentation Slide Deck | ⚪ `NOT_STARTED` | D3-T2, D3-T4 | Follow Topic 4 guidelines & DoD criteria |
| [ ] | **D3-T6** | **All** | Repository cleanup, README documentation & Final Code Freeze | ⚪ `NOT_STARTED` | D3-T5 | Tag release v1.0.0 |

---

## 🎯 Deliverable Artifacts Matrix

| Done | Deliverable | Target Path / Location | Owner | Target Completion | Current Status |
|:---:|---|---|:---:|:---:|:---:|
| [X] | **1. Processed Data** | `data/processed/train.json`, `val.json` | M1 | Day 1 (EOD) | 🟢 `COMPLETED` |
| [ ] | **2. Training Notebook** | `notebooks/train_llava_vqamed.ipynb` | M2 | Day 2 (Midday) | ⚪ Queued |
| [ ] | **3. LoRA Adapter Checkpoint** | `checkpoints/llava-med-qlora/` / HF Hub | M2 | Day 2 (EOD) | ⚪ Queued |
| [X] | **4. Evaluation Engine** | `src/evaluate.py` & `evaluation_results.json` | M3 | Day 1 (EOD) | 🟢 `COMPLETED` |
| [ ] | **5. Latency Profiler** | `src/benchmark_latency.py` & latency plots | M3 | Day 3 (Morning) | ⚪ Queued |
| [ ] | **6. CLI Inference Tool** | `src/predict.py` | M4 | Day 2 (EOD) | ⚪ Queued |
| [ ] | **7. Live Gradio Web App** | `demo/app.py` / Hugging Face Spaces | M4 | Day 3 (Midday) | ⚪ Queued |
| [ ] | **8. Final Report & Slides** | `reports/final_report.pdf`, `slides.pdf` | All | Day 3 (EOD) | ⚪ Queued |

---

## ⚠️ Blocker & Risk Radar

| Done | # | Risk / Blocker | Probability | Impact | Mitigation Strategy | Owner | Status |
|:---:|:---:|---|:---:|:---:|---|:---:|:---:|
| [ ] | **R1** | **Colab T4 Disconnection / Session Limit** | High | High | Save checkpoints every 100 steps directly to Google Drive. Keep batch size = 4 with gradient accumulation = 4. | M2 | 🟡 Active Monitoring |
| [ ] | **R2** | **VRAM OOM with 7B Model** | Med | High | Strict 4-bit NF4 (`load_in_4bit=True`, `bnb_4bit_compute_dtype=torch.float16`), freeze vision tower. | M2 | 🟢 Mitigated |
| [ ] | **R3** | **Zenodo Dataset Download Bottleneck** | Med | Med | Download raw archive once, upload directly to Google Drive / workspace mirror. | M1 | 🟡 In Progress |
| [ ] | **R4** | **HF Space Free Tier VRAM Limitation** | High | Med | Use Gradio `share=True` on Colab T4 as backup demo link if HF Space Free Tier CPU inference is too slow. | M4 | 🟢 Planned |

---

## ✅ Definition of Done (DoD) Checklist per Task
*Per team rules: No task is marked COMPLETED without satisfying all conditions:*
- [ ] Code executed and verified working without errors.
- [ ] Reproducibility tested (seeded random states, documented requirements).
- [ ] No hardcoded tokens, secrets, or API keys in source files.
- [ ] Code reviewed by at least one other team member.
- [ ] Documentation updated in README & task status marked in `status.md`.
