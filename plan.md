# 🚀 9-Day Team Sprint Execution Plan — Medical VQA System
### Topic 4: Medical Diagnostic Support Visual Question Answering
**Team Size:** 4 Members | **Duration:** 9 Days | **Architecture:** LLaVA-1.5-7B + QLoRA (4-bit NF4) | **Dataset:** VQA-Med-2019

---

## 📌 Executive Summary

This document serves as the single source of truth for the 4-member team executing the **Medical Visual Question Answering (VQA) System** project within a 9-day rapid development cycle. 

The goal is to deliver an end-to-end multi-modal diagnostic support system that:
1. Preprocesses medical imaging data (**VQA-Med-2019**: 3,200 training images, 12,792 QA pairs across Modality, Plane, Organ System, and Abnormality).
2. Fine-tunes **LLaVA-1.5-7B** using **QLoRA** on Google Colab (T4 GPU).
3. Evaluates clinical accuracy (**BLEU-1 + Exact Match Accuracy**) comparing **Zero-shot vs. Fine-tuned** models.
4. Conducts an 8-stage latency benchmark across 5 measurement periods (P50, P95, P99).
5. Deploys an interactive **Gradio web application** to **Hugging Face Spaces**.
6. Produces comprehensive technical documentation, presentation slides, and evaluation reports.

---

## 👥 Team Roles & Responsibilities

| Role | Title | Primary Focus & Domain Ownership |
|---|---|---|
| **Member 1** | **Data Engineering & Preprocessing Lead** | Dataset acquisition (Zenodo), data cleaning, EDA, conversion to LLaVA instruction JSON format (`data/processed/`), PyTorch dataset & dataloaders, question category categorization & error distribution analysis. |
| **Member 2** | **ML & Model Training Lead** | Colab T4 environment setup, LLaVA-1.5-7B loading with 4-bit NF4 quantization (`bitsandbytes`, `PEFT`), training loop configuration (`train_llava_vqamed.ipynb`), loss tracking, checkpoint saving & adapter export to Google Drive / Hugging Face Hub. |
| **Member 3** | **Evaluation & Benchmarking Lead** | Evaluation pipeline (`evaluate.py`), metric implementation (BLEU-1, Accuracy per category), Zero-shot baseline evaluation, latency profiling across 8 stages & 5 measurement periods (`benchmark_latency.py`), latency visualizations. |
| **Member 4** | **Inference, Demo & Delivery Lead** | End-to-end inference script (`predict.py`), Gradio UI design (`demo/app.py`), Hugging Face Spaces deployment, sample case curation, slide deck creation, demo video recording, and final report compilation. |

---

## 📊 RACI Matrix

- **R (Responsible):** The doer of the activity.
- **A (Accountable):** The person with final approval and accountability for correctness.
- **C (Consulted):** Provided input and domain consultation.
- **I (Informed):** Kept updated on progress and results.

| Work Item / Deliverable | Member 1 (Data) | Member 2 (Model) | Member 3 (Eval) | Member 4 (Demo/Report) |
|---|:---:|:---:|:---:|:---:|
| **Day 1 Team Architecture Walkthrough** | R / A | R / A | R / A | R / A |
| **Dataset Download & Raw Inspection** | **R / A** | I | C | I |
| **LLaVA Instruction Format (`train.json`, `val.json`)** | **R / A** | C | I | I |
| **Colab Environment & QLoRA Configuration** | C | **R / A** | I | I |
| **Zero-Shot LLaVA Baseline Run** | I | C | **R / A** | I |
| **Gradio Mock UI Shell & HF Space Setup** | I | I | I | **R / A** |
| **Model Fine-Tuning Execution & Checkpointing** | C | **R / A** | C | I |
| **8-Stage Latency Benchmark Script** | I | C | **R / A** | I |
| **Fine-Tuned Checkpoint Evaluation (BLEU & Acc)** | C | C | **R / A** | I |
| **Single-Image Inference Script (`predict.py`)** | I | C | I | **R / A** |
| **Gradio UI + Checkpoint Integration & Deployment** | I | C | C | **R / A** |
| **Error Analysis by Category (Modality/Plane/etc.)** | **R / A** | C | C | I |
| **Final Technical Report & Slide Deck** | C | C | C | **R / A** |
| **Code Freeze & Final Repository Polish** | R | R | R | **A** |

---

## 🗓️ 9-Day Master Timeline & Daily Breakdown

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 9-DAY SPRINT ROADMAP                                   │
├───────────────────┬───────────────────┬──────────────────────┬─────────────────────────┤
│ PHASE 1 (Day 1-2) │ PHASE 2 (Day 3-5) │  PHASE 3 (Day 6-7)   │    PHASE 4 (Day 8-9)    │
│ Foundations, Data │ Fine-Tuning &     │ Evaluation, Latency  │ Polish, Report,         │
│ & Zero-Shot Run   │ Training Pipeline │  & Demo Integration  │ Slides & Final Delivery │
└───────────────────┴───────────────────┴──────────────────────┴─────────────────────────┘
```

---

### 🔹 PHASE 1: Foundations, Environment Setup & Data Pipeline (Days 1–2)

#### 📅 **Day 1: Team Kickoff, Architecture Walkthrough & Environment Init**
* **🎯 Phase Objective:** Establish shared conceptual understanding, development standards, and repo initialization.
* **🤝 Shared Team Event (09:00 - 10:30 AM):** 
  - Joint walkthrough of [`LEARNING_ROADMAP.md`](file:///d:/diagnostic_support/LEARNING_ROADMAP.md) and [`README.md`](file:///d:/diagnostic_support/README.md).
  - Review LLaVA architecture: CLIP ViT-L/14 → 2-layer MLP projection → Mistral/LLaMA-7B causal LM.
  - Understand QLoRA 4-bit NF4 quantization mechanics and why it fits Google Colab T4 (15GB VRAM).
* **Individual Deliverables:**
  - [ ] **Member 1:** 
    - Write `src/data/download_dataset.py` to fetch VQA-Med-2019 from Zenodo.
    - Inspect raw directories (`data/raw/train/`, `data/raw/val/`) and verify image counts and QA pairs.
  - [ ] **Member 2:** 
    - Create `notebooks/train_llava_vqamed.ipynb` on Google Colab with T4 GPU runtime.
    - Verify library installations (`transformers`, `peft`, `bitsandbytes`, `accelerate`).
    - Test loading base model `liuhaotian/llava-v1.5-7b` in 4-bit precision.
  - [ ] **Member 3:** 
    - Create `src/evaluation/metrics.py` (implement BLEU-1 via NLTK/SacreBLEU and exact match Accuracy).
    - Write basic unit tests for evaluation metrics.
  - [ ] **Member 4:** 
    - Create repository layout (`src/`, `notebooks/`, `demo/`, `results/`, `checkpoints/`).
    - Create initial Gradio mockup `demo/app.py` with placeholder response and upload box.
    - Create Hugging Face Space placeholder.
* **🚦 Daily Sync Gate (05:30 PM):** All 4 members demonstrate local repo clone, working Python virtual environment, and Colab GPU connection.

---

#### 📅 **Day 2: Data Preprocessing, PyTorch Dataset & Zero-Shot Baseline**
* **🎯 Phase Objective:** Complete data formatting and establish the baseline Zero-shot performance benchmark.
* **Individual Deliverables:**
  - [ ] **Member 1:** 
    - Implement `src/data/preprocess.py` to parse raw QA text files into standard LLaVA format:
      ```json
      [
        {
          "id": "train_0001",
          "image": "train/images/synpic1001.jpg",
          "conversations": [
            {"from": "human", "value": "<image>\nWhat is the imaging modality?"},
            {"from": "gpt", "value": "X-Ray"}
          ]
        }
      ]
      ```
    - Generate `data/processed/train.json` (12,792 pairs) and `data/processed/val.json` (2,000 pairs).
    - Generate EDA summary (distribution of categories: Modality, Plane, Organ System, Abnormality).
  - [ ] **Member 2:** 
    - Implement `src/model/load_model.py` with `BitsAndBytesConfig` (4-bit, `nf4`, `torch.float16`).
    - Configure LoRA target modules (`q_proj`, `v_proj`, `k_proj`, `o_proj`, `r=16`, `alpha=32`).
    - Verify forward pass with dummy batch on Colab.
  - [ ] **Member 3:** 
    - Run **Zero-shot LLaVA-1.5-7B** on the validation set (`data/processed/val.json`).
    - Record initial zero-shot BLEU-1 and Accuracy scores as the benchmark baseline.
  - [ ] **Member 4:** 
    - Finalize Gradio layout with tabs (Diagnostic Query, Sample Gallery, Architecture Overview, Team Info).
    - Prepare 4 curated sample radiology images (1 Modality, 1 Plane, 1 Organ, 1 Abnormality) for demo gallery.
* **🚦 Daily Sync Gate (05:30 PM):** `data/processed/train.json` verified by Member 2; Baseline Zero-Shot numbers logged by Member 3.

---

### 🔹 PHASE 2: Fine-Tuning Execution & Performance Profiling (Days 3–5)

#### 📅 **Day 3: Training Pipeline Launch & Inference Engine Scaffold**
* **🎯 Phase Objective:** Initiate model fine-tuning on Colab and implement modular inference.
* **Individual Deliverables:**
  - [ ] **Member 1:** 
    - Implement PyTorch `Dataset` and `DataLoader` in `src/data/dataset.py` with token truncation and padding checks.
    - Write data validation sanity check script to catch corrupted/missing images.
  - [ ] **Member 2:** 
    - Implement training loop in `notebooks/train_llava_vqamed.ipynb` using HuggingFace `Trainer` / SFT.
    - Setup training parameters: `epochs=3`, `per_device_train_batch_size=2`, `gradient_accumulation_steps=8` (effective batch size = 16), `lr=2e-4`, `warmup_ratio=0.03`, `fp16=True`.
    - Run Trial 1 (Epoch 1) and inspect loss curve convergence.
  - [ ] **Member 3:** 
    - Scaffold `src/evaluation/benchmark_latency.py` with `benchmark_timer` context manager.
    - Define timing hooks for the 8 pipeline stages:
      1. Image Load
      2. Image Preprocessing (Resize & Normalize)
      3. CLIP Vision Encoder (`ViT-L/14`)
      4. Multi-Modal MLP Projection
      5. Text Tokenization
      6. LLM Prefill (Prompt + Image tokens)
      7. LLM Autoregressive Decode
      8. Detokenization & Output Clean
  - [ ] **Member 4:** 
    - Implement `src/inference/predict.py` with command-line arguments (`--image`, `--question`, `--model_path`, `--device`).
    - Connect inference logic to dummy/zero-shot model for pipeline testing.
* **🚦 Daily Sync Gate (05:30 PM):** Member 2 shares Colab Epoch 1 training logs; Member 3 shows timer output on test dummy input.

---

#### 📅 **Day 4: Hyperparameter Optimization & Latency Suite Buildout**
* **🎯 Phase Objective:** Optimize training stability, prevent overfitting, and expand benchmark harnesses.
* **Individual Deliverables:**
  - [ ] **Member 1:** 
    - Categorize validation set into 4 sub-files or indexed lists (`modality_val.json`, `plane_val.json`, `organ_val.json`, `abnormality_val.json`) for granular metric reporting.
  - [ ] **Member 2:** 
    - Run full 3-epoch QLoRA training on Colab T4.
    - Save intermediate checkpoints (`checkpoint-500`, `checkpoint-1000`, `checkpoint-final`) to Google Drive.
    - Track training loss and validation loss every 200 steps.
  - [ ] **Member 3:** 
    - Build test harnesses for the 5 latency measurement periods:
      1. **Cold Start** (first 5 inferences after model initialization)
      2. **Burst Load** (50 back-to-back fast requests)
      3. **Sustained Load** (200 continuous requests with throughput tracking)
      4. **Concurrent Requests** (simulated parallel requests)
      5. **Varied Sequence Lengths** (short vs long prompt/output tokens)
    - Implement automatic calculation of Mean, P50, P95, P99, and Std Dev.
  - [ ] **Member 4:** 
    - Create prompt engineering templates for clinical diagnostic queries (e.g., standardizing medical question prefixes).
    - Add Gradio UI components: confidence badges, category tags, latency readout timer in demo UI.
* **🚦 Daily Sync Gate (05:30 PM):** Checkpoint saved and verified on Google Drive; Benchmarking harness tested with simulated latency numbers.

---

#### 📅 **Day 5: Mid-Sprint Checkpoint & Full Pipeline Dry-Run**
* **🎯 Phase Objective:** Connect fine-tuned weights with evaluation and demo pipelines.
* **🤝 Shared Team Event (02:00 - 03:00 PM) — Mid-Sprint Knowledge Sync:**
  - Review fine-tuning training loss curves.
  - Review how the LoRA adapter is loaded on top of base LLaVA.
  - Test the fine-tuned adapter on 5 sample radiology queries live.
* **Individual Deliverables:**
  - [ ] **Member 1:** 
    - Review qualitative predictions from Epoch 3 and identify top 10 false positive / false negative edge cases.
  - [ ] **Member 2:** 
    - Export final merged/standalone LoRA adapter weights (`checkpoints/llava-vqamed-qlora/`).
    - Create Hugging Face Hub model upload script or direct Colab push.
  - [ ] **Member 3:** 
    - Run initial evaluation of fine-tuned checkpoint on validation set.
    - Compare initial scores against Day 2 Zero-Shot baseline.
  - [ ] **Member 4:** 
    - Hook fine-tuned model checkpoint into `demo/app.py`.
    - Verify Gradio demo functions locally or in Colab Gradio tunnel.
* **🚦 Daily Sync Gate (05:30 PM):** Full end-to-end dry run: Image Upload → Inference → Metric check verified by all 4 members.

---

### 🔹 PHASE 3: Evaluation, Benchmarking & Integration (Days 6–7)

#### 📅 **Day 6: Comprehensive Model Evaluation & Error Analysis**
* **🎯 Phase Objective:** Generate definitive evaluation numbers and deep clinical error insights.
* **Individual Deliverables:**
  - [ ] **Member 1:** 
    - Perform deep **Clinical Error Analysis**:
      - Why did the model fail on certain Abnormality questions?
      - Assess vocabulary mismatch (e.g., synonym usage: "fracture" vs "bone break").
      - Document clinical findings and failure taxonomy.
  - [ ] **Member 2:** 
    - Assist Member 3 with GPU runtime optimization (eval batching, memory pinning, torch.inference_mode).
    - Document model architecture hyperparameters and training compute cost in `results/training_summary.md`.
  - [ ] **Member 3:** 
    - Execute formal evaluation on all 2,000 validation pairs using `src/evaluation/evaluate.py`.
    - Generate breakdown tables:
      - Overall BLEU-1 & Accuracy
      - Category breakdown: Modality, Plane, Organ System, Abnormality
      - Comparison table: Zero-Shot vs. Fine-Tuned LLaVA vs. Literature SOTA
    - Save report to `results/evaluation_report.txt`.
  - [ ] **Member 4:** 
    - Refine Gradio UI with error handling (invalid file format, out-of-domain images).
    - Add example buttons for instant testing.
* **🚦 Daily Sync Gate (05:30 PM):** `results/evaluation_report.txt` finalized and approved by all members.

---

#### 📅 **Day 7: Full-Scale Latency Benchmark & Cloud Deployment**
* **🎯 Phase Objective:** Complete latency characterization and deploy live demo to Hugging Face Spaces.
* **Individual Deliverables:**
  - [ ] **Member 1:** 
    - Generate summary visualization charts (Accuracy by category bar chart, Training loss curve, Error distribution pie chart).
  - [ ] **Member 2:** 
    - Ensure adapter weights and model config are uploaded to Hugging Face Hub (`<hf-user>/llava-vqamed-qlora`).
  - [ ] **Member 3:** 
    - Execute full latency benchmarking script on T4 GPU:
      ```bash
      python src/evaluation/benchmark_latency.py \
        --model_path checkpoints/llava-vqamed-qlora/ \
        --data_path data/processed/val.json \
        --n_burst 50 --n_sustained 200 --n_cold 5 --n_concurrent 20 \
        --output_file results/latency_benchmark.txt
      ```
    - Generate latency breakdown waterfall chart (Image load vs Preprocessing vs CLIP vs MLP vs LLM Prefill vs LLM Decode).
  - [ ] **Member 4:** 
    - Deploy Gradio app to **Hugging Face Spaces**.
    - Verify public URL accessibility and test response latency on cloud.
    - Record a 2-minute high-quality screencast video demonstrating the live system.
* **🚦 Daily Sync Gate (05:30 PM):** Live Hugging Face Space URL verified; `results/latency_benchmark.txt` completed.

---

### 🔹 PHASE 4: Polish, Reporting, Presentation & Code Freeze (Days 8–9)

#### 📅 **Day 8: Technical Report, Slide Deck & Code Documentation**
* **🎯 Phase Objective:** Synthesize all technical assets into the final report and presentation slides.
* **Individual Deliverables:**
  - [ ] **Member 1:** 
    - Draft "Dataset & Preprocessing" and "Error Analysis & Clinical Discussion" sections in Final Report.
  - [ ] **Member 2:** 
    - Draft "Model Architecture, QLoRA Fine-Tuning & Quantization" section in Final Report.
  - [ ] **Member 3:** 
    - Draft "Evaluation Metrics, Zero-Shot vs Fine-Tuned Results, & 8-Stage Latency Benchmarking" section in Final Report.
  - [ ] **Member 4:** 
    - Assemble the master slide deck (12–15 slides covering: Problem Statement, Pipeline, Dataset, Architecture, Training, Results, Latency, Live Demo, Ethical & Clinical Limitations).
    - Draft "Demo System & User Interface" section in Final Report.
* **🚦 Daily Sync Gate (05:30 PM):** Slide deck v1 and Report Draft v1 completed and reviewed collectively.

---

#### 📅 **Day 9: Final Rehearsal, Code Freeze & Deliverable Hand-In**
* **🎯 Phase Objective:** Final verification, repository freeze, and presentation rehearsal.
* **Team Joint Activities:**
  - [ ] **09:00 - 11:00 AM:** Full presentation rehearsal (all 4 members present their respective sections).
  - [ ] **11:00 - 01:00 PM:** Peer-review and polish slide deck and technical report.
  - [ ] **02:00 - 04:00 PM:** **Code Freeze**:
    - Clean up repository (remove temp logs, check `.gitignore`, format code with `black` / `flake8`).
    - Verify `README.md` links, sample commands, and licenses.
    - Tag release `v1.0.0` on GitHub.
  - [ ] **04:00 - 05:00 PM:** Final submission package verification.
* **🎉 Final Deliverables Complete!**

---

## 🛠️ Technical Architecture & Pipeline Breakdown

```
                    ┌────────────────────────────────────────────────────────┐
                    │               MEDICAL VQA SYSTEM PIPELINE               │
                    └────────────────────────────────────────────────────────┘

    [Input Medical Image]                                    [Clinical Question]
     (X-Ray, CT, MRI, US)                                  ("What organ is shown?")
              │                                                        │
              ▼ (Stage 1: Load & Stage 2: Preprocess)                  ▼ (Stage 5: Tokenize)
    ┌──────────────────────┐                                 ┌──────────────────────┐
    │  CLIP Vision Encoder │ (Stage 3: ViT-L/14)             │  Text Tokenizer      │
    │  (336×336 px, Frozen)│                                 │  (LLaVA Tokenizer)   │
    └──────────┬───────────┘                                 └──────────┬───────────┘
               │ Visual Embedding                                       │ Text Tokens
               ▼                                                        │
    ┌──────────────────────┐                                            │
    │ 2-Layer MLP Adapter  │ (Stage 4: Trainable Projection)            │
    └──────────┬───────────┘                                            │
               │ Projected Visual Tokens                                │
               └──────────────────────┬─────────────────────────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │ Multi-Modal Token Fusion  │
                        └─────────────┬─────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │     LLaVA-1.5-7B LLM      │ (Stage 6: Prefill)
                        │  (QLoRA 4-bit NF4 INT4)   │ (Stage 7: Decode)
                        └─────────────┬─────────────┘
                                      │ Output Tokens
                                      ▼ (Stage 8: Detokenize)
                        ┌───────────────────────────┐
                        │   Diagnostic Response     │
                        │   e.g. "Brain" / "Axial"  │
                        └───────────────────────────┘
```

---

## 🔬 Target Evaluation & Benchmark Metrics

### 1. Accuracy Targets (Validation Set - 2,000 QA Pairs)

| Category | Questions in Val | Zero-Shot LLaVA Baseline (Expected) | Fine-Tuned LLaVA (Target) | Evaluation Metric |
|---|:---:|:---:|:---:|---|
| **Modality** | 500 | ~45% | **82% - 88%** | Exact Match / BLEU-1 |
| **Plane** | 500 | ~40% | **78% - 84%** | Exact Match / BLEU-1 |
| **Organ System** | 500 | ~35% | **70% - 76%** | Exact Match / BLEU-1 |
| **Abnormality** | 500 | ~20% | **60% - 68%** | Exact Match / BLEU-1 |
| **Overall** | **2,000** | **~35%** | **74% - 80%** | **BLEU-1 + Accuracy** |

### 2. Latency Profiling Targets (T4 GPU, Sustained N=200)

| Stage # | Pipeline Stage | Target P50 | Target P95 | Target P99 |
|:---:|---|:---:|:---:|:---:|
| 1 | Image Load (Disk/Network) | < 10 ms | < 15 ms | < 25 ms |
| 2 | Image Preprocessing & Normalize | < 25 ms | < 35 ms | < 50 ms |
| 3 | CLIP Vision Encoder (ViT-L/14) | < 150 ms | < 180 ms | < 220 ms |
| 4 | MLP Projection Layer | < 5 ms | < 8 ms | < 12 ms |
| 5 | Text Tokenization | < 3 ms | < 5 ms | < 8 ms |
| 6 | LLM Prefill (Prompt + Visual Embeddings) | < 500 ms | < 600 ms | < 750 ms |
| 7 | LLM Autoregressive Decode (Max 30 tokens) | < 1,100 ms | < 1,350 ms | < 1,600 ms |
| 8 | Detokenization & Output Clean | < 2 ms | < 4 ms | < 6 ms |
| **Total** | **End-to-End Inference Latency** | **< 1,800 ms** | **< 2,200 ms** | **< 2,650 ms** |

---

## ⚠️ Risk Management & Contingency Plan

| # | Identified Risk | Severity | Probability | Contingency & Mitigation Strategy |
|---|---|:---:|:---:|---|
| 1 | **Google Colab Free Tier Disconnect / GPU Timeout** | High | High | • Enable Google Drive auto-saving for checkpoints every 200 steps.<br>• Member 2 & Member 3 maintain separate Google accounts ready as backup.<br>• Keep training scripts runnable in modular resume-from-checkpoint mode. |
| 2 | **CUDA Out of Memory (OOM) on 15GB T4 GPU** | High | Medium | • Use 4-bit NormalFloat (`nf4`) quantization with `bnb_4bit_compute_dtype=torch.float16`.<br>• Use `gradient_checkpointing=True`.<br>• Cap batch size to 2 and use gradient accumulation steps = 8. |
| 3 | **Zenodo / Data Download Network Bottlenecks** | Medium | Low | • Download raw dataset once on Day 1, create a compressed backup zip on Google Drive.<br>• Share direct Google Drive download link among team members. |
| 4 | **Hugging Face Spaces Free CPU Resource Limit** | Medium | Medium | • For HF Spaces, run model in 4-bit CPU or optimized ONNX/LoRA adapter mode, or utilize Colab backend with Gradio tunnel for live GPU demo. |
| 5 | **Merge Conflicts in Git** | Medium | Medium | • Strictly follow GitHub Flow (one feature branch per component).<br>• Do not commit large binary checkpoints to Git (`.gitignore` `checkpoints/` and `data/`). |

---

## 🌿 Collaboration & Git Workflow

### Branching Convention
- `main`: Protected production-ready branch. Code must pass linting and run without errors.
- `feat/data-pipeline`: Owned by Member 1.
- `feat/model-training`: Owned by Member 2.
- `feat/evaluation-benchmarks`: Owned by Member 3.
- `feat/demo-ui-reporting`: Owned by Member 4.

### Daily Standup Template (15-Minute Sync at 05:30 PM)
Every member briefly answers 3 questions:
1. **What did I accomplish today?** (Demo code/output in 2 minutes)
2. **What will I accomplish tomorrow?**
3. **Are there any blockers or cross-member dependencies?**

---

## ✅ Definition of Done (DoD) Checklist

- [ ] All code runs cleanly on Python 3.10+ without unhandled exceptions.
- [ ] `data/processed/train.json` and `val.json` generated and verified against original 14,792 QA pairs.
- [ ] Model successfully fine-tuned with QLoRA and weights saved.
- [ ] `results/evaluation_report.txt` contains quantitative BLEU-1 and Accuracy scores comparing Zero-Shot vs Fine-Tuned across all 4 categories.
- [ ] `results/latency_benchmark.txt` contains timing for all 8 stages across Cold, Burst, Sustained, and Concurrent periods.
- [ ] Gradio demo deployed and functioning with live image upload and question answering.
- [ ] `README.md` is complete with reproduction commands, architecture diagram, and license disclosures.
- [ ] 15-slide presentation deck and project report finalized and rehearsed.
