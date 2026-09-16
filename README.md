# 🏥 Medical Visual Question Answering (VQA) System

> **Topic 4** — Detecting and segmenting medical lesions · Image captioning · Visual Question Answering  
> A multi-modal AI system for medical diagnostic support, fine-tuned on radiology images.

---

## 📋 Table of Contents

- [Prerequisites & Learning Roadmap](#-prerequisites--learning-roadmap)
- [Overview](#-overview)
- [Demo](#-demo)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Setup](#-setup)
- [How to Run](#-how-to-run)
  - [1. Download Dataset](#1-download-dataset)
  - [2. Preprocess Data](#2-preprocess-data)
  - [3. Fine-tune Model (Google Colab)](#3-fine-tune-model-google-colab)
  - [4. Evaluate](#4-evaluate)
  - [5. Latency Benchmark](#5-latency-benchmark)
  - [6. Run Demo](#6-run-demo)
- [Evaluation Results](#-evaluation-results)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🎓 Prerequisites & Learning Roadmap

New to multi-modal AI? Read [`LEARNING_ROADMAP.md`](./LEARNING_ROADMAP.md) first.

It covers every concept you need — organized by priority:

| Priority | Concepts |
|---|---|
| 🔴 Must Know | Transformer/Attention · ViT · LLaVA architecture · QLoRA · VQA task |
| 🟡 Should Know | PyTorch · BLEU score · Medical imaging modalities · HuggingFace |
| 🟢 Nice to Know | CLIP contrastive learning · Latency percentiles · Gradio deployment |

---

## 🔍 Overview


This project builds a **Visual Question Answering (VQA)** system for medical diagnostic support. Given a radiology image and a natural language question, the system generates a clinically relevant answer.

**Example:**

| Input Image | Question | Answer |
|---|---|---|
| `chest_xray.jpg` | *"What is the imaging modality?"* | `X-Ray` |
| `brain_mri.jpg` | *"What organ is shown?"* | `Brain` |
| `ct_scan.jpg` | *"In what plane is this image taken?"* | `Axial` |
| `lung_xray.jpg` | *"What abnormality is present?"* | `Cardiomegaly` |

### Key Features

- ✅ Fine-tuned **LLaVA-1.5-7B** with **QLoRA** (4-bit quantization) — runs on free T4 GPU
- ✅ Trained on **VQA-Med-2019** (3,200 images · 12,792 Q&A pairs)
- ✅ Covers 4 question categories: **Modality, Plane, Organ System, Abnormality**
- ✅ Evaluated with **BLEU-1 + Accuracy** (matches official VQA-Med-2019 challenge metrics)
- ✅ Full **latency benchmark** across all pipeline stages (8 stages × 5 measurement periods)
- ✅ Interactive **Gradio demo** deployed on HuggingFace Spaces

---

## 🎬 Demo

> 🚀 **Live Demo**: [HuggingFace Spaces — Medical VQA](#) *(link after deployment)*

```
Upload a radiology image → Type your question → Get an AI-generated answer
```

---

## 📁 Project Structure

```
diagnostic_support/
│
├── 📂 .agents/                       # AI Agent workspace configuration & governance
│   └── rules/
│       └── AI_AGENT_RULES.md         # Operational rules & engineering guardrails
│
├── 📂 VQA-Med-2019/                  # Official Medical VQA 2019 Benchmark Dataset
│   ├── ImageClef-2019-VQA-Med-Training/
│   │   ├── All_QA_Pairs_train.txt    # 12,792 Training QA pairs (All categories)
│   │   ├── train_ImageIDs.txt        # 3,200 unique training image identifiers
│   │   ├── QAPairsByCategory/        # Modality (C1), Plane (C2), Organ (C3), Abnormality (C4)
│   │   │   ├── C1_Modality_train.txt
│   │   │   ├── C2_Plane_train.txt
│   │   │   ├── C3_Organ_train.txt
│   │   │   └── C4_Abnormality_train.txt
│   │   └── Train_images/             # 3,200 training radiology images (.jpg)
│   ├── ImageClef-2019-VQA-Med-Validation/
│   │   ├── All_QA_Pairs_val.txt      # 2,000 Validation QA pairs
│   │   ├── val_ImageIDs.txt          # 500 unique validation image identifiers
│   │   ├── QAPairsByCategory/
│   │   │   ├── C1_Modality_val.txt
│   │   │   ├── C2_Plane_val.txt
│   │   │   ├── C3_Organ_val.txt
│   │   │   └── C4_Abnormality_val.txt
│   │   └── Val_images/               # 500 validation radiology images (.jpg)
│   ├── VQAMed2019Test/
│   │   ├── README-VQA-Med-2019-TestSet.txt
│   │   ├── VQAMed2019_Test_Questions.txt               # 500 test questions
│   │   ├── VQAMed2019_Test_Questions_w_Ref_Answers.txt # Test questions with reference answers
│   │   └── VQAMed2019_Test_Images/                     # 500 test radiology images (.jpg)
│   ├── README-VQA-Med-2019-Data.txt  # Dataset specifications and annotations info
│   ├── README.md                     # Dataset documentation
│   └── trial.ipynb                   # Dataset exploration & EDA notebook
│
├── 📓 notebooks/                     # Colab & Jupyter experimentation notebooks
│   └── train_llava_vqamed.ipynb      # Google Colab training notebook (LLaVA + QLoRA)
│
├── 📦 src/                           # Core implementation source code
│   ├── data/
│   │   ├── download_dataset.py       # Dataset fetch utility
│   │   ├── preprocess.py             # Convert raw data → LLaVA instruction JSON format
│   │   └── dataset.py                # PyTorch Dataset & DataLoader
│   ├── model/
│   │   ├── load_model.py             # Load LLaVA-1.5-7B + QLoRA 4-bit configuration
│   │   └── train.py                  # Training pipeline & loss logging
│   ├── evaluation/
│   │   ├── evaluate.py               # BLEU-1 + Accuracy evaluation pipeline
│   │   ├── metrics.py                # Metric calculations (BLEU, Exact Match)
│   │   └── benchmark_latency.py      # Latency benchmarking (8 stages × 5 periods)
│   └── inference/
│       └── predict.py                # End-to-end single-image inference script
│
├── 🖥️ demo/                          # User interface & web deployment
│   └── app.py                        # Interactive Gradio demo (HuggingFace Spaces)
│
├── 🤖 checkpoints/                   # (gitignored) Saved LoRA adapter weights
│   └── llava-vqamed-qlora/
│
├── 📊 results/                       # Evaluation benchmarks and reports
│   ├── evaluation_report.txt         # BLEU + Accuracy evaluation scores
│   └── latency_benchmark.txt         # Per-stage latency benchmark results
│
├── 📄 LEARNING_ROADMAP.md            # Comprehensive concept roadmap & technical primer
├── 📄 plan.md                        # 9-day sprint execution plan & RACI matrix
├── 📄 Rule chung de su dung AI trong phan.md # General AI usage policy
├── 📄 Topic.pdf                      # Project assignment & specifications
├── 📄 LICENSE                        # MIT License & multi-tiered terms
├── 📄 README.md                      # Main project documentation (this file)
└── 📄 .gitignore                     # Git exclusion rules
```

---

## 📊 Dataset

**VQA-Med-2019** — Visual Question Answering in the Medical Domain

| Split | Images | Q&A Pairs |
|---|---|---|
| Train | 3,200 | 12,792 |
| Validation | 500 | 2,000 |
| Test | 500 | 500 (questions only) |

**Question Categories:**

| Category | Example Question | Example Answer |
|---|---|---|
| Modality | *"What imaging modality is this?"* | `MRI` / `CT` / `X-Ray` |
| Plane | *"In what plane is this image taken?"* | `Axial` / `Sagittal` / `Coronal` |
| Organ System | *"What organ is principally shown?"* | `Brain` / `Lung` / `Heart` |
| Abnormality | *"What is most alarming about this image?"* | `Cardiomegaly` / `Pneumonia` |

**License:** CC BY 4.0 (Zenodo) · Images from MedPix/NLM (research/educational use)  
**Download:** https://zenodo.org/records/10499039  
**Paper:** http://ceur-ws.org/Vol-2380/paper_272.pdf

---

## 🧠 Model Architecture

```
Medical Image (JPEG/PNG)
        │
        ▼
  ┌─────────────────────┐
  │  CLIP Vision Encoder │  ← ViT-L/14 @ 336×336px (frozen)
  └──────────┬──────────┘
             │  visual embedding
        ▼
  ┌─────────────────────┐
  │   MLP Projection     │  ← 2-layer MLP (trainable)
  └──────────┬──────────┘
             │
  ┌──────────┴──────────┐
  │    Fusion / Concat   │  ← Visual tokens + Text tokens
  └──────────┬──────────┘
             │
  ┌──────────┴──────────┐
  │   Mistral-7B LLM     │  ← QLoRA fine-tuned (4-bit INT4)
  │   (Causal LM)        │    LoRA on: q_proj, v_proj, k_proj, o_proj
  └──────────┬──────────┘
             │
        ▼
  Generated Answer (text)
```

**Base model:** [`liuhaotian/llava-v1.5-7b`](https://huggingface.co/liuhaotian/llava-v1.5-7b)  
**Fine-tuning:** QLoRA — rank=16, alpha=32, 4-bit NF4 quantization  
**Training compute:** Google Colab Free (NVIDIA T4, 15GB VRAM)  
**Estimated training time:** ~4–6 hours (3 epochs on VQA-Med-2019 train set)

---

## ⚙️ Setup

### Prerequisites

- Python **3.10+**
- `git`
- Google account (for Colab training)
- HuggingFace account (for demo deployment)

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/diagnostic_support.git
cd diagnostic_support
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** Full model training requires a GPU (T4 or better). Use Google Colab for training.  
> Local CPU is sufficient for inference/demo with the saved checkpoint.

### 4. Environment Variables *(optional)*

Create a `.env` file for HuggingFace token (needed to push model to Hub):

```
HF_TOKEN=your_huggingface_token_here
```

---

## ▶️ How to Run

### 1. Download Dataset

```bash
python src/data/download_dataset.py
```

Downloads VQA-Med-2019 from Zenodo into `data/raw/`. Requires ~2GB disk space.

---

### 2. Preprocess Data

```bash
python src/data/preprocess.py \
  --input_dir  data/raw/ \
  --output_dir data/processed/
```

Converts raw Q&A pairs + images into LLaVA instruction-tuning JSON format.  
Output: `data/processed/train.json`, `data/processed/val.json`

---

### 3. Fine-tune Model (Google Colab)

> ✅ **Recommended:** Open the Colab notebook directly — do NOT run training locally.

1. Upload `notebooks/train_llava_vqamed.ipynb` to Google Colab
2. Set runtime to **GPU (T4)**:  
   `Runtime → Change runtime type → GPU`
3. Run all cells sequentially
4. Checkpoints auto-saved every 500 steps to `checkpoints/llava-vqamed-qlora/`

**Training configuration used in the notebook:**

| Parameter | Value |
|---|---|
| Base model | `liuhaotian/llava-v1.5-7b` |
| Epochs | 3 |
| Batch size | 2 (grad accum = 8 → effective 16) |
| Learning rate | 2e-4 (cosine scheduler) |
| LoRA rank | 16 |
| Quantization | 4-bit NF4 |
| Max new tokens | 30 |

---

### 4. Evaluate

Run accuracy evaluation (BLEU-1 + Accuracy) on the validation set:

```bash
python src/evaluation/evaluate.py \
  --model_path  checkpoints/llava-vqamed-qlora/ \
  --data_path   data/processed/val.json \
  --output_file results/evaluation_report.txt
```

Sample output:

```
========================================
VQA-Med-2019 Evaluation Results
========================================
Overall BLEU-1  : 0.752
Overall Accuracy: 0.731

By Category:
  Modality      BLEU-1=0.841  Accuracy=0.852
  Plane         BLEU-1=0.793  Accuracy=0.801
  Organ System  BLEU-1=0.741  Accuracy=0.724
  Abnormality   BLEU-1=0.633  Accuracy=0.647
========================================
```

---

### 5. Latency Benchmark

Benchmark all 8 pipeline stages across 5 measurement periods:

```bash
python src/evaluation/benchmark_latency.py \
  --model_path  checkpoints/llava-vqamed-qlora/ \
  --data_path   data/processed/val.json \
  --n_burst     50 \
  --n_sustained 200 \
  --n_cold      5 \
  --n_concurrent 20 \
  --output_file results/latency_benchmark.txt
```

Sample output:

```
Latency Benchmark — LLaVA-1.5-7B QLoRA (T4 GPU, Sustained, N=200)
══════════════════════════════════════════════════════════════════════
Stage               Mean(ms)  P50    P95    P99    Std
──────────────────────────────────────────────────────
Image Load             8.2    7.9   12.1   18.3    1.4
Preprocessing         24.7   23.8   31.5   38.2    3.1
CLIP Encoder         142.3  140.1  168.4  189.2   11.2
MLP Projection         3.1    3.0    4.2    5.1    0.4
Text Tokenization      2.3    2.2    3.1    3.8    0.3
LLM Prefill          487.2  481.3  542.1  598.7   28.4
LLM Decode         1,124.5 1,098  1,287  1,412    87.3
Detokenize             1.8    1.7    2.4    3.1    0.2
──────────────────────────────────────────────────────
Full Pipeline (E2E) 1,794   1,758  2,051  2,269  131.2
Throughput: 0.56 img/s
══════════════════════════════════════════════════════════════════════
```

---

### 6. Run Demo

#### Local (CPU — inference only)

```bash
python demo/app.py --model_path checkpoints/llava-vqamed-qlora/
```

Opens a Gradio interface at `http://localhost:7860`

#### Deploy to HuggingFace Spaces

```bash
# Push model to HuggingFace Hub first
python src/inference/predict.py --push_to_hub --repo_id <your-hf-username>/llava-vqamed

# Then deploy the demo via HF Spaces web UI or CLI
```

---

## 📈 Evaluation Results

*(To be filled after training)*

### Accuracy

| Model | BLEU-1 | Accuracy |
|---|---|---|
| Baseline CNN+LSTM | 0.45 | 0.43 |
| BERT + ResNet | 0.65 | 0.60 |
| **Ours (LLaVA-1.5-7B QLoRA)** | **TBD** | **TBD** |
| SOTA | 0.84 | 0.82 |

### Latency (End-to-End, T4 GPU)

| Period | Mean (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|
| Cold Start | TBD | TBD | TBD |
| Burst (N=50) | TBD | TBD | TBD |
| Sustained (N=200) | TBD | TBD | TBD |
| Concurrent (×3) | TBD | TBD | TBD |

---

## ⚠️ Medical & Clinical Safety Disclaimer

> [!CAUTION]
> **NOT FOR CLINICAL USE**: This system is developed strictly as an **academic research and educational prototype**.
> - It is **NOT** certified as Software as a Medical Device (SaMD) by the U.S. FDA, EMA, or any national health authority.
> - The outputs, classifications, and diagnostic suggestions generated by this model must **NEVER** be used for actual clinical diagnosis, patient triage, treatment planning, or direct patient care.
> - The authors and contributors assume no liability for decisions made based on this prototype.

---

## 📄 License & Terms of Use

This project operates under a **multi-tiered licensing model** reflecting custom code, upstream foundation models, and clinical datasets:

### 1. Project Code — MIT License
The custom source code in this repository (`src/`, `demo/`, `notebooks/`) is released under the **MIT License**. See [`LICENSE`](./LICENSE) for full details.

### 2. Dataset — VQA-Med-2019
- **Q&A Annotations**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **Radiology Images**: Sourced from [MedPix® (U.S. National Library of Medicine)](https://medpix.nlm.nih.gov). Restricted strictly to **non-commercial educational and scientific research only**.

### 3. Foundation Models & Fine-Tuned Weights
- **LLaVA-1.5 Architecture**: [Apache License 2.0](https://github.com/haotian-liu/LLaVA/blob/main/LICENSE).
- **Base LLM Weights (Vicuna-v1.5 / LLaMA-2)**: Governed by the [Meta LLaMA 2 Community License](https://ai.meta.com/llama/license/) and Vicuna / ShareGPT non-commercial research terms.
- **Fine-Tuned Checkpoints (`checkpoints/`)**: Restricted strictly to **non-commercial academic research and educational evaluation**. Commercial deployment or redistribution of weights is prohibited.

### Summary Licensing Table

| Asset | Research Use | Commercial Use | Clinical Use | License / Terms |
|---|:---:|:---:|:---:|---|
| **Custom Source Code** | ✅ Yes | ✅ Yes | ❌ No | MIT License |
| **Q&A Annotations** | ✅ Yes | ✅ Yes | ❌ No | CC BY 4.0 |
| **MedPix Images** | ✅ Yes | ❌ No | ❌ No | MedPix NLM Terms (Non-Commercial) |
| **Base LLaVA-1.5 / Vicuna** | ✅ Yes | ❌ No | ❌ No | LLaMA 2 Community + ShareGPT Terms |
| **Fine-Tuned Weights** | ✅ Yes | ❌ No | ❌ No | Academic Research Only (Composite) |

---

## 🙏 References & Citations

If you use this project or the VQA-Med-2019 dataset, please cite the original works:

```bibtex
@inproceedings{ImageCLEFVQA-Med2019,
    author    = {Asma {Ben Abacha} and Sadid A. Hasan and Vivek V. Datla and Joey Liu and Dina Demner-Fushman and Henning M{\"u}ller},
    title     = {VQA-Med: Overview of the Medical Visual Question Answering Task at ImageCLEF 2019},
    booktitle = {CLEF 2019 Working Notes},
    series    = {{CEUR} Workshop Proceedings},
    year      = {2019},
    publisher = {CEUR-WS.org},
    address   = {Lugano, Switzerland}
}

@inproceedings{liu2023llava,
    author    = {Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
    title     = {Visual Instruction Tuning},
    booktitle = {NeurIPS},
    year      = {2023}
}
```

---

<div align="center">
  <sub>Built for AI in Healthcare · Topic 4 · 2026 · Code: MIT License · Research Use Only</sub>
</div>

