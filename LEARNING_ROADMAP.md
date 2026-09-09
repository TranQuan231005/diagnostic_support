# 📚 Learning Roadmap — Medical VQA Project
### Mapped directly to Topic 4 requirements

> **Goal:** Understand every component shown in the topic PDF so you can
> read the code, run experiments, and explain the system end-to-end.

---

## 🗂️ What the Topic PDF Requires You to Understand

From the topic specification, the system pipeline is:

```
┌─────────────────────────────────────────────────────────────┐
│                    TOPIC 4 PIPELINE                          │
│                                                              │
│  Image ──→ CNN / ViT ──→ Visual Embedding ──┐               │
│                                              │               │
│                                         Fusion Module        │
│                                              │               │
│  Text ──→ Transformer ──→ Text Embedding ───┘               │
│                                              │               │
│                                              ▼               │
│                                    Caption / Answer          │
│                                                              │
│  Two tasks:                                                  │
│  • Captioning  — image → sentence description                │
│  • VQA         — image + question → answer                   │
│                                                              │
│  Suggested datasets: MS COCO Captions · VQA v2               │
│  Application: Medical Diagnostic Support                     │
└─────────────────────────────────────────────────────────────┘
```

Each box in this diagram is a **learning milestone**. This roadmap walks you
through them one by one, in dependency order.

---

## 🗺️ Full Roadmap at a Glance

```
PHASE 1 — Foundations (Week 1)
    └─ Python · NumPy · PyTorch · Neural Network basics

PHASE 2 — Image → CNN/ViT → Visual Embedding (Week 2)
    ├─ Convolutional Neural Networks (CNNs)
    ├─ Feature extraction (what is a visual embedding?)
    └─ Vision Transformer (ViT)

PHASE 3 — Text → Transformer → Text Embedding (Week 3)
    ├─ Word embeddings (Word2Vec, GloVe)
    ├─ Attention mechanism
    └─ Transformer architecture (BERT / GPT)

PHASE 4 — Fusion Module → Caption / Answer (Week 4)
    ├─ How fusion works (concatenation, cross-attention)
    ├─ Image Captioning task
    └─ Visual Question Answering (VQA) task

PHASE 5 — Datasets (Week 4–5)
    ├─ MS COCO Captions
    ├─ VQA v2
    └─ VQA-Med-2019 (our actual dataset)

PHASE 6 — Our Model: LLaVA + QLoRA (Week 5)
    ├─ CLIP (connects vision ↔ language)
    ├─ Large Language Models (LLMs)
    ├─ LLaVA architecture
    └─ QLoRA fine-tuning

PHASE 7 — Medical Domain (Week 6)
    └─ Medical imaging basics (X-Ray, MRI, CT, US)

PHASE 8 — Evaluation (Week 6)
    ├─ BLEU score
    ├─ Accuracy (exact match)
    └─ Latency benchmarking (P50/P95/P99)
```

---

## PHASE 1 — Foundations
### 🎯 Goal: Be able to write and read PyTorch training code

---

### 1.1 Python Essentials

**What you need to know:**

| Topic | Why it's needed in this project |
|---|---|
| Lists, dicts, loops | Loading Q&A pairs from dataset files |
| Classes (`__init__`, methods) | Writing `Dataset`, `Model` classes |
| File I/O (`open`, `json`) | Reading annotation JSON files |
| `with` statement | Context managers in benchmark timer |
| Decorators (`@contextmanager`) | Used in `benchmark_latency.py` |

**Resource:** [Python Official Tutorial](https://docs.python.org/3/tutorial/) — Chapters 3–9

**Checkpoint ✅** Can you write a class that reads a JSON file and returns
items one by one? If yes, move on.

---

### 1.2 NumPy

**What you need to know:**

| Topic | Why it's needed |
|---|---|
| `np.array`, shapes | Every image is a numpy array `(H, W, 3)` |
| Indexing, slicing | Selecting patches from images |
| Broadcasting | Computing metrics across batches |
| `np.mean`, `np.std` | Used in latency benchmark reporting |

**Resource:** [NumPy in 15 minutes](https://numpy.org/doc/stable/user/absolute_beginners.html)

---

### 1.3 PyTorch — The Deep Learning Framework

**What you need to know:**

| Topic | Why it's needed |
|---|---|
| `torch.Tensor` | Images and text become tensors |
| `nn.Module` | Every model is a subclass of this |
| `Dataset` & `DataLoader` | How we feed VQA-Med-2019 images into the model |
| `optimizer.step()` | Updates model weights during training |
| `loss.backward()` | Computes gradients |
| `torch.no_grad()` | Used during evaluation/inference (no gradient needed) |
| `.to("cuda")` | Moves tensors to GPU |

**Resource:** [PyTorch 60-Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

**Mini exercise:** Write a `Dataset` class that:
1. Loads an image file with `PIL.Image.open()`
2. Resizes it to `336×336`
3. Returns `(image_tensor, label)`

---

### 1.4 Neural Network Basics

**Key concepts:**

```
Input → [Hidden Layer 1] → [Hidden Layer 2] → Output
              ↑                  ↑
         weights W₁         weights W₂

Forward pass:  compute output from input
Loss:          measure how wrong the output is
Backward pass: compute gradient of loss w.r.t. every weight
Optimizer:     update weights to reduce loss (gradient descent)
```

**Resource:** [3Blue1Brown — Neural Networks series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (4 videos, ~1h total — the best visual explanation)

---

## PHASE 2 — Image → CNN/ViT → Visual Embedding
### 🎯 Goal: Understand how an image becomes a vector (visual embedding)

---

### 2.1 What is a Visual Embedding?

> An **embedding** is a list of numbers (a vector) that represents something.
> A **visual embedding** represents the content of an image.

```
chest_xray.jpg  →  [0.23, 0.87, 0.04, -0.56, 0.91, ...]
                    └──────────── 768 numbers ────────────┘
                    This vector captures: "chest", "bones", "X-ray texture"

brain_mri.jpg   →  [0.71, 0.12, 0.88, 0.33, -0.44, ...]
                    └──────────── 768 numbers ────────────┘
                    This vector captures: "brain", "MRI texture", "soft tissue"
```

Two images with similar content → similar vectors (close in vector space)
Two images with different content → different vectors (far apart)

---

### 2.2 Convolutional Neural Networks (CNNs)

**How it works:**

```
Image (3 × 336 × 336)                   ← 3 color channels, 336×336 pixels
    │
    ▼  Apply 64 filters (3×3)
Feature Map (64 × 334 × 334)            ← 64 patterns detected
    │
    ▼  MaxPool (2×2)
Feature Map (64 × 167 × 167)            ← downsampled
    │
    ▼  More conv layers...
Feature Map (512 × 10 × 10)
    │
    ▼  Global Average Pooling
Visual Embedding (512,)                  ← final 512-dimensional vector!
```

**Intuition:** Early layers detect edges. Middle layers detect shapes.
Late layers detect high-level concepts ("lung", "bone", "tumor").

**Resources:**
- [CS231n — CNNs for Visual Recognition](http://cs231n.stanford.edu/slides/2024/lecture_5.pdf) — Stanford slides
- [d2l.ai — Chapter 7: CNNs](https://d2l.ai/chapter_convolutional-neural-networks/index.html)

---

### 2.3 Vision Transformer (ViT)

> **The image encoder used in our project** (inside CLIP, inside LLaVA).

**Key idea:** Instead of convolutions, split the image into patches and treat each patch as a "token" — just like words in a sentence.

```
Image (336 × 336)
    │
    ▼  Split into 14×14 grid of 24×24 patches
196 patches
    │
    ▼  Embed each patch (linear projection)
196 vectors of size 768
    │
    ▼  Add positional embeddings (so model knows patch locations)
196 "visual tokens"  ← same format as text tokens!
    │
    ▼  Feed into standard Transformer encoder
196 contextualized vectors
    │
    ▼  Take the [CLS] token output
Visual Embedding (768,)  ← represents the whole image
```

**Why ViT instead of CNN?**
- Transformers scale better with data and compute
- Can be combined with text using the same architecture
- CLIP uses ViT-L/14 at 336px resolution

**Resources:**
- [An Image is Worth 16×16 Words (ViT paper)](https://arxiv.org/abs/2010.11929) — read the intro + Figure 1
- [ViT explained visually — lernapparat.de](https://lernapparat.de/visual-attention/)

---

## PHASE 3 — Text → Transformer → Text Embedding
### 🎯 Goal: Understand how a question becomes a vector (text embedding)

---

### 3.1 Tokenization

> Before a Transformer can process text, text must become numbers.

```
Question: "What organ is shown in this MRI?"
    │
    ▼  Tokenizer (vocabulary lookup)
Tokens:  ["What", "organ", "is", "shown", "in", "this", "MR", "##I", "?"]
    │
    ▼  Token IDs (index in vocabulary)
IDs:     [2054,   5652,   2003,  3491,    1999,  2023,  9827,  2072,  1029]
    │
    ▼  Embedding layer (look up vector for each ID)
Embeddings: 9 vectors, each of size 4096
```

---

### 3.2 Attention Mechanism

> The **core building block** of every modern AI model in this project.

**Intuition:** When processing the word "organ" in "What organ is shown?",
the attention mechanism lets it look at "shown" and "MRI" to understand context.

```
Query:  "organ" asks "who is relevant to me?"
Keys:   every other word raises its hand
Values: the actual information each word contributes

Attention score = softmax( Q · Kᵀ / √d )
Output = attention_score · V
```

**Visualized:**

```
"organ" attends to:
  "What"  →  0.05 (low — not very relevant)
  "organ" →  0.30 (itself)
  "shown" →  0.20
  "MRI"   →  0.40 (high — MRI tells us what kind of organ!)
  "?"     →  0.05
```

**Resource:** [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
→ **Read this first. It is the single best resource for understanding attention.**

---

### 3.3 Transformer Architecture

```
Text input: "What organ is shown?"
    │
    ▼  Tokenize + Embed
Token embeddings (9 × 4096)
    │
    ▼  Add positional encodings
Positional embeddings (9 × 4096)
    │
    ▼  Self-attention layer (each token attends to all others)
    ▼  Feed-forward layer
    ▼  × 32 Transformer layers
    │
Contextualized embeddings (9 × 4096)
    │
    ▼  Final layer output
Text Embedding (4096,)  ← represents the whole question
```

**Resources:**
- [Attention is All You Need — original paper](https://arxiv.org/abs/1706.03762)
- [Andrej Karpathy — Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) (3h — hands-on implementation)

---

## PHASE 4 — Fusion Module → Caption / Answer
### 🎯 Goal: Understand how image + text are combined to produce output

---

### 4.1 The Fusion Module

> This is the **bridge** between vision and language — the core of multi-modal AI.

The topic PDF says: `Fusion module → Caption / Answer`

**Three common fusion strategies:**

```
Strategy 1: Concatenation (simple)
    [visual_embedding | text_embedding] → Linear → Answer
    [768 dims         | 4096 dims     ] = 4864 dims

Strategy 2: Cross-attention (powerful)
    Q = text_embedding
    K = V = visual_embedding
    → Text tokens attend to image regions

Strategy 3: LLaVA approach (what we use)
    visual_embedding → MLP Projection → language-space vectors
    → Simply concatenate with text tokens as input to LLM
    → LLM naturally fuses both modalities
```

---

### 4.2 Image Captioning Task

> **Captioning:** Image → sentence describing the image (no question needed)

```
Input:   chest_xray.jpg
Output:  "Frontal chest radiograph showing cardiomegaly with bilateral
          pleural effusion."

Training data: (image, caption) pairs
Dataset:       MS COCO Captions (118K images, 5 captions each)
```

**How it works:**
```
Image → Visual Encoder → visual embedding
                              │
                              ▼
                        Decoder LLM:
                        START → "Frontal" → "chest" → "radiograph" → ...
                        (autoregressive: each word depends on previous words + image)
```

**Resource:** [Show and Tell — original captioning paper](https://arxiv.org/abs/1411.4555)

---

### 4.3 Visual Question Answering (VQA)

> **VQA:** Image + Question → Answer (our main task)

```
Input:   chest_xray.jpg + "What abnormality is shown?"
Output:  "Cardiomegaly"

Training data: (image, question, answer) triples
Dataset:       VQA v2 (265K images, 1.1M questions)
               VQA-Med-2019 (our actual dataset — medical domain)
```

**Two types of answers:**

| Type | Example | Evaluation |
|---|---|---|
| Closed-ended (classification) | "Brain" / "Axial" / "MRI" | Accuracy (exact match) |
| Open-ended (generation) | "Cardiomegaly with bilateral effusion" | BLEU score |

**VQA-Med-2019 is mostly closed-ended** → that's why accuracy works well.

**Resource:** [VQA: Visual Question Answering — original paper](https://arxiv.org/abs/1505.00468)

---

## PHASE 5 — Datasets
### 🎯 Goal: Know what data we train and evaluate on

---

### 5.1 MS COCO Captions *(mentioned in topic PDF)*

- **Size:** 118,287 training images, each with 5 human-written captions
- **Domain:** General (everyday objects, scenes)
- **Use:** Pretrain image captioning models before medical fine-tuning
- **Format:** `image_id → [caption1, caption2, caption3, caption4, caption5]`

---

### 5.2 VQA v2 *(mentioned in topic PDF)*

- **Size:** 265,016 images, ~1.1M questions, ~11M answers
- **Domain:** General (everyday images from MS COCO)
- **Use:** Pretrain VQA models on general visual reasoning
- **Design:** Each question has 2 complementary images to reduce language bias

---

### 5.3 VQA-Med-2019 *(our actual training dataset)*

- **Size:** 3,200 train / 500 val / 500 test images
- **Domain:** Radiology (MedPix / NLM)
- **Format:** `(image, question) → answer`
- **Categories:**

| Category | Example Question | Answer Type |
|---|---|---|
| Modality | "What imaging modality is this?" | MRI / CT / X-Ray / Ultrasound |
| Plane | "In what plane is this image?" | Axial / Sagittal / Coronal |
| Organ System | "What organ is principally shown?" | Brain / Lung / Heart / Liver |
| Abnormality | "What is most alarming?" | Free text (harder!) |

- **Download:** https://zenodo.org/records/10499039
- **License:** CC BY 4.0

---

## PHASE 6 — Our Model: LLaVA + QLoRA
### 🎯 Goal: Understand exactly what model we use and why

---

### 6.1 CLIP — Connecting Vision and Language

> OpenAI's model that puts images and text in the **same vector space**.
> We use CLIP as the frozen vision encoder inside LLaVA.

**Training idea:** Given 400M (image, caption) pairs from the internet:
- Push matching pairs **closer** in vector space
- Push non-matching pairs **further apart**

```
After CLIP training:

"chest X-Ray" text     → [0.2, 0.8, 0.1, ...]
 chest_xray.jpg image  → [0.2, 0.7, 0.1, ...]  ← very similar!

"brain MRI" text       → [0.7, 0.1, 0.9, ...]
 chest_xray.jpg image  → [0.2, 0.8, 0.1, ...]  ← very different!
```

**Why this matters for VQA:** Because CLIP gives us visual embeddings that
already "understand" medical terminology.

**Resource:** [CLIP paper](https://arxiv.org/abs/2103.00020) + [OpenAI blog](https://openai.com/research/clip)

---

### 6.2 Large Language Models (LLMs)

> **Mistral-7B** is the LLM backbone inside LLaVA-1.5.

**Key concepts you must understand:**

| Concept | Explanation |
|---|---|
| **Autoregressive generation** | Model generates one token at a time. Each token depends on all previous tokens |
| **Context window** | Max number of tokens the model can process at once (Mistral: 32K tokens) |
| **Prefill phase** | Process all input tokens at once (image + question) |
| **Decode phase** | Generate output tokens one by one (the answer) |
| **Temperature** | Controls randomness in generation (0 = deterministic) |

```
Prefill:  [image_tokens × 256] + ["What", "organ", "is", "shown", "?"]
           └─────────────────────────────────────────────────────────┘
                           Process all at once → slow first step

Decode:   → "Brain" (token 1)
          → "." (token 2)
          → <EOS> (stop token)
          └─── Generate one token per step → cumulative
```

**This is why we benchmark Prefill and Decode separately** in `benchmark_latency.py`!

**Resource:** [Andrej Karpathy — Intro to LLMs](https://www.youtube.com/watch?v=zjkBMFhNj_g) (1h)

---

### 6.3 LLaVA — The Exact Model We Use

> Combines CLIP ViT + MLP + Mistral-7B into one multi-modal model.

```
Full LLaVA Pipeline:

chest_xray.jpg
    │
    ▼  CLIP ViT-L/14 @ 336px (FROZEN — not trained)
256 visual embeddings (each 1024-dim)
    │
    ▼  MLP Projection (2 linear layers, TRAINABLE)
256 language-space embeddings (each 4096-dim)
    │
    ├───────────────────────────────────┐
    │                                   │
    │   Question tokens                 │
    │   ["What", "organ", "?"]          │
    │       ↓ Tokenize + Embed          │
    │   3 text embeddings (4096-dim)    │
    │                                   │
    └─── Concatenate ───────────────────┘
                    │
    [visual_token×256 | text_token×3]  = 259 total tokens
                    │
                    ▼  Mistral-7B (QLoRA fine-tuned)
                "Brain"
```

**Resource:**
- [LLaVA paper](https://arxiv.org/abs/2304.08485)
- [LLaVA-1.5 paper](https://arxiv.org/abs/2310.03744)
- [LLaVA GitHub](https://github.com/haotian-liu/LLaVA)

---

### 6.4 LoRA — Low-Rank Adaptation

> **Why:** We can't update all 7 billion weights of Mistral-7B — too much memory.
> **Solution:** Add small trainable matrices on top of frozen weights.

```
Original:  y = W · x         W is (4096 × 4096) = 16.7M parameters

LoRA:      y = W · x + (B · A) · x
                         A is (4096 × 16) = 65K parameters
                         B is (16 × 4096) = 65K parameters
                         rank r = 16

Only A and B are trained → 99% fewer parameters updated!
```

We apply LoRA to: `q_proj`, `v_proj`, `k_proj`, `o_proj` (attention matrices)

**Resource:** [LoRA paper](https://arxiv.org/abs/2106.09685)

---

### 6.5 QLoRA — Quantized LoRA

> **Why:** Even with LoRA, loading the 7B model in FP16 needs ~14GB VRAM (too much for T4).
> **Solution:** Quantize the frozen model to 4-bit integers → 4× memory reduction.

```
FP16 (normal):  each weight = 16 bits = 2 bytes
INT4 (quantized): each weight = 4 bits = 0.5 bytes
                                          ↑ 4× smaller!

Memory: LLaVA-7B FP16 = 14 GB  →  QLoRA INT4 ≈ 8 GB ✅ fits on T4
```

**Resource:** [QLoRA paper](https://arxiv.org/abs/2305.14314) + [HuggingFace 4-bit guide](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

---

## PHASE 7 — Medical Domain Basics
### 🎯 Goal: Recognize and name the 4 types of imaging in VQA-Med-2019

| Modality | Physics | Looks like | VQA-Med examples |
|---|---|---|---|
| **X-Ray** | Radiation through body | Black/white, bones bright | "chest X-ray", "bilateral infiltrates" |
| **CT Scan** | X-ray from multiple angles | Grayscale slices, detailed structure | "axial CT", "lung nodule" |
| **MRI** | Magnetic field + radio waves | Soft tissue contrast, brain detail | "T1 weighted", "sagittal MRI" |
| **Ultrasound** | Sound waves | Noisy/grainy, real-time | "fetal ultrasound", "gallbladder" |

**Planes (how the 3D body is sliced):**

```
        Coronal (front-back)
             ┃
    ─────────╋─────────  Axial (top-bottom horizontal)
             ┃
        Sagittal (left-right)
```

**Resource:** [Radiopaedia — Imaging modalities](https://radiopaedia.org/articles/imaging-modalities)

---

## PHASE 8 — Evaluation Metrics
### 🎯 Goal: Know how to measure if the model is doing well

---

### 8.1 BLEU Score

> Measures word overlap between predicted answer and ground truth.

```
Reference:   "lung"
Prediction:  "lung"        → BLEU-1 = 1.0  ✅ perfect
Prediction:  "left lung"   → BLEU-1 = 0.5  (1 of 2 words match)
Prediction:  "chest"       → BLEU-1 = 0.0  ❌ no overlap
```

**Why BLEU is suitable here:** VQA-Med-2019 answers are short (1-3 words).
For short answers, BLEU ≈ Accuracy anyway.

---

### 8.2 Accuracy (Exact Match)

> Strictest metric — prediction must match reference exactly.

```python
accuracy = (prediction.strip().lower() == reference.strip().lower())
# "Brain" == "brain"  → True  ✅
# "brain" == "Brain"  → True  ✅ (after lowercase)
# "the brain" == "brain" → False ❌
```

---

### 8.3 Latency Metrics (P50 / P95 / P99)

> Why percentiles matter more than averages for clinical tools.

```
100 requests, all took 1.0s — except 1 request that took 30s

Average = (99 × 1.0 + 30) / 100 = 1.29s  ← looks OK
P99     = 30s                              ← 1 in 100 doctors waits 30s!
```

| Percentile | Meaning |
|---|---|
| P50 (median) | Half of requests are faster than this |
| P95 | 95% of requests are faster than this |
| P99 | 99% of requests are faster than this |

**For a medical tool, P99 < 5s is our target.**

---

## ⏱️ Suggested Weekly Schedule

| Week | Phase | Daily time | Goal |
|---|---|---|---|
| Week 1 | Phase 1 | 1.5h/day | Write a working PyTorch training loop |
| Week 2 | Phase 2 | 1.5h/day | Understand CNN + ViT, extract image features |
| Week 3 | Phase 3 | 1.5h/day | Understand Transformer + implement attention from scratch |
| Week 4 | Phase 4 + 5 | 1.5h/day | Understand VQA task, load VQA-Med-2019 |
| Week 5 | Phase 6 | 2h/day | Understand LLaVA + QLoRA, run Colab notebook |
| Week 6 | Phase 7 + 8 | 1h/day | Medical imaging terms, run evaluation script |

---

## ✅ Self-Check Questions

After finishing the roadmap, you should be able to answer all of these:

**Phase 1–2:**
- [ ] What does a convolutional filter detect?
- [ ] How does ViT split an image into tokens?
- [ ] What is a visual embedding? Why is it useful?

**Phase 3:**
- [ ] What is "attention" doing mathematically?
- [ ] What is the difference between an encoder and a decoder Transformer?

**Phase 4:**
- [ ] What is the difference between Captioning and VQA?
- [ ] What does the fusion module do in the topic PDF pipeline?

**Phase 5:**
- [ ] What are the 4 question categories in VQA-Med-2019?
- [ ] How many training images does VQA-Med-2019 have?

**Phase 6:**
- [ ] How does LLaVA connect vision and language?
- [ ] Why do we use QLoRA instead of full fine-tuning?
- [ ] What is the difference between Prefill and Decode phases?

**Phase 7–8:**
- [ ] What is the difference between X-Ray and MRI?
- [ ] What is BLEU-1 measuring?
- [ ] Why do we report P99 latency instead of just average?
