# Walkthrough — Task D2-T6: Standalone CLI Inference Tool

## 📌 Accomplished Tasks

### 1. D2-T6: Core CLI Predictor Implementation
Implemented [`src/predict.py`](file:///d:/diagnostic_support/src/predict.py) containing:
- **`MedicalVQAPredictor`**: Unified inference engine capable of running:
  1. **Zero-Shot Base LLaVA-1.5-7B**: Evaluates out-of-the-box performance.
  2. **Fine-Tuned QLoRA Adapters**: Dynamically snaps on trained LoRA adapters (`--adapter-path`).
- **Clinical Question Categorizer**:
  - Uses regex word boundary patterns (`\bct\b`, `\bxr\b`) to reliably classify questions into Modality, Plane, Organ System, and Abnormality without false substring collisions (e.g. avoiding false matches in "depicted").
- **Deterministic Clinical Generation**:
  - Implements greedy decoding (`temperature=0.0`) by default for reproducible, hallucination-resistant clinical diagnosis.
- **Latency & Output Cleaning**:
  - Measures request latency in seconds.
  - Automatically strips conversation prefixes (`USER:`, `ASSISTANT:`, `<image>`) and special tokens.

---

### 2. Comprehensive Test Suite
Created [`tests/test_predict.py`](file:///d:/diagnostic_support/tests/test_predict.py) with 6 unit tests covering:
- Automatic question category inference across all 4 clinical categories.
- Simulated responses for all categories.
- Predictor execution using in-memory PIL images.
- Predictor execution using disk image file paths.
- Error handling when image files do not exist (`FileNotFoundError`).
- Dynamic adapter path metadata tracking.

---

## 🚦 Verification Results

### 1. Unit Tests
```powershell
python -m unittest tests/test_predict.py -v
# Output: Ran 6 tests in 0.173s - OK (100% Pass Rate)
```

### 2. Full Regression Suite (100% Pass Rate)
```powershell
python -m unittest discover tests -v
# Output: Ran 24 tests in 3.111s - OK across all 24 tests in the repository!
```

### 3. CLI Command Execution
```powershell
python src/predict.py --mock --question "What organ is principally shown in this chest scan?"
```
Terminal Output:
```text
=================================================================
🩺 Medical VQA Diagnostic Inference
Model: Base LLaVA-1.5-7B (Zero-Shot)
Target Image: temp_synthetic_scan.jpg
=================================================================
❓ Question:         What organ is principally shown in this chest scan?
📂 Category:         Organ System
🤖 Prediction:       lung
⏱️ Latency:          0.063 seconds
=================================================================
```

---

## 📈 Sprint Progress Snapshot (Halfway Mark!)

Updated [`status.md`](file:///d:/diagnostic_support/status.md):
- **Completed Tasks:** **9 / 18** (**50% Complete — Halfway Milestone**).
- **Day 1 Tasks:** 6/6 (100% Complete).
- **Day 2 Tasks:** 3/6 (D2-T1, D2-T4, D2-T6 Complete; D2-T2, D2-T3 next).
- **Deliverables:** **6 / 8 Key Artifacts Completed** (Processed Data, Training Notebook, Evaluation Engine, Latency Profiler, CLI Inference Tool, Gradio UI Shell).
