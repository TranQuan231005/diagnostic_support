# Walkthrough - Day 1 Deliverables (D1-T2 & D1-T5)

## 📌 Accomplished Tasks

### 1. D1-T2: Data Preprocessing Pipeline
- Implemented [`src/data/preprocess.py`](file:///d:/diagnostic_support/src/data/preprocess.py).
- Converted raw VQA-Med-2019 dataset to standard LLaVA instruction JSON format:
  - [`data/processed/train.json`](file:///d:/diagnostic_support/data/processed/train.json) (12,792 items)
  - [`data/processed/val.json`](file:///d:/diagnostic_support/data/processed/val.json) (2,000 items)
  - 4 Category validation splits: `val_modality.json`, `val_plane.json`, `val_organ.json`, `val_abnormality.json` (500 items each).
  - [`data/processed/eda_summary.json`](file:///d:/diagnostic_support/data/processed/eda_summary.json).

---

### 2. D1-T5: Standalone Evaluation & Metrics Engine
- Implemented [`src/evaluation/metrics.py`](file:///d:/diagnostic_support/src/evaluation/metrics.py) with:
  - **Exact Match (Accuracy):** Normalized comparison (case-insensitive, whitespace/punctuation stripped).
  - **BLEU-1:** Unigram precision with brevity penalty (official VQA-Med benchmark).
  - **BLEU-2:** Bigram serial sequence overlap for 2-word medical compound terms.
  - **ROUGE-L:** Longest Common Subsequence F1 score for versatile sequence alignment.
  - **Category Breakdown:** Aggregation across Modality, Plane, Organ System, and Abnormality.
- Implemented [`src/evaluation/evaluate.py`](file:///d:/diagnostic_support/src/evaluation/evaluate.py):
  - CLI runner to score model predictions against `val.json`.
  - Generates:
    - [`results/evaluation_summary.json`](file:///d:/diagnostic_support/results/evaluation_summary.json)
    - [`results/evaluation_report.md`](file:///d:/diagnostic_support/results/evaluation_report.md)
    - [`results/per_sample_predictions.csv`](file:///d:/diagnostic_support/results/per_sample_predictions.csv)
- Created and executed unit test suite [`tests/test_metrics.py`](file:///d:/diagnostic_support/tests/test_metrics.py): 6/6 tests passing (100% pass rate).

---

## 🚦 Status & Progress Snapshot
- Sprint progress updated to **17%** (3 of 18 tasks completed).
- [`status.md`](file:///d:/diagnostic_support/status.md) reflects current sprint state.
