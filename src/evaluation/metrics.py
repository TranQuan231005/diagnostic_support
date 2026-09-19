"""
Evaluation Metrics for Medical Diagnostic Visual Question Answering (VQA).

Includes:
- Exact Match (Normalized Accuracy)
- BLEU-1 (Unigram Word Overlap - Official VQA-Med Benchmark)
- BLEU-2 (Bigram Serial Sequence Overlap)
- ROUGE-L (Longest Common Subsequence F1)
- Category-Wise Benchmark Aggregation (Modality, Plane, Organ System, Abnormality)
"""

import re
import math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional


def normalize_answer(text: Optional[str]) -> str:
    """
    Standard text normalization for clinical VQA evaluation:
    1. Lowercase text.
    2. Replace multiple whitespaces/newlines with single space.
    3. Strip leading/trailing punctuation (.,!?:;"'), while keeping internal hyphens/slashes.
    """
    if text is None:
        return ""
    text = str(text).lower().strip()
    # Strip common markdown / quotes
    text = text.replace("`", "").replace('"', "").replace("'", "")
    # Remove punctuation from start/end
    text = re.sub(r"^[^\w\s]+|[^\w\s]+$", "", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Tokenizes normalized text into word tokens."""
    norm = normalize_answer(text)
    if not norm:
        return []
    # Split by whitespace, punctuation delimiters except internal hyphens
    tokens = re.findall(r"\b[\w-]+\b", norm)
    return tokens if tokens else norm.split()


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Computes binary Exact Match score (1.0 or 0.0) after normalization.
    """
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    if not norm_pred and not norm_gt:
        return 1.0
    return 1.0 if norm_pred == norm_gt else 0.0


def compute_n_gram_precision(
    pred_tokens: List[str],
    gt_tokens: List[str],
    n: int,
    smooth: bool = True
) -> float:
    """
    Calculates modified n-gram precision with Laplace-style / epsilon smoothing.
    """
    if len(pred_tokens) < n or len(gt_tokens) < n:
        if smooth:
            return 0.0
        return 0.0

    pred_ngrams = [tuple(pred_tokens[i:i + n]) for i in range(len(pred_tokens) - n + 1)]
    gt_ngrams = [tuple(gt_tokens[i:i + n]) for i in range(len(gt_tokens) - n + 1)]

    pred_counts = Counter(pred_ngrams)
    gt_counts = Counter(gt_ngrams)

    clipped_matches = 0
    for ngram, count in pred_counts.items():
        clipped_matches += min(count, gt_counts.get(ngram, 0))

    total_pred = len(pred_ngrams)
    if total_pred == 0:
        return 0.0

    if clipped_matches == 0 and smooth:
        # Small epsilon smoothing for zero-overlap in higher-order ngrams
        return 0.01 / total_pred

    return clipped_matches / total_pred


def compute_brevity_penalty(pred_len: int, gt_len: int) -> float:
    """Computes standard BLEU brevity penalty."""
    if pred_len == 0:
        return 0.0
    if pred_len >= gt_len:
        return 1.0
    return math.exp(1.0 - (gt_len / pred_len))


def compute_bleu_1(prediction: str, ground_truth: str) -> float:
    """
    Computes BLEU-1 score (Unigram precision with brevity penalty).
    Official evaluation metric for VQA-Med benchmark.
    """
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    p1 = compute_n_gram_precision(pred_tokens, gt_tokens, n=1, smooth=False)
    bp = compute_brevity_penalty(len(pred_tokens), len(gt_tokens))
    return round(p1 * bp, 4)


def compute_bleu_2(prediction: str, ground_truth: str) -> float:
    """
    Computes BLEU-2 score (Geometric mean of unigram and bigram precision).
    Evaluates 2-word serial sequence consistency.
    """
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    p1 = compute_n_gram_precision(pred_tokens, gt_tokens, n=1, smooth=True)
    p2 = compute_n_gram_precision(pred_tokens, gt_tokens, n=2, smooth=True)

    if p1 <= 0.0:
        return 0.0

    # Geometric mean: sqrt(p1 * p2) if p2 > 0 else p1 * 0.5
    if p2 > 0:
        geom_prec = math.sqrt(p1 * p2)
    else:
        geom_prec = p1 * 0.5

    bp = compute_brevity_penalty(len(pred_tokens), len(gt_tokens))
    return round(geom_prec * bp, 4)


def compute_rouge_l(prediction: str, ground_truth: str) -> float:
    """
    Computes ROUGE-L F1 score based on the Longest Common Subsequence (LCS).
    Naturally handles serial word ordering across both short and multi-word answers.
    """
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    m, n = len(pred_tokens), len(gt_tokens)
    # Dynamic programming table for LCS
    lcs_table = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gt_tokens[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])

    lcs_len = lcs_table[m][n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)


def compute_all_metrics(prediction: str, ground_truth: str) -> Dict[str, float]:
    """
    Calculates all metrics for a single prediction vs ground truth pair.
    """
    return {
        "exact_match": compute_exact_match(prediction, ground_truth),
        "bleu_1": compute_bleu_1(prediction, ground_truth),
        "bleu_2": compute_bleu_2(prediction, ground_truth),
        "rouge_l": compute_rouge_l(prediction, ground_truth),
    }


def evaluate_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates a full dataset of predictions.
    Each record must have:
      - 'prediction': str
      - 'ground_truth': str
      - 'category': Optional[str] (e.g. 'Modality', 'Plane', 'Organ System', 'Abnormality')

    Returns structured results with Overall metrics and Category Breakdowns.
    """
    if not records:
        return {
            "total_samples": 0,
            "overall": {"exact_match": 0.0, "bleu_1": 0.0, "bleu_2": 0.0, "rouge_l": 0.0},
            "by_category": {}
        }

    total_samples = len(records)
    metric_accumulators = defaultdict(float)
    cat_records = defaultdict(list)

    for rec in records:
        pred = rec.get("prediction", "")
        gt = rec.get("ground_truth", "")
        cat = rec.get("category", "General")

        m = compute_all_metrics(pred, gt)
        for k, v in m.items():
            metric_accumulators[k] += v

        cat_records[cat].append(m)

    # Compute Overall Averages
    overall_summary = {
        k: round(v / total_samples, 4)
        for k, v in metric_accumulators.items()
    }

    # Compute Category Breakdowns
    category_summary = {}
    for cat, metrics_list in cat_records.items():
        cat_count = len(metrics_list)
        cat_agg = defaultdict(float)
        for m in metrics_list:
            for k, v in m.items():
                cat_agg[k] += v

        category_summary[cat] = {
            "samples": cat_count,
            **{k: round(v / cat_count, 4) for k, v in cat_agg.items()}
        }

    return {
        "total_samples": total_samples,
        "overall": overall_summary,
        "by_category": category_summary
    }
