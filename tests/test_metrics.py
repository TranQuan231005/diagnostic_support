"""
Unit Tests for Medical VQA Evaluation Metrics Module.
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import (
    normalize_answer,
    compute_exact_match,
    compute_bleu_1,
    compute_bleu_2,
    compute_rouge_l,
    compute_all_metrics,
    evaluate_dataset,
)


class TestMetrics(unittest.TestCase):

    def test_normalize_answer(self):
        self.assertEqual(normalize_answer("  AXIAL.  "), "axial")
        self.assertEqual(normalize_answer("CT Scan:"), "ct scan")
        self.assertEqual(normalize_answer('"X-Ray"'), "x-ray")
        self.assertEqual(normalize_answer("t1 - weighted mri..."), "t1 - weighted mri")
        self.assertEqual(normalize_answer(None), "")

    def test_exact_match(self):
        # Exact match after normalization
        self.assertEqual(compute_exact_match("Axial", "axial."), 1.0)
        self.assertEqual(compute_exact_match("cta - ct angiography", "cta - ct angiography"), 1.0)
        # Mismatch
        self.assertEqual(compute_exact_match("Brain", "Liver"), 0.0)
        self.assertEqual(compute_exact_match("no", "yes"), 0.0)

    def test_bleu_1_unigram(self):
        # Perfect match
        self.assertAlmostEqual(compute_bleu_1("lung", "lung"), 1.0, places=2)
        # Partial match
        # Reference has 2 words, prediction matches 1 of 2
        score = compute_bleu_1("ct scan", "ct angiography")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)
        # Completely disjoint
        self.assertEqual(compute_bleu_1("mri", "ultrasound"), 0.0)

    def test_bleu_2_bigram(self):
        # Perfect compound match
        self.assertAlmostEqual(compute_bleu_2("ct angiography", "ct angiography"), 1.0, places=2)
        # Reversed words should score lower or zero for bigram
        self.assertLess(compute_bleu_2("angiography ct", "ct angiography"), 1.0)

    def test_rouge_l_sequence(self):
        # Exact match
        self.assertAlmostEqual(compute_rouge_l("glioblastoma multiforme", "glioblastoma multiforme"), 1.0, places=2)
        # Partial serial sequence
        score = compute_rouge_l("acute subdural hematoma", "subdural hematoma")
        self.assertGreater(score, 0.6)
        self.assertLessEqual(score, 1.0)

    def test_evaluate_dataset_breakdown(self):
        mock_records = [
            {"prediction": "MRI", "ground_truth": "mri", "category": "Modality"},
            {"prediction": "CT", "ground_truth": "x-ray", "category": "Modality"},
            {"prediction": "Axial", "ground_truth": "axial", "category": "Plane"},
            {"prediction": "Sagittal", "ground_truth": "sagittal.", "category": "Plane"},
            {"prediction": "Brain", "ground_truth": "brain", "category": "Organ System"},
            {"prediction": "Meningioma", "ground_truth": "glioblastoma", "category": "Abnormality"},
        ]

        results = evaluate_dataset(mock_records)
        self.assertEqual(results["total_samples"], 6)

        # Modality: 1 of 2 exact match -> 0.5
        self.assertEqual(results["by_category"]["Modality"]["exact_match"], 0.5)
        # Plane: 2 of 2 exact match -> 1.0
        self.assertEqual(results["by_category"]["Plane"]["exact_match"], 1.0)
        # Organ System: 1 of 1 -> 1.0
        self.assertEqual(results["by_category"]["Organ System"]["exact_match"], 1.0)
        # Abnormality: 0 of 1 -> 0.0
        self.assertEqual(results["by_category"]["Abnormality"]["exact_match"], 0.0)

        # Overall Exact Match: 4 of 6 -> 0.6667
        self.assertAlmostEqual(results["overall"]["exact_match"], 0.6667, places=3)


if __name__ == "__main__":
    unittest.main()
