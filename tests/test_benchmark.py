"""Unit tests for the 8-Stage Latency Benchmark Profiler."""

import json
import os
import shutil
import tempfile
import unittest
import numpy as np

from src.benchmark_latency import (
    CLINICAL_P99_THRESHOLD_SEC,
    STAGE_NAMES,
    BenchmarkReport,
    MedicalVQALatencyProfiler,
    StageMetrics,
    compute_percentiles,
)


class TestBenchmarkLatency(unittest.TestCase):
    """Test suite for latency benchmark statistics, stage recording, and reporting."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_compute_percentiles_empty(self):
        """Verify empty latency list produces safe zero-filled dictionary."""
        stats = compute_percentiles([])
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["mean"], 0.0)
        self.assertEqual(stats["p50"], 0.0)
        self.assertEqual(stats["p95"], 0.0)
        self.assertEqual(stats["p99"], 0.0)

    def test_compute_percentiles_known_distribution(self):
        """Verify percentiles calculation against a known sequence of 100 values."""
        values = list(range(1, 101))  # 1 to 100
        stats = compute_percentiles(values)

        self.assertEqual(stats["count"], 100)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 100.0)
        self.assertAlmostEqual(stats["mean"], 50.5, places=1)
        self.assertAlmostEqual(stats["p50"], 50.5, places=1)
        self.assertGreater(stats["p95"], 90.0)
        self.assertGreater(stats["p99"], 98.0)

    def test_profiler_single_pass_mock(self):
        """Verify mock profiler executes single pass and records all 8 stages."""
        profiler = MedicalVQALatencyProfiler(mock_mode=True, device="cpu")
        stage_times, tokens_gen = profiler.profile_single_pass("dummy.jpg", max_new_tokens=8)

        self.assertEqual(tokens_gen, 8)
        self.assertEqual(len(stage_times), 8)

        for name in STAGE_NAMES:
            self.assertIn(name, stage_times)
            self.assertGreater(stage_times[name], 0.0)

        # Stage 8 (End-to-End) should be roughly comparable to or greater than token decode
        self.assertGreaterEqual(stage_times[STAGE_NAMES[7]], stage_times[STAGE_NAMES[5]])

    def test_run_benchmark_mock_workflow(self):
        """Verify full benchmark execution, report aggregation, and Markdown formatting."""
        profiler = MedicalVQALatencyProfiler(mock_mode=True, device="cpu")
        report = profiler.run_benchmark(
            image_or_path=None,
            question="What is the organ?",
            num_runs=3,
            warmup_runs=1,
            max_new_tokens=4,
        )

        self.assertIsInstance(report, BenchmarkReport)
        self.assertEqual(report.num_runs, 3)
        self.assertEqual(report.warmup_runs, 1)
        self.assertIsInstance(report.clinical_p99_compliant, bool)
        self.assertEqual(len(report.stages), 8)

        for name in STAGE_NAMES:
            stage_metric = report.stages[name]
            self.assertIsInstance(stage_metric, StageMetrics)
            self.assertEqual(stage_metric.count, 3)
            self.assertGreater(stage_metric.p50_ms, 0.0)
            self.assertGreater(stage_metric.p99_ms, 0.0)

        # Verify Markdown table formatting
        table_md = profiler.format_markdown_table(report)
        self.assertIn("8-Stage Medical VQA Latency Benchmark Report", table_md)
        self.assertIn("Stage 1: Image Preprocess", table_md)
        self.assertIn("Stage 5: LLM Prefill (TTFT)", table_md)
        self.assertIn("Stage 8: End-to-End Total", table_md)
        self.assertIn("P50 (ms)", table_md)
        self.assertIn("P99 (ms)", table_md)

    def test_save_report_json(self):
        """Verify JSON export of benchmark results."""
        profiler = MedicalVQALatencyProfiler(mock_mode=True, device="cpu")
        report = profiler.run_benchmark(num_runs=2, warmup_runs=1, max_new_tokens=2)

        json_path = os.path.join(self.test_dir, "latency_report.json")
        profiler.save_report_json(report, json_path)

        self.assertTrue(os.path.isfile(json_path))
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("stages", data)
        self.assertIn("clinical_p99_compliant", data)
        self.assertEqual(len(data["stages"]), 8)
        self.assertIn("Stage 5: LLM Prefill (TTFT)", data["stages"])
        self.assertIn("p50_ms", data["stages"]["Stage 5: LLM Prefill (TTFT)"])


if __name__ == "__main__":
    unittest.main()
