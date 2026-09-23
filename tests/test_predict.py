"""Unit tests for the Standalone CLI Inference Tool (src/predict.py)."""

import os
import shutil
import tempfile
import unittest
from PIL import Image

from src.predict import (
    DEFAULT_MODEL_ID,
    MedicalVQAPredictor,
    PredictionResult,
    classify_question_category,
    simulate_clinical_answer,
)


class TestPredict(unittest.TestCase):
    """Test suite for question categorization, predictor execution, and error handling."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.test_dir, "test_scan.jpg")
        img = Image.new("RGB", (336, 336), color=(100, 100, 100))
        img.save(self.test_image_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_classify_question_category(self):
        """Verify automatic question category inference."""
        self.assertEqual(classify_question_category("What imaging modality was used?"), "Modality")
        self.assertEqual(classify_question_category("What kind of scan is this ct?"), "Modality")
        self.assertEqual(classify_question_category("In what plane is this image taken?"), "Plane")
        self.assertEqual(classify_question_category("Is this an axial view?"), "Plane")
        self.assertEqual(classify_question_category("What organ is principally shown?"), "Organ System")
        self.assertEqual(classify_question_category("What anatomical system is depicted in the chest?"), "Organ System")
        self.assertEqual(classify_question_category("What abnormality is seen in the lung?"), "Abnormality")
        self.assertEqual(classify_question_category("Is there a fracture or lesion?"), "Abnormality")
        self.assertEqual(classify_question_category("Is the patient male or female?"), "General Diagnostic")

    def test_simulate_clinical_answer(self):
        """Verify simulated responses across all categories."""
        categories = ["Modality", "Plane", "Organ System", "Abnormality"]
        for cat in categories:
            ans = simulate_clinical_answer(cat, "general question")
            self.assertIsInstance(ans, str)
            self.assertGreater(len(ans), 0)

    def test_predictor_mock_with_pil_image(self):
        """Verify predictor execution using in-memory PIL Image."""
        predictor = MedicalVQAPredictor(mock_mode=True, device="cpu")
        img = Image.new("RGB", (336, 336), color=(128, 128, 128))

        result = predictor.predict(
            image_or_path=img,
            question="What organ is shown?",
        )

        self.assertIsInstance(result, PredictionResult)
        self.assertEqual(result.question, "What organ is shown?")
        self.assertEqual(result.category, "Organ System")
        self.assertIsInstance(result.answer, str)
        self.assertGreater(len(result.answer), 0)
        self.assertGreater(result.latency_sec, 0.0)
        self.assertNotIn("USER:", result.answer)
        self.assertNotIn("ASSISTANT:", result.answer)

    def test_predictor_mock_with_file_path(self):
        """Verify predictor execution using file path."""
        predictor = MedicalVQAPredictor(mock_mode=True, device="cpu")
        result = predictor.predict(
            image_or_path=self.test_image_path,
            question="What imaging modality was used to take this image?",
        )

        self.assertIsInstance(result, PredictionResult)
        self.assertEqual(result.category, "Modality")
        self.assertGreater(result.latency_sec, 0.0)

    def test_predictor_nonexistent_image_raises_error(self):
        """Verify FileNotFoundError is raised when image file does not exist."""
        predictor = MedicalVQAPredictor(mock_mode=True, device="cpu")
        non_existent_path = os.path.join(self.test_dir, "does_not_exist.jpg")

        with self.assertRaises(FileNotFoundError):
            predictor.predict(
                image_or_path=non_existent_path,
                question="What is this?",
            )

    def test_predictor_custom_adapter_path_tracking(self):
        """Verify adapter path is recorded in result metadata."""
        custom_adapter = os.path.join(self.test_dir, "fake_adapter")
        predictor = MedicalVQAPredictor(
            adapter_path=custom_adapter,
            mock_mode=True,
            device="cpu",
        )
        result = predictor.predict(
            image_or_path=self.test_image_path,
            question="What is the abnormality?",
        )

        self.assertEqual(result.adapter_path, custom_adapter)
        self.assertEqual(result.category, "Abnormality")


if __name__ == "__main__":
    unittest.main()
