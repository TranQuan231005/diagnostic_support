"""Unit tests for LLaVA model loader, quantization configurations, and PEFT setups."""

import unittest
import torch
import torch.nn as nn

from src.model.load_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_LORA_TARGET_MODULES,
    get_quantization_config,
    get_lora_config,
    print_trainable_parameters,
    is_cuda_available,
)


class DummyLinearModel(nn.Module):
    """Simple linear module for testing trainable parameter computations."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

        # Freeze fc1 to simulate base model freezing
        for param in self.fc1.parameters():
            param.requires_grad = False


class TestModelLoad(unittest.TestCase):
    """Test suite for quantization and LoRA configuration builders."""

    def test_default_constants(self):
        """Verify default model ID and target module definitions."""
        self.assertEqual(DEFAULT_MODEL_ID, "llava-hf/llava-1.5-7b-hf")
        self.assertIn("q_proj", DEFAULT_LORA_TARGET_MODULES)
        self.assertIn("v_proj", DEFAULT_LORA_TARGET_MODULES)
        self.assertIn("k_proj", DEFAULT_LORA_TARGET_MODULES)
        self.assertIn("o_proj", DEFAULT_LORA_TARGET_MODULES)
        self.assertIn("gate_proj", DEFAULT_LORA_TARGET_MODULES)
        self.assertIn("up_proj", DEFAULT_LORA_TARGET_MODULES)
        self.assertIn("down_proj", DEFAULT_LORA_TARGET_MODULES)
        self.assertEqual(len(DEFAULT_LORA_TARGET_MODULES), 7)

    def test_quantization_config_builder(self):
        """Verify BitsAndBytesConfig properties when load_in_4bit is True."""
        config = get_quantization_config(
            load_in_4bit=True,
            quant_type="nf4",
            use_double_quant=True,
            compute_dtype=torch.float16,
        )
        self.assertIsNotNone(config)
        self.assertTrue(config.load_in_4bit)
        self.assertEqual(config.bnb_4bit_quant_type, "nf4")
        self.assertTrue(config.bnb_4bit_use_double_quant)
        self.assertEqual(config.bnb_4bit_compute_dtype, torch.float16)

    def test_quantization_config_disabled(self):
        """Verify get_quantization_config returns None when load_in_4bit is False."""
        config = get_quantization_config(load_in_4bit=False)
        self.assertIsNone(config)

    def test_lora_config_builder_default(self):
        """Verify default LoRA hyperparameters."""
        lora_config = get_lora_config()
        self.assertEqual(lora_config.r, 16)
        self.assertEqual(lora_config.lora_alpha, 32)
        self.assertEqual(lora_config.lora_dropout, 0.05)
        self.assertEqual(lora_config.bias, "none")
        self.assertEqual(lora_config.task_type, "CAUSAL_LM")
        self.assertEqual(set(lora_config.target_modules), set(DEFAULT_LORA_TARGET_MODULES))

    def test_lora_config_builder_custom(self):
        """Verify custom rank and target modules."""
        custom_targets = ["q_proj", "v_proj"]
        lora_config = get_lora_config(
            r=8,
            lora_alpha=16,
            target_modules=custom_targets,
            lora_dropout=0.1,
        )
        self.assertEqual(lora_config.r, 8)
        self.assertEqual(lora_config.lora_alpha, 16)
        self.assertEqual(lora_config.lora_dropout, 0.1)
        self.assertEqual(lora_config.target_modules, custom_targets)

    def test_print_trainable_parameters(self):
        """Verify parameter counting logic."""
        dummy_model = DummyLinearModel()
        stats = print_trainable_parameters(dummy_model)

        # fc1: 10*20 weights + 20 biases = 220 (frozen)
        # fc2: 20*5 weights + 5 biases = 105 (trainable)
        # Total: 325
        self.assertEqual(stats["all_params"], 325)
        self.assertEqual(stats["trainable_params"], 105)
        expected_pct = 100.0 * 105 / 325
        self.assertAlmostEqual(stats["trainable_percent"], expected_pct, places=4)

    def test_is_cuda_available_returns_bool(self):
        """Verify is_cuda_available returns a boolean value."""
        res = is_cuda_available()
        self.assertIsInstance(res, bool)


if __name__ == "__main__":
    unittest.main()
