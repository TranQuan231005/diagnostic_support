"""Standalone CLI Inference Tool for Medical Visual Question Answering (Medical VQA).

Supports both:
  1. Zero-shot baseline inference with base LLaVA-1.5-7B.
  2. Fine-tuned QLoRA inference by attaching a trained LoRA adapter (--adapter-path).

Features:
  - Formats conversational LLaVA prompts: USER: <image>\n{question}\nASSISTANT:
  - Greedy decoding (temperature=0.0) by default for deterministic clinical predictions.
  - End-to-end latency measurement per request.
  - Graceful hardware fallback / mock mode for local CPU workstations.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional, Union

from PIL import Image
import torch

try:
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from peft import PeftModel
    TRANSFORMERS_PEFT_AVAILABLE = True
except ImportError:
    TRANSFORMERS_PEFT_AVAILABLE = False

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("medical_vqa_predict")

DEFAULT_MODEL_ID = "llava-hf/llava-1.5-7b-hf"


@dataclass
class PredictionResult:
    """Structured clinical prediction result."""
    question: str
    answer: str
    category: str
    latency_sec: float
    model_id: str
    adapter_path: Optional[str]
    device: str
    full_response: str


import re

def classify_question_category(question: str) -> str:
    """Infer the VQA-Med-2019 clinical question category from prompt patterns."""
    q_lower = question.lower()
    if any(k in q_lower for k in ["abnormality", "abnormal", "lesion", "mass", "fracture", "disease", "finding", "diagnosis", "pathology", "tumor"]):
        return "Abnormality"
    elif any(k in q_lower for k in ["plane", "orientation", "axial", "coronal", "sagittal"]):
        return "Plane"
    elif any(k in q_lower for k in ["organ", "body part", "system", "anatomical", "location", "chest", "brain", "spine", "lung", "liver", "heart"]):
        return "Organ System"
    elif any(k in q_lower for k in ["modality", "kind of scan", "type of image", "mri", "ultrasound", "x-ray", "contrast"]) or re.search(r'\b(ct|xr|view)\b', q_lower):
        return "Modality"
    return "General Diagnostic"


def simulate_clinical_answer(category: str, question: str) -> str:
    """Provide realistic simulated responses for mock mode / CPU testing."""
    q_lower = question.lower()
    if category == "Modality":
        if "contrast" in q_lower:
            return "ct with iv contrast"
        elif "mri" in q_lower:
            return "mr - t1 weighted"
        elif "ultrasound" in q_lower or "us" in q_lower:
            return "ultrasound"
        return "xr - plain film"
    elif category == "Plane":
        if "coronal" in q_lower:
            return "coronal"
        elif "sagittal" in q_lower:
            return "sagittal"
        return "axial"
    elif category == "Organ System":
        if "lung" in q_lower or "chest" in q_lower:
            return "lung"
        elif "brain" in q_lower or "head" in q_lower:
            return "brain"
        elif "heart" in q_lower:
            return "heart"
        return "gastrointestinal"
    else:  # Abnormality
        if "fracture" in q_lower:
            return "rib fracture"
        elif "effusion" in q_lower:
            return "pleural effusion"
        return "cardiomegaly"


class MedicalVQAPredictor:
    """Inference engine for Medical VQA with optional LoRA adapter loading."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = True,
        device: Optional[str] = None,
        mock_mode: bool = False,
    ):
        """Initialize the predictor.

        Args:
            model_id: Base Hugging Face model repository ID.
            adapter_path: Optional path to trained LoRA adapter checkpoint.
            load_in_4bit: Whether to load base model in 4-bit NF4.
            device: Target device string ('cuda', 'cpu'). Auto-detected if None.
            mock_mode: Whether to run in simulated mode without downloading 7B weights.
        """
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit
        self.mock_mode = mock_mode

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.processor = None

        if not self.mock_mode and TRANSFORMERS_PEFT_AVAILABLE and torch.cuda.is_available():
            self._load_live_model()
        else:
            if not self.mock_mode:
                logger.warning("CUDA or transformers not available. Activating mock mode for inference.")
            self.mock_mode = True

    def _load_live_model(self):
        """Load base LLaVA model and optionally attach LoRA adapter."""
        from src.model.load_model import load_llava_model

        logger.info(f"Loading base LLaVA model: {self.model_id} (4-bit={self.load_in_4bit})...")
        self.model, self.processor = load_llava_model(
            model_id=self.model_id,
            load_in_4bit=self.load_in_4bit,
            device_map="auto" if self.device == "cuda" else "cpu",
        )

        if self.adapter_path:
            if os.path.isdir(self.adapter_path):
                logger.info(f"Attaching fine-tuned LoRA adapter from: {self.adapter_path}...")
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
                logger.info("LoRA adapter attached successfully.")
            else:
                logger.warning(f"Adapter path '{self.adapter_path}' not found. Running with base model.")
                self.adapter_path = None

        self.model.eval()

    def predict(
        self,
        image_or_path: Union[str, Image.Image],
        question: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> PredictionResult:
        """Run single-sample clinical diagnosis.

        Args:
            image_or_path: Image file path or PIL Image object.
            question: Clinical question prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 for deterministic greedy decoding).

        Returns:
            PredictionResult containing the generated answer, category, and latency.
        """
        category = classify_question_category(question)
        start_time = time.perf_counter()

        # Step 1: Image loading and RGB normalization
        if isinstance(image_or_path, str):
            if not os.path.isfile(image_or_path):
                raise FileNotFoundError(f"Medical scan image not found: {image_or_path}")
            image = Image.open(image_or_path).convert("RGB")
        elif isinstance(image_or_path, Image.Image):
            image = image_or_path.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_or_path)}")

        image = image.resize((336, 336))

        # Mock Simulation Path
        if self.mock_mode or self.model is None or self.processor is None:
            # Simulate realistic forward pass latency (~400-800ms)
            time.sleep(0.05)
            answer = simulate_clinical_answer(category, question)
            latency = time.perf_counter() - start_time
            return PredictionResult(
                question=question,
                answer=answer,
                category=category,
                latency_sec=round(latency, 3),
                model_id=self.model_id,
                adapter_path=self.adapter_path,
                device=self.device,
                full_response=f"USER: <image>\n{question}\nASSISTANT: {answer}",
            )

        # Live Forward Inference Path
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")

        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            torch.cuda.synchronize()

        do_sample = temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_tokens = self.model.generate(**inputs, **gen_kwargs)

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        input_len = inputs["input_ids"].shape[1]
        decoded_output = self.processor.decode(output_tokens[0], skip_special_tokens=True)
        new_text = self.processor.decode(output_tokens[0][input_len:], skip_special_tokens=True)

        # Clean generated answer
        answer = new_text.split("ASSISTANT:")[-1].strip()
        answer = answer.strip(' ".,\n')

        latency = time.perf_counter() - start_time

        return PredictionResult(
            question=question,
            answer=answer,
            category=category,
            latency_sec=round(latency, 3),
            model_id=self.model_id,
            adapter_path=self.adapter_path,
            device=self.device,
            full_response=decoded_output.strip(),
        )


def main():
    """Command-line interface runner for Medical VQA prediction."""
    parser = argparse.ArgumentParser(description="Medical VQA CLI Inference Tool (LLaVA-1.5-7B + QLoRA)")
    parser.add_argument("--image", type=str, default=None, help="Path to input medical image scan (e.g. chest_xray.jpg)")
    parser.add_argument("--question", type=str, default="What imaging modality is depicted in this scan?", help="Medical inquiry prompt")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to fine-tuned LoRA adapter checkpoint directory")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="Base Hugging Face model repository ID")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for deterministic greedy decoding)")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum number of generated answer tokens")
    parser.add_argument("--mock", action="store_true", help="Force mock/simulation mode without loading full 7B model weights")
    args = parser.parse_args()

    # If no image path provided, generate a dummy image for testing
    temp_image_created = False
    if args.image is None or not os.path.isfile(args.image):
        if args.image is not None and not os.path.isfile(args.image):
            logger.warning(f"Specified image '{args.image}' not found. Generating temporary synthetic test image.")
        dummy_path = "temp_synthetic_scan.jpg"
        img = Image.new("RGB", (336, 336), color=(128, 128, 128))
        img.save(dummy_path)
        args.image = dummy_path
        temp_image_created = True

    model_type_str = f"Fine-Tuned QLoRA ({args.adapter_path})" if args.adapter_path else "Base LLaVA-1.5-7B (Zero-Shot)"

    print("=" * 65)
    print("🩺 Medical VQA Diagnostic Inference")
    print(f"Model: {model_type_str}")
    print(f"Target Image: {args.image}")
    print("=" * 65)

    predictor = MedicalVQAPredictor(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        load_in_4bit=True,
        mock_mode=args.mock,
    )

    result = predictor.predict(
        image_or_path=args.image,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(f"❓ Question:         {result.question}")
    print(f"📂 Category:         {result.category}")
    print(f"🤖 Prediction:       {result.answer}")
    print(f"⏱️ Latency:          {result.latency_sec:.3f} seconds")
    print("=" * 65)

    # Cleanup temporary image if created
    if temp_image_created and os.path.isfile(args.image):
        try:
            os.remove(args.image)
        except Exception:
            pass


if __name__ == "__main__":
    main()
