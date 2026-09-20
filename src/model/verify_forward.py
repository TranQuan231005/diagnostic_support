"""CLI script to verify base LLaVA-1.5-7B 4-bit NF4 loading and forward pass execution."""

import argparse
import logging
import os
import sys
import time
from typing import Optional
from PIL import Image
import torch

from src.model.load_model import (
    DEFAULT_MODEL_ID,
    get_qlora_model,
    is_cuda_available,
    load_llava_model,
    print_trainable_parameters,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_forward")


def get_vram_usage_gb() -> Optional[float]:
    """Return current allocated CUDA VRAM in Gigabytes, or None if CUDA not available."""
    if not is_cuda_available():
        return None
    return torch.cuda.memory_allocated() / (1024 ** 3)


def get_max_vram_usage_gb() -> Optional[float]:
    """Return peak allocated CUDA VRAM in Gigabytes, or None if CUDA not available."""
    if not is_cuda_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def create_dummy_medical_image() -> Image.Image:
    """Create a synthetic 336x336 grayscale-like PIL image for testing."""
    img = Image.new("RGB", (336, 336), color=(128, 128, 128))
    return img


def run_forward_verification(
    model_id: str = DEFAULT_MODEL_ID,
    image_path: Optional[str] = None,
    question: str = "What imaging modality is depicted in this scan?",
    apply_qlora: bool = True,
    max_new_tokens: int = 32,
) -> dict:
    """Execute model loading, VRAM profiling, and single-sample forward pass verification.

    Args:
        model_id: HuggingFace model hub ID.
        image_path: Path to input image file, or None for dummy synthetic image.
        question: Medical inquiry prompt.
        apply_qlora: Whether to also test attaching the LoRA adapter.
        max_new_tokens: Max tokens to generate during forward test.

    Returns:
        Dictionary with verification results and metrics.
    """
    logger.info("=" * 65)
    logger.info("🩺 LLaVA-1.5-7B 4-bit NF4 Forward Pass Sanity Check")
    logger.info("=" * 65)

    cuda_present = is_cuda_available()
    device_desc = f"CUDA GPU ({torch.cuda.get_device_name(0)})" if cuda_present else "CPU (Fallback)"
    logger.info(f"Target Hardware: {device_desc}")

    initial_vram = get_vram_usage_gb()
    if initial_vram is not None:
        logger.info(f"Initial VRAM Allocated: {initial_vram:.2f} GB")

    # Step 1: Load Image
    if image_path and os.path.isfile(image_path):
        logger.info(f"Loading test image from: {image_path}")
        image = Image.open(image_path).convert("RGB")
    else:
        logger.info("No input image specified or file not found. Generating dummy synthetic 336x336 image.")
        image = create_dummy_medical_image()

    # Step 2: Load Model & Processor
    load_start = time.perf_counter()
    model, processor = load_llava_model(
        model_id=model_id,
        load_in_4bit=True,
        device_map="auto" if cuda_present else "cpu",
    )
    load_duration = time.perf_counter() - load_start
    logger.info(f"Base model loaded in {load_duration:.2f}s")

    post_load_vram = get_vram_usage_gb()
    if post_load_vram is not None:
        logger.info(f"Post-Load VRAM Allocated: {post_load_vram:.2f} GB")
        if post_load_vram <= 5.5:
            logger.info(f"✅ VRAM Target Achieved: {post_load_vram:.2f} GB <= 5.5 GB budget.")
        else:
            logger.warning(f"⚠️ VRAM ({post_load_vram:.2f} GB) exceeded 5.5 GB target threshold.")

    # Step 3: Test QLoRA Attachment (Optional)
    param_stats = {}
    if apply_qlora:
        logger.info("Attaching QLoRA Adapter for forward pass verification...")
        model = get_qlora_model(model)
        param_stats = print_trainable_parameters(model)

    # Step 4: Prepare Multimodal Inputs
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    logger.info(f"Formulated Prompt: {prompt!r}")

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    )

    if cuda_present:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Step 5: Forward Pass / Generation
    logger.info("Executing model forward pass & autoregressive generation...")
    model.eval()
    gen_start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    gen_duration = time.perf_counter() - gen_start

    # Step 6: Decode Output
    # Slice off input tokens to only get generated completion
    input_len = inputs["input_ids"].shape[1]
    generated_tokens = output_ids[:, input_len:]
    decoded_text = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()

    peak_vram = get_max_vram_usage_gb()
    logger.info("-" * 65)
    logger.info(f"Generated Output: {decoded_text!r}")
    logger.info(f"Inference Latency: {gen_duration:.3f}s ({max_new_tokens} tokens)")
    if peak_vram is not None:
        logger.info(f"Peak VRAM Usage: {peak_vram:.2f} GB")
    logger.info("=" * 65)
    logger.info("🎉 Forward pass verification completed successfully!")

    return {
        "status": "SUCCESS",
        "model_id": model_id,
        "device": device_desc,
        "load_time_sec": load_duration,
        "generation_time_sec": gen_duration,
        "post_load_vram_gb": post_load_vram,
        "peak_vram_gb": peak_vram,
        "generated_text": decoded_text,
        "trainable_param_stats": param_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify LLaVA-1.5-7B 4-bit forward pass and VRAM usage.")
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID or path (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to an image file (e.g. from VQA-Med-2019/Val_images/). If None, uses synthetic image.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What imaging modality is depicted in this scan?",
        help="Medical query question string.",
    )
    parser.add_argument(
        "--no_qlora",
        action="store_true",
        help="Skip attaching QLoRA adapter and only test base model forward pass.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Max new tokens to generate (default: 32).",
    )

    args = parser.parse_args()

    try:
        run_forward_verification(
            model_id=args.model_id,
            image_path=args.image_path,
            question=args.question,
            apply_qlora=not args.no_qlora,
            max_new_tokens=args.max_new_tokens,
        )
    except Exception as e:
        logger.error(f"Forward verification failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
