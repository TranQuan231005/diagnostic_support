"""Model loading, 4-bit quantization, and LoRA configuration package for Medical VQA."""

from src.model.load_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_LORA_TARGET_MODULES,
    get_quantization_config,
    get_lora_config,
    load_llava_model,
    get_qlora_model,
    print_trainable_parameters,
    is_cuda_available,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_LORA_TARGET_MODULES",
    "get_quantization_config",
    "get_lora_config",
    "load_llava_model",
    "get_qlora_model",
    "print_trainable_parameters",
    "is_cuda_available",
]
