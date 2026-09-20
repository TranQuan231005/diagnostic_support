"""LLaVA-1.5-7B Model Loader with 4-bit NF4 Quantization and QLoRA Configuration."""

import logging
from typing import Dict, List, Optional, Tuple, Union
import torch

try:
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        LlavaForConditionalGeneration,
    )
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    TRANSFORMERS_PEFT_AVAILABLE = True
except ImportError:
    from dataclasses import dataclass, field

    @dataclass
    class FallbackBitsAndBytesConfig:
        load_in_4bit: bool = True
        bnb_4bit_quant_type: str = "nf4"
        bnb_4bit_use_double_quant: bool = True
        bnb_4bit_compute_dtype: torch.dtype = torch.float16

        def to_dict(self):
            return {
                "load_in_4bit": self.load_in_4bit,
                "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
                "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
                "bnb_4bit_compute_dtype": str(self.bnb_4bit_compute_dtype),
            }

    @dataclass
    class FallbackLoraConfig:
        r: int = 16
        lora_alpha: int = 32
        target_modules: List[str] = field(default_factory=list)
        lora_dropout: float = 0.05
        bias: str = "none"
        task_type: str = "CAUSAL_LM"

    BitsAndBytesConfig = FallbackBitsAndBytesConfig
    LoraConfig = FallbackLoraConfig
    TaskType = None
    TRANSFORMERS_PEFT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default model identifiers and hyperparameters
DEFAULT_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def is_cuda_available() -> bool:
    """Check if CUDA GPU is available for 4-bit quantization."""
    return torch.cuda.is_available()


def get_quantization_config(
    load_in_4bit: bool = True,
    quant_type: str = "nf4",
    use_double_quant: bool = True,
    compute_dtype: torch.dtype = torch.float16,
) -> Optional[Union["BitsAndBytesConfig", "FallbackBitsAndBytesConfig"]]:
    """Create a 4-bit BitsAndBytes quantization configuration for QLoRA fine-tuning.

    Args:
        load_in_4bit: Whether to enable 4-bit quantization.
        quant_type: Quantization datatype, 'nf4' (NormalFloat4) or 'fp4'.
        use_double_quant: Enables nested quantization to save ~0.37 bits/param.
        compute_dtype: Computation dtype for 4-bit base weights during forward pass.

    Returns:
        BitsAndBytesConfig instance or None if load_in_4bit is False.
    """
    if not load_in_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: Union[str, "TaskType"] = "CAUSAL_LM",
) -> Union["LoraConfig", "FallbackLoraConfig"]:
    """Construct LoRA / PEFT configuration for LLaVA-1.5-7B fine-tuning.

    Args:
        r: LoRA attention dimension / rank.
        lora_alpha: LoRA scaling alpha hyperparameter.
        target_modules: List of module names to apply LoRA to. Defaults to all linear projections.
        lora_dropout: Dropout probability for LoRA layers.
        bias: Bias training strategy ('none', 'all', 'lora_only').
        task_type: Task type for PEFT model.

    Returns:
        Configured LoraConfig instance.
    """
    if target_modules is None:
        target_modules = list(DEFAULT_LORA_TARGET_MODULES)

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
    )


def print_trainable_parameters(model: torch.nn.Module) -> Dict[str, Union[int, float]]:
    """Compute and print trainable vs total parameters and their ratio.

    Args:
        model: PyTorch model or PEFT model.

    Returns:
        Dictionary with 'trainable_params', 'all_params', and 'trainable_percent'.
    """
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_percent = 100.0 * trainable_params / all_params if all_params > 0 else 0.0
    logger.info(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_params:,} || "
        f"Trainable %: {trainable_percent:.4f}%"
    )

    return {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percent": trainable_percent,
    }


def load_llava_model(
    model_id: str = DEFAULT_MODEL_ID,
    load_in_4bit: bool = True,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = True,
) -> Tuple["LlavaForConditionalGeneration", "AutoProcessor"]:
    """Load LLaVA-1.5-7B base model and processor with 4-bit NF4 quantization.

    Args:
        model_id: HuggingFace model hub ID or local path.
        load_in_4bit: Whether to load weights in 4-bit NF4 quantization.
        device_map: Accelerate device mapping ('auto', 'cuda:0', 'cpu').
        torch_dtype: Base computation torch dtype.
        trust_remote_code: Whether to allow custom code execution from hub.

    Returns:
        Tuple of (model, processor).
    """
    if not TRANSFORMERS_PEFT_AVAILABLE:
        raise ImportError(
            "transformers and peft are required. Please run: pip install transformers peft bitsandbytes"
        )

    has_gpu = is_cuda_available()
    quant_config = None

    if load_in_4bit:
        if has_gpu:
            logger.info("CUDA detected. Initializing 4-bit NF4 BitsAndBytes configuration...")
            quant_config = get_quantization_config(
                load_in_4bit=True,
                quant_type="nf4",
                use_double_quant=True,
                compute_dtype=torch_dtype,
            )
        else:
            logger.warning(
                "CUDA is not available on this machine. 4-bit BitsAndBytes quantization requires NVIDIA GPU. "
                "Falling back to unquantized CPU loading (device_map='cpu')."
            )
            device_map = "cpu"
            torch_dtype = torch.float32

    logger.info(f"Loading processor for: {model_id}")
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )

    logger.info(f"Loading base model: {model_id} (quantization={'4-bit NF4' if quant_config else 'None'})...")
    model_kwargs = {
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        **model_kwargs,
    )

    logger.info("Base LLaVA model loaded successfully.")
    return model, processor


def get_qlora_model(
    model: torch.nn.Module,
    lora_config: Optional["LoraConfig"] = None,
    gradient_checkpointing: bool = True,
) -> torch.nn.Module:
    """Prepare 4-bit quantized model for k-bit training and attach LoRA adapter.

    Args:
        model: Base quantized LlavaForConditionalGeneration model.
        lora_config: Optional custom LoraConfig. Defaults to all linear layers.
        gradient_checkpointing: Whether to enable gradient checkpointing for VRAM savings.

    Returns:
        PEFT-wrapped QLoRA model ready for fine-tuning.
    """
    if not TRANSFORMERS_PEFT_AVAILABLE:
        raise ImportError("peft is required. Please run: pip install peft")

    if lora_config is None:
        lora_config = get_lora_config()

    logger.info("Preparing model for k-bit training (freezing base weights, float32 layernorms)...")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=gradient_checkpointing,
    )

    logger.info(f"Attaching LoRA adapter (r={lora_config.r}, alpha={lora_config.lora_alpha})...")
    peft_model = get_peft_model(model, lora_config)

    print_trainable_parameters(peft_model)
    return peft_model
