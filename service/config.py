"""
llamacpp-agentified — Configuration
"""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    model_name: str = ""
    model_dir: str = "/data/models"
    config_dir: str = "/app/config"
    port: int = 8080
    gpu_layers: int = -1
    gpu_fraction: float = 1.0
    context_length: int = 0  # 0 = use model config default
    n_batch: int = 512
    n_threads: int = 0  # 0 = auto
    flash_attention: bool = True
    hf_token: str = ""


def get_config() -> Config:
    return Config(
        model_name=os.getenv("MODEL_NAME", ""),
        model_dir=os.getenv("MODEL_DIR", "/data/models"),
        config_dir=os.getenv("CONFIG_DIR", "/app/config"),
        port=int(os.getenv("SERVICE_PORT", "8080")),
        gpu_layers=int(os.getenv("GPU_LAYERS", "-1")),
        gpu_fraction=float(os.getenv("GPU_FRACTION", "1.0")),
        context_length=int(os.getenv("CONTEXT_LENGTH", "0")),
        n_batch=int(os.getenv("N_BATCH", "512")),
        n_threads=int(os.getenv("N_THREADS", "0")),
        flash_attention=os.getenv("FLASH_ATTENTION", "true").lower() == "true",
        hf_token=os.getenv("HF_TOKEN", ""),
    )
