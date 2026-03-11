"""
aify-llamacpp — Model Loader

Reads model config JSON, downloads GGUF from HuggingFace if needed,
and loads via llama-cpp-python.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.model_config: dict = {}
        self._model_path: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def is_embedding(self) -> bool:
        return self.model_config.get("type") == "embedding"

    @property
    def model_info(self) -> dict:
        if not self.model_config:
            return {"status": "not_configured"}
        info = {
            "name": self.config.model_name,
            "type": self.model_config.get("type", "chat"),
            "description": self.model_config.get("description", ""),
            "context_length": self.model_config.get("context_length", 0),
            "loaded": self.is_loaded,
        }
        if self.is_embedding:
            info["embedding_dims"] = self.model_config.get("embedding_dims")
        return info

    def _read_model_config(self) -> dict:
        """Read config/models/{MODEL_NAME}.json"""
        config_path = Path(self.config.config_dir) / "models" / f"{self.config.model_name}.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Model config not found: {config_path}. "
                f"Available: {[p.stem for p in (Path(self.config.config_dir) / 'models').glob('*.json')]}"
            )
        with open(config_path) as f:
            return json.load(f)

    def _download_model(self, repo: str, filename: str) -> str:
        """Download GGUF from HuggingFace if not cached."""
        model_path = Path(self.config.model_dir) / filename

        if model_path.exists():
            logger.info(f"Model already cached: {model_path}")
            return str(model_path)

        logger.info(f"Downloading {filename} from {repo}...")
        os.makedirs(self.config.model_dir, exist_ok=True)

        from huggingface_hub import hf_hub_download

        kwargs = {"repo_id": repo, "filename": filename, "local_dir": self.config.model_dir}
        if self.config.hf_token:
            kwargs["token"] = self.config.hf_token

        path = hf_hub_download(**kwargs)
        logger.info(f"Downloaded to: {path}")
        return str(path)

    def load(self):
        """Read config, download if needed, load model."""
        if not self.config.model_name:
            raise ValueError("MODEL_NAME not set")

        self.model_config = self._read_model_config()
        logger.info(f"Loading model: {self.config.model_name} ({self.model_config.get('description', '')})")

        # Download
        repo = self.model_config["repo"]
        filename = self.model_config["filename"]
        self._model_path = self._download_model(repo, filename)

        # Resolve parameters (env overrides > model config > defaults)
        ctx = self.config.context_length if self.config.context_length > 0 else self.model_config.get("context_length", 4096)
        gpu_layers = self.config.gpu_layers  # -1 = all, from env or default
        n_batch = self.config.n_batch
        n_threads = self.config.n_threads if self.config.n_threads > 0 else None
        is_embedding = self.model_config.get("type") == "embedding"
        chat_format = self.model_config.get("chat_format")

        from llama_cpp import Llama

        kwargs = {
            "model_path": self._model_path,
            "n_ctx": ctx,
            "n_gpu_layers": gpu_layers,
            "n_batch": n_batch,
            "flash_attn": self.config.flash_attention,
            "verbose": False,
        }
        if n_threads:
            kwargs["n_threads"] = n_threads
        if is_embedding:
            kwargs["embedding"] = True
        if chat_format:
            kwargs["chat_format"] = chat_format

        logger.info(f"Llama params: ctx={ctx}, gpu_layers={gpu_layers}, batch={n_batch}, embedding={is_embedding}")
        self.model = Llama(**kwargs)
        logger.info(f"Model loaded: {self.config.model_name}")

    def unload(self):
        """Free model resources."""
        if self.model is not None:
            del self.model
            self.model = None
            logger.info("Model unloaded")
