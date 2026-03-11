"""
aify-llamacpp — Entrypoint

1. Downloads the model from HuggingFace if not cached
2. Launches llama-server (official llama.cpp server) as a subprocess
3. Starts a lightweight FastAPI proxy for /health, /ready, /info, /v1/models
   The llama-server handles /v1/chat/completions, /v1/completions, /v1/embeddings natively
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("entrypoint")


def load_model_config(model_name: str, config_dir: str) -> dict:
    config_path = Path(config_dir) / "models" / f"{model_name}.json"
    if not config_path.exists():
        available = [p.stem for p in (Path(config_dir) / "models").glob("*.json")]
        logger.error(f"Model config not found: {config_path}. Available: {available}")
        sys.exit(1)
    with open(config_path) as f:
        return json.load(f)


def download_model(repo: str, filename: str, model_dir: str, hf_token: str = "") -> str:
    model_path = Path(model_dir) / filename
    if model_path.exists():
        logger.info(f"Model cached: {model_path}")
        return str(model_path)

    logger.info(f"Downloading {filename} from {repo}...")
    os.makedirs(model_dir, exist_ok=True)

    from huggingface_hub import hf_hub_download
    kwargs = {"repo_id": repo, "filename": filename, "local_dir": model_dir}
    if hf_token:
        kwargs["token"] = hf_token

    path = hf_hub_download(**kwargs)
    logger.info(f"Downloaded: {path}")
    return str(path)


def build_server_args(model_path: str, model_config: dict) -> list:
    """Build llama-server command-line arguments."""
    port = os.getenv("SERVICE_PORT", "8080")
    gpu_layers = int(os.getenv("GPU_LAYERS", "-1"))
    ctx = int(os.getenv("CONTEXT_LENGTH", "0"))
    n_batch = int(os.getenv("N_BATCH", "512"))
    n_threads = int(os.getenv("N_THREADS", "0"))
    flash_attn = os.getenv("FLASH_ATTENTION", "true").lower() == "true"

    if ctx == 0:
        ctx = model_config.get("context_length", 4096)

    args = [
        "llama-server",
        "--model", model_path,
        "--port", port,
        "--host", "0.0.0.0",
        "--ctx-size", str(ctx),
        "--batch-size", str(n_batch),
        "--n-gpu-layers", str(gpu_layers),
    ]

    if n_threads > 0:
        args.extend(["--threads", str(n_threads)])

    if flash_attn:
        args.extend(["--flash-attn", "on"])

    # Embedding model: enable embedding endpoint
    if model_config.get("type") == "embedding":
        args.append("--embedding")

    # Extra args from model config (e.g. --reasoning-budget, --no-mmap, --jinja, --cache-type-k)
    extra = model_config.get("extra_args", [])
    if extra:
        # Filter out args already set via env vars to avoid duplicates
        skip_next = False
        for i, arg in enumerate(extra):
            if skip_next:
                skip_next = False
                continue
            # Skip --flash-attn (handled above), --n-gpu-layers (handled above)
            if arg in ("--flash-attn", "-fa", "--n-gpu-layers", "-ngl"):
                skip_next = True  # skip the value too
                continue
            args.append(arg)

    return args


def main():
    model_name = os.getenv("MODEL_NAME", "")
    if not model_name:
        logger.error("MODEL_NAME not set")
        sys.exit(1)

    config_dir = os.getenv("CONFIG_DIR", "/app/config")
    model_dir = os.getenv("MODEL_DIR", "/data/models")
    hf_token = os.getenv("HF_TOKEN", "")

    # 1. Load config
    model_config = load_model_config(model_name, config_dir)
    logger.info(f"Model: {model_name} — {model_config.get('description', '')}")

    # 2. Download model
    model_path = download_model(
        model_config["repo"],
        model_config["filename"],
        model_dir,
        hf_token,
    )

    # 3. Build and launch llama-server
    args = build_server_args(model_path, model_config)
    logger.info(f"Starting: {' '.join(args)}")

    # Write model info for the management sidecar
    info_path = Path("/tmp/model_info.json")
    info_path.write_text(json.dumps({
        "name": model_name,
        "type": model_config.get("type", "chat"),
        "description": model_config.get("description", ""),
        "context_length": model_config.get("context_length", 4096),
        "embedding_dims": model_config.get("embedding_dims"),
        "model_path": model_path,
    }))

    # Forward signals to child process
    process = subprocess.Popen(args)

    def handle_signal(sig, frame):
        process.send_signal(sig)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Wait for process to exit
    sys.exit(process.wait())


if __name__ == "__main__":
    main()
