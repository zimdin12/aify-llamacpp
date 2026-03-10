# llamacpp-agentified

Self-contained LLM inference container powered by [llama.cpp](https://github.com/ggerganov/llama.cpp). Uses the official `llama-server` binary (multi-stage Docker build) for maximum model compatibility. Each container instance loads a single GGUF model and exposes an OpenAI-compatible API.

Designed to be spawned by [llamacpp-router-agentified](https://github.com/zimdin12/llamacpp-router-agentified) as a sub-container, but also works standalone.

## Quick Start

```bash
cp .env.example .env
# Edit .env — set MODEL_NAME to any config/models/*.json name

docker compose up -d --build

# Test
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-4b","messages":[{"role":"user","content":"Hello"}]}'
```

First startup downloads the model from HuggingFace — this may take a few minutes depending on model size and connection speed. Subsequent starts use the cached model from the data volume.

## Available Models

| Config Name | Model | Size | Type | Context |
|---|---|---|---|---|
| `qwen3.5-0.8b` | Qwen3.5 0.8B Q4_K_M | ~0.5 GB | chat | 8K |
| `qwen3-0.6b` | Qwen3 0.6B Q8_0 | ~0.7 GB | chat | 8K |
| `qwen3-1.7b` | Qwen3 1.7B Q4_K_M | ~1.1 GB | chat | 32K |
| `qwen3-4b` | Qwen3 4B Q4_K_M | ~2.5 GB | chat | 32K |
| `qwen3-8b` | Qwen3 8B Q4_K_M | ~4.9 GB | chat | 32K |
| `qwen3-embedding-0.6b` | Qwen3 Embedding 0.6B F16 | ~1.2 GB | embedding | 8K |
| `mistral-7b` | Mistral 7B Instruct v0.2 Q4_K_M | ~4.4 GB | chat | 32K |
| `phi-4-mini` | Phi-4 Mini Instruct Q4_K_M | ~2.4 GB | chat | 16K |

### Adding Custom Models

Create a JSON file in `config/models/`:

```json
{
  "repo": "TheBloke/SomeModel-GGUF",
  "filename": "some-model.Q4_K_M.gguf",
  "context_length": 8192,
  "gpu_layers": -1,
  "chat_format": "chatml",
  "description": "My custom model",
  "type": "chat",
  "embedding_dims": null
}
```

- `repo` — HuggingFace repo ID
- `filename` — GGUF file within the repo
- `type` — `"chat"` or `"embedding"`
- `chat_format` — chat template hint (usually `null` — llama-server auto-detects from GGUF metadata)
- `embedding_dims` — set for embedding models, `null` for chat models

Then set `MODEL_NAME=your-config-name` in `.env`.

## API Endpoints

### OpenAI-Compatible

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completion (streaming supported) |
| `/v1/completions` | POST | Text completion (streaming supported) |
| `/v1/embeddings` | POST | Text embeddings (embedding models only) |
| `/v1/models` | GET | List loaded model |

### Service

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check (200 if model loaded, 503 if loading) |
| `/ready` | GET | Readiness with model info |
| `/info` | GET | Full service discovery |
| `/docs` | GET | Swagger UI |

## GPU Support

The Docker image includes **both CPU and CUDA GPU backends** — no separate build needed. GPU is auto-detected at runtime; if no GPU is available it falls back to CPU automatically.

To enable GPU access, run with `--gpus all` (or use the router which passes GPUs automatically):

```bash
docker compose up -d --build
# GPU is detected automatically if available
```

**Benchmark (Qwen3.5 0.8B Q4_K_M):**

| | CPU | GPU (RTX 3050) |
|---|---|---|
| Prompt processing | 73 tok/s | 113 tok/s |
| Generation | 23 tok/s | 83 tok/s |

### GPU Configuration

| Env Var | Default | Description |
|---|---|---|
| `GPU_LAYERS` | `-1` | Number of layers offloaded to GPU (`-1` = all, `0` = CPU only) |
| `GPU_FRACTION` | `1.0` | GPU memory fraction (for sharing GPU between instances) |

## Configuration

All configuration is via environment variables (`.env` file):

| Env Var | Default | Description |
|---|---|---|
| `MODEL_NAME` | *(required)* | Model config name (without `.json`) |
| `MODEL_DIR` | `/data/models` | Model file cache directory |
| `SERVICE_PORT` | `8080` | API port |
| `CONTEXT_LENGTH` | `0` | Override context length (`0` = use model config) |
| `N_BATCH` | `512` | Batch size for prompt processing |
| `N_THREADS` | `0` | CPU thread count (`0` = auto) |
| `FLASH_ATTENTION` | `true` | Enable flash attention |
| `HF_TOKEN` | *(empty)* | HuggingFace token for gated models |

## Pre-downloading Models

To download a model without starting the server:

```bash
bash scripts/download_model.sh qwen3-4b
```

## Project Structure

```
llamacpp-agentified/
├── config/models/           # Model configs (repo, filename, params)
│   ├── qwen3-4b.json
│   ├── qwen3-8b.json
│   └── ...
├── service/
│   ├── entrypoint.py        # Downloads model, launches llama-server
│   ├── config.py            # Environment-based configuration
│   ├── main.py              # FastAPI app (alternative approach)
│   └── routers/
│       ├── health.py        # /health, /ready, /info
│       └── openai_compat.py # /v1/* OpenAI-compatible API
├── scripts/
│   └── download_model.sh    # Pre-download helper
├── Dockerfile               # Multi-stage: llama-server (CUDA) + Python 3.12
├── docker-compose.yml
└── .env.example
```

## Related Projects

- **[llamacpp-router-agentified](https://github.com/zimdin12/llamacpp-router-agentified)** — Ollama-like router that manages multiple llamacpp-agentified containers
- **[openmemory-agentified](https://github.com/zimdin12/openmemory-agentified)** — Hybrid memory system that can use this as its LLM backend
- **[agentify-container](https://github.com/zimdin12/agentify-container)** — The base template these projects build on
