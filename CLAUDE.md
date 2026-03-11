# aify-llamacpp

Self-contained LLM inference container. One model per container, selected via `MODEL_NAME` env var.

## Project Structure

- `service/main.py` — FastAPI app, loads model on startup
- `service/model_loader.py` — Downloads GGUF from HuggingFace, loads via llama-cpp-python
- `service/config.py` — Env-based config (MODEL_NAME, GPU_LAYERS, CONTEXT_LENGTH, etc.)
- `service/routers/openai_compat.py` — /v1/chat/completions, /v1/completions, /v1/embeddings
- `service/routers/health.py` — /health, /ready, /info
- `config/models/*.json` — Model catalog (repo, filename, context_length, type, chat_format)

## Adding Models

Create `config/models/<name>.json` with `repo`, `filename`, `context_length`, `gpu_layers`, `chat_format`, `type` ("chat" or "embedding"), `embedding_dims`.

## Key Patterns

- Models auto-download from HuggingFace on first start, cached in /data/models volume
- GPU: build with `CMAKE_ARGS=-DGGML_CUDA=on`
- Embedding models: set `"type": "embedding"` in config, use /v1/embeddings endpoint
- Streaming: SSE via `"stream": true` in request body
