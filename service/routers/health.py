"""Health endpoints for llamacpp-agentified."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request):
    loader = getattr(request.app.state, "loader", None)
    if loader and loader.is_loaded:
        return {"status": "healthy"}
    return JSONResponse({"status": "loading"}, status_code=503)


@router.get("/ready")
async def ready(request: Request):
    loader = getattr(request.app.state, "loader", None)
    if not loader:
        return {"status": "not_initialized"}
    return {
        "status": "ready" if loader.is_loaded else "loading",
        "model": loader.model_info,
    }


@router.get("/info")
async def info(request: Request):
    loader = getattr(request.app.state, "loader", None)
    config = getattr(request.app.state, "config", None)
    host = request.headers.get("host", "localhost:8080")
    base = f"http://{host}"

    return {
        "name": "llamacpp-agentified",
        "version": "1.0.0",
        "model": loader.model_info if loader else None,
        "endpoints": {
            "chat": f"{base}/v1/chat/completions",
            "completions": f"{base}/v1/completions",
            "embeddings": f"{base}/v1/embeddings",
            "models": f"{base}/v1/models",
            "health": f"{base}/health",
        },
        "config": {
            "gpu_layers": config.gpu_layers if config else None,
            "context_length": config.context_length if config else None,
            "flash_attention": config.flash_attention if config else None,
        },
    }
