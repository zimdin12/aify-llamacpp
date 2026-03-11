"""
aify-llamacpp — Main FastAPI Application

Self-contained LLM inference container. Loads a single model on startup,
exposes OpenAI-compatible API.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from service.config import get_config
from service.model_loader import ModelLoader

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )

    logger.info(f"Starting aify-llamacpp, model={config.model_name}")

    # Load model
    loader = ModelLoader(config)
    try:
        loader.load()
        app.state.loader = loader
        app.state.config = config
        logger.info(f"Ready: {loader.model_info}")
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        app.state.loader = loader  # still store so health endpoint can report
        app.state.config = config

    yield

    # Shutdown
    loader.unload()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="aify-llamacpp",
        version="1.0.0",
        description="Self-contained LLM inference via llama.cpp",
        lifespan=lifespan,
        docs_url="/docs",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from service.routers import health, openai_compat
    app.include_router(health.router)
    app.include_router(openai_compat.router)

    return app


app = create_app()
