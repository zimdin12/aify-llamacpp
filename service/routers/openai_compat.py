"""
OpenAI-compatible API endpoints for llamacpp-agentified.

Provides /v1/chat/completions, /v1/completions, /v1/embeddings, /v1/models.
"""

import json
import time
import uuid
import logging
from typing import List, Optional, Union

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai"])


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class CompletionRequest(BaseModel):
    model: str = ""
    prompt: Union[str, List[str]] = ""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = 256
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None


class EmbeddingRequest(BaseModel):
    model: str = ""
    input: Union[str, List[str]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_model(request: Request):
    loader = getattr(request.app.state, "loader", None)
    if not loader or not loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return loader


def _make_id():
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------

@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    loader = _get_model(request)
    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    kwargs = {
        "messages": messages,
        "temperature": body.temperature,
        "top_p": body.top_p,
        "frequency_penalty": body.frequency_penalty,
        "presence_penalty": body.presence_penalty,
    }
    if body.max_tokens is not None:
        kwargs["max_tokens"] = body.max_tokens
    if body.stop is not None:
        kwargs["stop"] = body.stop

    model_name = loader.config.model_name

    if body.stream:
        kwargs["stream"] = True

        def stream_generator():
            req_id = _make_id()
            for chunk in loader.model.create_chat_completion(**kwargs):
                # llama-cpp-python returns dicts in streaming mode
                if isinstance(chunk, dict):
                    chunk.setdefault("id", req_id)
                    chunk.setdefault("model", model_name)
                    yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # Non-streaming
    result = loader.model.create_chat_completion(**kwargs)
    if isinstance(result, dict):
        result.setdefault("id", _make_id())
        result.setdefault("model", model_name)
    return result


# ---------------------------------------------------------------------------
# /v1/completions
# ---------------------------------------------------------------------------

@router.post("/completions")
async def completions(body: CompletionRequest, request: Request):
    loader = _get_model(request)

    prompt = body.prompt if isinstance(body.prompt, str) else "\n".join(body.prompt)

    kwargs = {
        "prompt": prompt,
        "temperature": body.temperature,
        "top_p": body.top_p,
        "max_tokens": body.max_tokens,
    }
    if body.stop is not None:
        kwargs["stop"] = body.stop

    model_name = loader.config.model_name

    if body.stream:
        kwargs["stream"] = True

        def stream_generator():
            req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
            for chunk in loader.model.create_completion(**kwargs):
                if isinstance(chunk, dict):
                    chunk.setdefault("id", req_id)
                    chunk.setdefault("model", model_name)
                    yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    result = loader.model.create_completion(**kwargs)
    if isinstance(result, dict):
        result.setdefault("id", f"cmpl-{uuid.uuid4().hex[:12]}")
        result.setdefault("model", model_name)
    return result


# ---------------------------------------------------------------------------
# /v1/embeddings
# ---------------------------------------------------------------------------

@router.post("/embeddings")
async def embeddings(body: EmbeddingRequest, request: Request):
    loader = _get_model(request)

    if not loader.is_embedding:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{loader.config.model_name}' is type '{loader.model_config.get('type')}', not 'embedding'"
        )

    inputs = body.input if isinstance(body.input, list) else [body.input]

    data = []
    total_tokens = 0
    for i, text in enumerate(inputs):
        embedding = loader.model.embed(text)
        # llama-cpp-python embed() returns list or list of lists
        if embedding and isinstance(embedding[0], list):
            embedding = embedding[0]
        data.append({
            "object": "embedding",
            "embedding": embedding,
            "index": i,
        })
        total_tokens += len(text.split())  # rough estimate

    return {
        "object": "list",
        "data": data,
        "model": loader.config.model_name,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    }


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

@router.get("/models")
async def list_models(request: Request):
    loader = getattr(request.app.state, "loader", None)
    models = []
    if loader and loader.model_config:
        models.append({
            "id": loader.config.model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "llamacpp-agentified",
            "permission": [],
            "root": loader.config.model_name,
            "parent": None,
        })
    return {"object": "list", "data": models}
