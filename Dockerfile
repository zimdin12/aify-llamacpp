FROM python:3.12-slim

# Build deps for llama-cpp-python compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash service && \
    mkdir -p /app /data/models && \
    chown -R service:service /app /data

WORKDIR /app

# Python dependencies (no llama-cpp-python yet — installed separately for build arg control)
COPY service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# llama-cpp-python: override CMAKE_ARGS for GPU support
# CPU: docker build .
# CUDA: docker build --build-arg CMAKE_ARGS="-DGGML_CUDA=on" .
ARG CMAKE_ARGS=""
ARG FORCE_CMAKE=0
RUN CMAKE_ARGS="${CMAKE_ARGS}" FORCE_CMAKE="${FORCE_CMAKE}" \
    pip install --no-cache-dir llama-cpp-python>=0.3.0

COPY service/ ./service/
COPY config/ ./config/

VOLUME /data

EXPOSE 8080

USER service

# Long start period — first run downloads model
HEALTHCHECK --interval=15s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "-m", "uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8080"]
