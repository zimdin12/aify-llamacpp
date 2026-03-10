# ==============================================================================
# llamacpp-agentified — Self-contained LLM inference
# ==============================================================================
# Uses official llama.cpp server for inference (latest model support)
# + Python for model download and management
# CUDA image includes both CPU and GPU backends — auto-detects at runtime

# Stage 1: Get llama-server + libs from official CUDA image (supports CPU+GPU)
FROM ghcr.io/ggml-org/llama.cpp:server-cuda AS llama-server

# Stage 2: Build the service on CUDA base for runtime libs
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

# Install Python 3.12 and essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    curl libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# Copy llama-server binary and all shared libraries
COPY --from=llama-server /app/ /opt/llama/

# Make libs discoverable
RUN echo "/opt/llama" > /etc/ld.so.conf.d/llama.conf && ldconfig
# Symlink binary to PATH
RUN ln -s /opt/llama/llama-server /usr/local/bin/llama-server

RUN useradd -m -s /bin/bash service && \
    mkdir -p /app /data/models && \
    chown -R service:service /app /data

WORKDIR /app

COPY service/requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY service/ ./service/
COPY config/ ./config/

VOLUME /data

EXPOSE 8080

USER service

# Long start period — first run downloads model
HEALTHCHECK --interval=15s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "service/entrypoint.py"]
