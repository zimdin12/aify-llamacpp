# ==============================================================================
# llamacpp-agentified — Self-contained LLM inference
# ==============================================================================
# Uses official llama.cpp server for inference (latest model support)
# + Python for model download and management

# Stage 1: Get llama-server + libs from official image
FROM ghcr.io/ggml-org/llama.cpp:server AS llama-server

# Stage 2: Build the service
FROM python:3.12-slim

# Copy llama-server binary and all shared libraries (keep in /app like original)
COPY --from=llama-server /app/ /opt/llama/

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Make libs discoverable
RUN echo "/opt/llama" > /etc/ld.so.conf.d/llama.conf && ldconfig
# Symlink binary to PATH
RUN ln -s /opt/llama/llama-server /usr/local/bin/llama-server

RUN useradd -m -s /bin/bash service && \
    mkdir -p /app /data/models && \
    chown -R service:service /app /data

WORKDIR /app

COPY service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY service/ ./service/
COPY config/ ./config/

VOLUME /data

EXPOSE 8080

USER service

# Long start period — first run downloads model
HEALTHCHECK --interval=15s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "service/entrypoint.py"]
