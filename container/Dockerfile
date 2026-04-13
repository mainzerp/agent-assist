# ---- Builder stage: compile native wheels ----
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch BEFORE main requirements.
# sentence-transformers depends on torch; by pre-installing the CPU-only
# variant, pip will find torch already satisfied and skip the CUDA build.
# This eliminates ~4.3 GB of nvidia/cuda/triton libraries.
RUN pip install --no-cache-dir --prefix=/install \
    torch torchaudio torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Make pre-installed packages visible so pip skips torch on the next install
ENV PYTHONPATH=/install/lib/python3.12/site-packages

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Final stage: slim runtime ----
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/

# Create non-root user and data directory
RUN addgroup --system app && adduser --system --ingroup app app \
    && mkdir -p /data/chromadb /data/hf_cache \
    && chown -R app:app /data /app

# Entrypoint: fix permissions on mounted volumes then drop to non-root
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Install gosu for privilege drop
RUN apt-get update && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/*

# Default environment variables
ENV CONTAINER_HOST=0.0.0.0
ENV CONTAINER_PORT=8080
ENV LOG_LEVEL=INFO
ENV CHROMADB_PERSIST_DIR=/data/chromadb
ENV SQLITE_DB_PATH=/data/agent_assist.db
ENV HF_HOME=/data/hf_cache

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health')" || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--ws-ping-interval", "30", "--ws-ping-timeout", "10", "--timeout-graceful-shutdown", "30"]
