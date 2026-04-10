FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create data directory
RUN mkdir -p /data/chromadb

# Default environment variables
ENV CONTAINER_HOST=0.0.0.0
ENV CONTAINER_PORT=8080
ENV LOG_LEVEL=INFO
ENV CHROMADB_PERSIST_DIR=/data/chromadb
ENV SQLITE_DB_PATH=/data/agent_assist.db

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health')" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
