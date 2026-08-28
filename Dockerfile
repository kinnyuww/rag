FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_HOST=0.0.0.0 \
    RAG_PORT=18080 \
    RAG_DATA_DIR=/app/data \
    RAG_UPLOAD_DIR=/app/data/uploads \
    RAG_MODEL_CACHE_DIR=/app/models \
    RAG_SQLITE_PATH=/app/data/rag.db \
    RAG_QDRANT_PATH=/app/data/qdrant

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock* README.md ./
RUN pip install --no-cache-dir uv \
    && uv sync --system --no-dev

COPY app ./app
COPY static ./static
COPY scripts ./scripts
COPY eval ./eval

RUN mkdir -p /app/data/uploads /app/data/qdrant /app/models

EXPOSE 18080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "18080"]
