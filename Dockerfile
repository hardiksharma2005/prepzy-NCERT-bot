FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 MODEL_CACHE_DIR=/app/models HF_HUB_DISABLE_SYMLINKS_WARNING=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend backend
COPY data data
# Bake the embedding model into the image so cold starts don't download it.
RUN python -c "from backend.app.embeddings import embed_one; embed_one('warm up')"

EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
