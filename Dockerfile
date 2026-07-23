FROM python:3.11-slim

WORKDIR /srv

# System deps kept minimal; numpy/redis wheels don't need a compiler here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY redis_index/ ./redis_index/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/srv

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
