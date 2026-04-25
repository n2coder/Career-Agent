FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV WEB_CONCURRENCY=1
ENV PORT=8001

WORKDIR /app

# Minimal OS packages for common Python wheels/builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    antiword \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel jaraco.context && pip install -r /app/requirements.txt

COPY . /app

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8001} --workers ${WEB_CONCURRENCY}"]
