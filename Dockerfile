FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEMO_MODE=true \
    API_PORT=8877 \
    API_BASE_URL=http://127.0.0.1:8877 \
    KNOWLEDGE_DIR=/app/demo_knowledge_base

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . ./

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/chat_uploads \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 10000
CMD ["bash", "/app/deploy/render-start.sh"]
