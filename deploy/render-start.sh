#!/usr/bin/env bash
set -euo pipefail

PUBLIC_PORT="${PORT:-10000}"
API_PORT_VALUE="${API_PORT:-8877}"
API_PID=""
FRONTEND_PID=""

cleanup() {
  if [[ -n "${FRONTEND_PID}" ]] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    kill "${FRONTEND_PID}" 2>/dev/null || true
  fi
  if [[ -n "${API_PID}" ]] && kill -0 "${API_PID}" 2>/dev/null; then
    kill "${API_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[Beauty RAG] starting API on 127.0.0.1:${API_PORT_VALUE}"
python -m uvicorn main:app --host 127.0.0.1 --port "${API_PORT_VALUE}" &
API_PID=$!

echo "[Beauty RAG] waiting for the API readiness check"
ready=0
for _ in $(seq 1 180); do
  if ! kill -0 "${API_PID}" 2>/dev/null; then
    wait "${API_PID}" || true
    echo "[Beauty RAG] API exited before becoming ready" >&2
    exit 1
  fi
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${API_PORT_VALUE}/health', timeout=2)" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done

if [[ "${ready}" -ne 1 ]]; then
  echo "[Beauty RAG] API did not become ready within 6 minutes" >&2
  exit 1
fi

echo "[Beauty RAG] starting Streamlit on 0.0.0.0:${PUBLIC_PORT}"
python -m streamlit run app_frontend.py \
  --server.address=0.0.0.0 \
  --server.port="${PUBLIC_PORT}" \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false &
FRONTEND_PID=$!

wait -n "${API_PID}" "${FRONTEND_PID}"
