#!/usr/bin/env bash
# Start everything for local development. Run from the repo root: bash scripts/dev.sh
source .venv/bin/activate
HOST=$(ip route | awk '/default/{print $3}')
sed -i '/^RAG_OLLAMA_URL=/d' .env && echo "RAG_OLLAMA_URL=http://$HOST:11434" >> .env
docker compose up -d || { echo "Docker is not running: start Docker Desktop first"; exit 1; }
sleep 3
echo -n "qdrant: "; curl -sS -m 5 localhost:6333/readyz || echo "NOT READY"; echo
echo -n "ollama: "; curl -sS -m 5 "http://$HOST:11434" || echo "NOT READY"; echo
pkill -f "uvicorn app.main" 2>/dev/null
uvicorn app.main:app --port 8000 > /tmp/uvicorn.log 2>&1 &
trap 'pkill -f "uvicorn app.main"' EXIT
sleep 4
echo -n "api ready: "; curl -s localhost:8000/ready; echo
streamlit run ui/streamlit_app.py
