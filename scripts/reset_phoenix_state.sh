#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${COMPOSE_PROJECT_NAME:-$(basename "$(pwd)")}"
PHOENIX_VOLUME="${PROJECT_NAME}_phoenix_data"

echo "[reset-phoenix] project=${PROJECT_NAME}"
echo "[reset-phoenix] stopping phoenix container..."
docker compose stop phoenix >/dev/null 2>&1 || true
docker compose rm -sf phoenix >/dev/null 2>&1 || true

if docker volume inspect "${PHOENIX_VOLUME}" >/dev/null 2>&1; then
  echo "[reset-phoenix] removing volume ${PHOENIX_VOLUME}..."
  docker volume rm "${PHOENIX_VOLUME}" >/dev/null
else
  echo "[reset-phoenix] volume ${PHOENIX_VOLUME} not found, skipping remove."
fi

echo "[reset-phoenix] starting phoenix..."
docker compose up -d phoenix >/dev/null
echo "[reset-phoenix] done."
