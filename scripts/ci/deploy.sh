#!/usr/bin/env bash
set -eu

action="${1:-all}"
target_env="${TARGET_ENV:-dev}"

case "$target_env" in
  dev)
    deploy_dir="/opt/deploy/llm-data-analyst-dev"
    backend_port="${BACKEND_HEALTH_PORT:-8615}"
    frontend_port="${FRONTEND_HEALTH_PORT:-8613}"
    default_tag="dev-${CI_COMMIT_SHORT_SHA:-local}"
    ;;
  prod)
    deploy_dir="/opt/deploy/llm-data-analyst"
    backend_port="${BACKEND_HEALTH_PORT:-8605}"
    frontend_port="${FRONTEND_HEALTH_PORT:-8603}"
    default_tag="${CI_COMMIT_TAG:-}"
    ;;
  *)
    echo "TARGET_ENV must be dev or prod, got: $target_env" >&2
    exit 2
    ;;
esac

case "$action" in
  build|deploy|all) ;;
  *)
    echo "Usage: $0 [build|deploy|all]" >&2
    exit 2
    ;;
esac

image_tag="${IMAGE_TAG:-$default_tag}"
if [ -z "$image_tag" ]; then
  echo "IMAGE_TAG is required for prod unless the pipeline runs on a git tag." >&2
  exit 2
fi
export IMAGE_TAG="$image_tag"

case "$image_tag" in
  *[!A-Za-z0-9_.-]*)
    echo "IMAGE_TAG may contain only letters, digits, dot, underscore, and dash: $image_tag" >&2
    exit 2
    ;;
esac

if [ "$action" = "build" ] || [ "$action" = "all" ]; then
  docker build -f Dockerfile.backend -t "llm-data-analyst-backend:$image_tag" .
  docker build \
    -f Dockerfile.frontend \
    -t "llm-data-analyst-frontend:$image_tag" \
    --build-arg "USE_MIRROR=${USE_MIRROR:-False}" \
    --build-arg "NPM_REGISTRY=${NPM_REGISTRY:-}" \
    --build-arg "APP_BASE_PATH=${APP_BASE_PATH:-/}" \
    .
fi

if [ "$action" = "build" ]; then
  exit 0
fi

env_file="$deploy_dir/.env"
test -f "$deploy_dir/docker-compose.yaml"
test -f "$env_file"
docker image inspect "llm-data-analyst-backend:$image_tag" "llm-data-analyst-frontend:$image_tag" >/dev/null

if grep -q '^IMAGE_TAG=' "$env_file"; then
  sed -i "s/^IMAGE_TAG=.*/IMAGE_TAG=$image_tag/" "$env_file"
else
  printf '\nIMAGE_TAG=%s\n' "$image_tag" >> "$env_file"
fi

cd "$deploy_dir"
docker compose config --quiet
docker compose up -d --no-build backend frontend

curl -fsS --retry 20 --retry-all-errors --retry-connrefused --retry-delay 3 "http://127.0.0.1:$backend_port/health" >/dev/null
curl -fsS --retry 20 --retry-all-errors --retry-connrefused --retry-delay 3 "http://127.0.0.1:$frontend_port/" >/dev/null
docker compose ps
