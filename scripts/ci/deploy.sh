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
    --build-arg "VITE_BUILD_COMMIT=${CI_COMMIT_SHA:-local}" \
    .
fi

if [ "$action" = "build" ]; then
  exit 0
fi

env_file="$deploy_dir/.env"
compose_file="$deploy_dir/docker-compose.yaml"
semantic_compose_file="$deploy_dir/docker-compose.semantic-network.yaml"
test -f "$compose_file"
test -f "$env_file"
docker image inspect "llm-data-analyst-backend:$image_tag" "llm-data-analyst-frontend:$image_tag" >/dev/null

if ! cmp -s docker-compose.deploy.yaml "$compose_file"; then
  cp "$compose_file" "$compose_file.bak.pre-$image_tag"
fi
cp docker-compose.deploy.yaml "$compose_file"

compose_files=(-f "$compose_file")
semantic_network="$(sed -n 's/^SEMANTIC_DOCKER_NETWORK=//p' "$env_file" | tail -n 1 | tr -d '\r')"
if [ -n "$semantic_network" ]; then
  case "$semantic_network" in
    *[!A-Za-z0-9_.-]*)
      echo "SEMANTIC_DOCKER_NETWORK contains unsupported characters: $semantic_network" >&2
      exit 2
      ;;
  esac
  cp docker-compose.deploy.semantic-network.yaml "$semantic_compose_file"
  compose_files+=(-f "$semantic_compose_file")
fi

set_env_value() {
  local key="$1"
  local value="$2"
  if grep -q "^${key}=" "$env_file"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "$env_file"
  else
    printf '\n%s=%s\n' "$key" "$value" >> "$env_file"
  fi
}

set_env_value "IMAGE_TAG" "$image_tag"
if [ "$target_env" = "dev" ]; then
  set_env_value "SEMANTIC_METADATA_SCHEMA" "dev_semantic_layer"
  set_env_value "SEMANTIC_QDRANT_COLLECTION" "semantic_catalog_chunks_dev"
fi

cd "$deploy_dir"
docker compose "${compose_files[@]}" config --quiet
docker compose "${compose_files[@]}" up -d --no-build backend frontend

curl -fsS --retry 20 --retry-all-errors --retry-connrefused --retry-delay 3 "http://127.0.0.1:$backend_port/health" >/dev/null
curl -fsS --retry 20 --retry-all-errors --retry-connrefused --retry-delay 3 "http://127.0.0.1:$frontend_port/" >/dev/null
docker compose exec -T frontend grep -R -F -q "${CI_COMMIT_SHA:-local}" /usr/share/nginx/html/assets
docker compose ps
