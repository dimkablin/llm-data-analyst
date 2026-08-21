#!/usr/bin/env bash
set -eu

image_tag="${IMAGE_TAG:?IMAGE_TAG is required}"
deploy_dir="${CHRONOS_MCP_DEPLOY_DIR:-/opt/deploy/chronos-mcp-dev}"
network="${MCP_DOCKER_NETWORK:-llm-data-analyst-dev_default}"
env_file="$deploy_dir/.env"
compose_file="$deploy_dir/docker-compose.yaml"
secret_file="$deploy_dir/secrets/chronos_mcp_api_key"

docker build \
  --build-arg "PYTORCH_IMAGE=${CHRONOS_PYTORCH_IMAGE:-pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime}" \
  -t "chronos-mcp:$image_tag" \
  .

mkdir -p "$deploy_dir/secrets" "$deploy_dir/model_cache"
if [ -n "${CHRONOS_MCP_API_KEY:-}" ]; then
  umask 077
  printf '%s' "$CHRONOS_MCP_API_KEY" >"$secret_file"
elif [ ! -s "$secret_file" ]; then
  umask 077
  openssl rand -hex 32 >"$secret_file"
fi
test -s "$secret_file"
chmod 600 "$secret_file"
docker network inspect "$network" >/dev/null

touch "$env_file"
if grep -q '^IMAGE_TAG=' "$env_file"; then
  sed -i "s/^IMAGE_TAG=.*/IMAGE_TAG=$image_tag/" "$env_file"
else
  printf '\nIMAGE_TAG=%s\n' "$image_tag" >>"$env_file"
fi
if grep -q '^MCP_DOCKER_NETWORK=' "$env_file"; then
  sed -i "s/^MCP_DOCKER_NETWORK=.*/MCP_DOCKER_NETWORK=$network/" "$env_file"
else
  printf 'MCP_DOCKER_NETWORK=%s\n' "$network" >>"$env_file"
fi

cp docker-compose.deploy.yml "$compose_file"
docker compose --env-file "$env_file" -f "$compose_file" config --quiet
docker compose --env-file "$env_file" -f "$compose_file" up -d --no-build

for attempt in $(seq 1 20); do
  if docker compose --env-file "$env_file" -f "$compose_file" exec -T chronos-mcp \
    python -c 'import json,pathlib,urllib.request; payload=json.dumps({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"deploy-health","version":"1"}}}).encode(); key=pathlib.Path("/run/secrets/chronos_mcp_api_key").read_text().strip(); request=urllib.request.Request("http://127.0.0.1:8810/mcp",data=payload,headers={"Authorization":f"Bearer {key}","Content-Type":"application/json","Accept":"application/json, text/event-stream"}); urllib.request.urlopen(request,timeout=5).read()' \
    >/dev/null 2>&1; then
    break
  fi
  if [ "$attempt" -eq 20 ]; then
    docker compose --env-file "$env_file" -f "$compose_file" logs chronos-mcp
    exit 1
  fi
  sleep 3
done

docker compose --env-file "$env_file" -f "$compose_file" ps
