#!/usr/bin/env bash
set -eu

image_tag="${IMAGE_TAG:?IMAGE_TAG is required}"
deploy_dir="${SEARXNG_MCP_DEPLOY_DIR:-/opt/deploy/searxng-mcp-dev}"
network="${MCP_DOCKER_NETWORK:-llm-data-analyst-dev_default}"
env_file="$deploy_dir/.env"
compose_file="$deploy_dir/docker-compose.yaml"
secret_file="$deploy_dir/secrets/searxng_mcp_api_key"

docker build -t "searxng-mcp:$image_tag" .
docker build -f search_service.Dockerfile -t "searxng-search-service:$image_tag" .

mkdir -p "$deploy_dir/configs" "$deploy_dir/secrets"
if [ -n "${SEARXNG_MCP_API_KEY:-}" ]; then
  umask 077
  printf '%s' "$SEARXNG_MCP_API_KEY" >"$secret_file"
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
cp configs/searxng_settings.yml "$deploy_dir/configs/searxng_settings.yml"
docker compose --env-file "$env_file" -f "$compose_file" config --quiet
docker compose --env-file "$env_file" -f "$compose_file" up -d --no-build

for attempt in $(seq 1 20); do
  if docker compose --env-file "$env_file" -f "$compose_file" exec -T searxng-mcp \
    node -e 'const fs=require("fs"); const key=fs.readFileSync("/run/secrets/searxng_mcp_api_key","utf8").trim(); fetch("http://127.0.0.1:8811/mcp",{method:"POST",headers:{Authorization:`Bearer ${key}`,"Content-Type":"application/json",Accept:"application/json, text/event-stream"},body:JSON.stringify({jsonrpc:"2.0",id:1,method:"initialize",params:{protocolVersion:"2025-03-26",capabilities:{},clientInfo:{name:"deploy-health",version:"1"}}})}).then(response=>{if(!response.ok) throw new Error(String(response.status))}).catch(error=>{console.error(error);process.exit(1)})' \
    >/dev/null 2>&1; then
    break
  fi
  if [ "$attempt" -eq 20 ]; then
    docker compose --env-file "$env_file" -f "$compose_file" logs searxng-mcp
    exit 1
  fi
  sleep 3
done

docker compose --env-file "$env_file" -f "$compose_file" exec -T search-service \
  python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

docker compose --env-file "$env_file" -f "$compose_file" ps
