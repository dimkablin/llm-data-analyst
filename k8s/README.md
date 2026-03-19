# SciBox / K8s deploy

Этот каталог подготавливает `llm-data-analyst` к запуску в SciBox/k8s за reverse proxy c префиксом `/llm-data-analyst/`.

## Что важно

- Новый frontend собирается с base path на этапе build.
- Runtime nginx во frontend уже умеет жить под префиксом через `APP_BASE_PATH`.
- Ingress не должен срезать префикс. Он просто проксирует `/llm-data-analyst/*` во frontend service.
- Backend не требует отдельного `root_path`, потому что браузер общается только с frontend nginx, а он уже сам проксирует `/auth`, `/sessions`, `/db-connections`, `/observability` и `/phoenix`.

## Сборка образов

Backend:

```bash
docker build -f Dockerfile.backend -t ghcr.io/your-org/llm-data-analyst-backend:latest .
docker push ghcr.io/your-org/llm-data-analyst-backend:latest
```

Frontend для SciBox prefix deploy:

```bash
docker build -f Dockerfile.frontend \
  --build-arg VITE_BASE_PATH=/llm-data-analyst/ \
  -t ghcr.io/your-org/llm-data-analyst-frontend:latest .
docker push ghcr.io/your-org/llm-data-analyst-frontend:latest
```

Для обычного docker-compose локально `VITE_BASE_PATH` оставляйте `/`.

## Секреты

1. Скопируйте `k8s/secret.example.yaml` в свой секрет или создайте его вручную:

```bash
kubectl create secret generic llm-data-analyst-secrets \
  --from-literal=LLM_API_KEY=... \
  --from-literal=AUTH_DEFAULT_ADMIN_PASSWORD=... \
  --from-literal=DB_CONNECTIONS_ENCRYPTION_KEY_CURRENT=... \
  -n <your-namespace>
```

2. Если нужен корпоративный CA:

```bash
kubectl create secret generic llm-data-analyst-ca \
  --from-file=inno_pki_chain.pem=certs/ca/inno_pki_chain.pem \
  -n <your-namespace>
```

Backend mount уже предусмотрен в `deployment.yaml` в `/opt/certs`.

## Применение

```bash
kubectl -n <your-namespace> apply -f k8s/configmap.yaml
kubectl -n <your-namespace> apply -f k8s/pvc.yaml
kubectl -n <your-namespace> apply -f k8s/service.yaml
kubectl -n <your-namespace> apply -f k8s/deployment.yaml
kubectl -n <your-namespace> apply -f k8s/ingress.yaml
```

## Что проверить

1. Frontend открывается по:

```text
https://<host>/llm-data-analyst/
```

2. Phoenix открывается по:

```text
https://<host>/llm-data-analyst/phoenix/
```

3. Из браузера запросы идут не напрямую в backend, а через frontend nginx:

- `/llm-data-analyst/auth/login`
- `/llm-data-analyst/sessions`
- `/llm-data-analyst/db-connections`
- `/llm-data-analyst/observability/phoenix`

4. SSE не обрывается:

- чат стримит ответ через `/sessions/{id}/query/stream`
- в ingress включен `proxy-buffering: off`

## Почему этот вариант проще, чем в старом МР

В старом проекте приходилось держать отдельный nginx config для k8s с другими service names и ручной логикой prefix routing.  
В новом проекте это уже встроено в:

- `frontend/nginx.conf`
- `Dockerfile.frontend`
- `frontend/vite.config.ts`

Поэтому для SciBox здесь достаточно:

- собрать frontend с `VITE_BASE_PATH=/llm-data-analyst/`
- выставить runtime `APP_BASE_PATH=/llm-data-analyst/`
- не срезать prefix на ingress
