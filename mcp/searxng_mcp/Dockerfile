FROM node:25-alpine AS build

RUN apk add --no-cache git
WORKDIR /src
RUN git clone https://github.com/dimkablin/searxng-mcp-bridge.git . \
    && git checkout acbbfe7c57be06947d6da3d24284933857e8f2e6 \
    && npm ci \
    && npm run build \
    && npm prune --omit=dev --ignore-scripts

FROM node:25-alpine

LABEL org.opencontainers.image.source="https://github.com/dimkablin/searxng-mcp-bridge" \
      org.opencontainers.image.revision="acbbfe7c57be06947d6da3d24284933857e8f2e6"

COPY --from=build /src/build /usr/local/lib/node_modules/searxng-mcp-bridge/build
COPY --from=build /src/node_modules /usr/local/lib/node_modules/searxng-mcp-bridge/node_modules
COPY --from=build /src/package.json /usr/local/lib/node_modules/searxng-mcp-bridge/package.json
RUN ln -s /usr/local/lib/node_modules/searxng-mcp-bridge/build/index.js /usr/local/bin/mcp-searxng-bridge

ENV SEARXNG_INSTANCE_URL="http://searxng:8080"
ENV HOST="0.0.0.0"
ENV PORT="8811"
ENV TRANSPORT="http"

EXPOSE 8811

CMD ["mcp-searxng-bridge"]
