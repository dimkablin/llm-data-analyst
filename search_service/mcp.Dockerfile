# SearXNG MCP Bridge
# Based on https://github.com/nitish-raj/searxng-mcp-bridge

FROM node:20-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/nitish-raj/searxng-mcp-bridge.git .

RUN npm install

ENV SEARXNG_URL="http://searxng:8080"
ENV HOST="0.0.0.0"
ENV PORT="8001"

EXPOSE 8001

CMD ["npm", "start"]
