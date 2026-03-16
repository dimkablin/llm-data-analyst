FROM python:3.11

ARG USE_MIRROR
ARG MIRROR_PIP_INDEX_URL
ARG MIRROR_PIP_EXTRA_INDEX_URL
ARG MIRROR_PIP_TRUSTED_HOST

WORKDIR /app

COPY . .

RUN set -eux; \
    PIP_ARGS=""; \
    if [ "${USE_MIRROR}" = "True" ] || [ "${USE_MIRROR}" = "true" ] || [ "${USE_MIRROR}" = "1" ]; then \
      if [ -n "${MIRROR_PIP_INDEX_URL}" ]; then PIP_ARGS="${PIP_ARGS} --index-url ${MIRROR_PIP_INDEX_URL}"; fi; \
      if [ -n "${MIRROR_PIP_EXTRA_INDEX_URL}" ]; then PIP_ARGS="${PIP_ARGS} --extra-index-url ${MIRROR_PIP_EXTRA_INDEX_URL}"; fi; \
      if [ -n "${MIRROR_PIP_TRUSTED_HOST}" ]; then PIP_ARGS="${PIP_ARGS} --trusted-host ${MIRROR_PIP_TRUSTED_HOST}"; fi; \
    fi; \
    pip install ${PIP_ARGS} poetry; \
    poetry install --no-cache

EXPOSE 8609

CMD ["poetry", "run", "streamlit", "run", "app.py", "--server.port=8609", "--server.address=0.0.0.0"]
