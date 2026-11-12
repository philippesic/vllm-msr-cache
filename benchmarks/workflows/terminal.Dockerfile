ARG TASK_IMAGE
FROM ${TASK_IMAGE}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl tmux
COPY --from=ghcr.io/astral-sh/uv:0.9.5@sha256:f459f6f73a8c4ef5d69f4e6fbbdb8af751d6fa40ec34b39a1ab469acd6e289b7 /uv /root/.local/bin/uv
COPY --from=ghcr.io/astral-sh/uv:0.9.5@sha256:f459f6f73a8c4ef5d69f4e6fbbdb8af751d6fa40ec34b39a1ab469acd6e289b7 /uvx /root/.local/bin/uvx
RUN printf 'export PATH="/root/.local/bin:$PATH"\n' > /root/.local/bin/env
RUN /root/.local/bin/uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 pytest --version
RUN printf 'Acquire::Retries "0";\nAcquire::http::Timeout "1";\nAcquire::https::Timeout "1";\n' > /etc/apt/apt.conf.d/99-benchmark-offline
ENV UV_OFFLINE=1
