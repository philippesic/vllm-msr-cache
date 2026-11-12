FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
RUN apt-get update && apt-get install -y --no-install-recommends git ripgrep && rm -rf /var/lib/apt/lists/*
COPY source /testbed
WORKDIR /testbed
RUN git init && git add . && git -c user.name=Benchmark -c user.email=benchmark@localhost commit -m 'Freeze source documents'
