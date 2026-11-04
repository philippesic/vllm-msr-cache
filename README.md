# MSR KV Cache Eviction for vLLM

An experimental vLLM extension for exact KV-cache offload, semantic retention,
and causal prefetch. Mean semantic relevance (MSR) ranks cache candidates;
exact prefix, model, adapter, layout, and position identity determine reuse.

The general semantic-ranking and speculative-prefetch policies remain research
prototypes. A narrowly qualified CPU-advice/continuation-priority profile now
supports controlled asynchronous draft queues with human review on the tested
2080 Ti runtime. Fresh natural generation measured 6.10% higher throughput and
485 ms lower mean first-token latency versus matched native ARC with the same
priority; VRAM was unchanged and the p95 benefit remains uncertain. This is not
a semantic-ranking win or an interactive-chat qualification. See the
[results and operating limits](evidence/README.md) before selecting a profile.

## Develop

Use the [pinned runtime setup](vendor/VLLM.md) for Linux x86-64 / Python 3.12.
On the existing WSL host:

```bash
PY=/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python
make runtime-check PY="$PY"
make test PY="$PY"
```

These checks do not launch a model. See [development and benchmarking](docs/BENCHMARKING.md)
for the maintained smoke, screen, confirmation, and operational commands,
measured turnaround, and validation limits.
The generic `cache-affinity` package extra expresses compatibility; it does not
reproduce the patched runtime used by this repository.

## Use

The extension supplies `SemanticOffloadingConnector` and `SemanticOffloadingSpec`
through vLLM's KV-transfer configuration. See [configuration](docs/CONFIGURATION.md)
for examples and experimental controls. The helper is
`semantic_offload.config.msr_kv_transfer_config`.

Native adoption has narrow layout requirements, and the RTX 2080 Ti uses FP16.
Do not extrapolate a dense-model result to hybrid layouts, larger models, or
other hardware without qualification.

## Find your way

| Location | Purpose |
| --- | --- |
| [Current work](docs/STATUS.md) | Active tasks, owners, verified state, next steps |
| [Brief history](docs/PROJECT_HISTORY.md) | Completed work and useful negative findings |
| `semantic_offload/` | Installed extension: connector, policy, scoring, workers |
| `tests/` | Correctness and measurement checks |
| `benchmarks/` | Experiment runners, validators, and analysis |
| `harness/` | Reusable server, workload, policy, and metrics helpers |
| `tools/` | Local development utilities |
| `vendor/`, `patches/` | Pinned vLLM source, provenance, and upstream changes |
| `evidence/` | Compact retained results, protocols, and verification |
| `output/` | Ignored run artifacts and frozen experiments; may contain active work |
