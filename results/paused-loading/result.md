# A partial AWQ loader prototype reduced sampled startup peak but remains unqualified

2026-11-10 · historical · unresolved

Two completed loading pairs showed a 476 MiB lower sampled startup peak at roughly two seconds more loading time. This is partial evidence from a stopped prototype, not confirmation that the model fits a smaller GPU.

## Scope

2080 Ti/WSL, 7B AWQ at 1,280 GPU KV blocks; five completed configurations and a separately interrupted continuation were retained. Serving-memory qualification is a different result.

## Limits

Only two completed loader pairs, whole-device sampled peaks, no follow-on pressure or persistent qualification. The first loading prototype worsened startup memory. Historical source and reproduction inputs remain outside the active implementation.

## Decision

Preserve the partial positive and failed first approach without importing the prototype or treating it as an active task. Resumption requires a new explicit decision.

## Sources

- [output/gpu-cache-frontier-20261103/bounded-loading-v2/partial-progress-readonly.json](../sources/bacfe85fec581cae106a7d390533daaef02fe6add50689d3ca37be20015bb4c7.json)
- [output/gpu-cache-frontier-20261103/checkpoint.json](../sources/ba65bbc6ebaa70c2af780a8b896d813ef9aa317510f13d6c712bb6df79f566b8.json)
- [evidence/vram-frontier-2080ti-20261103/README.md](../sources/5862100cb703430d1ba5995f0717ccb97d4b20b8fa58d451f027b0cba74b6e1f.json)

Exact source hashes and figure inputs: [result.json](result.json).
