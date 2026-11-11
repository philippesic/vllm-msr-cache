# Exact offload and hint delivery pass the current functional checks

2026-11-10 · current · functional

The current engine-method integration passed 13 pinned-runtime CPU tests and three RTX 2080 Ti hardware tests. Restored and freshly computed continuations matched in both advice modes; authenticated HTTP hints reached the engine.

## Scope

Qwen2.5-0.5B-Instruct; RTX 2080 Ti, WSL2, dense FP16 TP1, eager TRITON_ATTN. Hardware execution was fresh; retained receipts were verified.

## Limits

These small functional cases do not establish performance, general answer quality, hardware fault recovery or other models/devices. CPU transfer-failure tests use fake handlers.

## Decision

Retain the core and its regression coverage. Expand qualification when implementation changes warrant it.

## Sources

- `current:f7ef1e39beaa550d604c5f32a1167c4489b5c8bd:evidence/core-qualification-20261110.md`

Exact source hashes and figure inputs: [result.json](result.json).
