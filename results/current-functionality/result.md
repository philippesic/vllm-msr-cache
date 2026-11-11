# Exact offload and hint delivery pass the current functional checks

2026-11-10 · current · functional

The current engine-method integration passed 13 pinned-runtime CPU tests and three RTX 2080 Ti hardware tests. Restored and freshly computed continuations matched in both advice modes; authenticated HTTP hints reached the engine.

## Scope

Qwen2.5-0.5B-Instruct; RTX 2080 Ti, WSL2, dense FP16 TP1, eager TRITON_ATTN. Hardware execution was fresh; retained receipts were verified.

## Limits

These small functional cases do not establish performance, general answer quality, hardware fault recovery or other models/devices. CPU transfer-failure tests use fake handlers. The pinned native ARC fallback limitation applies in both modes (see runtime docs). The later registration-refusal guards have CPU regression coverage; hardware tests were not rerun after that change.

## Decision

Retain the core and its regression coverage. Expand qualification when implementation changes warrant it.

## Correction

2026-11-10: clarified the existing upstream ARC limitation and the CPU-only validation boundary of subsequent registration refusal hardening; original measurements unchanged.

## Sources

- [evidence/core-qualification-20261110.md](../sources/5de064e7ed71e9ca3ac9b3b6fc4e4c4b7b1b60e666a81d5eaec2a15eb4c84a3b.json)

Exact source hashes and figure inputs: [result.json](result.json).
