# The upstream mmap transfer failure required the CUDA device alias

2026-11-10 · historical · functional

A controlled reproducer isolated illegal CUDA access to using a CPU mmap address where the batch-transfer path needed its CUDA device alias. Mapped-address stores and restores passed byte equality.

## Scope

RTX 2080 Ti/WSL2; 7,168 descriptors of 16 KiB. A separate three-request Qwen2.5-7B-AWQ serving probe restored pinned CPU KV and matched outputs.

## Limits

This is a specific compatibility result, not a blanket WSL limitation or a performance win. The external comparison used an explicitly patched upstream runtime.

## Decision

Preserve the cause and patch identity for future runtime comparisons; do not apply the old patch automatically to this project.

## Sources

- [evidence/upstream-pin-fix-20261104/README.md](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/upstream-pin-fix-20261104/README.md)
- [evidence/upstream-pin-fix-20261104/mapped-host-pointer.patch](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/upstream-pin-fix-20261104/mapped-host-pointer.patch)

Exact source hashes and figure inputs: [result.json](result.json).
