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

- [evidence/upstream-pin-fix-20261104/README.md](../sources/f9689985e97c8fa24f588d42904950e29a822c9b7ac23311bc028520c71aa131.json)
- [evidence/upstream-pin-fix-20261104/mapped-host-pointer.patch](../sources/398329ef70a547a81cb2c4a1e6423ead30cbecd5f72dcfe83f8554a674e3d29f.json)

Exact source hashes and figure inputs: [result.json](result.json).
