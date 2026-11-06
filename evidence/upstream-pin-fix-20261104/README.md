# Upstream pinned mmap transfer fix

A minimal experimental patch fixes the reproduced illegal CUDA access on this
RTX 2080 Ti / WSL host. The published upstream checkout and the five-system
comparison remain unchanged. This is a focused compatibility result, not a new
performance comparison or general runtime qualification.

## Established cause

For CUDA-registered shared mmap backing, the CPU address and CUDA device alias
are different on this host. Upstream builds batch-copy descriptors from the CPU
`Tensor.data_ptr()`. Its native batch-copy path faults with those descriptors.
Replacing only the host descriptors with the CUDA device alias makes the same
transfer pass. Ordinary PyTorch pinned allocations and PyTorch `copy_` on the
registered mmap already work. Changing the registration call from upstream's
ctypes wrapper to PyTorch's wrapper does not fix the fault.

The controlled reproducer uses 7,168 descriptors of 16 KiB each. The unmodified
registered-mmap batch store faults; mapped-address batch store and restore pass
byte equality. This establishes the address assumption as the cause of this
reproduced path; it does not establish a blanket WSL pinned-memory limitation.

## Change and verification

The [experimental patch](mapped-host-pointer.patch) queries
`cudaHostGetDevicePointer` once for each pinned CPU cache tensor, then uses that
address for transfer descriptor arithmetic. CPU tensor views stay unchanged.
It changes only Python code; it requires no CUDA compiler or replacement binary.
No existing supported config switch was found to select this address correction
while preserving pinned mmap backing.

A Qwen2.5-7B-Instruct-AWQ FP16 eager-serving probe passed with pinned memory enabled
and the older model runner. It submitted the same real repository-context prompt
three times, clearing only the GPU prefix cache before each return:

- 3 requests, 2,989 prompt tokens each, 32 generated tokens each; output token IDs
  matched exactly across all three requests.
- Read-only worker observation confirmed pinned mmap and 28 CPU tensors whose
  device aliases differed from their CPU pointers.
- 3 completed stores totaling 170,655,744 bytes.
- 2 completed CPU restores, each 170,655,744 bytes (341,311,488 bytes total).
- Explicit engine shutdown completed; no GPU compute clients remained and both
  GPU locks were available.

The source copy was explicitly experimental. Native libraries and submodule
files are linked to the retained published installation; the original checkout
remains clean. No manifest, existing policy, or recorded timing was relabeled.
The V2 runner, other GPUs/platforms, concurrent serving, and performance effects
were not tested. Priority-policy activation is established by the earlier
comparison, not by this transfer-focused probe. The probe uses forced GPU-cache
resets to guarantee restores; it does not claim representative eviction pressure.

## Evidence and reproduction

Raw evidence is retained in
[output/upstream-pin-fix-20261104](../../output/upstream-pin-fix-20261104/):

- [Result and cleanup](result.json), [source hashes](../../output/upstream-pin-fix-20261104/provenance.json).
- [Failing mmap batch transfer](../../output/upstream-pin-fix-20261104/transfer-mmap.log),
  [successful mapped-address transfer](../../output/upstream-pin-fix-20261104/transfer-mmap-alias.log),
  [successful PyTorch copy control](../../output/upstream-pin-fix-20261104/transfer-mmap-copy.log).
- [Serving log](../../output/upstream-pin-fix-20261104/serving-final.log),
  [completed-transfer observation](../../output/upstream-pin-fix-20261104/serving-transfers.jsonl),
  [frozen serving reproducer](../../output/upstream-pin-fix-20261104/serving_probe.py).

The first two serving setup attempts are retained: an incomplete archive omitted
submodule Python files, then 512 GPU blocks were one reserved block short of the
declared context. The successful run uses 1,024 blocks and 512 MiB CPU backing.
Useful isolated source and logs remain (~58 MiB); native binaries are reused by
symlink. No downloads, builds, commits, pushes, or full comparison reruns occurred.

From the repository root, the retained serving probe can be repeated with the
isolated interpreter and source (choose a fresh observation log):

```bash
env PYTHONPATH="$PWD/output/upstream-pin-fix-20261104/observer:$PWD/output/upstream-pin-fix-20261104/experimental-source" \
  VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_V2_MODEL_RUNNER=0 \
  VLLM_PLUGINS='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 \
  PIN_FIX_OBSERVATION="$PWD/output/upstream-pin-fix-20261104/repeat-transfers.jsonl" \
  output/five-way-20261104/runtimes/upstream-env/bin/python \
  output/upstream-pin-fix-20261104/serving_probe.py
```

The reproducer holds both GPU locks for its invocation and refuses an occupied
device. Its paths depend on the retained local model and published installation.
