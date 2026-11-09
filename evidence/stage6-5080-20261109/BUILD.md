# Declared RTX 5080 LMCache rebuild

The user authorized this architecture-specific rebuild on 2026-11-09. It keeps
LMCache 0.3.6 at the release source revision recorded in [the build receipt](lmcache-build.json).
The installed release embeds that revision, and all 178 release Python files
match it byte for byte. No LMCache kernel, algorithm or Python source was edited.
The existing Continuum/vLLM source pin is unchanged.

The new wheel targets `TORCH_CUDA_ARCH_LIST=12.0+PTX`, with the pinned torch
2.8.0+cu128 and CUDA 12.8.93 compiler. It is installed in the separate
`stage6-5080-runtimes-02/continuum-env` environment; the old environment and
November 8 failed campaign remain intact. ELF inspection finds five sm_120
images; PTX inspection finds CUDA PTX 8.7 for the same target.

Host build compatibility required GCC 14 instead of the unsupported default
GCC 15, and an isolated compiler sysroot with unmodified Ubuntu 24.04 glibc
2.39/kernel headers instead of the host glibc 2.42 headers. CUDA library headers
come from the pinned torch dependencies. The recipe records every downloaded
archive/header hash, compiler wrapper and relevant environment setting. These
are build setup repairs; they do not change the study design or serving policy.
Four failed preparation attempts and their logs are retained, as summarized in
the receipt. The fifth build succeeded.

The retained helper reproduces the build with a clean checkout and new receipt
directory (after reconstructing the pinned Continuum environment and installing
GCC 14):

```bash
python3 -m benchmarks.research_checks.stage6_5080_lmcache \
  --source /absolute/path/to/unchanged-lmcache-v0.3.6 \
  --python /absolute/path/to/new-continuum-env/bin/python \
  --toolkit /absolute/path/to/task-cuda-12.8 \
  --out /absolute/path/to/new-build-receipt \
  --original-python /absolute/path/to/original-continuum-env/bin/python
```

The original `runtime_setup` preparation is reused for the unchanged local and
upstream environments and clean pinned vLLM source trees. The new Continuum
environment was reconstructed from the exact exported `continuum.txt`, installed
editable against that same pinned vLLM checkout/precompiled base, then installed
with the rebuilt LMCache wheel using `--no-deps`. No dependency upgrade is part
of this repair. New transfer/import binds the resulting native-library identity.

The direct kernel smoke passed exact FP16 equality for 256 permuted token slots,
three layers and 393,216 pinned CPU bytes in both GPU-to-CPU and CPU-to-GPU
directions. It uses the formerly failing `multi_layer_kv_transfer` operation.
The smoke helper takes both GPU reservations and releases them on exit. This
proves that operation loads and transfers correctly in this bounded diagnostic;
live serving, campaign correctness, throughput and quality are separate checks.
