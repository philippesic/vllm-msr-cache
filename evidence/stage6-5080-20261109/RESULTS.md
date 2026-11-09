# RTX 5080 Stage 6 — rebuilt; serving validation pending

This fresh primary replication is authorized to rebuild pinned LMCache 0.3.6
for the RTX 5080 (`sm_120` plus PTX). The previous campaign remains closed and
unchanged. The rebuild succeeded. Binary inspection finds sm_120 kernels and PTX, and
an FP16 three-layer, 256-token pinned CPU/GPU round trip passed exact equality
in both directions (393,216 bytes). This proves the previously failing kernel
executes on the 5080; live Continuum serving and the full campaign remain pending.
There are no new measured performance results yet.

The comparison retains native ARC, candidate, Continuum and upstream vLLM
PR #59714, with the exported primary models, workloads, resource geometry,
order, repeat counts and analysis. Only LMCache's architecture-specific build
is an authorized setup deviation; source and algorithm remain pinned.
Cross-host comparisons cannot isolate GPU effects from runtime/host differences.

See [continuation instructions](CONTINUE.md) for active work and retained paths.

The [build receipt](lmcache-build.json) records the pinned source, CUDA/torch
versions, compiler and header identities, architecture list, wheel/binary hashes,
setup failures and repairs, and kernel proof. Exported CPU checks passed 240
benchmark tests, 31 lightweight tests and compilation.
