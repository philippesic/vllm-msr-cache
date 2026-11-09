# RTX 5080 Stage 6 — rebuild in progress

This fresh primary replication is authorized to rebuild pinned LMCache 0.3.6
for the RTX 5080 (`sm_120` plus PTX). The previous campaign remains closed and
unchanged. There are no new measured performance results yet.

The comparison retains native ARC, candidate, Continuum and upstream vLLM
PR #59714, with the exported primary models, workloads, resource geometry,
order, repeat counts and analysis. Only LMCache's architecture-specific build
is an authorized setup deviation; source and algorithm remain pinned.
Cross-host comparisons cannot isolate GPU effects from runtime/host differences.

See [continuation instructions](CONTINUE.md) for active work and retained paths.
