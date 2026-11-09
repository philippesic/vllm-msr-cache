# RTX 5080 Stage 6 — rebuilt; screening in progress

This fresh primary replication is authorized to rebuild pinned LMCache 0.3.6
for the RTX 5080 (`sm_120` plus PTX). The previous campaign remains closed and
unchanged. The rebuild succeeded. Binary inspection finds sm_120 kernels and PTX, and
an FP16 three-layer, 256-token pinned CPU/GPU round trip passed exact equality
in both directions (393,216 bytes). This proves the previously failing kernel
executes on the 5080; live Continuum then completed warmup and four requests with actual GPU KV
stores and verified pinned geometry/cleanup. The one-session diagnostic did not
exercise scheduler pins; full-screen policy activation remains required.
All four native preflights passed: restore/fresh-reference correctness for
every model and the anchor registry probe. All 223 preflight files verify.
Screening is running; confirmation and later phases remain pending.
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

The [new import](import.json) preserves all 22 cells and the frozen order/repeats:
54 screening, 524 confirmation, 38 overhead and six operational configurations.
All normalized corpora and profiles are unchanged. The [live smoke receipt](continuum-smoke.json)
retains its original policy-inconclusive verdict separately from successful
kernel/serving capability. These diagnostics are excluded from performance results.

[Native preflight evidence](preflight.json) preserves every check and seal.
Its correctness success does not establish throughput gains or qualification
of the full four-system matrix.

The [development interruption](recovery-interruption.json) passed owned cleanup
and sealing. Both original B01 arms remain retained; the resumed screen replaces
that whole block. The interrupted measurement is excluded from comparisons.

The [transfer seal audit](transfer-seals.json) passes for both original transfers,
the completion packet, execution bundle and new imported bundle. This is an
integrity result; campaign acquisition and answer-quality audits remain open.
