# RTX 5080 Stage 6 — rebuilt; screening in progress

This fresh primary replication is authorized to rebuild pinned LMCache 0.3.6
for the RTX 5080 (`sm_120` plus PTX). The previous campaign remains closed and
unchanged. The rebuild succeeded. Binary inspection finds sm_120 kernels and PTX, and
an FP16 three-layer, 256-token pinned CPU/GPU round trip passed exact equality
in both directions (393,216 bytes). This proves the previously failing kernel
executes on the 5080; live Continuum then completed warmup and four requests with actual GPU KV
stores and verified pinned geometry/cleanup. The one-session diagnostic did not
exercise scheduler pins. The separate full B06 Continuum screen has now passed
all 32 requests with 24 scheduler pin events, matching the rebuilt binary identity
and verifying pinned geometry and cleanup. Overall screening/qualification remains open.
All four native preflights passed: restore/fresh-reference correctness for
every model and the anchor registry probe. All 223 preflight files verify.
Screening is running; confirmation and later phases remain pending.
Repeated performance estimates remain pending.

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

The [whole-block recovery audit](recovery-complete.json) now passes: both B01
replacement arms completed and verified in one matched block. Screening advanced
to B02. This recovery result does not establish performance qualification.

[Full Continuum screen evidence](continuum-full-screen.json) confirms the declared
architecture rebuild works in the unchanged benchmark fixture. It is an advisory
screen result, not a repeated throughput or answer-quality conclusion.

The [anchor four-system B06 screen](anchor-screen.json) passed for native ARC,
candidate, Continuum and upstream. Native ARC actually restored CPU tokens and
had an eviction-affected return, establishing this screen pressure purpose.
This single development block does not provide the requested repeated estimates.

The [five Continuum screen receipts](continuum-all-screen.json) now all pass:
32, 64, 64, 80 and 32 completed requests with 24, 30, 35, 42 and 24 scheduler
pins respectively. Every attempt verifies its seal and clean release. All 12
Qwen screen blocks are complete; screening has moved to the secondary models.
The [B10 candidate control](screen-team-control.json) served all 20 requests
without CPU restores or evictions. Its original inconclusive status remains;
the frozen analysis accepts it as a GPU-resident control, not pressure evidence.
The [Ministral pressure screen](ministral-pressure-screen.json) passed for both
native arms. ARC had 35 CPU-eviction-affected returns among 48 returning requests,
and candidate had 29/48. This establishes that development fixture's pressure
purpose, without a repeated speed or quality conclusion.
The [Qwen3-4B pressure screen](qwen4b-pressure-screen.json) also passed: ARC had
41 CPU-eviction-affected returns among 48 returning requests, candidate 38/48.
All three models have positive development ARC pressure evidence; full screening
and the complete qualification gate remain open.

The [selected-screen source review](screen-review-index-06.json) covers 781 items
in a partial snapshot (289 complete contexts), retaining format, overlength,
repetition, wrong-subject, factual and attribution failures. The separate
[partial all-cap review](caps-screen-review-index-01.json) covers 13 sealed capped
responses, including intermediates and arms of unfinished whole blocks. Both
audits inspect actual preceding dependencies/tools and the archived system
instruction. Full acquisition and final source/capped-response audits remain
open; quality equivalence is not established and all timings remain included.
