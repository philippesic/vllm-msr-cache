# RTX 5080 Stage 6 — qualified; confirmation running

The pinned LMCache rebuild works on the RTX 5080. All five frozen Continuum
screens pass, all 54 required screening configurations form 22 complete matched
blocks, and the full functional qualification gate passes. The imported frozen
confirmation schedule is activated and running: 524 configurations in 210
matched blocks. Repeated throughput estimates and final latency comparisons are
still pending; overhead, operational and final audits have not completed.

This is a new campaign. The failed November 8 campaign and all 2080 Ti history
remain unchanged. The sole declared setup deviation is rebuilding the same
LMCache 0.3.6 source for `sm_120` plus PTX. Source and algorithm are unchanged;
the rebuilt Continuum environment retains the same package versions. The study
keeps native ARC, candidate, Continuum and upstream vLLM PR #59714, with the
exported models, workloads, resource geometry, order, repeat counts and analysis.
Cross-host differences cannot be attributed to the GPU alone.

The [build receipt](lmcache-build.json) records source/toolchain/architecture and
wheel/binary identities, setup repairs and exact pinned CPU/GPU round-trip proof:
three FP16 layers, 256 tokens and 393,216 bytes, equal in both directions. The
[short serving diagnostic](continuum-smoke.json) completed warmup and four requests
with GPU KV stores and pinned geometry/cleanup. Its original one-session policy
verdict remains inconclusive and it is excluded from performance measurements.
Exported CPU checks passed 240 benchmark tests, 31 lightweight tests and compilation.

The [five Continuum screen receipts](continuum-all-screen.json) record B06/B08/
B09/B11/B12 completing 32/64/64/80/32 requests with 24/30/35/42/24 scheduler pins.
Every seal and cleanup passes. This repairs the original warmup kernel blocker
across all frozen Continuum screen fixtures; it does not establish a speed gain.

[Native preflight](preflight.json) passed restore/fresh-reference token equality
and namespace/idempotence/future/stale checks for every model plus the registry
probe. [Qualification](qualification.json) passes native correctness, external
activation and pinned geometry, screen completeness, all three ARC pressure
gates, and the roomy reduction. [Activation](activation.json) preserves the exact
imported order, supplementary schedules, repeat/precision choices and acquisition
source, without replanning. This is functional, not production qualification.

Development pressure observations are retained with their scope:

- [Anchor Qwen3-8B](anchor-screen.json): ARC had one CPU-eviction-affected return;
  roomy ARC had zero, satisfying the prespecified reduction check.
- [Ministral3-3B](ministral-pressure-screen.json): ARC had 35 affected returns
  among 48 returning requests; candidate had 29/48.
- [Qwen3-4B](qwen4b-pressure-screen.json): ARC had 41/48; candidate had 38/48.
- [B10 candidate control](screen-team-control.json): all 20 requests completed
  without CPU restores/evictions. Its raw inconclusive status remains. The frozen
  analysis accepts the complete GPU-resident control, without pressure credit.

[Screen integrity](screen-complete.json) verifies all 6,160 sealed files across
58 verified, one deliberately interrupted and one raw inconclusive/failed attempt
in screen plus preflight. The unmatched original B01 ARC receives no comparison
credit. [Whole-block recovery](recovery-complete.json) passed owned cleanup and
replacement of both B01 arms. [Transfer seals](transfer-seals.json) and
[old-history integrity](history-integrity.json) pass; original evidence is retained.

The [partial selected source review](screen-review-index-06.json) covers 781 items
across 289 complete contexts: 190 source failures, 424 instruction failures and
53 outputs judged not useful, with duplicate contexts counted separately. These
are selected-review counts, not population error rates. Failures include malformed
or fenced JSON, wrong answers, unsupported additions/limitations, invented or
denied validation, and overlong/unfinished intermediates. Correct useful fragments
are separately recognized. The [partial all-cap review](caps-screen-review-index-01.json)
covers 13 capped responses at its earlier snapshot, including intermediates.
Both inspect actual preceding dependencies/tools and archived system instructions.
All timings and original objective results remain; quality equivalence is not
established. The final screen inventory has been prepared but still needs its
233 novel contexts reviewed. Confirmation and all-phase audits remain pending.

See [continuation instructions](CONTINUE.md) for the active owner, exact paths,
remaining phases and enabled health monitoring.
