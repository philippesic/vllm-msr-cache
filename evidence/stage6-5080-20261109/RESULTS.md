# RTX 5080 Stage 6 — qualified; confirmation running

The pinned LMCache rebuild works on the RTX 5080. All five frozen Continuum
screens pass, all 54 required screening configurations form 22 complete matched
blocks, and the full functional qualification gate passes. The imported frozen
confirmation schedule is activated and running: 524 configurations in 210
matched blocks. The first three native cells have complete repeated estimates below.
Headline comparisons with Continuum and the remaining cells are still pending;
overhead, operational and final audits have not completed.

Confirmation now has a retained [B06 activity failure](confirmation-b06-inactive-restore-01.json):
the first candidate process served all 32 requests and 24 tools with no caps and
clean cleanup, but recorded zero CPU restores and eight scheduler preemptions.
The frozen analysis rejects it; the zero-preemption GPU-resident-control exception
does not apply. Its four-system block therefore cannot receive comparison credit.
This is not an LMCache kernel-loading failure. Remaining planned measurements
continue without a favorable-activity retry, capacity change or analysis-rule
change. A complete final matrix is unproven and may remain unavailable.
The [finished B06 attempt inventory](confirmation-b06-attempts-complete.json)
verifies all 32 scheduled attempts and 2,648 sealed files: 1,024 completed
responses, zero caps and all coverage/cleanup checks passed. It has 31 valid
measurements and seven complete matched blocks, short of the required eight.
All planned repetitions ran once; the excluded block was not replaced.

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

The [completed-cell checkpoint](confirmation-three-cells.json) retains
all frozen repetitions for Qwen3-8B chat (B01), independent QA (B02) and document
summaries (B03). The [earlier two-cell receipt](confirmation-first-cells.json)
remains unchanged. The
candidate/ARC throughput ratio is a geometric mean of whole-process paired
ratios, with log Student-t 95% CIs; serving seconds exclude startup. All three
intervals include parity, so none resolves a throughput gain or loss. The
[current interim display](early-three-cells.html) and
[render receipt](early-three-cells-render.json) cover these three cells only.
[Extraction validation](early-cells-validation.json) records actual complete-cell
execution, exact prior-data preservation and guards against overwriting history
or importing from the wrong source directory.

| Cell | Process pairs | Candidate mean s | ARC mean s | Candidate/ARC throughput [95% CI] |
| --- | ---: | ---: | ---: | --- |
| B01 | 12 | 33.457 | 33.502 | 1.0014 [0.9989, 1.0038] |
| B02 | 6 | 6.220 | 6.268 | 1.0078 [0.9962, 1.0195] |
| B03 | 12 | 45.641 | 45.512 | 0.9972 [0.9896, 1.0048] |

Latency figures are milliseconds, averaged over per-process statistics.
Mean process p95 is not a pooled request p95 or a population-tail guarantee.

| Cell / system | Mean end-user TTFT | Mean process p95 TTFT | Mean completion | Mean process p95 completion |
| --- | ---: | ---: | ---: | ---: |
| B01 / candidate | 76.241 | 164.217 | 1068.861 | 1521.612 |
| B01 / ARC | 81.246 | 187.266 | 1076.579 | 1525.527 |
| B02 / candidate | 2822.843 | 5447.739 | 3137.215 | 5656.632 |
| B02 / ARC | 2837.121 | 5463.019 | 3153.108 | 5672.586 |
| B03 / candidate | 20807.810 | 39655.551 | 25531.067 | 44184.825 |
| B03 / ARC | 20754.448 | 39535.872 | 25461.058 | 44050.939 |

All 648 matched request texts and finish reasons agree between native arms
within these three fixed cohorts (1,296 recorded responses total). Each B01 arm
has 72 responses / 12 caps; each B02 arm has 384 responses / zero caps; each B03
arm has 192 responses / zero caps and 63,768 output tokens.
None of these cells records returning CPU restoration or eviction-affected returns;
these are controls, not evidence of pressured CPU-offload benefit. Output
identity is conditional on these requests and does not establish answer
correctness, population quality equivalence or results for other cells/systems.
The full 524-configuration confirmation and later phases remain incomplete.

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

The ongoing [confirmation integrity snapshot](confirmation-integrity-checkpoint-04.json)
verifies 15,456 sealed files, including 194 verified attempts, one interrupted
attempt and nine raw failed states (including accepted controls and the excluded
B06 attempt). All 2,192 frozen acquisition-source files still match the lock.
The [earlier checkpoint](confirmation-integrity-checkpoint-02.json)
also verified the 4,606-file execution bundle
unchanged. Active unsealed work and later attempts are outside each snapshot;
final campaign completion remains pending.

The [final screening source audit](screen-quality-complete.json) validates all
1,248 prescribed selected outputs across 522 distinct full contexts, alongside
objective checks on all 1,194 screening finals. It retains 401 source-support
failures, 725 instruction failures and 111 outputs judged not useful, counting
duplicate contexts separately. These are selected-review counts, not population
error rates. Failures include malformed/fenced JSON, wrong or incomplete answers,
unsupported additions/limitations, invented or denied validation, and overlong or
unfinished intermediates. Correct useful fragments are separately recognized.
[Explicit judgments](screen-source-judgments-07.json) bind the actual preceding
dependencies/tools and archived executed system instruction. The frozen audit
validator passed on an isolated materialized screen snapshot; the active
confirmation analysis was not changed. All timings and original objective results
remain, and quality equivalence is not established.

The [partial all-attempt cap review](caps-all-review-index-03.json) covers
163 capped responses present through early B08: 132 screen and 31 confirmation
responses, across 86 extended contexts (148 intermediates, 15 finals). It includes
interrupted captures and individual arms outside matched blocks. Archived
instructions, sealed corpus/node identities and raw prompt/workload digests
were separately checked. Seven responses are new since the prior inventory:
five reuse explicit confirmation08 judgments and two were explicitly read.
All 163 miss an instruction, 108 lack source support and 46 are not useful;
correct useful fragments are separately retained. These are cap-selected,
duplicate-inclusive counts, not population rates. The final inventory must still
include every ordinary attempt after all acquisition phases finish; this snapshot
still has a pending attempt and is not finalized. Confirmation
source review, overhead, operational and final campaign audits remain pending.

The [latest partial confirmation review](confirm-review-index-08.json) covers
2,243 selected outputs / 162 contexts through all B07 repetitions and the first
B08 four-system block, with 3,000 objective finals. All 39 novel PEP contexts
were explicitly judged, including four capped intermediates. All sixteen supplied
PEP source texts were read within their full-document/opening-excerpt boundaries;
[source bindings](confirm-source-components-08.json) retain exact repeated-text
reuse, with each new dependency/tool/answer context separately read.
Novel findings include 18 source-support failures, 25 instruction failures and
six outputs not useful: unfinished/repetitive caps, a repetitive final with the
wrong creation year, CSV/pickle confusion and unsupported limitation/runtime
claims. Useful supported fragments and accurate historical proposal rules remain.
Metadata validation checks five fields, not feature implementations or omitted
source content. Earlier contexts exactly reuse complete source/instruction/answer
judgments. The excluded first B06 block remains excluded; this source audit
does not repair its activity failure.
The earlier sixteen-document review found four summaries with unsupported claims
or lost uncertainty. Three summaries exceed 300 words in total, including one of
those four; earlier prose incorrectly said three additional summaries. Individual
ratings and prior counts are unchanged. Earlier QA and fictional-provenance
findings remain. The current snapshot has **133 source-support failures, 658
instruction failures and six not-useful outputs**. These counts include repeated
contexts. The [isolated validator](confirm-quality-checkpoint-08.json) checks every judgment
against the unchanged archived rubric; archived instructions were verified for
all 164 included attempts. These
duplicate-inclusive partial sample counts certify neither population accuracy
nor equivalence, and the final confirmation audit is pending. Original objective
results and all timings remain unchanged.

See [continuation instructions](CONTINUE.md) for the active owner, exact paths,
remaining phases and enabled health monitoring.
