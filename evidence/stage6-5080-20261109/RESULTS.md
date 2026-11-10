# RTX 5080 Stage 6 — qualified; confirmation running

The pinned LMCache rebuild works on the RTX 5080. All five frozen Continuum
screens pass, all 54 required screening configurations form 22 complete matched
blocks, and the full functional qualification gate passes. The imported frozen
confirmation schedule is activated and running: 524 configurations in 210
matched blocks. The first three native cells have complete repeated estimates below.
Headline comparisons with Continuum and the remaining cells are still pending;
overhead, operational and final audits have not completed.

Latest bounded health: **327/524 valid configurations and 120/210 matched
blocks**, with Qwen3-8B B12 running under the independent recovery service.
All eight B11 comparisons are complete. There are no new health issues beyond
the retained first B06 exclusion; progress and disk checks continue.

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

The [reporting validation](reporting-validation-03.json) covers 28 focused tests
and a synthetic CLI check for an explicitly incomplete presentation. The helper
keeps its default completion guard; its explicit option preserves missing frozen
blocks, actual pair counts, conditional estimates and failed completion flags.
This is preparation for reporting the terminal outcome, not final measured
extraction or a repair of the B06 activity failure.

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

The ongoing [confirmation integrity snapshot](confirmation-integrity-checkpoint-05.json)
verifies 26,096 sealed files, including 279 verified attempts, one interrupted
attempt and nine raw failed states (including accepted controls and the excluded
B06 attempt). All 2,192 frozen acquisition-source files still match the lock.
The [earlier checkpoint](confirmation-integrity-checkpoint-02.json)
also verified the 4,606-file execution bundle
unchanged. Active unsealed work and later attempts are outside each snapshot;
the ledger was unchanged during the latest check and the exact coordinator
remained live. All older checkpoints remain; final campaign completion is pending.

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

The [partial all-attempt cap review](caps-all-review-index-04.json) covers
**196 sealed capped responses / 119 extended contexts**: 132 screen, 64
confirmation; 177 intermediates and nineteen finals. It includes ordinary
individual arms outside matched blocks. Thirty-three responses are added since
[cap03](caps-all-review-index-03.json): 29 exactly match prior explicit confirmation
judgments, and four complete new contexts were explicitly read. One correctly
copies actual PASS but is not useful; another invents PASS before validation.
An eleven-word malformed repeated-token response is capped without exceeding
the whitespace-word limit. All four new contexts are not useful.

Selected duplicate-inclusive counts are **136 source-support failures, 196
instruction failures and 73 not useful**. Archived instructions in all sixty
cap-bearing attempts, sealed corpus/node/source/reference/role/tool branches,
objective fields and raw prompt/workload-source digests were separately checked.
Tokens were bound, not individually decoded. These are selected cap counts,
not population error rates. The inventory still has a pending attempt and is
not finalized; it must be regenerated after every acquisition phase finishes.
Timings, objective results and frozen exclusions remain unchanged.

The historical [partial cap review](caps-all-review-index-05.json) contains **215
responses / 135 contexts**: 132 screen and 83 confirmation, 194 intermediate and
21 final responses. Nineteen occurrences were added; three repeat an earlier
extended signature and sixteen exactly match full contexts already explicitly
judged in confirm12. No new answer rating was automatic. Selected counts are
**152 source-support failures, 214 instruction failures and 79 not useful**.
Source fidelity, requested format and usefulness are assessed separately.
[Bindings](caps-instruction-bindings-05.json) check all 68 cap-bearing archived
instruction sources and sealed corpus/node/prompt identities. Raw tokens were
bound, not individually decoded. Acquisition remains active and one attempt was
pending; this is not a final cap inventory or population error rate.

The latest [partial cap review](caps-all-review-index-06.json) contains **217
responses / 137 contexts**: 132 screen and 85 confirmation, 196 intermediate and
21 final responses. All earlier judgments exactly reuse cap05; the two additions
exactly match the explicitly reviewed confirm14 intermediates. They retain useful
facts but unsupported performance-test claims and intermediate word/command failures.
Selected counts are **154 source-support failures, 216 instruction failures and
79 not useful**. [Bindings](caps-instruction-bindings-06.json) check all 69
cap-bearing archived instruction sources and independently reconstruct actual
preceding dependencies/tools from sealed requests and tool files. Raw prompts
were bound, not individually decoded. Failed/interrupted/unmatched arms remain
included. One attempt was pending; the all-phase final inventory is still required.

The [historical partial confirmation review](confirm-review-index-11.json) adds the
first B09 four-system block after all twelve B08 blocks: **3,768 objective finals,
2,190 automatic final failures, 55 capped intermediates and 2,545 selected outputs
/ 234 complete contexts**. B09 is another PEP-investigation fixture. Fourteen
contexts absent from confirm10 (twelve finals, two capped intermediates) were
explicitly reviewed: thirteen new across prior confirmation snapshots and one
exact older context with the same judgments. Eleven supplied source strings,
repeated complete fields and individual preceding dependencies exactly match
prior explicit readings; [bindings](confirm-review-components-11.json) retain
this, with every new component/answer read separately and no automatic rating.
The [earlier completed-B08 snapshot](confirm-review-index-10.json) remains.

The added group has four source-support failures, eight instruction failures
and one output not useful. Findings retain fake tool-output excerpts, unsupported
modern comparisons, vague package restrictions and repetitive output. Historical
source scope, actual header validation, literal status formatting and usefulness
are assessed separately. Correct partial metadata can remain useful while
omitting the requested specification rule or concrete limitation. Earlier
wrong-subject, invented-validation, cap, QA, fictional-provenance and summary
findings remain in their snapshots, including the summary-word-count correction.

Selected duplicate-inclusive counts are **212 source-support failures, 894
instruction failures and 39 outputs not useful**. The
[isolated frozen validator](confirm-quality-checkpoint-11.json) accepts all 2,545
judgments; all 212 included archived instruction sources verify. These are
selected partial-review counts, not population error rates or quality equivalence.
The first B06 block remains excluded. Original objectives, timings and live
quality state are unchanged. Final confirmation audit is still pending.

The [historical partial confirmation review](confirm-review-index-12.json) includes
all twelve B09 blocks and the first B10 native pair: **4,480 objective finals,
2,337 automatic final failures, 76 capped intermediates, 2,777 selected outputs
/ 291 complete contexts**. All 59 new contexts versus the union of completed
earlier reviews were explicitly read. Fifteen supplied source strings match
earlier readings; four new team-source/test bodies were read in full.
[Bindings](confirm-review-components-12.json) retain exact repeated fields and
individual dependencies. Four contexts also match earlier explicit cap ratings.

The new group has **36 source failures, 47 instruction failures and 14 not
useful**. Wrong proposals, fabricated tool outcomes, unsupported runtime or
whole-document absence claims, caps and missing literal status remain visible.
B10 synthesis sometimes repeats claims that tests are absent despite actual
passes, or misses a boundary case already tested. Supported facts and legitimate
coverage limits retain credit independently of formatting failures.

Selected duplicate-inclusive counts are **264 source failures, 1,085 instruction
failures and 54 not useful**. All 258 archived instruction sources verify; the
[isolated frozen validator](confirm-quality-checkpoint-12.json) accepts all 2,777
judgments. These are partial selected-cohort counts, not population error rates
or quality equivalence. Original objectives, timings, live quality and the first
B06 exclusion remain unchanged. Final confirmation and all-phase cap audits
remain pending.

The [historical partial confirmation review](confirm-review-index-13.json) includes
all twelve B10 native pairs and two B11 four-system blocks: **4,824 objective
finals, 2,542 automatic final failures, 76 capped intermediates and 3,062 selected
outputs / 341 complete contexts**. Fifty new contexts—39 investigation finals,
eight syntheses and three QA finals—were explicitly read. Exact repeated fields
and preceding dependencies are retained in [bindings](confirm-review-components-13.json).

The new group has **21 source failures, 45 instruction failures and zero not
useful**. Incorrect restrictions copied from unrelated tests, unsupported custom
type exclusions, duplicate/empty-input claims contradicted by tests and test
internals confused with implementation remain visible. Correct behavior and
narrow coverage/storage limits retain usefulness independently of missing literal
status. Three supported QA answers obey JSON instructions but differ from the
short exact references; those automatic failures remain recorded.

Selected duplicate-inclusive counts are **332 source failures, 1,244 instruction
failures and 65 not useful**. The [isolated validator](confirm-quality-checkpoint-13.json)
accepts all 3,062 judgments and all 288 archived instruction sources verify.
This is a partial selected-cohort audit, not a population error rate or quality
equivalence. Timings, objectives, live quality and first-B06 exclusion are unchanged.
Final confirmation and all-phase audits remain pending.

The [historical partial confirmation review](confirm-review-index-14.json) includes
all eight B11 four-system blocks and the first B12 block: **5,624 objective
finals, 3,191 automatic final failures, 78 capped intermediates and 3,720 selected
outputs / 398 complete contexts**. All 61 new contexts (58 investigation finals,
two capped intermediates and one QA final) were explicitly read and judged.
Fourteen supplied source strings match prior readings; exact full-field and
individual-dependency [bindings](confirm-review-components-14.json) preserve
those matches, with every new component and answer separately reviewed.

The new group has **18 source failures, 60 instruction failures and one not
useful**. Unsupported input exclusions, performance validation, infinite-loop
predictions and a unique-lock claim remain visible. Source-supported type,
coverage and memory limits retain credit. Both capped intermediates contain
useful source facts but exceed the intermediate word limit. A supported JSON
answer differs from the short exact reference, whose automatic failure remains.

Selected duplicate-inclusive counts are **440 source failures, 1,612 instruction
failures and 67 not useful**. All 316 archived instruction sources verify; the
[isolated frozen validator](confirm-quality-checkpoint-14.json) accepts all 3,720
judgments. These are partial selected-cohort counts, not population rates or
quality equivalence. Original objectives, timings and live quality are unchanged;
failed/partial B11 setup attempts and first B06 remain outside comparison credit.
Final confirmation and all-phase audits remain pending.

A [declared orchestration repair](confirmation-host-interruption-02.json) records
the tool-host interruption after 298 valid confirmation configurations / 112
matched blocks. The recorded coordinator and all workers were absent, with no
GPU compute clients and free reservations. The interrupted B11 arm was sealed
and verified; its partial comparison has no timing credit. An independent user
service resumes the complete B11 block and remaining cells through the unchanged
frozen CLI, in original order. Earlier complete blocks are retained, and the
activity-invalid B06 block is deliberately not retried. This is setup recovery,
not a capability failure or study-design change. Acquisition remains in progress.

The first independent service exposed a [tool-PATH setup error](confirmation-tool-path-repair-03.json):
its default environment could not find the existing ripgrep executable. Eight
attempts failed and a ninth was deliberately interrupted after partial serving;
all nine seals and cleanup verify, and none receives timing credit. The corrected
service inherits the verified tool PATH and checks ripgrep/pinned pytest before
GPU acquisition. The service environment probe and live coordinator PATH checks
pass. After repair, upstream and Continuum each completed all 80 requests and 48
tools with clean cleanup and verified seals; the complete replacement block was
still pending at that checkpoint. Affected B11 comparisons are replaced as whole
blocks through the frozen CLI; all earlier evidence and the B06 activity
exclusion remain unchanged.

The subsequent [whole-block recovery proof](confirmation-b11-replacement-03.json)
passes: all four systems complete **320 requests and 192 tools** in total, with
common workload units, valid measurements, cleanup and seals. The frozen selector
chooses the new block and excludes the original partial and PATH-failed attempts.
Qwen3-8B B06 remains at its original 32 attempts. At that checkpoint health had **300/524 valid
configurations and 113/210 matched blocks**, with no new issue beyond the retained
activity exclusion and setup attempts awaiting valid replacement. This proves
recovery of that block, not completion of confirmation or the campaign.

All **364 planned Qwen3-8B confirmation configurations have executed**.
The [terminal model inventory](confirmation-qwen8b-attempts-complete.json)
verifies every retained seal: **377 attempts / 35,530 files**, including original
partial and setup-failed arms. There are **363 valid logical configurations and
129 of 130 required matched blocks**. The first B06 activity failure remains the
sole missing anchor block; all 32 B06 attempts ran once, without an activity retry.
Three valid original arms remain unmatched after whole-block setup recovery and
receive no extra comparison credit. Secondary-model and later phases continue;
this is model execution completion, not a completed confirmation matrix.

The [latest partial confirmation review](confirm-review-index-15.json) covers all
twelve B12 blocks: **5,976 objective finals, 3,476 automatic final failures,
78 capped intermediates and 4,024 selected outputs / 429 complete contexts**.
All 32 new investigation finals were explicitly reviewed; their five supplied
source strings exactly match earlier readings, with every new dependency and
answer separately read. [Bindings](confirm-review-components-15.json) retain
full-field and individual-dependency matches.

The new group has **five source failures, 31 instruction failures and zero not
useful**. Wrong test_none evidence, denial of an explicit error message and
unsupported infinite/custom-iterable restrictions remain visible. Explicitly
documented deprecation and abort behavior receive credit. Selected
duplicate-inclusive counts are **459 source failures, 1,908 instruction failures
and 67 not useful**. All 360 archived instruction sources verify, and the
[isolated frozen validator](confirm-quality-checkpoint-15.json) accepts all 4,024
judgments. These selected counts are not population quality rates or equivalence.
Original objectives, timings, live quality and first-B06 exclusion are unchanged.
The [inventory validation](anchor-inventory-validation.json) also reverifies the
original source and execution bundle after audit-generated caches were moved
outside the frozen source. Final confirmation and all-phase audits remain pending.

See [continuation instructions](CONTINUE.md) for the active owner, exact paths,
remaining phases and enabled health monitoring.
