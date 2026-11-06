# Stage 3: operating range (evaluation in progress)

All listed cells were selected before evaluation. Completed results follow below;
cells without results remain pending. The intended use case
and selection rules are in [OPERATING_SCOPE.md](OPERATING_SCOPE.md); answer checks
were fixed in [INDEPENDENT_QUALITY_RUBRIC.md](INDEPENDENT_QUALITY_RUBRIC.md).
The candidate settings remain unchanged after Stage 1. Primary comparator is
no-hint ARC; upstream is the strongest measured equal-budget Stage 2 substitute.
Continuum remains in the serialized-tool cell because its measured returning
latency was better despite slower batch completion on the development workload.

| Cell | Inputs and change | Repetitions per system |
| --- | --- | --- |
| Repository | Sixteen held-out more-itertools source/test investigations | 3 |
| Repository, cohosted memory | Same held-out tasks; half original GPU and CPU KV bytes | 3 |
| Documents | Sixteen held-out PEP search/parse/reference-check workflows | 3 |
| Documents, cohosted memory | Same held-out tasks; half original GPU and CPU KV bytes | 3 |
| Ample cache | Original tasks; CPU KV 3.5 GiB, four server sequences | 3 |
| CPU pressure | Original tasks; CPU KV 2.40625 GiB | 3 |
| Concurrency | Original tasks; twelve server sequences | 3 |
| Serialized tools | Original tasks; one real subprocess worker | 3 |
| Mixed traffic | Original tasks; eight hinted and eight explicitly unhinted sessions | 3 |
| Finite churn | Six sequential sixteen-session waves; mixed hints; staggered admission | 1 exposure |

Independent families retain the original full-budget result even if all useful
prefixes fit, plus one predeclared combined cohosting constraint. Do not pad
inputs or search capacities until a benefit appears. Fresh reserve tasks are
available if held-out findings drive a change. The chosen authored tasks are not
a sample of production agents and no weighted deployment-wide gain is defined.

The two public source revisions and all executed input hashes are recorded in
`output/wait-qualification-20261105/independent-source-identities.json`; each run
also archives its exact external inputs. Repository tools execute actual tests;
document tools compare parsed metadata with independently frozen references.
Live answers and output work may differ between systems. All offered work and
quality failures must remain in the result, without survivor-only speed claims.

Mixed groups omit local hint RPCs and published native retention metadata for
unhinted sessions, keeping shared continuation priority. Group workflow time,
progress, and descriptive returning-latency tails supplement batch timing.
The finite churn exposure has a predeclared 512 MiB summed-server-RSS growth
bound between first and final wave endpoints. It does not establish saturation,
unlimited bookkeeping bounds, a long-run failure rate, or production stability.

Raw manifests: `output/wait-qualification-20261105/stage3-*-spec.json`.
Selection receipt: `stage3-substitute-selection.json` in the same directory.

## First full-budget repository checkpoint (not the final estimate)

First triplet completed: ARC 66.32 s, wait 66.86 s, upstream draft 74.85 s.
The initial 0.815% wait loss to ARC was flagged and diagnosed before advancing
to another regime. Neither local arm had returning-prefix CPU evictions;
computed prompt tokens were 45,188 and 45,192, despite positive wait refresh
activity. Hint work cannot save evicted-prefix recomputation that did not occur.
The nominal overhead is inside the predeclared 2% neutral target, but one pair
does not establish that bound or resolve timing variation. Keep all fixed
repetitions and the separately predeclared half-byte cell; no policy retuning.
Detailed diagnostic receipt: `stage3-repository-initial-diagnosis.json`.

All 48 first-repetition final answers have an explicit source audit recorded in
`independent-manual-review.json`. Eleven wait finals exactly match their ARC
counterparts; their existing source audits are reused explicitly. Every answer
names the requested subject and correctly reports the actual passing tests, but
all conditions contain unsupported or contradicted content. Examples include
claims that `replace(count<0)` raises ValueError, `ichunked(n=0)` raises an error,
or that explicit zero-cache/inexact-step/duplicate-element tests do not exist.
The actual sources contradict these claims. Source-audit facets remain separate
from the frozen narrow scorer and no bad answer is excluded from timing. The
review is by Codex, not an independent human evaluator. This workload cannot
support a general task-quality or successful code-analysis claim.

## Complete initial full-budget repository set

All nine cases passed (576 responses, 432 real tools) and verified. ARC batch
times were 66.32, 66.06, 66.14 s; wait 66.86, 66.87, 66.18 s; upstream
74.85, 73.85, 74.22 s. No local case had returning-prefix CPU eviction pressure.
Wait's mean batch overhead was 0.70%, conditional 95% interval −0.76 to +2.19%.
This is a retained no-benefit result; the interval narrowly exceeds the 2%
neutral-overhead target. A **separate fixed-six ARC/wait confirmation** is selected
under the predeclared inconclusive-result rule, before constrained-cell results.
No input/policy change, pooling, or optional extension is allowed. Its objective
is overhead precision, not manufacturing a gain where there is no useful eviction.

Wait used 10.32% less batch time than the configured upstream draft (conditional
interval 8.42–12.18%), with 7,081–7,115 output tokens versus upstream's 7,475.
ARC used 7,045–7,115 tokens. The quality audit and differing output work prevent
describing this as equal-quality or isolated-policy superiority. All local paging
was small (at most about 1.1 MiB swap-out per case); gross host paging does not
explain away the observed small ARC loss. Compact summary/quality reports retain
all rows and raw responses.

## Fresh full-budget neutral confirmation

All twelve fresh ARC/wait cases passed and verified (768 responses / 576 real
tools). Every local case again had zero returning-prefix CPU evictions. Wait's
estimated overhead was **0.46%**, conditional paired 95% interval
**−0.16 to +1.08%**. This resolves the predeclared 2% neutral-overhead bound for
these frozen tasks/configuration, while establishing no useful timing gain.
Keep the original three-pair set separate. No further repetitions are selected.
All 32 first-repetition final answers exactly matched previously source-audited
answers; their audits were explicitly reused, bringing first-repetition coverage
to 80 finals including the initial three-system set. The same answer-quality
limitations remain. Raw result: `stage3-repository-neutral-confirm`.

The separately frozen repository cohosting cell follows unchanged: GPU/CPU byte
budgets halved together, same inputs, candidate, alternatives and scheduling.

## Instruction repair and fresh validation rule

The original repository instructions repeatedly elicited invented missing-test
claims and extra commands, across all systems. Preserve their timing and audits.
After the unchanged cohosting campaign, calibrate an explicit final-answer
instruction on four already-seen tasks (`replace`, `ichunked`, `seekable`,
`bucket`) in one ARC case. Calibration is for factual completion, never serving
settings or performance selection. It asks for four labeled lines, one supported
behavior, actual test status and one source-established input/resource/precision
boundary, explicitly forbidding invented defects and missing-coverage claims.

If adopted, freeze the same instruction for all systems on the untouched sixteen
repository reserve tasks. Compare ARC, wait and the same upstream draft in the
already-selected half-byte deployment geometry, three rotating repetitions. No
capacity or policy tuning, exclusion of incorrect answers, or pooling with original
evaluation/development data. Review every first-repetition final against actual
source and tools using the original factual rubric. The original full-budget
neutral result still applies to its original instructions. Document-family
instructions stay frozen unless their own actual evidence requires repair.

## Complete original-instruction cohosting set

All nine cases passed and verified (576 responses / 432 actual tools). Wait used
**9.77% less batch time than ARC**, conditional paired 95% interval **1.82–17.08%**.
All three pairs favored wait; ARC was 137.42/137.17/136.94 s and wait
121.24/121.72/128.49 s. ARC recomputed 156,865 prompt tokens each time and restored
zero CPU tokens; wait recomputed 116,198/114,870/106,467 and restored
40,672/41,936/50,400. The third wait case saved more computation but took longer,
so timing and retained tokens must not be presented as proportional. Host swap-out
was small in local cases (at most 26.6 MiB), with differing live output work.

Upstream was 122.92/123.61/122.59 s. Its estimated batch advantage over wait was
0.59%, with a wide interval −8.59 to +9.00%; this cell does not establish a winner
or the 3% competitiveness bound. Do not pool it with Stage 2 confirmation.
All 48 first-repetition final answers were source-audited, including explicitly
reused exact answers on identical frozen inputs. Unsupported/contradicted content
occurs in every condition, so the timing gain is for all offered workflows and
does not establish equal-quality code-analysis success. Fresh reserve validation
will test the revised application instruction; old outcomes remain development
evidence for that revision. No serving-policy change has been made.

The first four-task instruction calibration completed all 16 responses / 12 tools.
It eliminated the previously observed missing-test inventions, but two finals used
two semicolon rows instead of four labeled lines; `bucket` gave no concrete
limitation and `seekable` used overbroad "any index" wording. Preserve this attempt
as quality development only. Exactly one literal-template correction on the same
seen tasks is selected, with no further calibration extension until perfect
answers. Fresh reserve validation will retain remaining shortcomings under the
original factual rubric rather than silently changing its acceptance criteria.

The literal-template calibration completed 16 responses / 12 actual tools. All
four finals used four concise lines, gave a supported behavior and correct test
outcome, without invented defects. Two gave concrete supported boundaries; two
honestly said no limitation was established, which remains a missing required
element under the original rubric. Adopt this instruction unchanged for the
fresh reserve, with no further calibration or serving adjustment. The preflight
passed all 48 actual reserve tool commands before serving. Exact instruction,
corpus/spec/helper hashes are frozen in `stage3-reserve-premeasurement-manifest.json`.

Fresh reserve first-pair checkpoint (not final estimate): ARC 60.51 s, wait
56.46 s (6.69% less time). Computed prompt tokens 63,429 versus 54,773; restored
44,512 versus 54,416. Both completed 64 responses and 48 tools. 58/64 prompt-ID
sequences and 61/64 output-ID sequences match. Initial `interleave_evenly` and
`stagger` responses differ and affect later histories; the final `run_length`
answer differs only by capitalization/punctuation. The 16 final-answer source
audits therefore have the same facets: fifteen supported behaviors, one incorrect
`spy` statement (returned iterator includes the head, rather than just the rest),
and nine missing concrete limitations. Fifteen finals match exactly and the last
was explicitly reviewed. This improves instruction fidelity but does not turn
missing/incorrect elements into full task success. Finish fixed repetitions and
review the upstream first repetition before interpreting the reserve comparison.

The fresh reserve upstream first repetition completed in 61.12 s. Its sixteen
finals were also source-audited: the same incorrect `spy` behavior, eleven
missing concrete limitations, and an incorrect claim that default `scalar_types`
restricts all scalars to strings/bytes (`zip_broadcast` also broadcasts integer
scalars). All three systems accurately reported the passing tests and named the
subjects. These are descriptive first-repetition facets, not a scalar grade or
proof of quality equivalence in all repetitions. All 48 first-repetition reserve
finals are now accounted for.

## Fresh reserve validation, complete

All nine fresh-reserve cases passed and verified: 576 responses / 432 actual
tools, unchanged serving settings and fixed half-byte geometry. ARC batches were
60.51/60.37/60.39 s; wait 56.46/54.54/55.30 s. Wait used **8.27% less batch time**
(conditional paired 95% interval **4.49–11.89%**) and improved mean offered
first-token time by 0.89 s. ARC recomputed 63,429 prompt tokens in every run; wait
54,773/49,791/51,423. CPU-restored tokens were 44,512 for ARC and
54,416/58,144/56,512 for wait. This supports useful independent pressure coverage
after the application-instruction amendment, without pooling original evaluation.

Wait used 8.79% less batch time than the configured upstream draft (5.98–11.52%),
with 5,093–5,097 output tokens versus upstream's 5,366 each time. This is a
configured-system result with differing work, not isolated retention-policy
superiority. ARC output was 5,093 every time. Local swap-out was at most about
33 MiB, and at most 2.8 MiB in wait cases. All quality/reference/word/status
reports retain every answer. The audited missing/incorrect elements above prevent
a general successful-code-analysis or comprehensive equal-quality claim.

Mechanical review integrity passes for all 176 independent-family first-repetition
finals across eleven reviewed cases, including 91 exact-answer reuses on identical
frozen corpora. This verifies response identity and review bookkeeping, not the
author's factual judgments. Receipts: `manual-review-integrity.json` and
`independent-manual-review.json`. Document-family evaluation follows unchanged.

## Document instruction amendment, selected from first ARC source audit

The original full/half document cells stay unchanged through all fixed repetitions.
The first ARC finals contain incorrect limitations: PEP 498 conflates the allowed
`!=` expression operator with a backslash exception; PEP 451 wrongly retains
loader boilerplate responsibilities transferred by ModuleSpec; PEP 492 claims
missing compatibility/implementation discussion despite explicit sections; PEP 526
repeats a type-comment motivation as a limitation of the new syntax. Preserve these
and all other answers. Source/tool audits remain separate from timing eligibility.

After both original document cells, select exactly one ARC quality calibration
on already-seen PEPs 451/492/498/526 using a literal six-line template (PEP,
Status, Type, rule, actual test outcome, source/check limitation). This preserves
the original factual rubric, including its explicitly allowed metadata-only check
boundary. The instruction forbids inferring implementation support from Status.
If no document restriction is established, it may accurately state that metadata
checks do not establish implementation correctness. No serving changes or further
calibration extension. Freeze the same instruction for all systems on the sixteen
untouched PEP reserve tasks; validate three rotating ARC/wait/upstream cases in
the already-declared half-byte deployment. No capacity search or pooling with old
evaluation/development/calibration. Exact template and selected seen-task spec
are frozen in `stage3-documents-quality-calibration-premeasurement-manifest.json`
before fresh reserve outcomes.

## Original-instruction document family, full budget complete

All nine cases passed and verified (576 responses / 432 actual search/parser/check
tools). ARC batches were 143.45/143.75/143.95 s; wait 112.92/112.05/112.16 s. Wait
used **21.81% less batch time** (conditional paired 95% interval **20.67–22.93%**)
and reduced mean offered-to-first-token time by 3.29 s. ARC recomputed 126,240
prompt tokens per case; wait 74,266/74,487/74,593. Restored tokens were 81,568 for
ARC versus 133,584/133,360/133,264 for wait. Local swap-out was at most about
3.8 MiB; broad host paging does not explain away this result.

Wait used 5.42% less batch time than upstream draft (4.20–6.63%),
with 9,578–9,595 output tokens versus upstream's 9,316–9,342. ARC used 9,683.
This is a configured-system outcome with different work and the audited factual
limitations, not isolated-policy or equal-quality reasoning superiority.
All 48 first-repetition document finals are source-audited; 224 independent-family
finals across fourteen cases now have mechanically verified review records.
The frozen narrow scorer missed three clearly stated first-repetition PASS results
(ARC PEP484/526, wait PEP484); manual adjudications are separate and original
grades remain unchanged.

Workload shape matters: in the first ARC case, actual tool calls averaged 0.049 s
(p95 0.072 s), while tool-ready-to-first-token time averaged 17.08 s. The gain
includes retaining queued continuations; tool-only and return-queue-only advice
have not been isolated. Initial mean prompt length was about 2,856 tokens and
final mean about 4,019; these are authored always-returning pipelines, not a
sampled production return distribution. Context/tool/queue details are retained
in `stage3-documents-shape.json`. The fixed half-byte document cell follows.


## Half-byte document checkpoint (ongoing)

First pair: ARC 228.98 s, wait 217.46 s; 5.03% less time for wait,
not a final estimate. Both have native preemptions (10 / 15), so counters
with different request scopes must remain separate. ARC first-admission CPU hits
are zero despite 473,432,064 physical demand-load bytes; wait loads
3,775,528,960 bytes. Scheduled native computation is 211,650 / 177,156 tokens.
The terminal-episode load-attribution gap is retained rather than forced to zero.
Upstream first case is 217.87 s. Finish all three fixed rotations.

All 48 first-repetition finals are source-audited, now 272 independent-family
finals across 17 cases. Six upstream answers exactly reuse existing audited
answers on the identical corpus; ten new texts were checked. Shared failures
include false loader-responsibility and unsupported implementation-support
claims, an omitted optional epoch in PEP440 grammar, and generated extra commands
and overlong answers. Metadata outcomes remain correctly represented manually;
this does not establish equal comprehensive task quality. Review identity and
reuse bookkeeping passed (`manual-review-integrity.json`); factual judgments
remain author audits. Raw final answers and frozen narrow grades are unchanged.


ARC regression alarm, tight document repetition 2: wait 241.4487 s versus ARC
228.1500 s (approximately 5.83% overhead), after repetition 1 favored wait by
5.03%. Hold dispatch of new regimes pending diagnosis; finish the frozen current
three rotations rather than discard the negative pair. Second wait has 18 native
preemptions, 168,613 scheduled computed tokens and 4,980,211,712 physical load
bytes; it saves computation but has a much later tail. Host swap-out is zero.
Long inter-token gaps and lower whole-run GPU utilization are symptoms, not
proven causes or evidence of thermal throttling. New instructions/capacities
will not be selected to hide this cell.


Bounded diagnosis selected before the third original wait measurement: exactly
three rotating runs each of ARC, inactive hints and active hints, same tight
geometry/source/instructions/continuation priority. Use the existing tested
Stage1 no-op (validate ready keys but skip only ARC queue mutation). The native
step trace shows second wait has fewer large-token prefill intervals but longer
small-token intervals; these include async overlap and are not an exclusive
time decomposition. The new control tests machinery cost versus retention/restore
interactions. No capacity/retention tuning, no pooling, no indefinite extension.
This uses already-observed tasks as a diagnostic, not fresh held-out validation.
Frozen selection: `stage3-documents-pressure-control-premeasurement.json`.
The original current three rotations will complete first.


## Tight document original comparison, complete (adverse/inconclusive)

All nine cases passed and verified: 576 responses /432 actual tools. ARC
228.98/228.15/228.28 s; wait217.46/241.45/232.02 s. Wait geometric batch
overhead is **0.71%**, with time-reduction interval **-15.38 to+12.09%**.
The first pair favored wait5.03%; the next two lost5.83% and1.64%. This
configuration meets neither the primary gain gate nor a resolved neutral
overhead bound. Mean offered first-token time is0.11 s lower for wait despite
its slower batch tail; primary batch time governs. Upstream is217.87/217.51/
217.89 s; wait overhead versus it5.66%, wide interval. No competitiveness
claim for this cell. Do not pool it with the full-budget21.81% gain.

Wait native computed tokens177,156/168,613/173,978 (check raw summary scopes)
are lower, but transfers/preemption and small-token step intervals grow. The
step-interval analysis is descriptive and includes overlap, not exclusive kernel
profiling. Root cause remains unisolated; direct inactive-hint diagnosis starts
next. All48 first-repetition finals remain audited with quality failures.
The48 real reserve-document tool commands separately passed at this quiet
boundary; no reserve serving outcomes have been measured.


Pressure diagnostic first triplet (interim): ARC 229.06 s, inactive222.43 s,
active216.02 s. Inactive generated9,254 output tokens versus ARC9,410,
with slightly more native computation212,959 versus211,650. Its faster
batch is not proof of negative hint overhead: output work and arrival/scheduling
change, even without queue mutation. It has zero refreshed CPU keys as intended.
The active run's first-pair gain is5.69% versus ARC and2.88% versus inactive;
finish the fixed remaining rotations. All48 first-triplet finals audited;
320 independent-family finals across20cases,174 exact reuses, mechanical
integrity passed. The only new active text has a qualified PEP567 boundary: no
direct current Context reference/read-only Mapping, but ContextVar.set can
modify active values without Context.run. Missing/incorrect old elements remain.


Recurring active-tail variation: second inactive224.17 s, second active230.93 s
(before its paired ARC). Select exactly one additional operating-configuration
diagnostic before the last active outcome: three paired ARC/wait cases, same
half-byte memory/source/instructions, max running sequences reduced8->4 for both.
This tests the plausible GPU admission/preemption interaction as a deployable
mitigation; it changes no retention code or memory budget. No sequence/capacity
search, no pooling or automatic extension. The original8-sequence adverse cell
remains visible and unqualified. Frozen selection in
`stage3-documents-admission-diagnostic-premeasurement.json`. Finish the existing
three-condition diagnostic first. Fresh document reserve remains unchanged at
its previously selected8-sequence geometry; do not silently replace it.


## Tight document inactive-hint diagnosis, complete

All nine cases passed and verified: 576 responses /432 actual tools. ARC
229.06/229.06/228.30 s; inactive222.43/224.17/223.46 s; active216.02/230.93/
226.80 s. Active uses **1.88% less batch time** than ARC, with conditional
95% interval **-7.04 to+10.07%**: neither a useful-gain gate nor a resolved
neutral-overhead bound. Versus inactive, active overhead is0.51% (reduction
interval-8.47 to+6.86%). Inactive is2.38% faster than ARC (1.28–3.47%),
but generates shorter/different output and changes arrivals/scheduling; this is
a net live-system comparison, not negative exclusive hint-processing cost.
It correctly refreshes zero CPU keys. No retention benefit is established by
this diagnostic at the8-sequence tight geometry. The original adverse triplet
remains separate and unchanged; neither set substitutes for the other.

All48 first-triplet finals remain source-audited (320 cumulative finals/20cases,
174 exact reuses on identical frozen corpora). Raw format misses are retained
with separate explicit-PASS manual adjudications. Scope-specific native-compute/
terminal-episode/physical-transfer fields now appear in the summary, verified
against all nine raw original cohost cases without changing comparisons.

The previously frozen single-factor4-sequence ARC/wait diagnostic is active.
It tests a practical pressure mitigation, not a claim that the exact root cause
has been isolated. No further capacity/concurrency search has been selected.
After this finite check, continue the frozen quality/reserve/regime plan; this
one pressure geometry does not make the successful mechanism globally unworthy.


Fresh document range selection, before any reserve serving: the original full
setting also needs fresh validation after the instruction repair. Select both
full and already-selected half budgets on the same untouched16PEPs, each
with three rotating ARC/wait/upstream cases. Both are mandatory regardless of
results, no pooling or memory search. Full uses the original2560 GPU blocks/
2,818,572,288 CPU bytes; half remains1280/1,409,286,144 at8sequences.
This validates both original settings rather than treating the favorable old
full result as held-out success after using its answers for instruction repair.
`stage3-documents-reserve-range-selection.json` records selection now; after the
single seen-task calibration, freeze both actual specs before either reserve run.
No4-sequence reserve cell is selected by this amendment.


## Four-sequence tight-document mitigation, complete

All six cases passed and verified (384 responses/288 actual tools), changing
only maximum running sequences8->4 for both systems at the same half-byte
budgets. ARC236.51/237.06/236.41 s; wait213.96/208.95/221.29 s. Wait uses
**9.29% less batch time** (conditional paired95% interval **2.23–15.84%**);
all three pairs favor wait. Native computation is212,064 each ARC case versus
163,755/159,351/172,547 for wait. Outputs9,621 each ARC versus
9,521/9,535/9,720 for wait; live work is not byte-identical.

ARC has zero preemptions in all three; wait0/0/2. The configuration mitigates
the adverse behavior but does not eliminate every preemption or prove a unique
root cause. CPU competition remains (all48 returning baseline requests have
prior eviction), so this is not merely a no-reuse/neutral case. The exact
8-sequence original and inactive-control negatives remain separate and
unqualified. This is an already-seen development configuration diagnostic,
not fresh held-out qualification or a comparison against a4-sequence upstream
contender. Do not imply4 sequences is universally sufficient for8,192 contexts.

All32 first-pair finals are source-audited, bringing cumulative independent-family
review to352 finals/22cases with199 exact reuses; identity verification passed.
Shared unsupported facts/length failures remain. A single seen-PEP instruction
calibration now follows. Both original full and half byte geometries will be
frozen on the untouched reserve before either fresh document comparison starts.
No further concurrency/capacity search is selected. The mechanism remains worth
continuing; the adverse8-sequence geometry stays an explicit operating limit.


## Single document instruction calibration and reserve adoption

The four already-seen PEP calibration tasks completed and verified. All four
finals use the six literal lines, correct document metadata and observed PASS,
and fit70 words. Source audit retains the remaining limitations:451 provides
no concrete boundary;492 incorrectly implies absent implementation information;
498 quotes the globals/locals heading (implementation scope, not a ban on
explicit caller use) and correctly excludes binary f-strings;526 correctly
notes that bare annotations do not initialize values. No further calibration
was run. The single template is adopted unchanged, without a general answer
quality claim. The original rubric is unchanged.

Both fresh document reserve geometries were frozen before either execution:
the original full bytes and original half bytes, each at8 running sequences,
three rotated ARC/wait/upstream repetitions on the same16 untouched PEPs. Both
are mandatory regardless of result; no pooling, capacity search or selecting
the favorable cell. Selection/adoption/spec hashes remain in the linked raw
campaign directory (`documents-calibration-adoption.json` and
`stage3-documents-reserve-range-premeasurement.json`). Half-byte validation
is now running, followed by full bytes and all remaining original regimes.

Fresh half-budget reserve first pair: ARC220.304150110s versus
wait217.145287376s (1.43% less time, one pair only, below2% point target).
Native scheduled prefill224,772 versus185,695; preemptions16 versus9;
outputs8,358 versus8,260. No decision from this interim pair. All32 finals
are source-audited; wait10 exactly reuse the ARC audit. Both retain many
missing/unsupported limitations; wait's new PEP680 assertion that tomllib is
unavailable in Python3.11 contradicts that PEP's actual3.11 version field.
No answer is excluded. Upstream and two remaining pairs continue unchanged.

Fresh half-byte reserve second paired ARC loss alarm: wait221.812055932s
versus ARC220.980467467s (0.3763% overhead). Wait schedules183,272 native
prefill tokens versus224,772, has13 versus16 preemptions, but loads
3,641,573,376 versus41,287,680 physical CPU bytes. Output8,239 versus8,365.
The signed loss is small and two-pair direction remains unresolved; cache-token
savings alone do not establish a batch win. Host/transfer receipt:
`documents-reserve-second-pair-alarm.json`. No serving changes or further
capacity/concurrency search; finish the frozen third pair, retain this adverse
cell, then the already-selected full-byte validation. Earlier4-sequence
mitigation remains development, not a substitute. Separate gains justify
continued mechanism work under the user's explicit blocking criterion.


## Fresh half-budget document reserve, complete

All nine rotated cases passed and sealed verification/aggregate checks passed:
576 responses/432 actual tools. ARC220.304/220.980/221.042s;
wait217.145/221.812/216.942s; upstream216.640/217.307/216.961s.
Wait uses0.97551% less batch time versus ARC, conditional95% interval
**−2.00461 to+3.86856%**. This fails the useful2% point target, does not
resolve a gain, and narrowly fails to establish the2% neutral-overhead bound.
It is not a repeatable useful win. Do not round the adverse interval into
a margin pass. The second pair's0.3763% ARC loss is alarmed and diagnosed
above; paging is small, with no unique root cause established.

Upstream uses1.72390% less time than ARC (1.46049–1.98661% interval), below
the2% useful point target. Wait's actual time overhead versus upstream is
0.76152%, with conditional interval−2.01915 to+3.62111%; the3% secondary
competitiveness margin is unresolved here. Different live outputs remain:
ARC8,358–8,365; wait8,239–8,267; upstream8,019 each. Native local
preemptions ARC16/16/14 versus wait9/13/16. Mean offered FTL favors wait
by0.645s, but does not change the batch-time decision.

All48 first-repetition finals are source-audited, including unsupported
implementation-absence claims and a new false wait PEP680 version claim.
The upstream PEP585 final overstates implementation validation by the actual
metadata-only checks; PEP701 grammar permits nesting but does not guarantee
unbounded implementation depth. No equal-quality useful-task win is claimed.
Cumulative source-review coverage is400 finals/25cases; mechanical audit
identity verification passed.

No extra six-pair confirmation or further tuning is selected for this already
diagnosed tight8-sequence geometry. The finite comparison is inconclusive/
under-target, retained as an operating limitation. Other causal/live/fresh
repository evidence still makes mechanism work worthwhile. The already-frozen
full-byte reserve follows regardless of this outcome; the remaining original
regimes and Stage4 remain mandatory. No pooling or favorable-cell substitution.

Fresh full-byte reserve first pair: ARC146.225655100s versus
wait120.734447809s (17.43% less time, one pair only). Native scheduled
prefill139,020 versus89,569; output8,275 versus8,370; bothzero preemptions.
Physical CPU demand loads4,473,749,504 versus7,309,754,368bytes. Thus more
CPU transfer bytes alone do not explain the tight-cell loss; the interaction
with GPU occupancy/admission/scheduling remains a hypothesis. All32 first-pair
finals audited (28 exact prior reuses), shared unsupported/missing limitations
retained. Cumulative coverage432 finals/27cases; integrity verification passed.
No final decision before the three fixed pairs finish.


## Fresh full-budget document reserve, complete

All nine rotated cases passed and sealed artifact/aggregate checks passed:
576 responses/432 actual tools. ARC146.226/145.962/146.190s;
wait120.734/119.329/118.906s; upstream119.254/118.777/122.470s.
Wait uses**18.11592% less batch time than ARC**, conditional95% paired
interval**16.54800–19.65437%**. All three pairs favor wait, meeting the
predeclared narrow timing gate. Mean offered first-token latency is3.856s
lower. Local ARC/wait have zero preemptions throughout. Native scheduled
prefill ARC139,020/139,020/139,868 versuswait89,569/88,111/87,214;
output ARC8,275 each versuswait8,370/8,260/8,306. Low but unequal host
paging and live output differences remain in the reports.

Secondary is unresolved: wait's time ratio versus upstream is0.995822
(about0.42% less time), conditional interval0.942184–1.052514. Neither
a repeatable lead nor the3% competitiveness margin is established in this
three-pair cell. Upstream's mean offered FTL is0.292s lower than wait.
Do not transfer the original-workload six-pair competitiveness bound to this
new family. No extra repetitions are selected; report this precision limit.

All48 first-triplet finals are source-audited, including shared unsupported/
missing limitations. Some individual answers improve (upstream PEP701 nesting
boundary), others remain incorrect; no general equal-quality useful-task win
is established. Cumulative source-review448 finals/28cases, identity checks
passed. No answers are removed from batch timing.

This is fresh validation on the untouched reserve after one seen-task
instruction calibration; both full and half byte geometries were frozen
before either reserve ran. The strong full result and under-target/adverse
half result remain separate, without pooling or a favorable-cell substitution.
They support continuing the mechanism within a conditional operating scope,
while the eight-sequence tight-GPU geometry remains unqualified for benefit.
All remaining original regimes now follow, beginning with ample cache.


## Remaining original regimes, underway

The preselected ample-cache cell (3.5GiB CPU KV, four running sequences) is
active. Its first ARC case completes122.582624698s, zero preemptions and
zero returning requests with prior CPU-prefix eviction. CPU restores occur,
but no eviction-related recomputation opportunity is observed; retained CPU
copy activity alone is not evidence of work saved. This is a legitimate
neutral-overhead check. No settings or source changes were made for this
result. Constrained CPU, concurrency, serialized-tool, mixed and finite
six-wave cells remain mandatory, then all available Stage4 execution.

Ample-cache second paired loss alarm: wait118.604750s versus
ARC118.260457s (0.29113% overhead). Bothzero returning-prefix CPU
evictions; native scheduled prefill52,406 versus52,419; output7,508
versus7,657. The first two paired point estimates average essentiallyzero
difference, and the neutral-overhead bound awaits the final fixed pair.
Host/transfer receipt: `ample-second-pair-alarm.json`. No tuning; this
valid low-opportunity condition is not evidence of mechanism unworthiness.


## Ample-cache initial comparison complete; neutral confirmation selected

Nine cases/576responses/432tools passed and offline verified. ARC
122.583/118.260/118.536s; wait122.239/118.605/122.312s. Wait has
**1.05418% time overhead**, conditional95% interval−3.43648 to+5.75368%.
The useful-gain gate and2% neutral-overhead bound are unestablished. The
third pair loses3.18536% (second pair0.29113%): both alarms are retained.
All six local cases have zero preemptions and zero returning-prefix CPU
eviction; scheduled prefill52,405–52,419. Different live answers/critical
paths and modest host paging remain. Last wait summary-overlap final has
143tokens versus ARC88; final timing/CPU/tool/transfer details are in
`ample-final-regression-diagnosis.json`. No unique causal explanation is
established, and no exclusive hint CPU cost is inferred.

The predeclared separate fixed-six option is selected NOW, before any new
confirmation outcomes: twelve ARC/wait cases, alternating order, exact same
3.5GiB/four-sequence settings and original workload. No tuning or pooling
with initial three pairs; decide neutral scope from the fresh set. No
further extension selected. Selection and input hashes are in
`stage3-ample-neutral-confirm-selection.json`. This is a finite precision
check for an actual ARC-loss alarm, not running until a favorable result.
Remaining regime expansion resumes after this bounded check. Separate
strong causal/live/fresh-family gains keep further mechanism work worthwhile.

Initial wait/upstream time ratio0.995168 with95% interval0.973106–1.017731
is within the3% secondary margin on this low-pressure cell; no resolved
superiority or general task-quality equality. Upstream is not included in
the fresh neutral confirmation because the remaining question is ARC overhead.


## Ample-cache fresh neutral confirmation complete

All twelve cases/768responses/576real tools passed and offline hash/aggregate
verification succeeded. The separate six-pair estimate is **0.25270% wait
time overhead**, conditional95% interval−1.33908 to+1.87017%. It establishes
the predeclared2% neutral-overhead bound, with no resolved speedup. Do not
pool it with the original three pairs. All twelve local cases have zero
preemptions and no observed returning-prefix CPU eviction; native scheduled
prefill is52,396–52,419tokens. This is an overhead result without observed
eviction-related opportunity, not retained useful work saved.

Three individual ARC-loss alarms remain:1.39820%,0.89788% and2.31606%
(repeats2,3,5); the other three favor wait. Native work/output/paging/CPU
covariates attach to `stage3-ample-neutral-confirm-pair-alarms.json`. No
unique causal hint-processing cost is established. The final pair has nearly
identical output lengths and little swap-in, so those alone cannot dismiss
its loss. The full fixed-set interval, not selecting its favorable pairs,
supports the conditional neutral bound. No policy tuning or further
repetitions selected. Remaining constrained-CPU, concurrency, serialized-tool,
mixed and finite-churn cells proceed; Stage4 remains mandatory.


## Reduced CPU capacity

The preselected2,583,691,264-byte CPU cell completed nine cases/576responses/
432real tools and offline hash/aggregate verification. ARC batches are
115.706/115.921/117.396s; wait103.959/106.912/101.259s. Wait uses
**10.59060% less batch time**, conditional95% interval2.77085–17.78144%,
with all three pairs favorable and mean offered first-token latency1.195s
lower. Native scheduled prefill is82,712/89,571/89,698 for ARC versus
61,416/64,399/59,608 for wait. ARC preemptions0/0/6 versus wait0/0/0
and different generated output lengths are retained; do not call this
an isolated fixed-output policy effect. First ARC and wait both have21
returning requests with prior CPU eviction, showing that a request-level
eviction count alone does not measure how much recomputation is saved.

The upstream draft is11.77% faster than ARC in this cell. Wait/upstream
time ratio1.01337194 has conditional95% interval0.94334387–1.08859847: a
1.34% wait-overhead point estimate, with neither secondary lead nor the3%
competitiveness bound established. No extra repetitions or tuning selected
for that secondary question. Local/upstream paging differs, all raw work
and answer-check limitations remain, and original-workload quality scope
is unchanged. The primary ARC result supports continuing the remaining
preselected regimes. See `cpu-pressure-decision.json` and the sealed run.


## Twelve-sequence concurrency

Nine cases/576responses/432real tools passed and offline hash/aggregate
verification succeeded. ARC102.255/103.814/102.325s versus wait
100.980/98.946/100.599s: **2.55320% less batch time**, conditional95%
interval−2.23805 to+7.11991%. All three paired points favor wait, but neither
the useful-gain gate nor the2% neutral-overhead bound is established. No
additional confirmation or tuning is selected; report this cell as
inconclusive rather than rejecting the mechanism or advertising a win.

All six local cases have zero preemptions. ARC native prefill
61,812/65,221/64,153 versus wait55,109/55,842/54,822; output totals
7,466/7,492/7,432 versus7,458/7,488/7,459. First ARC peak running reaches12
and11returning requests have prior CPU-prefix eviction, so the selected
concurrency is exercised and opportunity exists. Output/work and unequal
paging remain; original-workload answer checks do not grade all reasoning.

Wait/upstream time ratio0.997013 has conditional95% interval
0.965591–1.029458, establishing the3% secondary competitiveness bound for
this cell without a resolved lead. Mean offered first-token latency is
0.163s lower than ARC and0.545s lower than upstream. This is a conditional
configured-system comparison, not isolated cross-runtime policy superiority.
See `concurrency-decision.json`; remaining serialized-tool, mixed and
finite-churn cells remain mandatory before Stage4.

### Serialized tool execution

The preselected one-worker tool regime completed twelve verified cases (768 model
responses / 576 real tools), including Continuum because its returning-request
latency was strongest in the original comparison. Wait uses **3.65% less batch
time than ARC** (conditional95% interval −1.27 to+8.33%), all three pairs favorable.
A useful gain remains unresolved; the2% neutral bound is met in this cell.
Upstream uses **5.59% less time than wait** (2.17–8.89%) and Continuum **10.32%
less** (7.25–13.30%). These secondary gaps are measured workload limits, not noise
or a universal lead. The first Continuum pair generated more output than wait
(7745 versus7495 tokens) and had lower returning FTL (3.773 versus6.854seconds),
so shorter answers alone cannot explain that result. Different live outputs,
runtimes, scheduling, chunk sizes and backend paths remain. No pair lost to ARC;
no extra repetitions or policy optimization selected. [Decision](../../output/wait-qualification-20261105/serialized-tools-decision.json).

### Mixed hinted and unhinted traffic

Nine verified cases completed576responses/432tools, with eight hinted and eight
unhinted workflows. Wait uses **3.28% less batch time than ARC** (conditional95%
interval −0.26 to+6.70%), all three pair points favorable; useful gain remains
unresolved, within2% neutral-overhead bound. Hinted mean workflow ratio is0.95660
(interval0.89953–1.01729); unhinted0.98133 (0.95048–1.01319). Both upper intervals
remain within the predeclared5% regression margin. All offered workflows completed;
no paired group returning-FTL p95 regression exceeds the descriptive10% limit. Hinted p95 is nevertheless5.03% worse geometrically (paired regressions8.25%,1.08%,5.90%); unhinted p95 is1.77% lower. A better batch mean does not imply uniformly better latency tails.
Upstream uses **4.74% less time than wait** (3.88–5.59%), a repeated secondary
limit. No extension or policy tuning selected. [Decision and tails](../../output/wait-qualification-20261105/mixed-decision.json).

### Finite session-churn exposure

All three retained six-wave cases passed artifact/aggregate verification:1152
responses/864real tools,96completed workflows per system, zero failed responses.
Total measured batch spans: ARC518.654s,wait488.490s,upstream470.362s. Wait uses
**5.82% less time than ARC**; upstream uses **3.71% less than wait**. These are
descriptive single-exposure comparisons, not six independent paired repeats,
a saturation benchmark or a production failure-rate certificate. Native prefill
is262952versus219419tokens for ARC/wait, with0preemptions in both; outputs45636
versus44857and physical CPU load15.868GBversus16.732GB differ.

All first-to-final wave endpoint summed-RSS changes meet the predeclared512MiB
growth bound. ARC falls6022.54→5854.79MiB; wait6763.52→6383.20MiB. Wait has higher
absolute endpoint RSS despite equal configured KV budgets; shared mappings and
resident-page differences prevent treating this as unique allocated memory or an
isolated metadata cost. Completion/owned cleanup passed; this agent-tier campaign
does not run direct disconnect/abort/drain/recovery checks. Those remain required
in Stage4 operational cells. Ordinary bounded-registry and audited research
profiles retain their documented different lifetime limits. [Decision, groups and all memory endpoints](../../output/wait-qualification-20261105/soak-decision.json).

All selected Stage3 execution is now complete. Gains survive fresh independent
repository/document tasks in selected pressure regimes; tight documents, ample
cache and several secondary-system comparisons limit the operating range. No
universal advantage, arbitrary-family aggregate, equal useful-answer quality,
indefinite stability or broad runtime support is established. Continue Stage4
functional/performance/integration checks; the mechanism is not scientifically
unworthy and does not meet the user's abandonment criterion.
