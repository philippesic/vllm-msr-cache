# Stage 2: configured systems, substitutes and resource cost

The four-system checkpoint completed all sixteen fresh-process cases: 1,024 EOS
responses and 768 actual tools, with verified policy activation and cleanup.
Wait-hint reduced workflow batch time by **7.51% versus no-hint ARC**, conditional
paired 95% interval **4.87–10.08%**. It was close to upstream; the small difference
has no resolved winner. Substitute/resource and same-runtime controls are
reported separately below; the fresh secondary confirmation is complete. This
supports a narrow local value claim, not general superiority or a finished
cross-environment adoption decision.

| Configured system | Geometric mean batch seconds | Mean offered-to-first-token seconds | Mean tool-ready-to-first-token seconds |
| --- | ---: | ---: | ---: |
| ARC without hints | 108.28 | 7.80 | 5.67 |
| Wait-hint ARC | 100.14 | 7.23 | 4.92 |
| Pinned upstream draft + declared host-pointer fix | 99.85 | 7.64 | 5.34 |
| Published Continuum preview, decode publication on | 108.50 | 8.54 | 4.16 |

Four complete paired repetitions, with balanced position/predecessor order within
blocks. Startup is excluded; actual tools, queues, advice and observation are
included. Continuum has better returning-request latency here despite slower
whole batches, so a batch-time result does not imply superiority on every metric.

Wait's batch-time reduction versus Continuum is 7.70% (4.90–10.42%). Wait's time
overhead versus upstream is 0.29%, with an interval from −2.74% to +3.43%. The latter
is slightly too wide to qualify the predeclared 3% competitiveness margin. A
**separate fresh fixed-six confirmation** is selected under the protocol's
inconclusive-result rule, with no pooling or optional extension of this set.

All arms share model/FP16/context, explicit CPU/GPU byte budgets, task instructions
including declared fixed bash tools, and common arrival/workflow logic. Every case
checks actual allocation geometry. Upstream uses native CPU ARC, native retention
metadata and forwarded continuation priority, V2 runner, pinned backing, and the
frozen verified host-pointer compatibility patch. Continuum retains its published
scheduler/LMCache backend and has supported decode publication enabled. Different
runtime versions, runners, chunk sizes, input-pinning paths, instrumentation and
schedulers remain; these are configured-system results, not isolated policy ranks.
The published preview's generated-tool detection activated (21 pins per case),
not a fabricated promise of one pin for every declared tool.

Host paging is material and unequal: ARC swap-out ranged 62–475 MiB, wait 0.4–576
MiB, upstream 0.1–51 MiB, Continuum 0.2–10 MiB. All four ARC pairs favor wait, but
these live timings cannot establish what the gaps would be under matched paging.
The low-paging identical-token replay in Stage 1 supplies separate causal support;
do not substitute it for the live cross-system outcome.

Outputs differ: ARC produced 7,504–7,579 tokens per case; wait 7,496–7,645;
upstream 7,517–7,668; Continuum 7,807–7,809. All offered workflows/tools completed.
The frozen status scorer reports 64/64 for ARC, wait and Continuum, and 60/64 for
upstream; all four upstream misses are the same quoted-JSON PASS representation,
manually read as correct and retained separately without changing the original
score. Shared unsupported coverage assertions and word-limit failures remain in
[the quality audit](QUALITY.md). General task-quality equivalence is unestablished.

Raw immutable run: `output/wait-qualification-20261105/stage2-published/`.
Compact reports: `stage2-published-summary.json` and `stage2-published-quality.json`
in the same parent. Offline artifact verification is at the end of the campaign
log. Source archives, executed compatibility source, package/model identities,
per-case effective settings, allocation receipts, raw answers and all measurements
are retained. This result is conditional on the frozen authored workload and host.

## Resource checkpoint, first rotated triplet

At the larger CPU budget, ARC finished in 97.69 s versus wait's 105.06 s at the
original budget, while native LRU at the original budget took 107.94 s. This is a
retained resource-cost counterexample, not an equal-memory loss to ARC. Effective
allocation receipts confirm identical 2,348,810,240 GPU KV bytes and respectively
2,818,572,288 versus 3,523,215,360 CPU KV bytes (25% more CPU for ARC).
Extra-memory ARC recomputed 52,365 prompt tokens versus wait's 55,923, and had no
returning requests with prior CPU-prefix eviction versus wait's 12. Its output was
7,475 versus 7,533 tokens. Paging also differed (about 6 versus 269 MiB swap-out),
so do not attribute the entire observed gap to retention/capacity. These first
cases diagnose a plausible cache-fit boundary; the complete repeated triplets
must determine the resource comparison. No result is removed or silently retuned.

## Complete resource/substitute campaign

All nine cases passed, including 576 requests and all real tools. Wait batch times
were 105.06, 101.61, 101.21 s; LRU 107.94, 105.92, 105.72 s; extra-memory ARC
97.69, 100.69, 99.58 s. Wait used 3.67% less batch time than equal-budget LRU
(conditional paired 95% interval 1.49–5.80%). Extra-memory ARC used 3.22% less
time than wait, with a wide interval spanning −5.55 to +11.25%; every observed
pair favors extra memory. It avoided returning-prefix CPU evictions in all three
cases. This is an observed resource tradeoff, with paging/output confounding and
insufficient precision for a resolved population direction, not an equal-memory
ARC failure. The complete result and summary are retained under `stage2-resource`.

The predeclared equal-budget substitute selection chooses upstream: geometric
mean batch time 99.85 s, versus LRU 106.52, plain ARC 108.28 and Continuum 108.50.
Selection uses complete repetitions and retains activation/quality limitations;
it does not assert isolated policy superiority. Stage 3 manifests compare ARC,
wait and upstream; the serialized-tool cell also retains Continuum because of its
return-latency strength. All ten manifests are frozen before candidate evaluation.

Before the next campaign, mixed published requests were corrected to omit native
retention metadata on explicitly unhinted sessions. Arrival-based continuation
priority stays shared. HTTP/client checks verify this distinction for both
published systems. The lifecycle-link negative check now also rejects intervening
CPU recommits. Benchmark checks: 128 passed. Native/control checks use the patched
runtime; an initial wheel-only invocation failed native-publication checks as
documented by the setup instructions and was replaced with the proper launcher.

## Upstream native feature-off control

All six fresh cases passed and verified (384 workflow responses). Omit only native
`kv_hints`; leave CPU ARC, priority scheduling, bridge, task instructions and byte
budgets unchanged. Every off-case omitted metadata on every workflow request and
recorded zero positive retention updates; every on-case recorded 48 updates.
Off batch times were 104.97, 106.17, 106.87 s; on 100.73, 101.91, 99.19 s.
On used 5.09% less batch time (conditional paired 95% interval 0.42–9.54%).
Outputs and paging differed: off output totals 7,578–7,698, on 7,518–7,542;
swap-out ranged approximately 0.05–241 MiB off and 4.25–263 MiB on. This is a
within-runtime live retention benefit, not a fixed-work effect or a ranking of
policies against different baselines. The separate exact-token local replay
addresses comparative recomputation. Raw run, source and summary are retained
as `stage2-upstream-control` and `stage2-upstream-control-summary.json`.

## Fresh six-pair secondary confirmation

All twelve fresh cases passed and verified: 768 EOS responses and 576 real tools,
with active retention and complete work. This is the separately selected set;
it is not pooled with the original four repetitions. Geometric mean batch time
was 101.31 s for wait and 100.71 s for the configured upstream draft.
Wait's time overhead was **0.60%**, conditional paired 95% interval
**−0.69 to +1.90%**. The interval fits the predeclared 3% competitiveness margin,
with no resolved superior system. The earlier approximately 1% exploratory lead
over upstream is not established by these measurements.

Wait used 7,476–7,532 output tokens per batch, upstream 7,523–7,587. Upstream's
mean offered-to-first-token time was 0.258 s higher in the paired comparison,
despite its slightly shorter batches. Paging remained unequal: wait swap-out
16.8–389.4 MiB versus upstream approximately 0.01–4.77 MiB. These are configured
live-system results with different outputs/runtime/observation costs. The frozen
status scorer reports wait 96/96 and upstream 95/96; the single upstream miss is
again quoted JSON PASS and receives a separate manual adjudication without
changing the frozen score. General answer-quality equivalence remains unproven.

Raw source-frozen result: `stage2-secondary-confirm`; compact summary and narrow
quality report have the same prefix in the parent output directory.

The Stage 2 outcome justifies continuing: wait beats same-budget no-hint ARC and
LRU, is faster in batch time than configured Continuum, and stays competitive
with the configured upstream draft on this workload. Upstream already supplies a
credible retention option; these results do not establish a compelling adoption
advantage over it, isolated policy superiority, or research novelty. Extra RAM
can remove this workload's observed ARC eviction pressure. Those tradeoffs remain
part of the decision as independent workloads and compatibility are evaluated.
