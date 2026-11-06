# Wait-hint staged qualification — 2026-11-05

## Authority

The user authorized all four stages of docs/WAIT_HINT_PRODUCT_PLAN.md, routine
repairs and continued diagnosis while unavailable. ARC is primary; upstream and
Continuum are secondary. Stop scientific work only if credible evidence makes
further work on this mechanism unworthy. No deployment/push, shared configuration
changes or resumption of paused AWQ work. Numerical margins and finite budgets
below are experiment-owner choices under this authority, not user-supplied numbers.

## Predeclared decisions

Primary: batch completion time for all offered workflows, including actual tools,
queues and hints, excluding startup and identical tiny warmup. Secondary: offered/
return first-token latency, token throughput, recomputation/restores, evictions,
transfer bytes, memory/CPU, failures and quality. Whole runs are paired units;
use paired log-ratio Student intervals. Do not count requests as independent runs.
All attempts, output differences and failed tasks remain visible. No survivor-only
speed claims. Correct test-status reporting alone does not establish task quality.

Primary-profile useful improvement: >=2% less batch time and a 95% paired interval
resolving the direction. Start with three repetitions as a developmental checkpoint.
If inconclusive, a separate fixed six-repetition confirmation is authorized by this
predeclared rule; decide from that fresh set and report both separately. Do not
extend until a favorable outcome appears. A loss to ARC triggers an alarm and
inspection of setup/host state, output work, activity, transfers and queues;
a single noisy pair does not establish failure. Repairs require a causal hypothesis,
preserved negatives and fresh validation. Neutral/no-reuse regimes target <=2%
batch overhead; larger observed losses get diagnosis. Secondary competitiveness:
repeatable advantage or within 3% batch time, with equal useful task completion.
Wide intervals mean unresolved, not a tie or proof of harmlessness.

## Stage 1

Freeze Qwen2.5-7B-Instruct-AWQ / FP16 / RTX 2080 Ti, 8,192 context, eager execution,
2,560 GPU blocks (2,240 MiB), 2,688 MiB CPU KV, 16 repository workflows, 8 server
sequences, 4 tool workers, 8-second continuation credit. Disable semantic/SVD/
speculative prefetch. Three fresh-process runs per condition, rotating no hints,
inactive hints and active hints. Inactive control follows ready-prefix validation
but skips only ARC queue mutation. No tuning between cases. Equal native tracing
in primary conditions; extra retention tracing is off in primary measurements.

Select the latest fair comparison's wait trace before variants. Diagnostic replay
forces exact prompt/continuation token IDs while executing normal model forwards,
replays recorded tool queue+execution delays and retains causal turn dependencies.
Future completion information never reaches retention. Three rotated repetitions
of ARC/full wait/GPU-gated wait/single initial refresh. Label all forced-token and
recorded-delay evidence as diagnostic, not live application performance. A separate
trace links specific keys' ordering/eviction/reuse; its cost is not primary timing.

## Stage 2

Reuse corrected fairness settings and published pins/compatibility patch; enable
Continuum decode publication. Four repetitions of all four systems with balanced
position/predecessor order. Check actual byte budgets and activation; preserve
runtime/backend/paging differences. Separately compare ARC with 25% more CPU KV
against equal-budget wait (three repetitions), labelled resource-cost comparison.
Add native LRU without hints as the simplest existing retention substitute. The
resource/substitute campaign rotates wait, LRU and ARC with 25% more CPU across
three fresh repetitions; its comparisons remain separate from the four-system
primary. These selections are frozen before any Stage 2 measurement.
Use clean same-runtime feature-off controls where available; do not invent isolated
algorithmic superiority across runtimes. Review final answers' source support and
limitations separately from the existing normalized-status-v2 scorer.

## Stage 3

Freeze two independent authored families before measurement: a different repository
with real searches/tests and document analysis with actual parsing/search/check
tools. No production/autonomous-agent claims. Separate calibration and held-out
inputs. Freeze settings before evaluation; if evaluation drives a repair, retain it
as development evidence and use fresh validation. Three paired repetitions per
primary family versus ARC and the strongest credible Stage 2 alternative.
Predeclare bounded regimes covering ample cache, constrained CPU cache, increased
concurrency, uneven waits, and mixed hinted/unhinted traffic. Include cancellation,
duplicate/stale advice and changed returns as functional checks. Persistent-server
waves exercise churn, abort/drain/recovery and resource growth. No long-run SLA or
failure-rate claim from a short soak. Retain neutral and adverse conditions.

## Stage 4

Inventory available models/environments. Start with another local model and a
distinct architecture if compatible. Convert byte budgets by documented geometry,
not searches for winning capacities. Additional GPU/native-Linux/runtime cells
need real access; unavailable cells remain untested, not mechanism failures.
Test imports/settings, actual CPU stores/restores, identity/namespace/stale hints,
cancellation, off switch and trusted fresh-computation references. Separate
functional-only and performance-qualified cells. No cross-product, universal
benefit or independent reproduction claims without actual evidence.

## Preservation

Raw runs/backups: output/wait-qualification-20261105/. Reusable code in benchmarks/;
compact findings here. Initial archive preserves preexisting dirty work. Freeze
and check executed sources at every dispatch; no code edits while campaigns run.
Supervisor reserves common and legacy GPU locks throughout idle gaps. Reap only
owned processes and retain useful failed/negative evidence.

Stage 2 native feature-off diagnostic (fixed before Stage 2): upstream supplies a
clean control by omitting only kv_hints from every workflow request, preserving
native CPU ARC, application continuation priority, scheduler, bridge, byte budgets
and task instructions. Three paired fresh-process repetitions alternate order;
verify absent metadata and zero positive priority applications. Compare native
retention on/off within that runtime separately from full-system ranking.
Continuum couples pinning to its scheduler/job lifecycle and generated tool-call
detection; this campaign does not claim a clean retention-only off control for it.
