# Stage 5 optimization protocol

Authorized 2026-11-05: sustained optimization in every category until the remaining
search mostly produces speculative non-results. This supersedes the proposed
six/four/two total limit: those are round sizes, not a completion criterion.
No Stage 6 generalization claim or deployment is authorized here. The user's
subsequent replacement AGENTS instructions authorize coherent milestone commits
and pushes to the established user-owned remote. Checkpoint the saved completed
foundation separately from Stage 5 changes; preserve unrelated active work.

## Controls and decision rules

Freeze preexisting source before editing (`output/stage5-optimization-20261105/source-before.tar.gz`).
Retain plain no-hint ARC and unchanged wait-hint behavior. New switches default off.
Compare equal model/runtime/cache budgets and offered work. Shared ARC plumbing
improvements require both no-hint and wait variants to separate their contribution.
Diagnostic identical-token replay is development evidence, never live quality evidence.
Live output differences remain explicit. Historical cases reused here are development
inputs, regardless of their original held-out labels. Stage 6 requires fresh inputs.

Primary outcome: completed workflow throughput (fixed offered work / batch time).
Secondary: token rate, returning/initial TTFT, tails, CPU cost, recomputation,
restores/transfers, preemption, memory and hinted/unhinted completion fairness.
Starting useful-change margin: 2% throughput over unchanged wait. Neutral loss margin:
2% throughput; fairness margin: 5% mean completion time per cohort. These are selection
criteria, not universal guarantees. No pooled average may hide an adverse workload.
Small-sample tails are descriptive. Inconclusive intervals are reported as such.

## Search sequence

1. Correctness/property checks and CPU cost profiles for all implementation variants.
2. Six initial policy variants: recent queue, frequent queue, bounded boost, leading
   prefix cap, coalescing cooldown, pressure gating; plain ARC/current wait controls.
3. Extend each category using measured activity and mechanism: prefix fraction/order,
   session budgets, client refresh cadence/one-shot, GPU gate, efficient eviction,
   operational audit costs, admission/concurrency and prefill batch scheduling.
4. Targeted combinations and repeated full-matrix development comparison for finalists.
5. Operational correctness, selected source/config freeze, category-by-category
   exhaustion ledger, retained losing/invalid/inconclusive attempts and cleanup.

Each GPU tranche declares cases/order before running, generally one exploratory
repetition then three or six fixed comparison repetitions. No extension until a win;
additional tranches answer a documented new diagnostic question. Screen inactivity
or noise calls for representative follow-up, not rejection. Total time is adaptive:
a wall-time checkpoint is not completion. Pause only on user request.

## Matrix and evidence

Repository/tool continuations under CPU eviction pressure; tight documents;
mixed hinted/unhinted; ample cache; uneven waits/duplicate/abandoned hints. Use
causal replay for fast policy search and live tools for finalists. Admission changes
apply equally to no-hint/hint controls and stay separate from retention changes.
The pinned 7B AWQ model on the 2080 Ti/WSL2 host is the development platform.
Maintain supported dtype/backend, GPU reservations and owned-process cleanup.

Run source is frozen; no executed-source changes during a tranche. Every run keeps
validated plans, raw receipts, negative outcomes and offline verification. New reusable
code lives under benchmarks/research_checks or semantic_offload; raw runs in output.
No pending/expired advice may acquire GPU ownership or speculative transfers.
Queue membership, ghost history, adaptation and block references must be unchanged
by advice; any exception must be labeled a different policy.

## Exhaustion criterion

Each category needs concrete tested alternatives or a measured bottleneck bound
showing why further work has little credible headroom, plus a record of remaining
ideas. At least one evidence-driven follow-up round after initial screening.
Conclude only once credible independent changes and combinations have been tested,
remaining gains are unresolved/smaller than practical margins or demand substantially
new mechanisms, and the selected behavior has passed repeated and operational checks.

## Declared development tranches

- Calibration: ARC, unchanged wait, linear-eviction wait on historical identical-token repository replay; one pass, all completed and verified.
- Policy round 1: two independent cells (repository pressure and ample cache), each ARC/current wait plus recent-only, frequent-only, half boost, half prefix, two-second key cooldown and 95% occupancy gate. One exploratory pass per cell.
- Category round 2: repository replay, ARC/current wait plus quarter boost, quarter prefix, 128-block prefix, suffix ordering, 512/1024 per-session budgets, 0.25/2/5-second client cadence, initial-only advice, GPU residency gate/bridge, five-second key cooldown and 75% occupancy gate. Order fixed by seed 503 before execution. These screens nominate follow-ups; no single noisy result is a final policy verdict.

## Repeated-comparison decision rule (declared before repeats)

A performance promotion needs a primary workflow-throughput point improvement of
at least2% over unchanged wait and a conditional paired95% interval excluding no
improvement. Critical neutral/adverse cells need a throughput-ratio lower bound
of0.98 and hinted/unhinted mean-completion-ratio upper bounds of1.05; unresolved
guards prevent broad promotion. A simpler/lower-cost equivalent may be chosen only
with the same noninferiority guards and explicit measured cost reduction. These
are development selection rules, not selection-adjusted population confidence;
fresh Stage6 cases are still required. No pooling with screens. Three-repetition
follow-ups are fixed in advance; six-run final comparisons, if warranted by a
new selection decision, are separately declared before execution.

Round2 was extended to17 arms with queue-only before launch. All17 passed
measurement validation. The apparent GPU-bridge gain had zero extra suffix-key
activation and remains inconclusive; a smaller-CPU/larger-GPU cell is declared.
Whole-session budgets lost reuse, motivating fresh per-wait budgets and explicit
abandoned-advice stress. The policy never receives the abandonment label.

The operating tranche completed 19 cases, then stopped before workload submission
on an unsupported 16-second continuation credit. The public gateway accepts only
0/2/8 seconds; all three supported values had measurements. The failed attempt is
retained. A preflight contract check now catches this before model startup; the
gateway contract is unchanged. A fresh 8-second default reference is separately
declared, not a retry of the unsupported setting.

Follow-up plans v2 add live initial-RPC/tool overlap with a mandatory join before
the continuation, acknowledgement-aware duplicate suppression, mixed abandoned
advice, uneven waits, tight live documents, prompt-only storage controls, and a
contemporaneous combination screen. Per-cohort completion analysis uses designated
corpus groups in both ARC and wait arms; abandoned publications add neither
requests nor time to their already completed owner's workflow. Selecting screens
remain separate from the fixed repeated pressure-gate comparison.


## Final scope adjustment and eviction diagnostic

The user narrowed remaining breadth and explicitly prioritized following the CPU
eviction speedup into real serving. Completed repeated timing finalists and live
document/cadence panels bring the checkpoint to 123 valid GPU cases; the two
invalid configuration attempts remain separate. No new default is selected.

The bounded remaining pass is prompt storage (five arms), a contemporaneous
ARC/wait/combined scheduling screen (three arms), and matched audit-off ARC/wait
operational controls. The unexecuted twelve-case audit repeat and nine-case uneven
pressure repeat are deferred under this narrower scope, not counted as completed
or statistically rejected. A three-arm unforced functional probe checks defaults
and the opt-in linear-eviction/CPU-grace combination.

Eviction diagnosis profiles primary and tight cache cells with ARC, unchanged
wait, and linear-eviction wait. This is attribution, not a performance selection
round. Every call records native victim-selection body wall/thread CPU time,
victim count, protected-set and queue sizes. Bounded exact before-states preserve
real byte keys, block references, ghost order and adaptive target for offline
native/linear replay. Snapshot and logging overhead is separate; inclusive store
time must not be added to eviction body time. Offline iterator-visit counts are
separate from timed plain-queue replays. Host elapsed cost is not a strict
end-to-end bound because GPU overlap and scheduler changes can affect results.
Actual cost and state shapes determine any additional eviction follow-up.
