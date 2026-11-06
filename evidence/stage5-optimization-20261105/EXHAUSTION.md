# Category exhaustion ledger — active, not a completion claim

| Category | Concrete search | Follow-up required before closure |
| --- | --- | --- |
| Queue ordering | Both/recent/frequent, quarter/half/full boost | Repeat credible variants under pressure and mixed traffic; inspect actual advice activity |
| Prefix selection | Quarter/half/full/128 blocks, root/suffix order | Check useful restored work, prefix gaps and displacement |
| Cadence/timing | 0.25/1/2/5 seconds, initial only, 2/5 second per-key coalescing | Uneven waits and queue refresh activity; live tools for finalists |
| Pressure gating | 75/95% CPU occupancy, GPU-resident gate, GPU prefix bridge | Ample, pressure and tight cases; misses from incorrectly suppressed hints |
| Session budgets | 512/1024 cumulative refreshes, bounded state | Mixed traffic fairness and abandoned/duplicate advice correctness |
| Plumbing | Differentially equivalent single-pass ARC eviction; audit cost; hint no-op | Serving controls with and without hints; CPU/RPC overhead bounds |
| Scheduling/transfer | Admission 4/8/12, prefill batch sizes, continuation credit | Matched ARC/wait controls; preemption and transfers; separate from policy gain |
| Combinations | Only mechanisms surviving isolated tests | Repeat complete development matrix and operational checks |

No category is exhausted yet. An inactive or noisy screen cannot close a category.
Each remaining idea will be labeled measured loss, unresolved, equivalent, outside
this mechanism, or speculative, with its evidence and explanation.

## First pressure screen

All eight arms completed identical-token work and verified with active cache
competition. No candidate point estimate cleared the 2% useful-change margin
over unchanged wait. Recent-only had very little applicable activity; half-prefix
and two-second cooldown increased recomputation. Frequent-only and 95% occupancy
gating stayed close to unchanged wait. This is a single-pass development screen,
not statistical rejection. Neutral cases and the independent category panel remain
pending. See [assessment](../../output/stage5-optimization-20261105/round1-pressure-assessment.json).

## Second policy panel and targeted follow-ups

The neutral ample-cache cell and all 17 second-panel cases completed and verified.
The ample cell had identical computed work and no eviction competition, so it is
an overhead control, not evidence against ordering rules. Whole-session refresh
quotas and aggressive prefix truncation increased recomputation in the pressure
cell. Per-wait quotas and a 75% prefix remain concrete follow-ups in a mixed
workload with abandoned-future advice; the policy receives no abandonment label.

Longer client refresh intervals reduced publication volume without an established
throughput gain. Initial-only and queue-only publication lost useful reuse. The
remaining timing tests cover uneven waits, nomination-window length, removal of
immediate duplicate publications, and actual publication/tool overlap.

An apparent GPU-prefix bridge improvement had zero extra suffix-key activation.
It is unresolved, not a mechanism win. A smaller CPU cache relative to GPU cache
is the declared activation follow-up. The 75% occupancy gate's apparent gain also
needs its declared repeated comparison; its computed work was almost unchanged.

Linear ARC eviction is substantially cheaper in isolated CPU stress tests but has
no established serving gain. Tight-cache matched controls, an advice no-op, and
audit-off operational runs separate eviction, publication, and diagnostic costs.
Scheduling tests change admission, prefill batching, and continuation credit in
matched ARC/wait pairs. These operating changes cannot be credited to retention.

The search remains active. The first 36 completed GPU cases are development
measurements, not final selection or broad qualification.

## Operating and live follow-ups

Nineteen operating cases completed before an unsupported priority setting stopped
dispatch without workload requests. The supported 0/2/8-second settings were all
measured; a new preflight rejects unsupported settings before model loading. The
failed attempt remains retained. A fresh default subsequently matched the fastest
small scheduling changes, so their initial points do not establish improvement.

All 21 fourth-round cases completed and verified. The retargeted GPU bridge still
had zero extra suffix-key activation despite 592 GPU lookups. Its general value is
unresolved; it does not explain measured timing differences in these two cells.
The same-admission extra-CPU-cache reference removed eviction competition and
remaining excess prefill work, with modest additional speed. It uses more memory
and is neither an eligible policy winner nor a strict timing bound.

Five live publication arms completed actual tools and all narrow test-outcome
checks. Acknowledgement-aware duplicate suppression reduced publications with
essentially unchanged workflow time; joined overlap and during-tool publication
did not improve the screen. Natural outputs differ. See
[live screen](../../output/stage5-optimization-20261105/live-publication-screen.json).

The mixed abandoned-advice panel activated four false-future owners. Caps reduced
unnecessary refreshes but also lost ordinary reuse; none improved total workflow
time. Both designated cohorts completed faster with unchanged wait than ARC in
this single pass. This is activity-backed adverse screening, not repeated fairness
acceptance. See [mixed screen](../../output/stage5-optimization-20261105/abandoned-screen.json).

Seventy-six GPU cases have passed measurement checks, with one retained unsupported
configuration failure. Fixed repeats, timing, uneven waits, documents, storage,
combinations, and final operational/functional checks still prevent closure.

## Fixed pressure-gate repeats

Three fixed repetitions did not reproduce the selecting screen's apparent gate
gain: throughput versus unchanged wait was 0.9999×, conditional 95% interval
0.9785–1.0217×. Unchanged wait versus ARC was 1.0615×, interval 1.0499–1.0732×.
The gate is not selected. These reused development inputs do not provide fresh
population confirmation. [Repeated assessment](../../output/stage5-optimization-20261105/pressure-repeat-assessment.json).

The following three-second declaration attempt was rejected by the strict hint
API. Its completed requests do not make it a valid policy comparison. It is
retained as the second invalid-setting failure. Preflight now checks the actual
hint contract. The replacement experiment uses supported declarations and an
explicit default-off CPU-advice delivery grace; it changes no GPU staging lease.
The prompt-storage panel also includes a matching prompt-scope/full-store control,
because completed-transcript publication requires storing decoded tails.
