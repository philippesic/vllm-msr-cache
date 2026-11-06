# Adverse cases, alarms and unresolved scope

No observed result justifies rejecting CPU wait retention as a mechanism. Live
controls, identical-token replay and fresh independent cache-pressure tasks show
benefits. This does **not** establish that enabling it everywhere beats plain ARC.
Every completed ARC/wait pair with a slower wait point estimate is retained in a
`*-pair-alarms.json` receipt, alongside whole-run estimates and uncertainty.

| Case | Observed limitation | Diagnosis and disposition |
| --- | --- | --- |
| Ample cache | Initial three-pair overhead estimate1.05% was imprecise; one pair lost3.19%. Separate fixed-six confirmation estimates0.25% overhead, interval−1.34 to+1.87% | No eviction/preemption opportunity in the initial cell; hints add work without demonstrated saved recomputation. Confirmation meets the declared 2% neutral bound, with individual losses retained. No speedup claim |
| Tight eight-sequence documents | Initial development cell includes a 5.83% pair loss. Fresh reserve estimates0.98% less time, interval−2.00 to+3.87%, narrowly outside the 2% neutral bound | GPU admission, restore work, preemption and decode completion interact; transfer volume alone is not a cause. A preselected four-sequence diagnostic improves9.29%, interval2.23–15.84%, but remains development evidence without fresh confirmation. No universal admission fix claimed |
| Original independent repository, full cache | Separate fixed-six estimate0.46% overhead, interval−0.16 to+1.08% | Fits the 2% neutral bound; this result is retained alongside the favorable constrained fresh reserve. It does not prove benefit on every repository task |
| Twelve active sequences |2.55% less time, interval−2.24 to+7.12% | All three pair points favor wait, but neither useful-gain nor2% neutral precision is established. No favorable extension or parameter search |
| Serialized tools |3.65% less time than ARC, interval−1.27 to+8.33%; upstream and Continuum use 5.59% and 10.32% less time than wait | Secondary system losses are real retained observations. Runtime/scheduler/output differences prevent an isolated policy conclusion; Continuum also has better returning latency in this regime |
| Mixed hinted/unhinted sessions |3.28% less time than ARC, interval−0.26 to+6.70%; hinted returning p95 increases5.03% | Both groups' mean completion bounds pass the declared 5% margin, but the tail estimate is adverse and descriptive. Upstream uses 4.74% less batch time than wait |
| Additional CPU cache | Plain ARC with 25% more CPU cache has a favorable mean relative to wait at the original budget | Unequal-resource counterexample, not an equal-memory retention loss; removes eviction pressure and weakens the adoption case when RAM is cheap |

Percentages describe complete batch time, not throughput increase. Intervals are
conditional whole-run95% intervals on the fixed tested inputs. Separate original
and confirmation cohorts are not pooled. Strong gains elsewhere cannot cancel an
adverse cell through an invented workload average. See [operating range](STAGE3.md)
and the retained alarm/diagnosis receipts for native work, outputs and paging.

Functional-probe timing is a different category: the candidate performs extra
protocol-validation RPCs. Its one-pair raw timing losses remain visible but do
not measure matched application speed. Every functional acceptance still requires
actual restores or intentional abandonment, fresh full computation, greedy-output
agreement, protocol checks, allocation evidence and operational recovery.

The newer published runtime is an **unsupported API boundary**: actual extension
imports fail and required native hooks are absent. Other GPUs, native Linux,
BF16, distributed execution and hybrid/sliding-window attention remain untested.
Those limits restrict claims; they do not overturn the measured mechanism.

Stage 4 adds no ARC-loss pair in any of the four selected performance cells. Graph-mode gain is resolved; full-budget 1.5B has no CPU-load opportunity and fails to establish the 2% neutral bound. Constrained 1.5B and the public audit-off profile meet that neutral bound, but their useful gains remain unresolved. All three have positive points with unequal output and answer-quality limits. See [Stage4](STAGE4.md) for every interval and disposition. Production failure rates, unlimited identity churn, general answer-quality
equality, research novelty and population-wide superiority remain unestablished.
