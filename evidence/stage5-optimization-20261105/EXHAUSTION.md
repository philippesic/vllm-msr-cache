# Category decisions — Stage 5 complete

The decision is to retain the existing wait-hint rule. No new serving candidate
met the declared promotion requirements. The user reduced the breadth of the
remaining pass and explicitly prioritized diagnosing eviction; that lead was
followed through CPU stress, actual serving clocks, real cache-state replay and
larger prefill batches on real documents. This is a stopping decision for this
local development search, not a proof of global optimality.

| Category | Concrete alternatives and follow-up | Finding and decision | What remains |
| --- | --- | --- | --- |
| Queue choice and ordering | Both/recent/frequent queues; root/suffix order; quarter/half/full recency boost; pressure, ample and live-document cells | No qualified gain. Eligible returning keys were mostly in the frequent queue, limiting recent-only activity. Partial boost also lost reuse or added sorting work. Keep the original full refresh. | Recent-heavy workloads remain a workload question; inactivity here is not a general rejection. Finer boost grids have no convincing measured lead. |
| Prefix selection | Quarter/half/three-quarter/full prefix; 128-block cap; root/suffix ordering; mixed abandoned advice | Truncation generally lost useful restore opportunities. No qualified throughput/fairness benefit. Keep full eligible prefixes. | A learned or cost-based block selector would be a new mechanism, not another justified fraction sweep. |
| Cadence and delivery | 0.25/0.5/1/2/5-second cadence; initial-only, queue-only and tool-only; 50/250/750ms declarations; 750/2250ms CPU-only delivery grace; live overlap and duplicate suppression | Tool-only and timing/grace/eviction finalists failed fixed-repeat selection. Grace genuinely accepted late CPU advice without GPU reservations. Reduced cadence saved publications but could substantially increase recomputation. Keep current timing. | Adaptive cadence may depend on workload conditions; further fixed intervals have weak support. Broader timing guesses cannot use actual future return information. |
| Pressure and GPU residency | 75/95% occupancy gates; GPU-resident suppression; GPU-prefix bridge; ample and smaller-CPU/larger-GPU follow-ups | The repeated 75% gate comparison was neutral. Bridge extra-key activation stayed zero even after retargeting. No new default. | Bridge value is unresolved, not disproved. A separate uneven-wait gate point was not confirmed; additional repeats were deferred under the user's narrower scope. |
| Advice budgets and displacement | 512/1024 cumulative session quotas; 512/1024/2048 per-wait quotas; bounded cooldown state; mixed hinted/unhinted traffic with abandoned-future advice | Caps reduced false-future refresh work but lost normal reuse. The policy received no abandonment label. Neither total time nor cohort evidence qualified a promotion. | Better abandonment prediction requires new causal signals and fresh fairness tests. Blindly tightening quotas is unsupported. |
| Eviction implementation | Native versus single-pass ARC; randomized differential checks; protected-prefix CPU stress; six serving profiles; real-document batches up to 250 victims | Exact-state CPU gains are real, but native eviction remained below 0.1% of workflow time even in the larger-batch diagnostic. Keep linear eviction optional; no serving promotion. | A single-victim fast path could save microseconds. Faster GPUs or substantially different cache/batch distributions justify a new profile, not more tuning against the same tiny local cost. |
| Publication plumbing and audit | Scheduler/digest CPU profiles; no-op hints; acknowledgement-aware duplicate suppression; joined RPC/tool overlap; live tools; audit-on/off screen and matched audit-off controls | No new qualified serving gain. CPU hot-loop costs were small; they exclude network/scheduler waits. Audit-off passed abort/recovery controls and already exists as a production-profile option. Missing audit records cannot be interpreted as missing cache work. | New transport batching or APIs need a measured bottleneck and correctness design. The broad audit repeat panel was deferred; the bounded control pass is complete. |
| Admission and scheduling | 4/6/8/12 sequences; 512/1024/2048 prefill batches; supported 0/2/8-second continuation credit; fresh default reference; combined linear/512/FCFS screen | Small favorable initial points were matched by a fresh default. The final combination gained only about 1.04% in its screen, below the 2% useful margin. No setting is promoted. | Hardware/workload-specific settings remain possible. More neighboring values on these reused cases are low-confidence tuning. The 4096-token batch was an eviction diagnostic, not a selected operating profile. |
| Storage and combinations | Full versus prompt-only storage with ARC and wait; full-store/prompt-scope control; fixed timing/grace/eviction repeats; final scheduling interaction | Storage variants did not beat unchanged wait. Neither repeated timing finalist qualified, and the final scheduling interaction was below the useful margin. Keep the existing storage and publication scope. | More elaborate admission/storage policies are new mechanisms. No untested combination of the weak local effects has a strong current rationale. |

## Fixed repeats, kept separate from selecting screens

Three fixed pressure-gate repetitions measured existing wait versus ARC at
1.0615× throughput (conditional 95% interval 1.0499–1.0732×). The gate versus
unchanged wait was 0.9999× (0.9785–1.0217×), so it was not selected.

Three fixed timing-finalist repetitions measured tool-only publication at
1.0121× unchanged wait (0.9613–1.0655×) and tool-only plus CPU grace and linear
eviction at 1.0154× (0.9636–1.0699×). Neither reached the 2% point threshold or
excluded no improvement. Four arms were rotated over three repetitions, not a
complete four-position cycle. These are reused-input development intervals,
not fresh population confidence or a correction for all searched candidates.

## Adverse and activation follow-ups

The ample-cache cell had identical computed work and no eviction competition;
it is an overhead control. A same-admission larger-CPU-cache reference removed
remaining excess prefill with modest timing headroom, but uses more memory and
is not an eligible policy winner or strict wall-time bound.

The mixed abandoned-advice panel activated four false-future owners and 46
phantom publications with unchanged wait. Quotas reduced unnecessary refreshes
but also reduced ordinary reuse. Both designated cohorts completed faster with
unchanged wait than ARC in that single pass; this is not repeated fairness
acceptance. The bridge retarget saw 592 GPU lookups but still no extra suffix
calls/keys. Neither an inactive screen nor a noisy timing point counts as a
mechanism win or a general rejection.

Live publication variants completed real tools and their narrow task-outcome
checks. Output counts differed, so they are not equal-work causal comparisons.
The live tight-document screen took 172.62 s for ARC, 160.41 s for unchanged wait,
161.15 s for the pressure gate, 170.50 s for two-second refresh and 176.19 s for
half boost. Output counts also differed. Two-second refresh increased executed
prefill from 144,000 to 177,389 tokens; partial boost increased preemptions.
The tight identical-token cadence follow-up also favored unchanged wait in that
pass. Reducing publication work can lose valuable retention.

## Final bounded passes

All five identical-token storage cases passed: full-store ARC 184.56 s,
prompt-only ARC 183.92 s, unchanged wait 172.72 s, prompt-scope/full-store wait
189.13 s, and prompt-only wait 177.09 s. Both shorter-prefix variants had active
refreshes. Unchanged wait executed 138,778 prefill tokens, versus 173,848 for
prompt-scope/full storage and 159,618 for prompt-only storage. The control
separates publication coverage from storage. No variant nominates a better
serving configuration.

The contemporaneous scheduling interaction took 184.71 s for ARC, 180.11 s for
unchanged wait and 178.27 s for linear eviction plus 512-token prefill batches
plus FCFS. Its 1.04% point gain is below the useful margin. A single pass does
not establish improvement or noninferiority, and it is not pooled with earlier
isolated screens.

Matched audit-off controls took 102.39 s for wait and 108.49 s for ARC. Both
completed an actual backend abort, drain and recovery generation. This is one
workload wave plus probes, not the full multi-wave operational qualification
tier. This pair does not itself estimate audit-on/off overhead. Audit-off
removes native attribution history and is an existing option.

The eviction follow-up is detailed in [EVICTION.md](EVICTION.md). It reproduces
the CPU lead on actual states and deliberately larger real-document prefill
batches, while measuring the small absolute time available to save.

## Retained invalid attempts and scope limits

An unsupported 16-second continuation credit stopped dispatch without workload
requests. A 3-second wait declaration was rejected by the strict public hint
contract; its completed requests do not make a valid policy comparison. Both
attempts remain retained. Preflight now validates the actual contracts; neither
public enum was widened. CPU nomination grace is a separate default-off option
and extends no GPU staging lease.

The user narrowed the closeout: the unexecuted twelve-case audit repeat and
nine-case uneven-gate repeat were explicitly deferred, not counted as completed
or rejected. No candidate passed enough evidence to justify broader promotion
guards. The final unforced functional closeout passed all nine waves and 48 assertions,
with actual CPU restore, fresh generation agreement and operational recovery.
All 47 captured eviction states replayed identically.

The next credible step is fresh Stage 6 workloads and broader environments with
the unchanged candidate, revisiting a specific mechanism only if activity and
cost measurements supply a new reason. Further brute-force parameter hunting
within the tested mechanisms mostly offers uncertain small gains, inactive paths
or more elaborate unmeasured mechanisms.
