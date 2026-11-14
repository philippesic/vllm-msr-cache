# Frozen 2080 Ti screening procedure

The accepted [evidence plan](pr-evidence-plan.md) owns all decision thresholds and
precedence. This procedure fills its execution details before any GPU acquisition.
It authorizes no later stage. The machine-readable Stage 1 exit is
[`stage1-freeze.json`](../benchmarks/pr-evidence/stage1-freeze.json).

## Stage 1 exit

Both routes are feasible on the frozen runtime. A4t is an out-of-tree CachePolicy
using read-only private scheduler state; it needs no delivery core change. A4p uses
the same policy mechanism plus the single hashed delivery utility. The common Gemma
Turing patch and common benchmark endpoint/observer apply as recorded to every arm.
GPU admission additionally exposed the native registered-mmap address failure;
the matched `pr-mapped-host-pointer` compatibility patch applies to every arm in
both runtimes, before any efficacy observation. It translates pinned CPU addresses
to their CUDA device aliases without changing keys, policies or transfer ordering.
Private-state dependence limits portability to this exact vLLM revision. Neither
CPU feasibility nor the upstream wheel installation admits a model on the GPU.

Freeze **hold-only**, for both ARC and LRU: native recency remains unchanged. No
promotion or dead-first ordering enters this screen; those optional mechanisms are
not selected. Hold the largest ready, group-complete reachable prefix, admit later
stores until the original deadline, release on return/expiry/cancellation, and yield
earliest deadlines before a feasible native store could fail. Physical held keys
count once across sessions and occupy at most 50% of CPU slots. Limits are 4,096
observed/held sessions, 256 actions per envelope/observation, 60-second TTL clamp,
32 MiB charged hold metadata and 32 MiB charged observation/action metadata. These
are explicit accounting limits; measured process memory is reported separately.

A4t attaches a 12-second request TTL to every adopting request, including final
turns; its deadline starts at native request finish. A4p publishes after the last
response token, using only the current recorded assistant's nonterminal tool action;
`submit` is terminal. Its deadline starts at engine acceptance. It may hint a final
truncated turn. Both use the same eight preassigned adopters. Neither inspects the
next turn or true gap. Duplicate action IDs replay without extending deadlines.
A3/A3p receive their respective matched messages with advice disabled. The freeze
identifies policy files, the patched core hash, native CPU A3p/A3 eviction identity
and delivery-cost samples, W1 manifest, model revisions and dependency tuple.

The production-size bookkeeping cost remains a reported limitation, not an invented
CPU stopping threshold. The Qwen diverged-local-hit component persists on main;
its full scheduler behavior is checked in model admission. Public filing is deferred
because Philip did not authorize Stage 2b. LMCache 0.5.6 fails dependency admission;
no competitor lane is in this screen. A compatible replacement is required before
any future comparison. No simulator is built: its estimated engineering cost cannot
replace the required GPU evidence in this phase.

## Hardware admission and budgets

Keep one cumulative authorization ledger at `artifacts/runs/pr-evidence-2080ti`.
Maximum charged startup-to-cleanup time is 3,600 seconds for admission and 28,800
seconds for the screen, including calibration, failed attempts and diagnostics.
Before every job, require a fresh idle `nvidia-smi` observation. A busy GPU causes
waiting, not a takeover. Each job's maximum duration plus 60 seconds cleanup must
fit its remaining phase budget. A paired decision block must fit in its entirety
before its first arm starts; never spend the last budget on selected arms.

First run the A2 correctness probe for D8, HM and HS in that order, initially at
1 GiB. If native geometry/residency shows that the 4,097-token probe exceeds that
pool, retain the attempt and retry at the next 2/4 GiB candidate containing it,
subject to RAM and the same admission budget. This is a capacity correction before
efficacy, not a changed model/route setting chosen for benefit. The initial per-job
ceiling is 600 seconds. For models that pass, run the same byte/continuation probe on A1, A3,
A4t, A3p, A4p and A5, in model order, with a ceiling of the greater of 120 seconds
and twice that model's first successful startup-to-cleanup duration, capped at 600.
Before starting another job, reserve enough remaining admission time for it. A
model/route missing required correctness evidence is not admitted for that route;
report missing arms explicitly. Common runtime/model failures do not become evidence
against holding. Correctness failures attributable to a route block that route.

The probe uses fixed seed 11991 for token IDs and requests prime, resident and CPU
restore continuations. Require successful copy provenance, poisoned-destination
byte identity in every group, actual CPU reuse, the same resident/restore cached
boundary, and identical generated IDs at that boundary. Fresh-versus-cached output
identity is diagnostic because different compute boundaries can change numerics.
Record actual GPU block counts, group geometry, cache dtypes, transfer regions,
loading/staging peaks and partial failures. No performance claim uses the synchronous
byte detector. Compiler/kernel caches live under `artifacts/cache/pr-evidence`,
shared across arms; admission may populate them. Each cell resets model/KV state,
not the common compiled-kernel cache. Remaining lazy compilation stays in the cold
finite-horizon observations and is not selectively trimmed. The existing CPU diverged-hit observation is a known Qwen limitation;
if a targeted full-scheduler probe cannot fit admission time, label it unqualified.

Stop and retain a job if available host RAM falls below 2 GiB or new swap appears.
Pre-existing global swap is recorded; it is not treated as new experiment swap.
Owned-process swap growth and host swap-out growth are checked. An arm-induced
resource failure remains an outcome. Other GPU processes or material outside host
CPU activity invalidate performance measurement: material means over 10% of one CPU
core averaged across the cell, or over 25% for ten consecutive seconds, excluding
the recorded runner/server/client tree. Retain the counters and diagnose apparent
interference before attributing it to an arm. Windows activity outside WSL remains
a visibility limitation. Do not run CPU checks alongside decision cells.

## Baseline-only sizing and pool selection

Use only calibration repositories and seeds 12001–12003. The following sizing
passes are calibration observations, never decision evidence. Start each cell with
a fresh server/cache. All sixteen sessions and all one-shots before the common cell
end are measured; there is no Stage 3 warm-up or post-warm-up cohort in this bounded
Stage 2 screen. Report cold-start, finite-horizon behavior.

1. At native ARC (A2), use the smallest 1/2/4 GiB pool containing the largest eligible
   calibration conversation, arrival rate 0.2 sessions/s, no one-shots and a
   240-second horizon, for all three seeds. Compute each seed's mean session duration
   from scheduled arrival to completion or deadline, including failed sessions at
   their observed termination/deadline. The target arrival rate is eight divided by
   the median of these three means. It targets overlap; report actual overlap too.
2. Freeze the one-shot rate at 10% of the total offered request count divided by
   total sizing time. Set a common horizon to the larger of 120 seconds and the
   ceiling of the largest sixteen-session arrival time across all eleven specified
   calibration/decision seeds plus twice the pooled sizing p95 session duration,
   capped at 900 seconds. This uses arrival draws, not decision outcomes. If sixteen
   arrivals do not fit, the rate/horizon is not admissible; do not silently omit them.
3. Run A2 for all three calibration seeds at this rate, one-shot rate and horizon,
   using the same reference pool. The working set is the median of the three
   time-averaged unique reachable native-key footprints. Current submitted prefixes
   contribute; future turns do not. Window/recurrent keys and physical sharing are
   respected. Include the independent one-shot traffic. Multiply unique chunk counts
   by native allocator bytes per chunk; report residency separately.
4. Rank 1/2/4 GiB candidates containing the largest eligible conversation by distance
   from half this working set, breaking ties toward the smaller pool. Validate the
   first candidate with complete seven-arm calibration blocks for all three seeds.
   Every arm must retain at least 2 GiB available RAM and add no swap, including
   loading/staging. If a candidate fails resources, retain that complete failed block
   and try the next candidate in the fixed ranking; rerun all three paired seeds at
   that candidate. Do not use benefit to choose a pool. If none fits, report resource
   infeasibility. Fit predictions from admission are estimates, never substitutes
   for the all-arm observations.
5. At the selected pool, the better baseline is A1 or A2 with lower pooled hinted
   recompute share across the three full calibration blocks; exact ties choose A2.
   Freeze model-specific rate, horizon, one-shot rate, pool, working-set ratio and
   baseline in a committed calibration record before decision seeds. Calibration
   resource failures and noncompletion remain visible.

No mechanism, TTL, cap, decision threshold or trace inclusion rule is adjusted in
these steps. If calibration consumes the budget, report decision cells as not run.
The initial complete-block upper quote is seven times the common horizon plus the
model's admission startup/cleanup allowance. Re-quote from actual charged durations
after the first complete decision block, including remaining calibration, startup
and contingency. Only begin another block if it fits. Priority is W1 D8, then HM,
then HS, each targeting seeds 12101–12108; then the historical constrained W2 bridge
and its 12-second variant. W2 cannot change a W1 verdict.

## Statistics and missing evidence

The replication unit is the paired seed. Recompute each statistic inside 10,000
paired bootstrap draws. Resample complete seed blocks, never individual requests.
Use deterministic random draws keyed by model, workload and statistic. Report 95%
BCa intervals and the seed-clustered delta-method sensitivity for the same pooled
estimand. No pseudocounts, post-result interval switching or selected-seed deletion.

Use frozen inclusion shares to post-stratify the sampled episodes. Within each
cohort and stratum, divide summed request/token quantities by the number of planned
sampled sessions in that stratum, multiply by its frozen inclusion share, and sum.
Mean TTFT/ITL divides the resulting latency total by the resulting return count;
recompute share divides the recomputed-token total by the eligible-token total.
One-shots are one stratum. Recompute these totals and weights inside every seed
resample. This is a pooled, episode-population-weighted estimand, not an average of
per-seed ratios. Report raw counts and unweighted pooled summaries beside it.

TTFT starts at scheduled submission; client mean token ITL is the guardrail ITL,
with chunk ITL and native engine ITL reported separately. Only complete responses
enter descriptive latency means. Failed/unoffered requests and failed sessions stay
in their denominators and are reported next to every summary. A required cohort
with censoring cannot establish a speed ratio or efficacy STOP; retain descriptive
survivor means, error counts and an INCONCLUSIVE efficacy label unless a blocking
correctness/resource/error guardrail takes precedence. Primary adopters remain the
preassigned sessions regardless of whether their hints were accepted or eligible.

BCa uses half the tied bootstrap mass in the bias estimate and leave-one-seed-out
jackknife acceleration. Fewer than three complete paired seeds, a missing required
stratum/cohort, undefined denominator, nonfinite adjusted quantile or degenerate
bootstrap/jackknife makes that required interval unavailable. Do not substitute a
percentile interval. For ratio sensitivity use the analytic seed-clustered influence
of the same post-stratified pooled numerator and denominator on the log ratio;
transform its normal 95% interval back. For absolute recompute changes use the
corresponding pooled-difference influence and normal 95% interval. Apply the
finite-cluster variance correction n/(n-1). A disagreement across a decision
threshold gives INCONCLUSIVE. Missing required evidence cannot establish CONTINUE.

Apply the plan's 15% minimum worthwhile TTFT benefit, 20% opportunity label, A/A
bands and harm margins unchanged. A3 minus A2 supplies TTL's A/A band; A3p minus A2
supplies post-response's, and A3p minus A3 reports patch overhead. An A/A interval
excluding zero triggers CPU identity re-verification and diagnosis; it is not alone
an identity defect. Compare both routes against both native baselines, with the
calibration-selected baseline deciding. Report the paired whole-route difference,
first-turn latency, session completion, one-shot harms, failed sessions, evictions,
CPU/GPU reuse, recomputation, transfers, hint outcomes and abandoned/truncated ends.
Report first-task-occurrence sensitivity and cross-session reuse separately.

Evaluate each route/model/workload under the accepted precedence: correctness or
added store/resource/error harm blocks; then efficacy STOP; then CONTINUE; otherwise
INCONCLUSIVE. Aggregate only with the plan's any-model CONTINUE/all-models-run STOP
rules, explicitly naming missing models and unrun cells. A budget stop is a resource
decision. The screen's multiplicity and small-seed precision limits remain explicit.
Philip's fixed Stage 3 harm limit and permission for Stage 2b remain undecided.
