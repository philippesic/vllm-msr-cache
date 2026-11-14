# Evidence plan for a vLLM wait-hint pull request

Status: accepted plan, authorized by Philip on 2026-11-13 for Stage 1 and the local
2080 Ti phase only (1 hour admission, 8 hours screen maximum). CPU feasibility passes for both routes; [the Stage 1 exit](pr-evidence-screen.md)
freezes their implementation and the bounded screening procedure. GPU admission is
pending; see [the execution boundary](pr-evidence-execution.md).
Later stages remain unauthorized. Historical proposed caps below are estimates;
Philip's explicit authorization supersedes those estimates, not the decision rules.
Initially reviewed on
2026-11-13 through two structured reviews: the stages and decision rules (Part A) by
GPT-6 Astra and Claude Fable 5.2, the concrete workloads, models, comparators and
machine tiers (Part C) by GPT-6 Astra and GPT-6.1 Sol. Every reviewer explicitly
accepted that revision. A later debate between GPT-6 Astra and GPT-6.1 Sol reviewed
the dual delivery-route revision; both explicitly accepted it at 9a8e0acf as a staged
planning document, with Philip's harm limit and the named freeze gates as caveats. Part B lists interpretation rules. Part C owns concrete execution
settings; missing admission/protocol details must be frozen before dependent runs.

Background facts verified against upstream vLLM:
vLLM's CPU offload default eviction policy is `lru` (both at the pin and on main,
`vllm/v1/kv_offload/cpu/spec.py`); all our earlier results used opt-in ARC.
`ReqContext` on main carries `kv_hints` but no session id; policies are constructed
with capacity only.

## Part A. Stages and decision rules

### Stage 0. Record (no compute)
Completed in d8cf1d63: committed the plan and corrected the campaign record (seed
overlap with design data; upstream baseline change; LRU, not ARC, is the default).

### Stage 1. Feasibility spike on vLLM main (CPU only; needs authorization)
Freeze one upstream commit. Answer in writing, with CPU tests:
1. Hint lifecycle. When the client knows of a wait (often only after the response
   ends in a tool call), how the hint reaches the offload manager, when the deadline
   starts, how ready chunks become eligible (stores may complete after request
   finish), how a return or expiry releases holds, idempotency/duplicates/cancel.
   Candidates: (a) a TTL hint attached to the request ("hold this session's prefix for
   T after this request finishes"), delivered to the policy through
   on_request_finished with the request's ordered key chain and ReqContext.kv_hints;
   a pure plugin cannot see the finish reason, so (a) is unconditional and every
   last turn becomes an abandoned hold, bounded by the cap and earliest-deadline
   release and measured by the abandoned-hint cases; same shape as Anthropic/Dynamo
   cache_control TTL; (b) a hint-only request/endpoint carrying a KvHintsEnvelope
   after the response (true post-response delivery); (c) a minimal core hook (e.g.
   finish reason in ReqContext, making (a) conditional; engine finish reasons do not
   identify tool calls, which the API layer parses, so (c) reduces to (b)). Both
   routes are built and compared: (a) as the pure plugin (arm A4t, request TTL)
   and (b) as the plugin plus a minimal, hashed core patch (arm A4p,
      post-response, the current product's delivery shape). Their whole-route
   difference (information, timing, abandoned holds and patch overhead together) is
   a reported finding for the RFC, and keeps a TTL failure from being read as
   evidence that holding does not help.
2. Plugin sufficiency. Can an out-of-tree CachePolicy see what it needs: session
   identity (via hint payload), the session's key chain across requests, chunks that
   hit the GPU prefix cache and are never observed by the policy, shared-prefix chunks
   stored by other sessions, window geometry per group? A4t must work as a pure
   plugin; A4p adds only the post-response delivery patch. At Stage 1 exit each
   route is classified independently as feasible, infeasible, or needing a revised
   interface. An infeasible route gets no efficacy verdict and is reported as not
   run. If A4t needs any core change it loses the pure-plugin label and the protocol
   is revised before GPU; a core change needed by both routes applies to both and
   every later result states it.
3. Replacing removed `touch`: show the promote step preserves intended ARC semantics.
4. Make the hold policy-agnostic if feasible (LRU+hint as well as ARC+hint); if not,
   the claim states the hint requires ARC.
5. Hint-off identity: on a recorded schedule, the plugin with hints disabled produces
   the same eviction sequence as stock ARC (and LRU if applicable).
6. Replay simulator: port it to main's manager only if the Stage 1 estimate shows it
   costs less than the GPU work it would replace; it is not required evidence. If
   built, it may estimate the A4t/A4p/perfect-information gap on pilot traces before
   GPU (diagnostic only, never a claim) and is otherwise used only after GPU rank
   agreement is shown (Stage 3).
7. Qwen diverged-GPU-hit lookup: deterministic reproduction on main (scheduler-level
   CPU test if possible, else a tiny GPU probe charged to Stage 2). If it persists,
   file it upstream as a separate bug; it caps any Qwen result.
8. Overhead at production scale: microbenchmark evict/hold bookkeeping with ~1M
   chunks and ~1k held sessions; report p99 evict time; fix ARCHolds.active() union
   rebuild if needed.
9. Trace source admission: name one public agent-trajectory source with revision,
   inspect whether it has usable per-step timing/tool gaps and token content; define
   sampling/truncation and a held-out group; define the causal hint generator
   (deadline predicted from information available at hint time) and a labelled
   perfect-information reference, not a guaranteed upper bound. Fallback if no public
   source has per-step timing: real trace token content with a declared tool-latency
   distribution (Part C), labelled
   semi-synthetic; it may serve as the admitted target-workload case only with that
   label carried into every claim.
Exit: a written go/no-go for Stage 2 with each route's feasibility class, the
route contract (Part C.1), the patch hash and patched-control identity/overhead. Missing
trace admission, feasible models or runtime blocks the runs that depend on them.

### Stage 2. Feasibility screen on the 2080 Ti (needs authorization; proposed cap ~4-5 GPU h, a resource limit, not an evidence requirement)
- Runtime: frozen upstream commit + our plugin; A4p additionally applies the hashed
  delivery patch, as does its control A3p; every other arm runs unpatched.
- Models and priority: D8 dense first, then the admitted Mamba and sliding-window
  hybrids, as in Part C. Failed admission is reported as not run; a missing D8 screen
  does not qualify the later D8 primary comparison.
- Arms: LRU (default), ARC, ARC+plugin with hints off (doubles as the A/A arm and the
  GPU hint-off identity check), ARC+plugin with request-TTL hints (A4t),
  ARC+plugin with post-response hints (A4p) (and LRU+plugin if Stage 1 makes the
  hold policy-agnostic). A4t and A4p use the same mechanism variant, hold cap and
  preassigned adopting sessions; they differ in delivery and the information each
  route may use (Part C.1). A3p, the patched runtime receiving the same
  post-response messages with advice disabled, controls A4p's patch: it supplies
  A4p's A/A band and hint-off identity, and runs in every block that runs A4p. Pools follow the three-seed calibration rule in Part C; calibration
  seeds never decide efficacy; budgets are reported in GiB and as working-set ratio.
- Workloads: (T) W1 first, then (S) the existing closed-loop W2 pressure screen.
  Target eight fresh paired seeds per admitted model/workload, subject to the cap;
  incomplete coverage is reported. Earlier A/A noise motivates a precision check,
  but does not establish the power of this new workload.
- Definitions fixed before runs (per model and workload, paired by seed; every
  statistic is computed inside each paired bootstrap resample over seeds, 10,000
  resamples, BCa intervals):
  - better baseline = whichever of LRU and ARC has lower pooled hinted recompute
    share over the calibration seeds (fixed before decision seeds run);
  - every benefit and harm statistic is computed separately for each hinted arm
    (A4t, A4p) against the better baseline;
  - TTFT benefit = 1 - (mean hinted-return TTFT with the hinted arm) / (same with
    the better baseline); positive is better;
  - recompute change = pooled hinted recompute share with the hinted arm minus the
    better baseline, in absolute points; negative is better;
  - route difference = TTFT benefit of A4p minus that of A4t, with its paired
    interval: the whole-route difference (information, timing, abandoned holds and
    patch overhead), not an isolated estimate of a core hook's benefit; descriptive,
    reported for the RFC, never a decision input;
  - harm metrics (guardrails): unhinted recompute share (absolute points),
    unhinted and one-shot mean TTFT and mean ITL (relative change, positive is
    worse);
  - A/A band, per metric (including each harm metric) = the 95% CI half-width of
    ARC+plugin(hints off) minus ARC, in that metric's units (A3 for A4t; A3p for
    A4p, whose patch overhead A3p minus A3 is reported). CPU eviction identity is
    required; a GPU interval excluding zero triggers diagnosis under Part B.2,
    rather than automatically proving an identity defect;
  - guardrail margin per harm metric = max(A/A band, 5% relative for TTFT/ITL or
    2 absolute points for recompute share);
  - minimum worthwhile benefit (MWB) = TTFT benefit of 0.15;
  - opportunity = under the better baseline at least 20% of hinted returns' reusable
    prefix tokens are recomputed.
- Outcomes, evaluated separately for A4t and A4p and reported per delivery route
  (an A4p-only CONTINUE is labelled "requires a core delivery hook"). Each route
  gets a target-workload verdict from W1 and a separate mechanism verdict from W2.
  Progression follows the W1 verdict only: W2 success shows mechanism potential and
  cannot override a W1 STOP without Philip's explicit scope-change decision; W2
  never turns a W1 INCONCLUSIVE into CONTINUE.
  CONTINUE (per workload) if on at least one model: TTFT benefit point estimate
  >= 0.15 and its CI lower bound > 0; recompute change CI upper bound < 0; for every
  harm metric the point estimate <= its guardrail margin and the CI upper bound
  <= twice the margin; zero added store failures; hint-off identity holds.
  STOP for efficacy, per feasible route, only if that route's cells are valid and
  its hint-off identity holds (A3 for A4t, A3p for A4p), and on the admitted
  target-workload case (T) with opportunity present, its TTFT benefit CI upper bound
  is below 0.15 on every model run. Each route's verdict is independent: either route
  may STOP while the other is CONTINUE, INCONCLUSIVE or infeasible; the idea as a
  whole stops only when every feasible route stops. A STOP applies only at the screened scale (2080 Ti, contexts <= 6k tokens)
  and is reported that way.
  INCONCLUSIVE otherwise, including failure to meet CONTINUE, insufficient
  precision, absent opportunity, or no admitted case T. Precedence: a correctness
  failure or added store failure/resource exhaustion in the route blocks it; then
  STOP; then CONTINUE; else INCONCLUSIVE. INCONCLUSIVE goes to the RFC
  with the evidence as it is and a decision by Philip.
  A stop for budget or time is reported as a resource decision, not as evidence of
  ineffectiveness. Multiple models/workloads and two delivery arms raise the
  false-CONTINUE rate; accepted for a screen and stated with the result.
- Run priority and complete paired blocks follow Part C. Runs never started because
  of the cap are reported as not run.
- Measurement validity, treatment failures and opportunity follow Part C.4. An
  arm-induced server failure or ineffective hint remains evidence, not a reason to
  discard the cell.

### Stage 2b. Early RFC (no compute)
Right after Stage 2 (if CONTINUE or INCONCLUSIVE): short vLLM issue/RFC presenting
both delivery routes and the measured delivery gap (does a post-response hook earn
a core change over request TTL?), the hint action semantics, aligned with TTL `cache_control` semantics (Anthropic/Dynamo) so one
hint could drive GPU pinning and CPU hold; validation, expiry, idempotency,
tenant/session ownership, best-effort behavior, server-side deadline clamp; whether
maintainers want an upstream policy or an external plugin/example. Maintainer feedback
shapes Stage 3 before major spending. Scale-down branch: if maintainers prefer an
external plugin or example, Stage 3 shrinks to the central point on one platform
with optional secondary arms dropped. Correctness, hint-off control and protected
traffic guardrails remain mandatory. If there is no maintainer response within
3 weeks, propose the plugin/example scope to Philip; silence neither grants
maintainer endorsement nor authorizes the next stage.

### Stage 3. PR-grade benchmark (protocol written and committed before any run)
3a. Protocol document (committed by hash before confirmation runs) containing:
- Primary treatment: A4, which is A4t or A4p, chosen after Stage 2b from the Stage 2
  route verdicts and maintainer feedback (A4p only if a core delivery hook is an
  acceptable upstream route) and frozen in the protocol before any Stage 3
  calibration. The other delivery arm runs at the central point as a secondary
  with the same seeds as the primary (with A3p whenever A4p runs); their paired
  whole-route difference is a reported finding, not a second chance at support.
- Claim scope: SUPPORTED means A4 improves the specified native CPU-offload
  configuration. Claims beyond it (better than GPU-only serving, than a tuned
  admission filter, or than an automatic hold) need primary-grade paired evidence
  (primary seeds and decision rule) against that comparator; three-seed comparisons
  are descriptive.
- Primary outcome: SLO-miss ratio = (share of sessions missing the SLO with
  A4) / (share missing it with the better of LRU/ARC), at the central load,
  50% hint adoption, causal hints. A session misses if any return turn exceeds the
  TTFT SLO or it errors/times out; misses stay in denominators. SLO = 2x the median
  return-turn TTFT of the large-pool arm at low load in calibration, frozen before
  confirmation. Central load = the session arrival rate where the better of LRU/ARC
  attains 80-90% session success in calibration (pre-fixed rule). Goodput (sessions
  per hour meeting the SLO) is reported beside it with its ceiling
  (1 / baseline success share) stated.
- Minimum counted sessions: each run counts at least the number of post-warm-up
  sessions needed for an expected 20 baseline misses (set from the pilot), and
  enough observations in every harm cohort (unhinted returns, one-shots) to meet the
  frozen harm precision; the larger requirement wins. Run length and GPU hours are
  re-estimated from that (expect a multiple of the sizing below; the pilot re-quotes
  before the 5080 and rented stages).
- Dilution diagnostic: report the share of baseline misses associated with hinted
  evicted prefixes, without treating it as a causal attribution or hard floor;
  reduced queueing can benefit other sessions. The 0.75 target remains fixed, and
  the hinted-cohort miss ratio is a key secondary.
- Key secondaries: highest tested sustainable arrival rate (>= 90% success and
  bounded backlog; not a located maximum), p50/p95 return TTFT, session completion time, recomputed tokens split into
  GPU reuse / CPU reuse / evicted / never stored, transfer bytes, restore GB/s and
  prefill tokens/s per platform (break-even), iso-latency memory (pool LRU/ARC need to
  match ARC+plugin; full pool curve including where gain is zero), host CPU time,
  metadata bytes and hint traffic per arm, actual hinted request and token shares
  beside the 50% session adoption.
- Guardrails: unhinted and one-shot p95 TTFT and mean ITL, zero added store failures,
  and 0% adoption identity (CPU identity test + GPU A/A). The previous proposed harm
  rule and the pending fixed-margin alternative are in Part C.5; resolve before the
  protocol is frozen. Demonstrated harm (the point estimate beyond the margin) and
  unestablished safety (only the upper bound beyond it) are reported distinctly;
  neither supports release. Failure and correctness rules are in Part C.4.
- Decision: replication unit = workload seed/trace sample paired across arms; the
  miss ratio is the ratio of pooled miss shares, its 95% interval from a paired
  bootstrap over seeds with BCa correction, with the seed-clustered delta-method
  sensitivity for the same pooled ratio in Part B.3. Better baseline = lower primary miss
  share in calibration, fixed before confirmation. SUPPORTED if the CI upper bound of the SLO-miss ratio
  is below 0.75 and guardrails hold; NEGATIVE if the CI lower bound is above 0.75
  (a 25% miss reduction ruled out); otherwise INCONCLUSIVE. Guardrails, regardless
  of the primary: a point estimate beyond the approved margin, or any added store
  failure, resource exhaustion or error, is demonstrated harm (NOT SUPPORTED); an
  upper bound beyond the margin with the point estimate within it is unestablished
  safety (INCONCLUSIVE). Neither permits release.
  Seeds: pilot variance sets the count, minimum 8 for primary cells, 3 for curves;
  maximum budget stated; held-out trace groups.
- Traffic: Poisson session arrivals; within a session the next turn is sent at
  completion + tool gap (causal); TTFT from scheduled send time; record intended
  arrival, actual submission, queue delay, completion. Common warm-up, cohort, drain
  and non-stationary labels follow Part C.3; there is no restart at first eviction.
  No artificial GPU cache cap or two-request limit in PR-grade runs; pressure comes
  from session count. Working set defined in unique reachable KV bytes across groups.
- Workloads: (W1, primary) replayed agent traces: natural system/tool prefixes,
  declared synthetic long-tailed gaps, recorded token content and output lengths replayed
  deterministically, hints from the causal generator including early, late (return
  after expiry), abandoned, duplicate and cancelled hints, deadline sweep; (W2,
  mechanism) synthetic pressure workload; (W3, neutral control) vLLM's own multi-turn
  benchmark without hints. Inactive/no-opportunity cases are retained.
- Core matrix: 3 models (dense, sliding-window hybrid, Mamba hybrid) x arms {LRU, ARC,
  ARC+plugin hints off (A/A and identity), A4} x W1 at central load (8 seeds) + W1 at low and overload
  (3 seeds) + W2 at central load (3 seeds). This is an outline, not an admitted
  schedule or cost quote; Part C fixes model priority and requires a full costed manifest.
- Secondary comparisons at the central point only (3 seeds): no offload, large-pool
  control (not an oracle), GPU TTL pin (Continuum code if it runs, else labelled "our
  implementation"), claim-scoped LMCache, perfect-information hint reference. With
  all primary seeds: the other delivery arm (and A3p), and the automatic hold
  (Part C.2). Their cost is included in the manifest, not a separate assumed allowance.
- Ablations: only those needed to substantiate the selected mechanism (Part C.2).
  Pool sweep (0.25-2x) and adoption sweep (0/50/100%) run on GPU at their listed
  points; the simulator, if built, may prioritize further points only after rank
  agreement with GPU on >= 16 pre-chosen cells, and any reported conclusion needs
  GPU confirmation. Hint-free policy ideas stay outside this feature's readiness
  gate (Stage 4).
- Abuse: greedy-tenant test with a pre-set bound on harm to honest unhinted traffic;
  server deadline clamp; behavior under TieringOffloadingSpec stated.
- Threats tracked: tuned pressure band (publish full curves), hinted gain taken from
  unhinted (all-request primary), queueing hiding savings (report recompute, TTFT and
  completion together), cold start, simulator drift, small-context artifacts (PR runs
  at longer contexts only after admission per Part C), hardware crossover, forking paths (protocol
  hash), state carry-over (clear() between cells, interleaved arm order in time),
  busy-time-union throughput estimate retired (throughput measured directly).
3b. Pilot on the 2080 Ti: one authentic path end to end, failure cases, variance.
3c. RTX 5080: core matrix with longer contexts and larger models.
3d. Rented datacenter GPU: scoped confirmation at admitted contexts, under Part C's
    progression rule and a firm quote first.

### Stage 4. Upstream proposals and release
- Two independent proposals: hint-free cache-policy improvements, only if a residual
  gain on main is GPU-confirmed; and the wait-hint feature. Policy and HTTP exposure
  are pieces of the wait-hint proposal, possibly dependent PRs, not those two tracks.
- Stage 2 supports an RFC or explicitly experimental draft. A performance-backed
  wait-hint PR requires the scoped Stage 3 result, tests, documentation, reproducible
  commands through the maintained entry point, and a results table including failures.
  Use an external plugin/example if that is the agreed delivery route. A4p-based
  claims require the core delivery hook to be proposed with the feature; a pure-plugin
  claim rests on A4t only.
- Before a supported release, qualify the actual client-to-engine hint path, server
  opt-in, validation, tenant/session ownership, expiry, cancellation, resource limits,
  disable/rollback behavior and bounded sustained churn/abandonment recovery. A
  benchmark-only injection path cannot qualify the public interface. Freeze exposure
  duration and resource-growth limits before the operating test; a short probe does
  not establish long-term reliability.
- Require full local acceptance, the new frozen-runtime CPU lane, and transfer and
  same-boundary continuation gates on each claimed model/hardware configuration.
  Record the supported version/configuration matrix and known exclusions. Rebase or
  runtime changes need an impact review and affected requalification; old receipts
  do not silently qualify a new upstream head.
- Record the result under the result profile, verify retained evidence can recompute
  claims, and update current docs and FINDINGS. A performance verdict, engineering
  readiness and publication approval are distinct. Claim scope follows the evidence
  under Part C.5; the harm limit is still pending. No publication is authorized here.
- Related work cited: Dynamo cache-control TTL, Continuum, MORI, InferCept.

## Part B. Interpretation rules

Earlier review caveats, reconciled with Part C and the final review:
1. Warm-up: use the common cutoff and no-restart rule in Part C.3; measure occupancy
   change as a fraction of capacity and keep no-pressure and late-pressure cases.
2. A/A-CI-includes-0 identity check is multiplicity-prone across ~6-7 metrics: treat an
   exclusion as a trigger to re-verify the Stage 1 CPU identity and inspect, not an
   automatic defect.
3. Eight seeds may give unstable intervals: report a sensitivity interval for the
   same estimand beside BCa. For pooled ratios (miss ratio, TTFT benefit) it is a
   seed-clustered delta-method interval on the log of the pooled ratio, weighting
   seeds by their counted sessions or returns; for pooled absolute changes (e.g.
   recompute share in points) it is a seed-clustered delta-method interval for the
   same pooled difference, with each share's numerator and denominator summed over
   seeds. Disagreement on a decision threshold => INCONCLUSIVE. A geometric mean of
   per-seed ratios or an unweighted mean of per-seed differences is a different
   estimand: report it only as a labelled secondary, never as a veto. Predeclare the exact
   computation and behavior for zero misses, undefined ratios, degenerate bootstrap
   or jackknife samples, and missing observations; no post-result method switching
   or silent pseudocounts. Undefined required evidence cannot establish support.
4. Harm metrics compare each hinted arm with the better baseline; A/A band measured against
   ARC even when LRU is better (state it).
5. Report benefit against both LRU and ARC, not only the selected better baseline.
6. Schedule Stage 2 in complete paired-seed blocks; re-quote after the first block.
7. Stage 2 retains its screening harm rule. Stage 3's release margin is pending
   Philip's decision in Part C.5.
8. Opportunity (>= 20%) is the pooled point estimate on the decision seeds.
9. STOP also requires valid cells and hint-off identity.
10. Multiplicity in CONTINUE, STOP scope, semi-synthetic labelling of case T: as in Part A.

Scope limits:
- Stage 2 is a screen; efficacy STOP limited to the tested scale; absent opportunity or
  inadequate precision stays INCONCLUSIVE.
- Trace admission, feasibility and the pilot resolve protocol details before dependent
  runs; simulator conclusions need GPU confirmation; spending estimates provisional.
- Acceptance does not authorize implementation or compute.

## Part C. Concrete workloads, models, comparators and stages

### 0. Frozen inputs
- vLLM main e9281bd939d4ade90d6aacf4b2319aa89ea46c39; prefix caching on;
  prefix_cache_retention_interval=0 (main default); TP=1; no speculative decoding, no
  LoRA; unquantized KV.
- Models (Hugging Face revisions; local snapshots identity-checked at admission):
  - D8: Qwen/Qwen3-8B-AWQ (revision pinned at admission) - primary dense model on
    all three platforms (see 0.1).
  - D4: Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554 BF16 -
    unquantized dense on the 5080.
  - D8B: Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218 BF16 - rented only.
  - HM: cyankiwi/Qwen3.5-4B-AWQ-4bit@d3e78b50c727b1f34802f440ca84b25781625352 (Mamba
    hybrid). HS: axteam/gemma-4-12B-it-AWQ-4bit@7c781be42ce8d29d0975b857f6d0c0b2558d9b63
    (sliding-window hybrid; Turing attention patch applied identically to all arms).
  - Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306 and smaller:
    functional fixtures only, no performance claims.
- 0.1 Why D8 instead of the 1.5B model both reviewers proposed for the 2080 Ti:
  measured from local configs, KV bytes/token are 28,672 for Qwen2.5-1.5B and 147,456
  for Qwen3-8B. On 11 GB at utilization 0.85 the 1.5B model leaves room for an
  estimated ~185k GPU KV tokens (sixteen 6,144-token prefixes occupy ~2.6 GiB for 1.5B
  versus ~13.5 GiB for D8), so the 1.5B model is unlikely to create substantial GPU
  eviction in a short screen unless the GPU cache is artificially capped. Qwen3-8B-AWQ (5.7 GB
  weights) leaves an estimated ~18k GPU tokens: natural GPU pressure with 4+ concurrent
  6k-token sessions, a common production size, and it runs on all three platforms as a
  cross-hardware anchor. A CPU pool of 2 GiB holds ~14.6k D8 tokens, 4 GiB ~29k. Actual
  allocated GPU KV blocks and residency are recorded at admission; no-pressure outcomes
  stay visible. D8 activation/KV dtype is FP16 on every platform for the anchor
  comparison (hardware-specific kernels still limit causal hardware attribution). D8's
  native context limit is 40,960 positions (no RoPE scaling in its config).
- Content: SWE-bench/SWE-smith-trajectories@08e109b4a59eaeebf80e4675cd125d42e7ac99a4,
  split `tool` (messages with tool actions/observations; no timestamps). Freeze row IDs.
- Upstream control: vllm benchmarks/multi_turn at the frozen commit, checked-in
  generator config, frozen pg1184.txt SHA-256.
- LMCache: 0.5.6 (tag 3e11e10812b448142c6cdb177a3058652bd826fa) as first candidate;
  if incompatible with frozen main, select and freeze a verified replacement commit
  before any comparison.

### 1. Workloads
W1 (primary, semi-synthetic agent replay):
- Sampling: dedupe by trajectory/task identity (tool/xml/ticks renderings are not
  independent); group by repository; SHA-256(repo) mod 10: 0-1 calibration, 2-3 pilot,
  4-9 confirmation; within each partition sort trajectory IDs by SHA-256 and admit all
  eligible distinct tasks up to 2,048, no success filter; freeze manifest, exclusions,
  inclusion rates and context/turn distributions before GPU.
- Stage 2 and all engineering/variance pilots use only the pilot partition;
  calibration uses only calibration repositories. Confirmation repositories remain
  untouched by performance-driven development. Freeze the repository-ID mapping,
  deduplication choice among multiple trajectories for one task, tokenizer/chat
  template and tool-schema rendering, output-length tokenization, and context-stratum
  assignment at admission. Report the number of distinct repositories/tasks in each
  partition and source-level success selection, even though we add no success filter.
  The pinned dataset card's narrative and split counts differ; row inspection, not
  its narrative alone, must establish the admitted population.
- Task reuse: each seed orders its partition's tasks by SHA-256(seed, task ID) and
  assigns them to sessions in that order; if a run needs more sessions than distinct
  tasks, it cycles through the same order (deterministic reuse). Session IDs are
  distinct from task IDs; the reuse count per run is reported, and a sensitivity
  restricted to first occurrences of each task within the run is reported beside the
  primary. Across-seed intervals describe schedule variation on this fixed corpus;
  reshuffling the same tasks does not create independent evidence about new repositories.
  Report cross-session reuse from repeated tasks, including warm-up reuse, separately
  from within-session return reuse. First-occurrence filtering alone does not remove
  effects of duplicates already present in the shared cache.
- Replay semantics ("transcript replay"): each turn sends the canonical recorded
  prompt (natural system/tool prefixes and observations preserved, no manufactured
  prefix), requests the recorded assistant output length (cap 256 on the 2080 Ti,
  native up to 2048 elsewhere; cap frequency reported) with ignore_eos. Generated text
  is not fed back; reuse is measured as the exact token LCP with the session's
  previous requests and the group-aligned eligible boundary; lost reusable suffix
  reported. A live mini-swe-agent lane on the 4 SWE-bench tasks is an application
  sanity check only, deferred from the minimum PR package. Freeze its task IDs, graders, budgets, pairing and treatment of
  quality regressions before running it. Replay supports serving-latency claims on
  recorded prompts, not faster problem solving or unchanged application quality.
- Episodes: chronological; 2080 Ti: first turns until prompt+output > 6144, stop at
  first overflow, 2-8 turns; 5080/rented: context strata <= 8k, 8-16k, 16-32k (64k
  rented only); never delete transcript middles.
- Gaps (declared, not production): min(300, LogNormal(ln 4 s, 0.8)), drawn per turn
  from a seeded stream independent of arms; sensitivities median 0.5 s and 30 s.
- Hints (causal): per-session adoption 50% primary (0/100 sensitivities); adopting
  sessions are preassigned by seed and identical for A4t, A4p and their controls,
  whatever hints are accepted. TTL fixed at 12 s (3x the declared median gap) in
  every cell, including the gap sensitivities, so those test robustness, not
  retuning; TTL sensitivities 2 s and 60 s; server clamp 60 s.
  Perfect-information hint (TTL = true gap) is a labelled oracle reference,
  secondary only (not assumed to be an upper bound).
- Route contract, frozen at Stage 1 exit:

  | | A4t (request TTL) | A4p (post-response) |
  |---|---|---|
  | Information | Session adoption only | Plus the current recorded assistant turn |
  | Eligible | Every request of an adopting session, final included | Turns whose recorded assistant message contains a tool action other than a declared terminal action (e.g. submit) |
  | Published | With the request | After the client receives the response's last token, plus measured transport delay |
  | Deadline origin | Engine request finish | Engine hint acceptance |
  | Stores finishing later | Join the hold when ready, until the deadline | Same |

  Neither route may use whether another recorded turn exists, the next prompt or the
  true gap. Replay discards generated text, so A4p's signal comes from the recorded
  turn, not live tool-call parsing; benchmark injection of either route does not
  qualify the public interface (Stage 4).
- Episode ends: report natural trajectory completion, benchmark truncation
  (context/turn limits) and injected abandonment separately, because truncated
  ends become abandoned holds for both routes. Robustness abandonment percentages are
  additional abandonment on top of these. The route difference is claimed only for
  the admitted (truncated) population; the 5080 strata include longer, more often
  complete episodes as a sensitivity.
- Gap/TTL sensitivities (0.5 s and 30 s median gaps; 2 s and 60 s TTL) are required
  at the central point with 3 seeds on the primary model for any claim beyond the
  4 s distribution; otherwise the claim is limited to that distribution.
- Robustness variants (separate cells, never mixed into the primary): abandonment
  0/34/75%, 10% duplicate hints, 10% cancellations, late returns (gap > TTL forced),
  greedy tenant sending 90% of hints, +1,024-token common-prefix variant.
- One-shot stream for harm guardrails: independent Poisson stream at an absolute rate
  fixed per platform from baseline calibration (10% of the baseline arm's calibrated
  request rate), identical across arms; 2,048 input / 128 output tokens.
- Context-stratum weights are frozen in the manifest (strata weighted by their
  inclusion share); task reuse follows the rule above, within and across seeds.
W2 (mechanism): the exact existing pressure.json generator (historical bridge, labelled
synthetic; keeps its 60 s hint) plus a 12 s-TTL variant; long-context variant for the
5080/rented: initial 16,384-24,576, growth 512-2,048, outputs 128-256, 4-8 turns, max
40,960 for Qwen3 models (65,536 only for a model whose native or separately validated
context-extension configuration admits it), no compaction.
W3 (neutral control): upstream multi_turn, 24 conversations, 12-18 messages, no hints;
integration/control only, not a capacity benchmark.

### 2. Arms and comparators
Core (every primary cell; offload_prompt_only=True explicitly for A1-A4, the main
default, whereas the old connector forced False): A1 stock OffloadingConnector LRU;
A2 stock ARC; A3 ARC+plugin hints off (A/A and identity); A4t ARC+plugin with
request-TTL hints (pure plugin); A4p ARC+plugin with post-response hints (plugin plus
the hashed delivery patch). Stage 2 runs both; Stage 3 names one as A4 (Part A) and
runs the other at the central point. A3p: patched runtime with matched
post-response messages and advice disabled (patch control; eviction identity with
A3 on CPU, overhead measured). Both hinted arms use exactly one mechanism variant (hold-only, promote+hold, or promote+hold+dead-first) frozen at
the end of Stage 1 from feasibility and CPU results only, never selected after GPU
benefits are seen. Preserve the current 50% CPU hold cap for the primary, counting
the union of ready physical chunks once, with earliest-deadline yield before a store
would fail. Freeze metadata/session/action limits, overflow/rejection behavior and
tenant accounting at Stage 1 exit; cap sweeps are separate secondary cells.
A5 LRU+plugin if Stage 1
shows the hold is policy-agnostic, with central-point coverage at primary seeds
because LRU is the default; otherwise the claim is stated as ARC-dependent.
A6 automatic hold (primary seeds, central point): the same mechanism, TTL, physical
cap and release rules applied by the server after every request, without client
advice. It tests whether client information earns a new interface; a matched hold
budget diagnostic may separate selectivity from resource use.
Required at the central point (3 seeds unless noted):
- No CPU offload (GPU prefix cache only).
- Hint-free admission: A1 and A2 with store_threshold=2, max_tracker_size=64,000.
- Matched all-False sensitivity: offload_prompt_only=False for A1, A2 and A4 together.
- Large-pool capacity control and the pool curve (0.25/0.5/1/2x frozen working set,
  where RAM permits).
- Ablations: only those that isolate components of the selected mechanism
  (hold-only always; promote and dead-first only if selected or claimed). If
  dead-first is selected, add a small compaction/branch-return sensitivity.
- GPU TTL pin ("our GPU TTL implementation", patch hashed; Continuum fork
  Hanchenli/vllm-continuum@316a58794a6ff86b216e579b74fd56ed0c5a911f inspected first,
  never called Continuum): same causal hint, resident pages only, cap 50% of GPU KV,
  release on return/expiry, earliest-deadline yield under allocation pressure. Run
  before the full matrix. Cap sweep and GPU pin + CPU hold are deferred unless the
  RFC makes them consequential.
Claim-scoped (required only for the claims named):
- LMCache (any superiority claim over LMCache): LMCacheMPConnector, lmcache_driven,
  kv_both, CPU-only L1, LRU, separate object groups, no disk/remote/compression/
  blending, matched usable KV bytes, total host/staging memory reported, chunk size
  derived from admitted layout (256 dense), observability enabled; actual usable GPU KV
  blocks recorded for LMCache and the native arms: a policy-isolating lane at matched
  usable GPU KV capacity plus a natural-deployment lane at the same hardware memory
  envelope, each labelled; restored-byte, continuation, eviction and
  counter-reconciliation checks per admitted layout; an incompatible or unobservable
  lane supports no ranking. LMCache pin/unpin with matched causal hints only if pinning
  is verified in the selected MP connector version to protect the local CPU tier. WSL lane:
  exactly lmcache-peer, lmcache-copy, lmcache-allocator, LMCACHE_WSL_VMM=1, expandable
  segments, labelled, no general ranking; native-Linux lane unpatched, default
  transfer path.
- SGLang HiCache pinning (Dynamo's agent hints are SGLang-backed, protect host
  entries, pin budget default 0, TTL clamp 300-3600 s): one dense model with its own
  no-pin control; required before claiming CPU wait-hold is absent from competing
  systems.
Optional: full Continuum scheduler, MORI, InferCept, NVMe/remote tiers, KVBM, TP>1,
quantized KV. CPU-only TieringOffloadingSpec lifecycle test in Stage 1.

### 3. Stage plans
Stage 1 (CPU only; no GPU use):
- Hold-only feasibility first on LRU and ARC; promote and dead-first as extras.
- Deterministic tests: hint before execution, TTL from finish, stores completing after
  finish (mark_evictable as readiness hook), return, expiry, duplicate action IDs,
  cancellation, reset, failed transfer, failed-eviction atomicity; shared keys owned by
  several sessions (hold cap counts the union; fix current hold() double counting);
  GPU-only prefix hits (does the policy see the complete reusable CPU prefix?);
  window/recurrent checkpoints and forks ("locally unreachable", not globally dead);
  promotion without touch; hint-off identical eviction sequences for LRU and ARC;
  capacities 1/2/16/1,024 chunks and hold shares 0/0.25/0.5.
- Bookkeeping microbenchmark: 1e4/1e5/1e6 chunks x 0/100/1,000 held sessions;
  p50/p99 evict latency and metadata bytes.
- Qwen diverged-GPU-hit reproduction at scheduler level on CPU where possible;
  source-level check of Turing kernel support and Gemma patch applicability; runtime
  builds and CPU tests of the frozen runtime.
- New open-loop streaming load generator (the existing proxy serializes requests and is
  non-streaming); actual per-group store/load/eviction events and cache-source hit
  accounting (GPU vs CPU) replace inferred eviction and aggregate cached tokens;
  process-tree PSS, page faults, swap and transfer bandwidth sampling.
- Admit the W1 manifest, LMCache candidate and compatibility tuple (CPU-checkable
  parts), freeze the shared A4t/A4p mechanism variant and route contract, hash the
  A4p delivery patch, and show A3p's eviction sequence equals A3's on recorded
  schedules with its overhead measured.
- Produce a staged implementation estimate before building: distinguish the minimum
  end-to-end screen path from simulator/competitor/full-matrix work. Extend
  tools.benchmark and the current receipt/verification lifecycle; add no independent
  runner. The existing release profile is unqualified, targets the old runtime and
  cannot execute or certify this plan. Update the benchmark contract and guide when
  implementing the new protocol; do not resume release-01.
Stage 2 admission block (2080 Ti, GPU; separately budgeted, proposed <= 1 GPU h,
before any performance cell): runtime build on sm_75; model probes for D8, HM, HS;
restore correctness (restored bytes, same-boundary continuation) per model; actual GPU
KV blocks and residency; diverged-hit probe if not settled on CPU; LMCache layout
checks if the LMCache lane is in scope; loading and staging memory peaks. A model that
fails admission is "not run", not evidence against the mechanism.
Stage 2 (2080 Ti screen; Part A rules and Part B caveats):
- FP16 activations/KV, eager, gpu_memory_utilization 0.85 every arm, max_model_len
  6144, batched tokens 2048 (512 for HS, consistent across its arms).
- Run families in priority order: T = W1 on D8, then T on HM, then T on HS, then W2
  bridge (keeps its old 2-request limit and Qwen 36-block override, labelled
  constrained).
- T settings: no GPU-block override; max_num_seqs 16; 16 sessions per cell at a
  calibration-frozen arrival rate targeting 8 overlapping sessions; 900 s ceiling
  including drain; unfinished sessions are failures.
- Pools (deterministic, paired; same rule on every platform with its own candidates):
  working set = unique reachable KV bytes across overlapping sessions, summarized as the
  median over calibration seeds of the time-averaged value under the baseline arm.
  Among candidates (2080 Ti: 1, 2, 4 GiB) that hold the largest eligible conversation
  and meet resource limits (including loading/staging peaks, >= 2 GiB available RAM,
  no swap) for ALL compared arms, select the one nearest 0.5x that working set; ties
  choose the smaller. Freeze it across arms. If any arm cannot fit, report that
  failure and rerun the complete paired comparison at the next qualifying budget; if
  none qualifies, report resource infeasibility.
- Arms A1-A3, A4t, A3p, A4p (+A5 if feasible); the A4p route adds two arms per
  block, so the cap covers fewer seeds or models, re-quoted after the first block. Seeds: calibration 12001-12003; decision 12101-12108,
  checked against the exposure ledger; complete paired blocks, randomized arm order;
  re-quote after the first block; 4-5 GPU h cap.
Stage 3a/3b (protocol and 2080 Ti pilot):
- Primary comparison named now: W1, D8 on the RTX 5080, central load, 50%
  adoption, A4 vs the better of A1/A2. Other models/platforms are supporting results,
  not additional chances to declare overall support. A rented-GPU confirmation, if it
  happens, is a second pre-registered comparison (D8B, W1).
- Open-loop Poisson session arrivals; next turn at completion + gap; arrival times and
  gap draws paired across arms (realized timestamps may differ).
- Platform-local calibration: arrival rate, SLO, working set, pool, one-shot rate,
  A/A band and variance are calibrated separately on each platform; the 2080 Ti pilot
  never sets them for the 5080 or H100.
- Before platform calibration, commit an ordered calibration algorithm with an
  initial reference arm/load/pool, large-pool SLO reference policy, tie-breaks,
  stopping/resource limits and treatment of infeasible candidates. Baseline selection,
  central load and working set currently depend on one another; resolve that order
  in the protocol rather than choosing a favorable combination afterward. Select
  Stage 3's better baseline by its calibrated primary miss share, not Stage 2's
  recompute share; compare both native arms under the same frozen settings.
  Use calibration/pilot partitions on the 5080 and rented hardware, never confirmation
  data, to set local variance, session counts and seed count. Specify a numeric
  sample-size rule and maximum before those pilots; a ceiling that cannot attain
  the required precision yields INCONCLUSIVE, not a relaxed threshold.
- Load calibration: rates {0.25, 0.5, 1, 2}/D (D = low-load mean session service
  demand excluding gaps), expand geometrically until baseline success falls below 85%,
  then bisect 3 times; central = success nearest 85%; also 0.5x and 1.25x central
  ("high load"; "overload" only when measured backlog growth shows it).
- Measured count per run N, separate from warm-up arrivals, is the largest of:
  200; ceil(20 / pilot baseline miss share); and, for each required harm cohort
  (unhinted returns, one-shots; TTFT p95 and ITL), the sessions or run duration the
  frozen sizing algorithm needs for that cohort's interval to meet its precision
  ceiling at the pilot variance. The efficacy formula is a floor, not the sizing
  rule.
- Paired schedule: before a block runs, one seeded schedule per tested load (session
  IDs, task assignments, arrival times, gap draws, hint adoption, one-shot arrivals)
  is generated at that load's actual arrival rate and replayed identically for every
  arm. Session arrivals are generated through the Nth session arriving after the
  six-window warm-up deadline (so N post-cutoff arrivals exist whatever the cutoff);
  background session and one-shot arrivals then continue through that Nth arrival time
  plus the drain timeout, which is the common endpoint for every arm. Sessions other
  than the measured cohort run but are not counted. If the run budget cannot cover the
  whole schedule, the attempt is retained and the comparison is INCONCLUSIVE; the
  cohort is never shortened. Only acquisition interruption or insufficient authorized
  run time counts as resource incompleteness; sessions unfinished at the scheduled
  endpoint of a run that reached it are failures of their arm.
- Warm-up and measured cohort: windows of max(120 s, 3x pilot mean session duration).
  Stability between two consecutive windows requires all of: CPU-pool occupancy changes
  by < 2% of capacity (absolute, so an empty pool is stable); median return TTFT changes
  by < 10%, evaluated only when both windows contain >= 10 return turns (when the
  schedule offers fewer than 10 returns in a window the criterion is not applicable;
  when it offers >= 10 but fewer complete, the criterion is unmet); backlog (queued +
  running requests) changes by < max(10%, 2 requests), zero backlog in both windows
  counting as stable, with no monotone growth across three windows.
- Common cutoff = the latest stabilization point among the block's core arms (A1, A2,
  A3, A4) on the shared schedule clock, applied unchanged to secondary arms so adding
  a comparator cannot move the primary cohort; bounded by the 6-window deadline; if any core arm has not stabilized by
  then, the cutoff is the deadline and the block is labelled non-stationary and
  retained. There is no restart: if the first eviction in any arm occurs after the
  cutoff, the block is labelled late-pressure, its eviction time is recorded, and the
  measured cohort is also reported split before/after that time as a secondary.
- The measured cohort = the first N scheduled sessions arriving after the cutoff,
  identical session IDs for every arm. Drain timeout after the last measured arrival =
  max(600 s, 3x pilot p95 session duration); unfinished measured sessions are
  failures. If acquisition is interrupted or authorized run time runs out before the
  scheduled endpoint, the attempt is retained and the comparison is INCONCLUSIVE; the cohort is never shortened and no
  arm is omitted. No-pressure is labelled by absence of evictions in the measured
  cohort.
- Harm margin and A/A band frozen from calibration with a precision ceiling: if the
  A/A band exceeds 10% relative TTFT or 5 points recompute, the comparison is
  INCONCLUSIVE for inadequate precision. Define the ITL precision ceiling and the
  minimum observations for each harm cohort in the protocol too. This is a noise
  check, not permission to increase the pending release harm limit.
- 0.75 miss-ratio target kept as the declared worthwhile effect; the dilution estimate
  is an observational diagnostic, not a hard floor.
- Pilot seeds 13001-13003; confirmation 13101-13108 (count raised only by a
  predeclared pilot rule). Pilot on the 2080 Ti with D8 at <= 6k only.
- Cost admission: the shared schedule always runs through the six-window warm-up
  deadline, arrival of N more sessions, then the drain timeout, even if stability is
  reached early. With minimum 120-second windows and 600-second drain this is at
  least 22 minutes plus arrival time per arm, before startup/calibration. Retire the
  old six-minute-run estimate. Quote the complete prioritized cell manifest, warm-up,
  arrivals, drain, startup, admission, calibration, diagnostics and contingency before
  authorizing each platform; map every required cell to the claim it protects and
  predeclare the drop order (claim-scoped and optional cells first, then curves,
  never the primary cohort, its controls or harm cohorts) for when the budget cannot
  reach primary precision. The outlined core matrix alone exceeds 70 GPU hours at
  the minimum run length. Under a cap, run complete paired blocks and retain
  incomplete attempts; reduce optional scope before confirmation, never its cohort
  after seeing results. Hours-long acquisition needs a progress healthcheck automation
  at a declared cadence, stopped when the job ends, as required by AGENTS.md.
Stage 3c (RTX 5080):
- Models in order: D8 (anchor, same AWQ build as the 2080 Ti), D4 BF16, HM and HS if
  admitted; host RAM measured first.
- gpu_memory_utilization 0.90, CUDA graphs where qualified, max_num_seqs 64, batched
  tokens 4096; max_model_len 32768 (within D8/D4 native limits) only after a
  longest-request restore and concurrent-serving probe passes; longer contexts only for
  a model whose native or separately frozen and validated context-extension
  configuration admits them, as a separate configuration; declared shorter context if a
  probe fails.
- Pools 2/4/8 GiB conditional on host headroom; chosen by the same paired rule with
  platform-local calibration.
- Order: W1 central with A1-A4 (primary), the other delivery arm (and A3p), A5 and
  A6, GPU TTL pin, gap/TTL sensitivities, required comparators,
  then curves, robustness variants, W2 long-context, W3.
Stage 3d (rented, only if 3c is SUPPORTED, or Philip explicitly chooses to explore a
positive but INCONCLUSIVE result; firm quote first):
- One H100 80GB, native Linux, >= 128 GB host RAM, TP=1.
- D8B BF16 (second pre-registered comparison), D8 anchor, one hybrid; contexts 32k, and
  64k only under a separately validated context-extension configuration (D8B's native
  limit is checked from its pinned config); utilization 0.90; batched tokens 8192; max_num_seqs 128; pools
  8/16/32 GiB; native-Linux LMCache lane; restore-vs-recompute break-even measured.

### 4. Cell validity (gates), treatment outcomes (evidence), opportunity (label)
Validity gates (infrastructure and measurement only): input/config/runtime hashes;
complete offered/admitted/completed accounting and captures; streaming client TTFT and
ITL recorded; actual per-group store/load/eviction events recorded; no external
interference (competing GPU activity, host processes outside the experiment) and no
capture failure. An invalid cell is rerun as a complete paired block and the invalid
attempt is retained.
Treatment outcomes (always evidence, never invalidating): store failures, resource
exhaustion (OOM, swap, allocation failure), timeouts and errors caused by an arm count
in its results; any added store failure, resource exhaustion or error rate above the
baseline caused by a hinted arm fails the guardrail and prevents a positive verdict. Hint
acceptance vs effective held bytes, restored-byte and same-boundary continuation checks
are reported per arm; a correctness failure in an arm is a blocking result for that
arm.
Opportunity (label only, never a validity gate): evictions of returning sessions'
chunks, reusable prefix share, residency. Zero-eviction or zero-benefit valid cells
stay in results.

Before acquisition, qualify the streaming instrument end to end: verify scheduled
versus submitted times, first generated token versus role-only/empty stream events,
token-versus-chunk ITL semantics, backpressure, late clients, cancellation and server
death. The client must not silently serialize traffic. Use hand-verifiable event
traces and an injected delay/failure to show the verifier detects a known regression;
measure event-capture overhead against a minimally instrumented control. The SLO
primary is return-TTFT compliance, not total-session responsiveness: also report first
turn latency, decode/ITL, session completion time and failure denominators.

Freeze per-request and session deadlines and how an early failure terminates later
turns. Failed measured sessions stay in the primary denominator; latency summaries
must show their censoring/failure counts so survivor-only tails cannot imply safety.
Freeze which one-shots belong to the guardrail measurement interval. Define finite
horizon conclusions for non-stationary cells; they cannot establish sustainable
capacity. Publish the rule for combining disagreements across the 5080 primary,
supporting models and any rented confirmation; a failed replication remains visible.

Retain every attempt with its protocol/config/source identity and terminal state.
Resume only never-started paired blocks; source or protocol changes start a new
campaign, and an invalid-block rerun keeps its predecessor. Exercise interruption
and verification before long acquisition. Hardware admission and claim scope are
recorded per platform; passing on the 2080 Ti does not qualify the 5080 or H100.

### 5. Release judgment and remaining decisions
- Philip's direction on 2026-11-13: competitor superiority is desirable, but the
  eventual release claim depends on results; use judgment on scope. Establish native
  vLLM value first, then prioritize qualified competitor comparisons before broader
  sweeps when that evidence justifies continuing. A scoped PR/plugin remains useful
  without a superiority result. Each competitor claim needs its own predeclared
  comparison, paired uncertainty and resource matching; passing the native-vLLM
  primary does not establish superiority over another system. Keep unfavorable
  competitor results visible. Compute budgets still need separate authorization.
- Release harm limit: the original noise-dependent proposal permits a point estimate
  up to max(A/A band, 5%) and an interval upper bound twice that margin; with the
  TTFT precision ceiling it can allow 10% measured harm and 20% upper uncertainty.
  Recommend a fixed 5% margin for unhinted/one-shot p95 TTFT and mean ITL, requiring
  the upper confidence bound to stay within it. Insufficient precision would be
  INCONCLUSIVE. Noise sets the evidence required, not the harm tolerated. The
  final reviewers (GPT-6 Astra, GPT-6.1 Sol) both support the fixed limit. Pending
  Philip's decision; resolve it before Stage 3 calibration and sample sizing.
- Why protect unhinted traffic: server opt-in and per-session hints do not partition
  the shared CPU cache or GPU execution queue. Protecting one session can evict
  another session's prefix and add recomputation/queueing. Yield-before-store-failure
  prevents a capacity deadlock, not latency interference. The primary's 50% adoption
  intentionally tests mixed traffic; 0% tests disabled behavior and 100% tests full
  adoption. A deployment restricted to isolated, fully participating workloads can
  support a narrower claim, but is not evidence of safety for mixed deployments.
  Philip asked for this distinction to be explained; no numeric harm tolerance has
  been accepted yet.
- WSL memory limit (~15 GB) caps CPU pools at a few GiB on Machine A; raising it is a
  Windows-side change only Philip can approve. Without it, Stage 2 pools stay <= 4 GiB.
- Stage authorizations and budgets remain separate. The bounded Stage 1/2080 Ti
  authorization is recorded at the top of this document. No later GPU platform,
  paid compute, memory-limit change or external posting is authorized.

Final-review source checks: [frozen vLLM CPU spec](https://github.com/vllm-project/vllm/blob/e9281bd939d4ade90d6aacf4b2319aa89ea46c39/vllm/v1/kv_offload/cpu/spec.py)
confirms the LRU default and configurable policy module; the
[pinned trajectory card](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/blob/08e109b4a59eaeebf80e4675cd125d42e7ac99a4/README.md)
supplies the source schema, not runtime compatibility or population qualification.
