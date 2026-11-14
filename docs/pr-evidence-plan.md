# Evidence plan for a vLLM wait-hint pull request

Status: agreed plan, not authorized for implementation or compute. Each stage needs
Philip's authorization; see [the task](../tasks/pr-evidence.md). Settled on
2026-11-13 through two structured reviews: the stages and decision rules (Part A) by
GPT-6 Astra and Claude Fable 5.2, the concrete workloads, models, comparators and
machine tiers (Part C) by GPT-6 Astra and GPT-6.1 Sol. Every reviewer explicitly
accepted the final text. Part B lists caveats they accepted, to settle before runs.
Where Part C is more specific than Part A, Part C governs.

Background facts verified against upstream vLLM:
vLLM's CPU offload default eviction policy is `lru` (both at the pin and on main,
`vllm/v1/kv_offload/cpu/spec.py`); all our earlier results used opt-in ARC.
`ReqContext` on main carries `kv_hints` but no session id; policies are constructed
with capacity only.

## Part A. Stages and decision rules

### Stage 0. Record (no compute)
Commit the plan; correct the campaign record (seed overlap with design data;
upstream baseline change; LRU, not ARC, is the default).

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
   finish reason in ReqContext, making (a) conditional). Default: (a) for the pure
   plugin; (b)/(c) are RFC refinements, not prerequisites.
2. Plugin sufficiency. Can an out-of-tree CachePolicy see what it needs: session
   identity (via hint payload), the session's key chain across requests, chunks that
   hit the GPU prefix cache and are never observed by the policy, shared-prefix chunks
   stored by other sessions, window geometry per group? Decide explicitly: pure
   plugin, or plugin plus a minimal core patch; every later result states which.
3. Replacing removed `touch`: show the promote step preserves intended ARC semantics.
4. Make the hold policy-agnostic if feasible (LRU+hint as well as ARC+hint); if not,
   the claim states the hint requires ARC.
5. Hint-off identity: on a recorded schedule, the plugin with hints disabled produces
   the same eviction sequence as stock ARC (and LRU if applicable).
6. Port the replay simulator to main's manager; it is used only after GPU rank
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
   perfect-information upper bound. Fallback if no public source has per-step timing:
   real trace token content with a published tool-latency distribution, labelled
   semi-synthetic; it may serve as the admitted target-workload case only with that
   label carried into every claim.
Exit: a written go/no-go for Stage 2 with the chosen plugin/patch shape. Missing
trace admission, feasible models or runtime blocks the runs that depend on them.

### Stage 2. Feasibility screen on the 2080 Ti (needs authorization; proposed cap ~4-5 GPU h, a resource limit, not an evidence requirement)
- Runtime: frozen upstream commit + our plugin (or plugin + minimal patch).
- Models: Gemma-4-12B AWQ (sliding window) and Qwen3.5-4B AWQ (Mamba hybrid) required;
  one dense full-attention model included if a model probe admits one that fits. If
  none is admitted, any Stage 2 claim is restricted to hybrids and dense coverage
  moves to Stage 3 confirmation.
- Arms: LRU (default), ARC, ARC+plugin with hints off (doubles as the A/A arm and the
  GPU hint-off identity check), ARC+plugin (and LRU+plugin if Stage 1 makes the hold
  policy-agnostic). Pools recalibrated by a rule fixed beforehand on one calibration
  seed not used for decisions; reported in GB and as working-set ratio.
- Workloads: (S) the existing closed-loop pressure benchmark, labelled a screen, 8
  fresh paired seeds per model (proposed; chosen because A/A noise reached 8-27
  points, so fewer seeds would almost surely be INCONCLUSIVE); (T) the admitted
  target-workload case from Stage 1 on at least one model, paired seeds, as many as
  the cap allows (target 8).
- Definitions fixed before runs (per model and workload, paired by seed; every
  statistic is computed inside each paired bootstrap resample over seeds, 10,000
  resamples, BCa intervals):
  - better baseline = whichever of LRU and ARC has lower pooled hinted recompute
    share on the calibration seed (fixed before decision seeds run);
  - TTFT benefit = 1 - (mean hinted-return TTFT with ARC+plugin) / (same with the
    better baseline); positive is better;
  - recompute change = pooled hinted recompute share with ARC+plugin minus the
    better baseline, in absolute points; negative is better;
  - harm metrics (guardrails): unhinted recompute share (absolute points),
    unhinted and one-shot mean TTFT and mean ITL (relative change, positive is
    worse);
  - A/A band, per metric (including each harm metric) = the 95% CI half-width of
    ARC+plugin(hints off) minus ARC, in that metric's units; the hint-off identity
    check additionally requires each A/A CI to include 0 (a systematic offset is an
    identity defect that blocks interpretation, not noise);
  - guardrail margin per harm metric = max(A/A band, 5% relative for TTFT/ITL or
    2 absolute points for recompute share);
  - minimum worthwhile benefit (MWB) = TTFT benefit of 0.15;
  - opportunity = under the better baseline at least 20% of hinted returns' reusable
    prefix tokens are recomputed.
- Outcomes:
  CONTINUE if on at least one model and workload: TTFT benefit point estimate
  >= 0.15 and its CI lower bound > 0; recompute change CI upper bound < 0; for every
  harm metric the point estimate <= its guardrail margin and the CI upper bound
  <= twice the margin; zero added store failures; hint-off identity holds.
  STOP for efficacy only if, on the admitted target-workload case (T) with
  opportunity present, the TTFT benefit CI upper bound is below 0.15 on every model
  run. A STOP applies only at the screened scale (2080 Ti, contexts <= 6k tokens)
  and is reported that way.
  INCONCLUSIVE otherwise, including failure to meet CONTINUE, insufficient
  precision, absent opportunity, or no admitted case T. INCONCLUSIVE goes to the RFC
  with the evidence as it is and a decision by Philip.
  A stop for budget or time is reported as a resource decision, not as evidence of
  ineffectiveness. Multiple models/workloads raise the false-CONTINUE rate; accepted
  for a screen and stated with the result.
- Run priority under the cap (fixed beforehand): case T first, then Gemma and Qwen
  on workload S, then the dense model, then LRU+plugin; runs never started because
  of the cap are reported as not run.
- Measurement validity separated from opportunity: validity checks (server healthy,
  hints acknowledged, captures complete, counters consistent with scheduler and
  eviction events) gate a cell; lack of opportunity does not invalidate a cell and
  stays in the results.

### Stage 2b. Early RFC (no compute)
Right after Stage 2 (if CONTINUE or INCONCLUSIVE): short vLLM issue/RFC on the hint
action semantics, aligned with TTL `cache_control` semantics (Anthropic/Dynamo) so one
hint could drive GPU pinning and CPU hold; validation, expiry, idempotency,
tenant/session ownership, best-effort behavior, server-side deadline clamp; whether
maintainers want an upstream policy or an external plugin/example. Maintainer feedback
shapes Stage 3 before major spending. Scale-down branch: if maintainers prefer an
external plugin or example, Stage 3 shrinks to the central point on one platform
with the secondary arms dropped. If there is no maintainer response within 3 weeks,
proceed on the plugin/example scope.

### Stage 3. PR-grade benchmark (protocol written and committed before any run)
3a. Protocol document (committed by hash before confirmation runs) containing:
- Primary outcome: SLO-miss ratio = (share of sessions missing the SLO with
  ARC+plugin) / (share missing it with the better of LRU/ARC), at the central load,
  50% hint adoption, causal hints. A session misses if any return turn exceeds the
  TTFT SLO or it errors/times out; misses stay in denominators. SLO = 2x the median
  return-turn TTFT of the large-pool arm at low load in calibration, frozen before
  confirmation. Central load = the session arrival rate where the better of LRU/ARC
  attains 80-90% session success in calibration (pre-fixed rule). Goodput (sessions
  per hour meeting the SLO) is reported beside it with its ceiling
  (1 / baseline success share) stated.
- Minimum counted sessions: each run counts at least the number of post-warm-up
  sessions needed for an expected 20 baseline misses (set from the pilot); run
  length and GPU hours are re-estimated from that (expect a multiple of the sizing
  below; the pilot re-quotes before the 5080 and rented stages).
- Dilution floor: at 50% adoption, if unhinted sessions do not benefit and some
  misses are queueing-caused, the miss ratio cannot fall below about
  1 - (share of baseline misses that are hinted and eviction-caused). The pilot
  measures this share and confirms 0.75 is reachable at the planned session counts;
  any adjustment is made before the protocol hash, never after. The floor is stated
  beside the result, and the hinted-cohort miss ratio is a key secondary.
- Key secondaries: maximum sustainable arrival rate (>= 90% success and bounded
  backlog), p50/p95 return TTFT, session completion time, recomputed tokens split into
  GPU reuse / CPU reuse / evicted / never stored, transfer bytes, restore GB/s and
  prefill tokens/s per platform (break-even), iso-latency memory (pool LRU/ARC need to
  match ARC+plugin; full pool curve including where gain is zero).
- Guardrails: unhinted and one-shot p95 TTFT and mean ITL not worse than
  max(A/A 95% band, 5%); zero added store failures; 0% adoption equals stock (CPU
  identity test + GPU A/A).
- Decision: replication unit = workload seed/trace sample paired across arms; the
  miss ratio is the ratio of pooled miss shares, its 95% interval from a paired
  bootstrap over seeds with BCa correction (or a t-interval on the log ratio if BCa
  is unstable), method fixed in the protocol. Better baseline = lower primary miss
  share in calibration, fixed before confirmation. SUPPORTED if the CI upper bound of the SLO-miss ratio
  is below 0.75 and guardrails hold; NEGATIVE if the CI lower bound is above 0.75
  (a 25% miss reduction ruled out); otherwise INCONCLUSIVE. A guardrail failure
  makes the result NOT SUPPORTED regardless of the primary, reported as harm.
  Seeds: pilot variance sets the count, minimum 8 for primary cells, 3 for curves;
  maximum budget stated; held-out trace groups.
- Traffic: Poisson session arrivals; within a session the next turn is sent at
  completion + tool gap (causal); TTFT from scheduled send time; record intended
  arrival, actual submission, queue delay, completion. Warm-up ends at the
  earlier of (a) the pool being full and evicting, or (b) a predeclared stationarity
  criterion: CPU-pool occupancy and rolling return-turn TTFT each change by less
  than 5% between two consecutive fixed-length windows, with a fixed maximum
  warm-up time; cells that end warm-up by (b) without evictions are kept and
  labelled no-pressure. Cells whose backlog keeps growing after warm-up are marked
  overloaded.
  No artificial GPU cache cap or two-request limit in PR-grade runs; pressure comes
  from session count. Working set defined in unique reachable KV bytes across groups.
- Workloads: (W1, primary) replayed agent traces: shared system/tool prefix across
  sessions, real long-tailed gaps, recorded token content and output lengths replayed
  deterministically, hints from the causal generator including early, late (return
  after expiry), abandoned, duplicate and cancelled hints, deadline sweep; (W2,
  mechanism) synthetic pressure workload; (W3, neutral control) vLLM's own multi-turn
  benchmark without hints. Inactive/no-opportunity cases are retained.
- Core matrix: 3 models (dense, sliding-window hybrid, Mamba hybrid) x arms {LRU, ARC,
  ARC+plugin hints off (A/A and identity), ARC+plugin} x W1 at central load (8 seeds) + W1 at low and overload
  (3 seeds) + W2 at central load (3 seeds). Roughly 190 runs, ~19 GPU h per platform at
  ~6 min/run (to be re-estimated from pilot).
- Secondary comparisons at the central point only (3 seeds): no offload, large-pool
  control (not an oracle), GPU TTL pin (Continuum code if it runs, else labelled "our
  implementation"), pin+hold combined, LMCache as informational only, perfect-
  information hint upper bound. ~5 GPU h.
- Ablations (promote-only, hold-only, dead-first-only, promote+hold), pool sweep
  (0.25-2x) and adoption sweep (0/50/100%) and the hint-free head-chunk-aging idea:
  run in the ported simulator, used for prioritization only after rank agreement
  with GPU on >= 16 pre-chosen cells; any reported ablation or sweep conclusion
  needs GPU confirmation of the selected points.
- Abuse: greedy-tenant test with a pre-set bound on harm to honest unhinted traffic;
  server deadline clamp; behavior under TieringOffloadingSpec stated.
- Threats tracked: tuned pressure band (publish full curves), hinted gain taken from
  unhinted (all-request primary), queueing hiding savings (report recompute, TTFT and
  completion together), cold start, simulator drift, small-context artifacts (PR runs
  at >= 32k contexts on 5080/datacenter), hardware crossover, forking paths (protocol
  hash), state carry-over (clear() between cells, interleaved arm order in time),
  busy-time-union throughput estimate retired (throughput measured directly).
3b. Pilot on the 2080 Ti: one authentic path end to end, failure cases, variance.
3c. RTX 5080: core matrix with longer contexts and larger models.
3d. Rented datacenter GPU: core + secondary at >= 32k contexts, only if 3c is
    SUPPORTED or strongly positive; firm quote first.

### Stage 4. Upstream PRs
- PR A (if Stage 1/2 show value): the policy (plugin-shaped, or minimal core patch),
  unit tests under tests/v1/kv_offload/cpu/policies, docs section, reproducible
  benchmark script and results table.
- PR B (separate): HTTP exposure of kv_hints behind a server opt-in with clamp and
  validation.
- The hint-free part is pursued only if the simulator shows a residual gain on main
  (e.g. head-chunk aging) and a GPU check confirms it.
- Related work cited: Dynamo cache-control TTL, Continuum, MORI, InferCept.

## Part B. Accepted caveats

From Claude Fable 5.2:
1. Warm-up (b): measure occupancy change as a fraction of capacity; window >= several
   mean session durations; if the first eviction comes after a (b) end, restart the
   measured interval there; label no-pressure by absence of evictions in the measured
   interval.
2. A/A-CI-includes-0 identity check is multiplicity-prone across ~6-7 metrics: treat an
   exclusion as a trigger to re-verify the Stage 1 CPU identity and inspect, not an
   automatic defect.
3. BCa at 8 seeds under-covers: report a paired t-interval (log ratio for TTFT benefit)
   beside it; disagreement on a threshold => INCONCLUSIVE.
4. Harm metrics compare ARC+plugin with the better baseline; A/A band measured against
   ARC even when LRU is better (state it).
5. Report benefit against both LRU and ARC, not only the selected better baseline.
6. Schedule Stage 2 in complete paired-seed blocks; re-quote after the first block.
7. Carry the Stage 2 guardrail rule (point estimate <= margin, CI upper <= 2x margin)
   into Stage 3.
8. Opportunity (>= 20%) is the pooled point estimate on the decision seeds.
9. STOP also requires valid cells and hint-off identity.
10. Multiplicity in CONTINUE, STOP scope, semi-synthetic labelling of case T: as in Part A.

From GPT-6 Astra:
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
- Task reuse: each seed orders its partition's tasks by SHA-256(seed, task ID) and
  assigns them to sessions in that order; if a run needs more sessions than distinct
  tasks, it cycles through the same order (deterministic reuse). Session IDs are
  distinct from task IDs; the reuse count per run is reported, and a sensitivity
  restricted to first occurrences of each task within the run is reported beside the
  primary.
- Replay semantics ("transcript replay"): each turn sends the canonical recorded
  prompt (natural system/tool prefixes and observations preserved, no manufactured
  prefix), requests the recorded assistant output length (cap 256 on the 2080 Ti,
  native up to 2048 elsewhere; cap frequency reported) with ignore_eos. Generated text
  is not fed back; reuse is measured as the exact token LCP with the session's
  previous requests and the group-aligned eligible boundary; lost reusable suffix
  reported. A live mini-swe-agent lane on the 4 SWE-bench tasks is an application
  sanity check only.
- Episodes: chronological; 2080 Ti: first turns until prompt+output > 6144, stop at
  first overflow, 2-8 turns; 5080/rented: context strata <= 8k, 8-16k, 16-32k (64k
  rented only); never delete transcript middles.
- Gaps (declared, not production): min(300, LogNormal(ln 4 s, 0.8)), drawn per turn
  from a seeded stream independent of arms; sensitivities median 0.5 s and 30 s.
- Hints (causal): per-session adoption 50% primary (0/100 sensitivities); every
  request in an opted-in session, including the final one, carries a fixed TTL of 12 s
  (3x the declared median gap); sensitivities 2 s and 60 s;
  server clamp 60 s. Perfect-information hint (TTL = true gap) is a labelled oracle
  reference, secondary only (not assumed to be an upper bound).
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
A2 stock ARC; A3 ARC+plugin hints off (A/A and identity); A4 ARC+plugin with exactly
one mechanism variant (hold-only, promote+hold, or promote+hold+dead-first) frozen at
the end of Stage 1 from feasibility and CPU results only, never selected after GPU
benefits are seen. A5 LRU+plugin if Stage 1
shows the hold is policy-agnostic; otherwise the claim is stated as ARC-dependent.
Required at the central point (3 seeds unless noted):
- No CPU offload (GPU prefix cache only).
- Hint-free admission: A1 and A2 with store_threshold=2, max_tracker_size=64,000.
- Matched all-False sensitivity: offload_prompt_only=False for A1, A2 and A4 together.
- Large-pool capacity control and the pool curve (0.25/0.5/1/2x frozen working set,
  where RAM permits).
- Ablations: promote-only, hold-only, promote+hold (dead-first only if claimed).
- GPU TTL pin ("our GPU TTL implementation", patch hashed; Continuum fork
  Hanchenli/vllm-continuum@316a58794a6ff86b216e579b74fd56ed0c5a911f inspected first,
  never called Continuum): same causal hint, resident pages only, cap 50% of GPU KV
  (sweep 10/25/50%), release on return/expiry, earliest-deadline yield under
  allocation pressure; and GPU pin + CPU hold. Run before the full matrix.
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
  parts), and freeze the A4 mechanism variant.
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
- Arms A1-A4 (+A5 if feasible). Seeds: calibration 12001-12003; decision 12101-12108,
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
- Load calibration: rates {0.25, 0.5, 1, 2}/D (D = low-load mean session service
  demand excluding gaps), expand geometrically until baseline success falls below 85%,
  then bisect 3 times; central = success nearest 85%; also 0.5x and 1.25x central
  ("high load"; "overload" only when measured backlog growth shows it).
- Measured count per run N = max(200, ceil(20 / pilot baseline miss share)) sessions,
  separate from warm-up arrivals.
- Paired schedule: before a block runs, one seeded schedule per tested load (session
  IDs, task assignments, arrival times, gap draws, hint adoption, one-shot arrivals)
  is generated at that load's actual arrival rate and replayed identically for every
  arm. Session arrivals are generated through the Nth session arriving after the
  six-window warm-up deadline (so N post-cutoff arrivals exist whatever the cutoff);
  background session and one-shot arrivals then continue through that Nth arrival time
  plus the drain timeout, which is the common endpoint for every arm. Sessions other
  than the measured cohort run but are not counted. If the run budget cannot cover the
  whole schedule, the attempt is retained and the comparison is INCONCLUSIVE; the
  cohort is never shortened.
- Warm-up and measured cohort: windows of max(120 s, 3x pilot mean session duration).
  Stability between two consecutive windows requires all of: CPU-pool occupancy changes
  by < 2% of capacity (absolute, so an empty pool is stable); median return TTFT changes
  by < 10%, evaluated only when both windows contain >= 10 return turns (when the
  schedule offers fewer than 10 returns in a window the criterion is not applicable;
  when it offers >= 10 but fewer complete, the criterion is unmet); backlog (queued +
  running requests) changes by < max(10%, 2 requests), zero backlog in both windows
  counting as stable, with no monotone growth across three windows.
- Common cutoff = the latest stabilization point among the block's arms on the shared
  schedule clock, bounded by the 6-window deadline; if any arm has not stabilized by
  then, the cutoff is the deadline and the block is labelled non-stationary and
  retained. There is no restart: if the first eviction in any arm occurs after the
  cutoff, the block is labelled late-pressure, its eviction time is recorded, and the
  measured cohort is also reported split before/after that time as a secondary.
- The measured cohort = the first N scheduled sessions arriving after the cutoff,
  identical session IDs for every arm. Drain timeout after the last measured arrival =
  max(600 s, 3x pilot p95 session duration); unfinished measured sessions are
  failures. If the cohort cannot be completed within the run budget, the attempt is
  retained and the comparison is INCONCLUSIVE; the cohort is never shortened and no
  arm is omitted. No-pressure is labelled by absence of evictions in the measured
  cohort.
- Harm margin and A/A band frozen from calibration with a precision ceiling: if the
  A/A band exceeds 10% relative TTFT or 5 points recompute, the comparison is
  INCONCLUSIVE for inadequate precision.
- 0.75 miss-ratio target kept as the declared worthwhile effect; the dilution estimate
  is an observational diagnostic, not a hard floor.
- Pilot seeds 13001-13003; confirmation 13101-13108 (count raised only by a
  predeclared pilot rule). Pilot on the 2080 Ti with D8 at <= 6k only.
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
- Order: W1 central with A1-A4 (primary), GPU TTL and pin+hold, required comparators,
  then curves, robustness variants, W2 long-context, W3.
Stage 3d (rented, only if 3c is SUPPORTED or strongly positive; firm quote first):
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
baseline caused by A4 fails the guardrail and prevents a positive verdict. Hint
acceptance vs effective held bytes, restored-byte and same-boundary continuation checks
are reported per arm; a correctness failure in an arm is a blocking result for that
arm.
Opportunity (label only, never a validity gate): evictions of returning sessions'
chunks, reusable prefix share, residency. Zero-eviction or zero-benefit valid cells
stay in results.

### 5. Open decisions for Philip
- WSL memory limit (~15 GB) caps CPU pools at a few GiB on Machine A; raising it is a
  Windows-side change only Philip can approve. Without it, Stage 2 pools stay <= 4 GiB.
- LMCache and SGLang lanes are claim-scoped; decide later which claims the PR makes.
