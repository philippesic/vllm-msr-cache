# MSR KV Cache Eviction for vLLM

An experimental vLLM extension for exact KV-cache offload, semantic retention,
and causal prefetch. The project explores **mean semantic relevance (MSR)** as a
cache policy alongside LRU and ARC.

This is a research prototype, not a production-ready
cache backend. The accepted experiments preserve exact output and KV identity,
but the combined MSR-retention and semantic-prefetch system has **not** shown a
repeatable provider benefit against matched ARC/LRU controls.

## Current standing

The November 2026 RTX 2080 Ti follow-up found a small, conditional
**throughput/VRAM improvement** using causal wait hints to refresh CPU ARC
recency, with speculative GPU copying disabled. This is CPU retention advice,
not a content-semantic ranking win. It remains opt-in because latency worsened.

The initial low-usefulness diagnosis found copy-and-reclaim churn: 41 of 42
completed speculative copies were reclaimed for demand, with a median
reservation lifetime of 146 ms. Only 136.7 MB of 4.321 GB copied was attributed
to intended-request consumption (3.16%). This does not isolate ranking as the
cause. Higher utilization in later staging experiments did not establish a
serving benefit; the selected CPU-only profile removes that speculative traffic.

Twelve fresh paired workflows across three fixed public SWE traces compared
2,048 candidate GPU blocks with 2,112 ARC blocks, using the same 3.5 GiB CPU
cache. All 1,728 requests passed strict physical proofs and exact recorded
input/output-token checks. Pilots and earlier failed confirmations were excluded.

| Measurement | Candidate versus ARC |
| --- | --- |
| Output throughput | +0.64%; paired 95% interval +0.09% to +1.20% |
| Serving-load VRAM | 56 MiB less in every pair, under allocated, reserved and sampled device peaks |
| Throughput / device GiB | +1.34%; paired 95% interval +0.78% to +1.90% |
| Mean first-token latency | 233 ms worse; paired 95% interval 98–369 ms worse |
| p95 first-token latency | 1.43 s worse; paired 95% interval 0.07–2.78 s worse |

These are Qwen2.5-7B-Instruct AWQ / FP16 serving-compute replay results on
this WSL host: normal model forwards with recorded output tokens, 24 sessions,
8 active requests and 3 turns. They do not establish unconstrained-generation
quality, other hardware performance, or startup-memory requirements. A separate
intrusive audit checked all 28 KV components in all 583 serving transfers
(9.74 GB), including one-byte corruption controls and ordered physical IDs.
The integrated checkout then passed another 144 strict, exact-token requests;
394 focused tests passed with the benchmark's required block-pool runtime API.

See the [curves](evidence/cpu-wait-recency-2080ti-20261102/curves.png),
[paired measurements](evidence/cpu-wait-recency-2080ti-20261102/confirmation-pairs.csv)
and [result manifest](evidence/cpu-wait-recency-2080ti-20261102/result.json).
The dependency-free [aggregate verifier](evidence/cpu-wait-recency-2080ti-20261102/verify_aggregates.py)
recomputes the reported throughput intervals and memory savings from the CSV.
The full frozen protocol and compressed raw evidence remain in ignored
`output/prefetch-native-20261102/`.

On an existing instrumented ARC configuration with
`prefetch_trigger="known_prefix_wait_hint"`, set
`wait_hint_cpu_retention_only=true`. It defaults to `false` and requires a JSON
boolean. The validated hint refreshes only the contiguous, ready CPU prefix
within its existing ARC queues; it preserves ghost history, the adaptive target,
references and normal eviction eligibility. It creates no GPU reservation or
speculative transfer. The existing global `prefetch_budget_fraction=0` switch
also disables this path; the measured profile uses `0.6` while reserving zero
GPU blocks. The [complete tested connector configuration](evidence/cpu-wait-recency-2080ti-20261102/tested-connector-config.json)
includes its pinned model/layout namespace and the existing wait-hint settings.
Do not interpret zero copied bytes as 100% prefetch utilization: that fraction
has a zero denominator.

The live-generation optimization follow-up is still under evaluation; its
[protocol and retained failures](evidence/live-optimization-2080ti-20261103/protocol.json)
are separate from the replay result above. No new production qualification is
claimed yet.

The experimental `wait_hint_cpu_retention_gpu_gate=true` option suppresses a
CPU recency refresh while the exact full prefix is already cached on the GPU.
Partial GPU coverage uses the original ready CPU prefix; an unavailable index
falls back to ordinary CPU advice. It changes no GPU references, allocations,
eviction order, or bytes. GPU coverage can disappear after a skipped refresh,
and an already evicted CPU copy cannot be restored by advice.

This option defaults to false and requires CPU retention-only mode and the
native-adoption layout guard: dense attention, TP1, one KV group, and one cache
block per offload chunk. It cannot be combined with
`wait_hint_cpu_retention_gpu_aware`. The
[live discovery](evidence/live-optimization-2080ti-20261103/gpu-gate-root-verification.json)
is promising but includes a throughput regression on one input set; fresh-input
confirmation and separate byte qualification are still required.

A new opt-in `preservation_audit=false` setting removes full request histories
and wait-hint logging for the narrow instrumented ARC / CPU-only advice profile.
Keep `preservation_events=true`: operational transfer failure handling still
uses it. Configuration validation rejects semantic scoring, speculative staging,
and lifecycle/admission policies in this mode. It emits **no request preservation
proofs**; correctness qualification must be performed separately. Auditing remains
on by default.

In this mode, `observed_session_registry_size` defaults to 4,096 session identities.
Once full, new-session hints return `no_action/session_registry_capacity`, while
inference continues. Generation records do not expire, preserving rejection of
stale generations. Cache reset opens a new registry epoch. A deployment that
continually creates new session identities must account for this capacity and
reset/recycle behavior; this is not unlimited session retention. Current native
failure tests and GPU work cover one GPU, not tensor-parallel serving.

For performance measurements, `SEMANTIC_ALLOCATOR_AUDIT=0` skips detailed allocator
inspection on each forward. Native CUDA peak counters remain available at export;
the switch does not suppress transfer failure handling. No-probe workers also
skip unused summary streams and completion events, retaining native copy fences.

The earlier E04 admission result was policy-agnostic. In E04, native
`store_threshold=2` reduced writes by 99.790% and improved the semantic arm's
mean TTFT by 22.50%, ITL by 19.78%, throughput by 14.93%, and makespan by 12.00%
versus its neutral configuration. Matched ARC also benefited from threshold 2
and remained ahead of semantic by point estimate: semantic TTFT was 3.17% worse,
ITL 4.76% worse, throughput 2.70% lower, and makespan 5.12% longer. Those cells
performed no restores or prefetches, so E04 supports a generic admission guard,
not a semantic-policy or prefetch win.

The final combined-system campaign reached exact causal semantic selection and
consumption, then closed as a scoped negative:

- A26 repeated three matched 48-request traces. MSR+semantic executed 120,174
  prompt tokens versus 119,918 for ARC/LRU in every trace (+0.2135%), used 3.56%
  more measured model-runner time than ARC, and transferred 12.40% more H2D data.
- A27 used one genuinely disjoint 48-request trace. MSR+semantic selected and
  consumed the returning source, but tied ARC/LRU on prompt work while using
  4.94% more model-runner time, with 6.23% worse median TTFT and 6.37% lower
  throughput than ARC.

These results rule out the tested revisions and inputs as release-benefit
evidence; they do not establish that MSR can never help. The
[curated evidence snapshot](evidence/README.md) gives the full denominators,
controls, hashes, and limitations.

## Architecture

MSR ranks exact KV candidates using compact summaries derived from observed
attention queries. Retention manages CPU-resident exact KV; prefetch can stage a
causally selected prefix back to GPU before demand. Exact prefix, model, adapter,
layout, and position identity—not semantic similarity—determine whether KV may
be installed and reused.

```mermaid
flowchart LR
    Q[Observed attention queries] --> S[MSR scoring]
    S --> R[Retention policy]
    R --> D[(Exact KV in CPU DRAM)]
    S --> P[Prefetch selection]
    H[Session signals] --> P
    D --> P
    P --> G[GPU staging]
    D -->|Demand restore| I[Exact match and install]
    G --> I
    I --> C[Request consumes exact KV]
```

The package is named `semantic-offload` and targets a narrow prerelease vLLM
compatibility range. Reproduction requires pinned extension and vLLM revisions.
The tested DRAM comparisons explicitly set `store_threshold=2`; it is not a
global runtime default.

## How to use

Install the extension with its supported vLLM range:

```bash
python -m pip install -e '.[cache-affinity]'
```

MSR is selected through vLLM's KV connector configuration. The public helper
returns the complete configuration, including mean-semantic retention and its
paired semantic prefetch:

```python
import json

from semantic_offload.config import msr_kv_transfer_config

config = msr_kv_transfer_config(
    cpu_bytes_to_use=8 * 1024**3,  # 8 GiB CPU KV tier
)
print(json.dumps(config))
```

Pass that JSON to `vllm serve`:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --kv-transfer-config '{
    "kv_connector": "SemanticOffloadingConnector",
    "kv_connector_module_path": "semantic_offload.connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "spec_name": "SemanticOffloadingSpec",
      "spec_module_path": "semantic_offload.spec",
      "cpu_bytes_to_use": 8589934592,
      "method": "mean",
      "store_threshold": 2,
      "prefetch_budget_fraction": 0.05,
      "prefetch_selector": "semantic",
      "prefetch_score_max_age_ms": 1000
    }
  }'
```

The helper defaults to preemption-triggered semantic prefetch. Set
`prefetch_budget_fraction` to control how much of the GPU block pool may be
reserved speculatively; it must be greater than zero to enable prefetch. The
current implementation requires pipeline and context parallel sizes of one,
does not support MLA KV caches, and requires the vLLM version range declared in
`pyproject.toml`. Treat the configuration as experimental and pin both projects
when reproducing results.

### Experimental scoring controls

The existing scoring behavior remains the default: `scoring_mode="synchronous"`,
`scoring_schedule="continuous"`, `capture_stride=1`, and `scoring_search="flat"`.

For the single-worker ARC wait-hint experiment, `scoring_schedule="opportunity"`
retains bounded last-query snapshots instead of scoring every generation step.
Use `capture_stride=8` to sample fewer query-extraction steps. When at least two
eligible sources compete for insufficient staging space, the scheduler requests
scores only for their DRAM-resident prefix keys. When every source fits, ranking
is bypassed. A missing or expired snapshot retains an unscored fallback; scoring
can defer speculative submission for at most 50 ms, checked at scheduler steps.
Demand serving is never held for a score. Snapshot bounds are
`query_snapshot_max_age_ms=1000` and `query_snapshot_max_requests=256`. This mode
requires `instrumentation_only=true`, `eviction_policy="arc"`,
`shared_prefetch_scorer=true`, and mean-key/mean-head TP=1 scoring. It supports
synchronous scoring or experimental `scoring_mode="asynchronous"` with flat search.
It does not provide semantic retention updates. `scoring_schedule="capture_only"`
is an attribution control, not a serving recommendation.

Asynchronous scoring requires `preservation_events=true` and one dense KV group.
One executor owns the scoring stream. Pending jobs are bounded by
`async_scoring_max_queue_items=2` and `async_scoring_max_snapshot_bytes=67108864`;
the byte limit covers retained job tensors and conservative workspace, not the
whole process, durable cache, query bank, Python metadata or allocator reservation.
Overflow drops the newest observation. Results retain their original feature
timestamp and expire within `async_scoring_max_snapshot_age_ms=1000` (also bounded
by the query lease in opportunity mode). Request generations, cache residency
versions and reset epochs are checked through final consumption. Normal completion
preserves a bounded final-query lease; abort/reset invalidate stale results.
Pending CUDA host waits query the event and sleep for 500 microseconds between
checks. This reduces active CPU waiting but can delay result availability; OS
scheduling can add more than one interval. Cancellation retains live GPU inputs
until their completion fences finish.

The scoring audit measured about 71% less readiness-plus-executor CPU per job in
the continuous diagnostic. Its final opportunity-mode serving comparison still
failed the ARC target: throughput was 2.27–2.71% lower and median first-token latency
12.22–12.81% higher. Those products had no constrained scoring opportunities.
This supports experimental work reduction, not dependable ARC parity, ranking
utility or lower VRAM use. Qualification is limited to the tested dense TP1/PP1
eager runtime. See the local [audit report](output/scoring-audit/REPORT.md) and
[independent Astra review](output/scoring-audit/astra-acceptance.md) when retained.

`prefetch_preserve_gpu_prefix_cache=true` is a later experimental opt-in
(default `false`). It reserves speculative staging only from the contiguous
unhashed head of the native free-block queue, preserving reusable GPU prefix
entries for ordinary demand allocation. It can defer prefetch and reduce its
coverage; unsupported queue layouts defer staging safely.

A three-seed confirmation measured regular throughput at 81.252 versus ARC's
80.806 output tokens/s (+0.552%), with all six regular pairs positive. The
higher-concurrency aggregate was 0.168% below ARC, including one pair 1.588%
slower; one regular pair's median first-token latency worsened 16.480%. Earlier
prefetch also matched ARC in the allocator screen, so the option's causal speed
benefit remains unproven. Real prefetch consumption remained enabled, but these
products had no competitive semantic scoring jobs. See the retained
[throughput report](output/throughput-parity/REPORT.md) for results and limits.

`scoring_search="ivf"` enables approximate cluster routing followed by the existing
exact mean-key/mean-head dot-product score on visited rows. Defaults are
`search_min_pool=128`, `search_clusters=16`, `search_probes=4`, and
`search_exploration=8`. Small pools use the flat scan. The index rebuilds when the
summary pool changes; its maintenance cost must be included in comparisons.
Unvisited blocks receive no score or freshness update. Rotating exploration
reduces persistent omission on a stable pool; it does not guarantee recall under
churn. Shortlist ranks change rolling-score weights relative to exhaustive
ranking. Candidate search never authorizes reuse of inexact KV or arbitrary
noncontiguous prefixes. These controls are experimental and have no presumed
serving benefit.

Set `SEMANTIC_PROFILE_DIR` to write bounded per-process component aggregates.
They distinguish host wall time, thread CPU time, counters and sampled CUDA-stream
intervals (`SEMANTIC_PROFILE_CUDA_STRIDE=16`). CUDA samples are read only after
completion; the profiler adds no per-step synchronization. Nested scopes and
concurrent streams overlap, so totals are not additive. Histogram quantiles are
duration bounds. Compare observer-off runs for serving results and use separate
profiled runs for attribution. Final flushes retain unfinished-sample counts.

### Scoring efficiency trial

A bounded follow-up compared five observer-off configurations on cached dense
Qwen3-4B with six simulated agents and 12 background requests (45 requests per
configuration). Current semantic scoring produced 81.26 output tokens/s; work
reduction produced 81.89 (+0.78%), and both controls enabled produced 81.84.
Matched ARC demand remained ahead at 82.34. Median release-to-first-token time
was 3.149 s for current semantic, 2.738 s with work reduction and 2.743 s for ARC.
These are one-run point estimates; completion-dependent arrival/cache histories
and prompt work differ, so these are not isolated scoring-cost or significance
claims.

Only 11 blocks entered DRAM, with no competing eligible sessions. Production
IVF correctly stayed on flat fallback. A separate static replay of real query
and summary tensors forced IVF on this tiny pool: it was costlier than scanning,
with 100% top-1/top-3 recall but 88.0% mean top-8 recall over 24 correlated query
rows. Large-pool crossover remains unmeasured; keep the index disabled for this
workload.

A separate complete diagnostic placed 13.22 s of cumulative host time inside
construction of the tiny GPU gather-index tensor, versus 0.195 s inside scoring.
The upload span includes possible driver waiting/spinning for previous GPU work;
it is not 13 s of additional GPU computation or reclaimable latency. Device,
host and nested timing ranges overlap. This identifies a capture/synchronization
investigation seam rather than evidence that vector comparisons dominate.
Per-request results, complete component/queue/serialization measurements, exact
source pins and reproduction material are retained outside Git under
`output/scoring-efficiency/`. All nine real serving runs preserved exact output
and native KV identities; native direct adoption was not exercised because
prefetched prefixes were partial. This follow-up establishes no capacity or
production-policy win.

### Direct adoption of prefetched blocks

The experimental `wait_hint_native_adoption` option defaults to `false`. With
`prefetch_trigger="known_prefix_wait_hint"`, setting it to `true` lets a returning
request adopt the same physical GPU blocks that hold its completed prefetched
prefix, avoiding a second prefix allocation and the staging-to-destination copy.
It requires the companion [vLLM patch](patches/vllm-native-prefix-adoption.patch);
check the patch against the runtime checkout with `git apply --check` before
applying it. The patch was CPU-tested on the recovered prerelease runtime used
for the October opportunity screen, not every release in the dependency range.

The path supports a complete exact prefix with dense causal BF16 or FP16 KV,
one worker, one KV group, and one hash per block. It publishes blocks only after
transfer completion and returning-request identity checks. Native cache lookup
then owns allocation, sharing, and eviction. Eviction before admission falls
back to demand restoration; unsupported configurations retain the existing
staging path. Missing runtime support also retains that path.

CPU tests verify allocator ownership and physical block reuse. A bounded GPU
validation also confirmed real transfer completion, the same physical prefix
blocks in the worker's CUDA block table, and exact generated outputs. Three
alternating-order pairs with lighter logging measured 41.9–42.3 ms from request
release to first token, versus 70.9–71.4 ms for matched project ARC demand
restoration. The test supplied the exact target prefix 750 ms in advance.
Background generation took 6.8–17.6 ms longer over roughly eight seconds; no
background-neutrality claim is made. Broader repeatability, capacity gains and
deployable prediction remain unverified. This option is a prototype and does not
change the default path.

Two opt-in controls support full-prefix adoption under cache pressure.
`wait_hint_coalesce_exact_source=true` acknowledges a refresh of the same owner,
source, namespace and generation with the existing `canonical_event_id`. The
producer must bind that canonical ID to the returning request. This reuses the
original lease without extending its expiry or refreshing score timestamps.
`prefetch_require_full_prefix=true` submits only a complete CPU-resident prefix
that fits the current staging budget and has no keys already cached on the GPU.
The scheduler rechecks these conditions before allocation. Full-prefix staging
also waits while admitted requests still have prompt processing outstanding;
decoding does not close this gate. This gives known prompt growth priority and
can shorten the available prefetch window. This mode can skip useful partial
prefetches, but avoids allocating a partial prefix that cannot use native
adoption. Both options default to `false`.

`wait_hint_completed_transcript=true` additionally registers full blocks of a
completed request's already computed transcript as a known source. It requires
`offload_prompt_only=false` and the supported dense TP1 native-adoption layout.
The declared input source remains valid. The extra source uses actual computed
tokens after excluding in-flight work, exact native hashes and its own transcript
digest; it cannot include the last sampled token before that token has computed
KV. Source registration does not make an unfinished CPU store ready. Clients
must nominate only transcript tokens they have received and bind the matching
source identity. Incremental offload increases CPU storage and transfer traffic;
source-query freshness and speculative staging budgets still apply. The option
defaults to `false`.

`prefix_mean_prefetch_scoring=true` is a separate experimental policy for ARC
retention with semantic prefetch selection. It requires synchronous opportunity
scoring, the `mean` method, mean head aggregation and one worker. It uses a
common fresh query from active work to compare exact candidate prefixes by raw
mean dot product. Owned FP32 prefix centroids avoid per-block score readback and
rank smoothing; at most 16 centroids and 1 MiB of centroid tensor backing are
cached. The scalar transport rechecks query/source leases, key order, residency
versions, namespace, generation and expiry. Normal query completion preserves
only its existing bounded lease; abort, reset or source reuse invalidates it.
Actual staging retains the full-prefix and admission checks above. This metric
changes ranking policy and does not promise the original block-level EMA quality.
The option defaults to `false`.

When demand admission cannot fit, completed legacy prefetch scratch may be
discarded so demand can load its exact CPU keys normally. In-flight copies,
pending splices and bound wait-hint protection retain their existing ownership
rules. Discarding scratch can require another CPU transfer or recomputation if
the CPU copy has since been evicted.

GPU KV storage is allocated at engine startup. Releasing cached block references
does not release that allocation. Memory-capacity comparisons must reduce the
configured GPU block pool and measure device memory while serving the workload;
free block counts alone do not establish additional VRAM availability.

A separate synthetic load trial used six agent sessions and twelve background
requests on Qwen3-4B, with live semantic capture/scoring and natural cache
admission. ARC demand produced 82.4 output tokens/s, request-order prefetch 82.3,
and semantic prefetch 77.8. Only one or two blocks became eligible per decision;
no full prefix used native adoption and no semantic decision had competing
eligible sessions. This trial showed no end-to-end semantic benefit. Each arm
ran once on generated sessions, not a production trace. The separate CPU-cost
optimizations were excluded from these measurements.

## Capture and summary overhead

`SEMANTIC_CAPTURE_BACKEND=attention` selects an opt-in hook on the probe
attention implementation instead of intercepting every PyTorch operation.
The default remains `dispatch`. Both use the same request boundaries, last-token
queries and KV-head grouping. The attention hook requires the runtime's dynamic
`impl.forward(layer, query, ...)` contract; validate it on the installed vLLM.
GPU serving validation here covers FP16 eager Triton attention on the RTX 2080 Ti,
not every compiled backend. Full CUDA-graph replay still has no Python capture.

`SEMANTIC_CAPTURE_STEPS=multi_token` optionally skips batches where no request
has more than one scheduled token. Mixed batches retain all nonempty request
rows. This is a token-count filter, so it also skips single-token prefills.
The default is `all`; skipped batches still advance `capture_stride`. Fewer
observations can leave relevance stale, and an all-single-token workload can
produce no observations under this option. It preserves the dense scoring
formula at observed steps, not the original EMA trajectory or ranking quality.

`SEMANTIC_SUMMARY_FIELDS=required` is an experimental mean-only payload for
mean scoring with continuous or opportunity schedules. It computes the same FP32
centroid reduction while omitting unused minimum, maximum and deviation tensors.
Other scoring methods and capture-only mode are rejected; the default `all`
preserves the complete summary payload. GPU checks found identical centroids and
rankings in the exercised FP16 layouts. Reduced summary storage does not by itself
establish a serving throughput or overall VRAM improvement.

Summary construction batches compatible key layouts and KV heads. Scoring-cache
insertion uses one stack per required field; removal uses ordered slices rather
than uploading a CPU mask. Exact KV identities and transfer fencing remain
unchanged. Summary slices can retain entire batch reduction buffers, and a
contiguous scoring-cache slice can retain its previous storage. Equal configured
KV budgets therefore do not imply equal metadata allocations.

Reproduction scripts, paired measurements, quality diagnostics and rejected
candidates are retained locally under `output/scoring-overhead-20261101/`.

## Earlier individual comparison

In one earlier individual comparison, MSR semantic prefetch consumed the staged exact KV and reduced median time to first token by 0.80% versus ARC. Prompt work and transfer volume were unchanged.

| Metric | ARC | MSR | MSR vs. ARC |
|---|---:|---:|---:|
| Executed prompt tokens | 94,748 | 94,748 | 0.00% |
| Median TTFT | 166.088 ms | 164.756 ms | **−0.80%** |
| Makespan | Baseline | — | +0.17% |
| Throughput | Baseline | — | −0.17% |
| H2D transfer | 2,752,512 bytes | 2,752,512 bytes | 0.00% |
| GPU-reused tokens | 448 | 448 | 0 |
| DRAM-restored tokens | 96 | 96 | 0 |

## Repository map

```text
.
├── semantic_offload/
│   ├── config.py                 Public vLLM/MSR configuration helper
│   ├── connector.py              Exact KV movement and semantic prefetch
│   ├── spec.py                   vLLM offloading spec and runtime options
│   ├── manager.py                Admission, residency, and eviction manager
│   ├── policy.py                 MSR retention policy
│   ├── index.py                  KV summaries and semantic scoring
│   ├── worker.py                 Worker-side scoring and transfers
│   ├── query_capture.py          Attention-query capture
│   └── wait_hint.py              Optional causal wait-hint schema
├── benchmarks/
│   ├── eviction_prefetch_campaign.py
│   ├── msr_combined_prefetch_workload.py
│   └── msr_combined_prefetch_attribution.py
├── harness/
│   ├── policies.py               LRU, ARC, and semantic configurations
│   └── server.py                 vLLM server launcher
├── tests/                        Correctness and measurement validation
├── evidence/
│   ├── README.md                 Results, denominators, and limitations
│   ├── e04/                      Threshold-2 admission report
│   └── a27/                      Final combined-system report and audit
├── docs/
│   ├── PROJECT_HISTORY.md        Experiment and milestone timeline
│   └── FUTURE_WORK.md            Untried optimizations and measurement gaps
├── pyproject.toml
└── uv.lock
```

See the [project history](docs/PROJECT_HISTORY.md) for the experiment timeline
and [future work](docs/FUTURE_WORK.md) for the unfinished optimization and
measurement backlog.
