# Configuration

Experimental options; a setting being available does not establish a serving benefit.
See [current results](../evidence/README.md) for tested configurations and limits.

## Qualified CPU-advice / continuation-priority profile

The [exact exercised profile](../evidence/live-optimization-2080ti-20261103/qualified-live-profile.json)
contains the full connector JSON, serving arguments, runtime identity and
operating contract. It is a separate opt-in CPU ARC profile; the generic MSR
helper below does not reproduce it. Preserve its GPU-gated CPU retention-only
settings, `preservation_audit=false`, `preservation_events=true`, native priority
scheduling and layout guards. The measured credit is 8,000 ms:

```python
from semantic_offload.request_priority import gateway_priority

# gateway_epoch_ns is shared across requests for the server, not recreated per turn.
priority = gateway_priority(submitted_ns, gateway_epoch_ns, turn_index, 8000)
```

Credit an actually released continuation only. Hints describe a completed exact
prefix and an observed pending wait; they must not reveal future timing or text.
The tested producer used a constant 750 ms estimate with 1,000 ms refreshes.
Manage the default 4,096-session registry lifetime: new-session advice declines
at capacity, inference continues, and entries do not expire automatically.
The claim binds the measured installed-wheel/overlay runtime, not the separately
qualified benchmark suite/runtime. The suite does not reproduce this research
profile merely by starting the same model. Use for asynchronous human-reviewed drafts with
tolerance for the [measured cold and burst delays](../evidence/README.md).


For repository development on Linux x86-64 / the 2080 Ti WSL host, use the
[pinned runtime setup](../vendor/VLLM.md). It includes the patched source required
by the native CPU-advice experiments. The generic package installation below
expresses the supported version range; it does not reproduce that runtime:

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

Generic serving example (choose hardware-compatible flags and the pinned runtime
for repository work; this is not the 2080 Ti benchmark protocol):

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

`prefetch_preserve_gpu_prefix_cache=true` is a later experimental opt-in
(default `false`). It reserves speculative staging only from the contiguous
unhashed head of the native free-block queue, preserving reusable GPU prefix
entries for ordinary demand allocation. It can defer prefetch and reduce its
coverage; unsupported queue layouts defer staging safely.

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

## CPU retention advice

The separate CPU-only ARC profile uses `instrumentation_only=true`,
`eviction_policy="arc"`, `prefetch_trigger="known_prefix_wait_hint"`, and
`wait_hint_cpu_retention_only=true`. Use the complete
[tested configuration](../evidence/cpu-wait-recency-2080ti-20261102/tested-connector-config.json)
as the reference for that experiment, not the MSR helper above. Its
`prefetch_budget_fraction=0.6` enables the path while CPU-only mode creates no
speculative GPU reservations; a zero budget disables the path.

Newer opt-ins include `wait_hint_cpu_retention_gpu_gate` (requires CPU-only mode
and the native-adoption layout guard), and `preservation_audit=false` for the
narrow instrumentation-only CPU-advice profile. Disabling audit removes request
preservation proofs; retain `preservation_events=true` for operational failure
handling. These controls are still under live qualification. Full compatibility
constraints and defaults are enforced in `semantic_offload/spec.py`.

## Benchmark activity telemetry

`benchmark_telemetry=true` enables a native scalar counter for scheduler batches
that actually apply semantic relevance updates. The maintained suite enables it
on the candidate and retains the before/after Prometheus snapshots. It defaults
to false outside the suite. This proves applied scoring activity, not ranking
quality, speculative-copy usefulness, or cache-hit correctness. See the
[benchmarking guide](BENCHMARKING.md) for the four tiers and their evidence limits.

## Isolated KV SVD experiment

`kv_svd_rank` defaults to `0` (off). Ranks 1–15 enable lossy SVD compression
of actual FP16 K/V pages in the semantic worker; this is separate from semantic
summary/predictor features. The experimental codec supports uniform dense pages,
head size 128, block size 16, and one CPU block per GPU block. It retains FP16
factors only on CPU and reconstructs full GPU pages before transfer completion.
Scheduler ownership remains fenced; no speculative or demand reader may consume
unfinished data. Transfers currently synchronize and include codec work in their
reported duration. `kv_svd_trace` optionally records allocation, physical factor
bytes, operation time and reconstruction residuals to a local JSONL file.

`kv_svd_expand_capacity=true` uses the same nominal CPU byte budget for more
compressed slots, including the readiness bitmap. Default false keeps the native
slot count and saves host payload bytes instead. Policy/summary metadata and
transient codec workspace are additional and must be measured. The solver batches
at most 256 matrices to bound the observed workspace spike. Rank 4 failed the
retrieval quality screen; rank 8 passing a small assertion suite is not general
quality qualification. All options remain experimental and default off. See
[serving experiment](../evidence/semantic-svd-serving-20261104/PROTOCOL.md).
