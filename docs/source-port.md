# Minimal source port

The product performs exact native CPU KV offload. A completed transcript may
register an observed prefix; a later causal wait hint may refresh the ready CPU
prefix in native ARC. There is no scoring, GPU prefetch, approximate reuse,
request-priority change or separate policy implementation.

## Ownership and boundaries

| Owner | Responsibility |
| --- | --- |
| contracts.py | Closed versioned registration/hint schemas, namespace and canonical digest |
| config.py | Closed product settings and resolved mechanism attribution |
| observed.py | Bounded scheduler-owned observations and replay state |
| advice.py | Causal/identity/replay validation; advice-disabled control |
| vllm_adapter/arc.py | Ready-prefix scan and MRU movement within each existing ARC queue |
| vllm_adapter/connector.py | Native construction, computed-token bounds, registration and scheduler cleanup |
| vllm_adapter/transfers.py | Submission/completion failure reporting and failed-load destinations |
| vllm_adapter/endpoint.py | Authenticated HTTP boundary into engine RPC |
| tools/runtime_verify.py | Functional attempt admission, source snapshot and terminal receipt |

Pure product code cannot import runtime/development packages or the adapter.
Native transfer bytes, allocation, reference counts, readiness and exact block-key
matching belong to vLLM. Advice changes queue order only. Each key stays in its
own T1/T2 queue; the root is most recent within its queue. An absent or pending
block ends the eligible prefix; active ready references remain unchanged.

## Construction contract

Use the pinned runtime and the overlay launcher described in [runtime](runtime.md).
The connector accepts exactly one dense FP16 full-attention group, TP1, without
LoRA or speculative decoding. It requires `kv_load_failure_policy="recompute"`.
Unsupported construction and optimized Python execution refuse explicitly.

Settings are `namespace`, positive `cpu_bytes_to_use`, Boolean `advice_enabled`,
bounded positive `max_sessions`/`max_events`, optional `eviction_policy="arc"` and
`store_threshold=0`. Defaults are advice disabled, 4096 sessions and 256 events per
session. The constructor selects native CPU ARC directly and full-transcript
stores. Dead/unknown historical keys refuse; there is no plugin-spec selector.
The control uses this same connector with advice disabled and all correctness
and failure guards retained. Unloaded stock ARC is a distinct, defective reference.

Finished stores stop at computed tokens minus in-flight tokens and round down to
complete chunks. The sampled but uncomputed final token never becomes reusable KV.
A successful transfer alone permits native readiness. Failed stores release their
slots without publication. Failed loads report invalid GPU destinations, release
CPU references and invalidate scheduler mappings; native recompute handles affected
requests. Exceptions without a safe completion fence remain fail-stop. CPU fake
handlers test failure bookkeeping; real byte transfers require hardware.

## Registration and hints

Pass this in an ordinary generation request's `kv_transfer_params.wait_hint`:

```json
{"schema":"wait-registration-v2","scope":"opaque-client-scope","session":"conversation","incarnation":"unique-instance","generation":1}
```

Completion returns `kv_transfer_params.wait_hint` with registration acceptance,
complete-token boundary, prefix digest, observed monotonic timestamp and engine
clock domain. Registration is performed on both advice modes. Invalid registration
returns a refusal without crashing the engine. Reusing a session with another
incarnation refuses; use a new session identity for a new client instance.

A hint has schema `wait-hint-v2`, the same scope/session/incarnation/generation,
an event ID, exact prefix/boundary/namespace, the returned `observed_ns`, and
`published_ns`, `deadline_ns`, `clock_domain`. Names and schemas are closed;
Booleans are not integer generations. Published/deadline times use the same host
kernel's monotonic clock. Remote-clock clients are unsupported. Deadline is at
most 60 seconds after publication. A reset changes the engine clock domain.

Namespace fields are `model_revision`, `tokenizer_revision`, `adapter_namespace`,
`layout_namespace`. The prefix digest is SHA256 of UTF-8 JSON containing
`namespace` and `token_ids`, recursively sorted keys, separators `,` and `:`,
without whitespace. A fixed externally calculated vector is tested. Namespace
labels organize observations; native exact block keys still determine reuse.

Advice requires the exact server observation timestamp and identity, a causal
live deadline and matching engine clock. An exact replay returns its original
acknowledgement without another mutation; changed content for an event refuses.
Event capacity refuses further new events. Session capacity retires the entire
least recently active session and counts retirement. New registration after
retirement has a new server observation timestamp, invalidating old hints even
when a caller reuses names/generations. Reset clears observations and replay state.

POST hints to `/v1/wait-hints`. The endpoint uses explicit server API keys and
refuses startup without them; `VLLM_PLUGINS=wait_hint` enables its installed entry.
It checks authentication before JSON/schema processing and routes through the
engine owner. Valid identity acknowledgements include ready CPU prefix length,
acknowledged and newly refreshed counts, replay flag, decision and resolved mechanisms. Replayed events preserve their acknowledged count but report zero new refreshes. This supports a single
trusted operator: scope prevents client-name collisions, and API access does not
provide tenant isolation or establish permission to run an experiment.

Supported readers are `wait-registration-v2`, `wait-hint-v2`, `wait-hint-ack-v2`,
`functional-admission-v1`, `functional-terminal-v1`, `runtime-observation-v1`.
Unknown versions refuse. There is no legacy reader or automatic migration. A
future migration must create a new artifact with lineage and preserve its source.

## Historical disposition

| Historical path/switch | Minimal port disposition and reason |
| --- | --- |
| `_calc_num_offloadable_tokens` | Reimplemented computed-token safety bound in both modes |
| `request_finished` / native adoption | Explicit versioned transcript registration in both modes |
| `publish_wait_hint` | Narrow causal observed-prefix advice; one explicit Boolean mode |
| `reset_cache` | Native reset plus observation/replay clearing and new clock domain |
| Failure worker callbacks / `_cleanup_failed_job` | Narrow store/load metadata and cleanup retained independently of preservation telemetry |
| `prefetch_trigger`, budget and disable flag | Replaced by `advice_enabled`; no prefetch mechanism is implied |
| `preservation_events`, `instrumentation_only` | Failure safety always active; no alternate telemetry-selected manager |
| Shared-local-prefix / publish-at-lookup keys | Previously unread; now rejected |
| Scoring, prefetch/splice/staging/preservation overrides | Excluded; no staging allocations, scoring context or preservation instrumentation exists here |
| Admission manager mixins | Excluded with admission filtering off; native manager owns all selected prepare/complete operations |
| Native block publication / allocation epochs | Excluded; registration uses native request hashes and completed-token extent, not block-pool publication |
| Lookup hook / idle-drain patch | Excluded; historical methods were absent in the selected extension; native pending-push work drives stores |
| Abort receipt / CUDA attribution patches | Excluded as research instrumentation |
| Hint RPC patch | Only publication methods in the four attested native Python files |

The stock pin's T2-empty ARC fallback limitation is deliberately retained in both
modes. The small native regression records it explicitly. It bounds this port's
functional qualification; it is not a claim that historical results were unaffected.
A fix/repin requires a new identity and qualification of both modes.
