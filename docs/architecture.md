# Architecture

The product performs exact native CPU KV offload. A completed transcript may
register an observed prefix; a later causal wait hint may protect the ready CPU
prefix in native ARC. There is no scoring, GPU prefetch, approximate reuse or
request-priority change; the advice only promotes, temporarily withholds ready keys
from native eviction and evicts a hinted session's unreachable keys first.

## Ownership and boundaries

| Owner | Responsibility |
| --- | --- |
| `contracts.py` | Closed versioned registration/hint schemas, namespace and canonical digest |
| `config.py` | Closed product settings and resolved mechanism attribution |
| `observed.py` | Bounded scheduler-owned observations and replay state |
| `advice.py` | Causal, identity and replay validation; advice-disabled control |
| `vllm_adapter/arc.py` | Ready-prefix scan, promotion, bounded holds and unreachable-first eviction |
| `vllm_adapter/replay.py` | Native manager binding for the CPU trace-replay simulator |
| `vllm_adapter/connector.py` | Native construction, computed-token bounds, registration and scheduler cleanup |
| `vllm_adapter/transfers.py` | Submission/completion failure reporting and failed-load destinations |
| `vllm_adapter/endpoint.py` | Authenticated HTTP boundary into engine RPC |
| `tools/runtime_verify.py` | Functional attempt admission, source snapshot and terminal receipt |

Pure product code cannot import runtime or development packages, or the adapter.
vLLM owns transfer bytes, allocation, reference counts, readiness and exact block-key
matching. Advice changes queue order only. Each key stays in its own T1/T2 queue,
and the root is most recent within its queue. An absent or pending block ends the
eligible prefix; active ready references are unchanged.

For multiple cache groups, observations share a token boundary aligned to every
group’s chunk size. Refresh requires the full-attention prefix and each other group’s
required window/checkpoint at that boundary. A missing or pending required group
reduces the boundary or refuses refresh. Recurrent storage retains the native state
precision, which may differ from attention KV precision. Group key lists are charged
to the same observation memory budget.

## Construction

Use the pinned runtime and launcher described in [runtime](runtime.md). The
connector accepts FP16 full-attention groups, optional FP16 sliding-window groups,
and aligned native recurrent checkpoints, TP1, without LoRA or speculative decoding.
At least one full-attention group is required. Dense qualification remains the
established scope; hybrid/sliding qualification is ongoing in [benchmarks](benchmarks.md).
The connector requires `kv_load_failure_policy="recompute"`. Unsupported
construction and optimized Python execution refuse explicitly.

Settings are `namespace`, positive `cpu_bytes_to_use`, Boolean `advice_enabled`,
bounded positive `max_sessions`/`max_events`, and
`metadata_bytes`. Defaults are
advice disabled, 4096 sessions, 256 events per session and a 64 MiB metadata budget
(maximum 256 MiB). ARC and unfiltered stores are fixed choices, not settings.
Retained observations and events are charged by recursive Python object size plus
1024 bytes per record/container entry, counting shared values repeatedly. This is a
conservative admission budget, not a process RSS limit. Capacity refusals preserve
existing state and never refresh ARC. Keys are at most 256 bytes and boundaries at
most one million tokens. Replacement, retirement and reset reclaim the charge. The constructor selects native CPU ARC and full-transcript stores.
Registration and hint acknowledgements expose exact retained session/event counts,
charged bytes, configured limits and lifetime retirements. These owner-maintained
counts distinguish reclaimed records from allocator/RSS behavior. Registration also
reports native CPU chunk bytes/capacity and the full-attention prefix's minimum
chunk demand; it does not assert transfer readiness or a performance benefit.
Unknown keys refuse. The control is this same connector with advice disabled and every
correctness and failure guard retained.

Finished stores stop at computed tokens minus in-flight tokens and round down to
complete chunks, so a sampled but uncomputed final token never becomes reusable KV.
Only a successful transfer permits native readiness. Failed stores release their
slots without publication. Failed loads report invalid GPU destinations, release CPU
references and invalidate scheduler mappings; native recompute handles affected
requests. Exceptions without a safe completion fence stop the engine.

## Registration and hints

Pass this in an ordinary generation request's `kv_transfer_params.wait_hint`:

```json
{"schema":"wait-registration-v2","scope":"opaque-client-scope","session":"conversation","incarnation":"unique-instance","generation":1}
```

A completion with registration returns `kv_transfer_params.wait_hint`. Successful
observation construction includes registration acceptance,
complete-token boundary, prefix digest, observed monotonic timestamp and engine
clock domain. Registration happens in both advice modes. Invalid registration
returns a refusal without crashing the engine. Within an incarnation, each turn
must register a strictly greater generation. A new incarnation replaces the old session and its replay state. Registration refusals
include a reason. Untracked requests, no complete computed chunk, missing exact
keys, and boundaries above 1,000,000 tokens return `registered: false` with a reason
without observation fields. Native request completion and offload are unchanged;
requests without registration keep the native response.

A hint has schema `wait-hint-v2`, the same scope/session/incarnation/generation,
an event ID, exact prefix/boundary/namespace, the returned `observed_ns`, and
`published_ns`, `deadline_ns`, `clock_domain`. Names and schemas are closed;
Booleans are not integer generations. Published and deadline times use the engine
host's monotonic clock; remote-clock clients are unsupported. The deadline is at
most 60 seconds after publication. A cache reset changes the engine clock domain.

Namespace fields are `model_revision`, `tokenizer_revision`, `adapter_namespace`,
`layout_namespace`. The prefix digest is SHA-256 of UTF-8 JSON containing
`namespace` and `token_ids`, recursively sorted keys, separators `,` and `:`, no
whitespace. A fixed externally calculated vector is tested. Namespace labels
organize observations; native exact block keys still determine reuse.

Advice requires the exact server observation timestamp and identity, a live
deadline for a new event and the matching engine clock. The advice mechanism is
reachable-hold. It applies native touch semantics to the ready prefix (ready T1
keys move to T2, T2 keys to most recent) and then holds those keys out of eviction
until the session's next request with the same scope/session, the hint deadline,
replacement or retirement of its observation, or a cache reset. Held keys are
capped at half the CPU pool; a hint that would exceed the cap promotes without
holding. When an eviction cannot be satisfied with holds in place, holds are
released one at a time, earliest deadline first, so holds never cause a store
failure. The hint also marks the session's unreachable keys: sliding-window and
recurrent-state keys before the window its ready boundary needs (for Gemma,
local-attention keys more than 1,024 tokens back; for Qwen, recurrent checkpoints
before the last one). A return at that boundary can never use them, so evictions
take marked keys that are idle and unprotected before asking native ARC for the
rest, all or nothing. A return at a shorter boundary (for example after context
compaction) may need marked keys and then recomputes. Reachable-hold replaced the
earlier recency hint under a rule fixed before its engine runs
([result](../results/reachable-hold-decide/result.md)); recency and promote-hold were retired after
promote-hold missed two earlier rules
([first](../results/promote-hold-advice/result.md),
[settle](../results/promote-hold-settle/result.md)). An exact replay returns
its original decision, refreshed count and ready-prefix count even after expiry, with `replayed=true` and zero newly refreshed keys. It
never mutates ARC again. Changed content for an existing event refuses. This applies
in both advice modes. Replay state lasts only for the current observation and
engine clock domain; a new generation, incarnation, retirement or reset invalidates
it. Event capacity refuses further new events. Session capacity retires the least
recently active session and counts the retirement. A new registration gets a
new server timestamp, which invalidates older hints. Reset clears observations and
replay state.

POST hints to `/v1/wait-hints`. The endpoint requires explicit server API keys and
refuses startup without them; `VLLM_PLUGINS=wait_hint` enables the installed entry.
It authenticates before parsing and routes through the engine owner. Acknowledgements
(`wait-hint-ack-v3`) include ready CPU prefix length, acknowledged and newly
refreshed counts, held keys, replay flag, decision and resolved mechanisms
(`wait-hint-mechanisms-v2`, field `advice`). API keys define one trusted operator, not
separate tenants.

Supported schema readers are `wait-registration-v2`, `wait-hint-v2`,
`wait-hint-ack-v3`, `functional-admission-v1`, `functional-terminal-v1` and
`runtime-observation-v1`. Unknown versions refuse. A future migration creates a new
artifact with lineage and keeps its source.

## Known runtime limitation

The pinned vLLM ARC policy can fail to evict when T2 has no eligible candidate,
without retrying eligible T1 entries. Upstream fixed this in
[vllm#55461](https://github.com/vllm-project/vllm/pull/55461). It is deliberately
retained in both modes and pinned by a regression test. Fixing or repinning needs
a new runtime identity and requalification of both modes.
