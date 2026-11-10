# Minimal port qualification, 10 November 2026

This record covers functional correctness and one bounded agent workflow. It does
not establish a retention-policy win, throughput gain, quality equivalence or
causal improvement in agent productivity. Historical performance findings retain
their original scope in [FINDINGS](../FINDINGS.md).

## Implemented scope

The source is now `wait_hint_offload`: pure closed contracts/configuration,
scheduler-owned bounded observations and causal ARC advice, plus a thin adapter
to one pinned native runtime. The same connector with advice disabled is the
functional control. Both modes retain computed-token guards and explicit
store/load failure handling. Semantic ranking, prefetch and approximate reuse
remain absent. See [source protocol and historical dispositions](../docs/source-port.md).

The foundation now permits ordinary RPC dispatch and genuine vLLM block names,
while retaining tested reservation/API and import prohibitions. Portable checks,
strict pinned-runtime type/CPU checks, and explicit hardware qualification are
separate commands. Discovery-manifest updates are explicit and print their diff.
The production dependency boundary remains independent of benchmark/tool code.

The installed runtime is a hash-locked separate environment with a retained wheel,
private-seam hashes and four temporary RPC publication patches. Installed vLLM is
not modified. Direct wheel-only construction, wrong versions, drift and optimized
Python refuse. Functional attempts freeze dirty/untracked source bytes, record
actual loaded inputs, enforce finite execution with zero automatic retries, and
retain terminal failures. Receipt hashes detect drift; they grant no authority.

## Executed acceptance

Raw attempts and development logs are retained outside Git at
`/home/philip/.local/share/wait-hint-port-work/` on this host. Each functional
attempt contains its exact source snapshot, admission, execution log, observed
inputs and terminal record. Paths below are local dependencies, not remote backups.

| Lane | Retained evidence | Actual scope |
| --- | --- | --- |
| Portable | `development-logs/wait-port-portable-final.log` | 166 CPU tests and deliberate-violation/control proofs; nine groups pass, native-owned-source and runtime qualification explicitly omitted |
| Strict runtime | `development-logs/wait-port-runtime-lane6.log`, `runtime-cpu-06/` | Strict types on all 12 adapter/runtime-test source files; 12 real pinned-runtime CPU tests; CUDA uninitialized |
| Hardware | `hardware-04/`, `hardware-04-wrapper.log`, `hardware-04-sanity.json` | Three real hardware tests; external historical ownership wrapper; RTX 2080 Ti/WSL2; dense FP16, TP1, TRITON_ATTN, eager execution, FCFS |
| Setup | `development-logs/wait-port-install.log`, `development-logs/wait-port-lock.log`, retained `wheels/vllm.whl` | Hash-required installation from the locked closure; installed dependency consistency checked; archived native wheel |

The portable log above precedes the final coherent checkpoint; the normal commit
hook reruns full acceptance on the committed source. CPU fixtures construct real
native ARC/manager/scheduler/RPC bodies and fake only transfer handlers or utility
transport where necessary. They cannot prove CUDA streams, device fences or bytes.
Small independent oracles include an externally calculated digest vector, explicit
ARC ordering/victim expectations, historical sampled-token and failed-load cases,
and HTTP/schema producer-consumer conformance. Optimized execution cannot remove
the safety guards because construction and the launcher reject it explicitly.

Hardware observations were:

- Two CPU slots were filled from GPU blocks 3 and 1. Advice retained the root;
  another store evicted the unhinted key. Restored GPU destinations 6 and 5 exactly
  matched source blocks 3 and 2, including native slot/key order and readiness.
- A completed 96-token transcript registered six full blocks. After resetting
  only GPU prefix state, restoration reused 96 tokens. Resetting both caches
  produced zero cached tokens; restored and fresh output token IDs matched.
  Both advice modes produced the same tested output. This is a tiny functional
  case, not general model-quality equivalence.
- A real API server returned 401 for unauthenticated advice, 400 for malformed
  schema and 200 with six refreshed keys for valid advice through async RPC.
  Engine/server cleanup completed. Optional unused DeepGEMM import warnings and
  an upstream resource-tracker shutdown warning remain in the logs.

## Adverse attempts and limits

All earlier attempt directories remain. `runtime-cpu-03` failed because an added
test was missing from the frozen discovery manifest; this was repaired explicitly.
`hardware-03` failed after an over-restrictive guard rejected the pinned model's
native default runner. The guard was removed, and `hardware-04` re-executed all
three cases. Earlier portable fixture/manifest and strict-type failures remain in
`development-logs/`; they were not rewritten into successful attempts.

The native pin's T2-empty ARC eviction fallback defect is deliberately retained
in both modes and characterized by a regression. This bounds the port's behavior;
it does not establish that historical measurements were unaffected. Transfer
submission/completion failures are injected only into CPU fake handlers. Real
hardware fault recovery and power-loss durability are unqualified. Unsupported
layouts, distributed execution, other models/hosts and the RTX 5080 are unqualified.
Clients must share the engine host's monotonic clock. Authentication is a trusted
operator access boundary, not tenant isolation or scientific authorization.

The verifier owns only its child session and exact process birth identity. It is
not a persistent research coordinator, scheduler, global storage service or GPU
reservation implementation. External cooperating ownership and ambient occupancy
checks remain necessary. Disk observations are headroom checks, not OS capacity
reservations. No active/shared data or historical evidence was reclaimed.

## Fresh-agent checkpoint

The public task is [exercise v1](../docs/agent-exercise.md). Its outcome is appended
after the source checkpoint so the evaluated agent receives a pinned, clean source.
One trial can establish only that workflow's usability, with product correctness,
truthful reporting and causal productivity treated as separate claims.

## Deferred work

Benchmark workloads, tiers, metrics, statistics and performance comparisons remain
for the separate redesign. The benchmark interface stays explicitly unavailable.
High-level research operation, persistent retries/wakeups, optional skills and
interface preferences are also deferred. No shared framework, GPU scheduler or
project-local reservation machinery was introduced. The historical repository and
its running/frozen evidence were preserved.
