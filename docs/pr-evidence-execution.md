# Frozen-main evidence execution

The accepted [plan](pr-evidence-plan.md) owns the decision rules. Philip authorized
Stage 1 and the local 2080 Ti admission/screen on 2026-11-13, with a 1-hour admission
and 8-hour screen ceiling. Later stages and public posting remain unauthorized.

## Implementation estimate before building

These are engineering estimates, not measured runtime or commitments of GPU time.
The minimum end-to-end screen requires, in dependency order:

| Work | Estimated engineering time | Gate |
| --- | --- | --- |
| Frozen interface inspection and CPU feasibility reproductions | 2–6 hours | At least one feasible route |
| Isolated runtime build, hold plugin, delivery patch and deterministic lifecycle tests | 12–24 hours | Identity, completeness and readiness |
| W1 admission, streaming client, group events and instrument qualification | 12–24 hours | Frozen inputs and trustworthy measurement |
| Admission, calibration, paired acquisition and verification | 4–8 hours plus authorized GPU exposure | Hardware and budget |
| Statistical report, source retention and closeout | 3–6 hours | Independently recomputable verdicts |

Minimum screen path: approximately 33–68 engineering hours, conditional on the
interface and build gates. This excludes Stage 3 and competitor comparisons.
A simulator port would add an estimated 8–16 hours and cannot replace required
hardware evidence before rank agreement; do not build it for this screen.
Competitor integrations/full-matrix work would add at least several engineering
days plus separately authorized compute and are outside this phase.

Stage 1 failures are classified per route. If both routes require an interface
revision, stop before GPU and retain the precise missing contract; do not silently
expand the delivery patch or revive the old connector as a pure policy plugin.

## Feasibility boundary

The CPU reproduction uses the exact upstream wheel
`0.34.0rc1.dev465+ge9281bd93`, in `artifacts/venvs/pr-evidence`. It verifies nine
relevant installed Python files against the clean frozen checkout. This is an
isolated wheel installation and CPU interface test, not a local CUDA build or a
full runtime qualification. The original supported runtime remains unchanged.

The public CachePolicy callbacks do not provide request arrival or the complete
key chain for GPU-hit prefixes. CPUOffloadingManager.on_new_request creates manager
state without calling the policy. Its finish callback passes only keys observed
through CPU load/store operations. The scheduler's CHUNK_LEVEL store handler skips
GPU-hit chunks. Tests reproduce this for both native policies, including a prefix
shared by a different session. Successful stores completing after finish do call
mark_evictable; that part of the proposed readiness design is viable.

The missing keys exist in private ReqContext._offload_key_positions; the concrete
context also reaches the private scheduler. Private introspection could recover
keys and group geometry. Detecting returns without a callback would additionally
require retaining that scheduler reference and polling its active-request state
before eviction. This is a possible different integration, not an impossibility
proof about all Python plugins. It does not establish the proposed CachePolicy
lifecycle contract, especially for tier-originated contexts without that scheduler.
Changing to REQUEST_LEVEL offers GPU-hit chunks again but changes storage behavior,
is controlled by the manager rather than the policy, and still lacks arrival
notification. Releasing at request finish is too late; sending a cancellation at
return changes A4t's request-only delivery and adds behavior beyond A4p's
post-response-only patch. Monkey-patching the manager changes the core contract.

Neither route has yet been ruled infeasible. The initial interpretation that both
needed a revised interface was too narrow: the accepted plan permits a pinned
out-of-tree policy, including private state reads. The next qualification tests
read-only scheduler access to recover complete key/group information and release
holds before eviction on a session return. Polling cost and tiering behavior must
be measured, rather than assuming this path fails. No GPU acquisition is admitted
until these and the other Stage 1 requirements pass.

The [diagnostic result](../results/pr-evidence-feasibility/result.md) records the
observations and explicitly corrects the initial stop interpretation. Its original
source record remains immutable. The active work is in [the task](../tasks/pr-evidence.md).

## Hold-only prototype

The candidate out-of-tree policies live in `vllm_adapter/policy.py` and load through
native CachePolicyFactory. They require the exact new runtime and delegate native
readiness, references and storage. Private scheduler state is read only. Holds
release lazily before eviction when a newer registered generation is visible;
older pending requests cannot release newer holds. A read-only membership union
avoids rebuilding the held-key set for each eviction. Failed native evictions first
check whether the unheld request could succeed before releasing leases; feasible
pressure releases earliest deadlines and prevents later readiness from silently
reinstating a yielded lease.

This is a CPU-tested prototype, not the Stage 1 freeze or hardware qualification.
Current candidate limits are half the physical chunks, 4096 observations, 256 action
IDs per observation, a 60-second deadline clamp, and separate 32 MiB conservative
charges for observations/actions and leases. Shared keys count once against physical
capacity. Duplicate action content replays; changed content refuses. The closed
session/action schemas live in `policy_protocol.py`; request TTL is carried in a
KvHintsEnvelope and registration in `kv_transfer_params.wait_hint_policy`.
All limits and route behavior still need the complete Stage 1 tests and cost
measurements before freezing for GPU.

`pr-feasibility --hold-policy` exercises native ARC/LRU policies, disabled controls
and the post-response policy method through the same receipt lifecycle. It is not a
client-to-engine post-response test. The retained probe checks late readiness,
shared GPU-only prefixes, action replay/cancellation/expiry, yield before store
failure, failed-store cleanup, reset and disabled-policy eviction identity. CPU geometry checks now include full attention plus recurrent partial-tail
checkpoints, sliding windows at the selected reusable boundary, shared anchors and
branch-specific state. The snapshot derives partial-tail keys from the existing
native block hashes; it never registers keys or changes scheduler state. Every
required group must be ready. Speculative groups and unsupported partial-tail
alignment refuse advice. Tiering's primary CPU manager preserves the concrete
scheduler context for ordinary engine requests; its finish/late-store/return/reset
lifecycle passes. Tier-originated requests without scheduler/session registration
receive no hold. Remote tiers and threaded transfer races are not qualified.

`pr-feasibility --bookkeeping` retains a fixed matrix of 10k/100k/1M chunks by
0/100/1,000 held sessions for ARC and LRU. It measures 200 one-chunk evictions per
cell, all raw samples, nearest-rank p50/p99, finish/hold cost and recursively counted
Python metadata. Up to 32 distinct keys per session occupy the oldest cache entries;
the 10k/1k case uses five keys each to respect the 50% physical cap. Sixteen unrelated
active registrations exercise return polling. Logical time is fixed so TTL expiry
cannot silently shrink the measured holds. Native chunk storage and the read-only
scheduler are excluded from plugin metadata. This is an adversarial bookkeeping
measurement, not a serving-throughput claim.

The first complete matrix (`artifacts/runs/pr-bookkeeping-01`) found p99 eviction
latency at 1M chunks/1k sessions of 14.9 ms ARC and 81.5 ms LRU, versus 7.6 and 27.0
microseconds with no holds. Owned Python metadata was 18.2 MB. Membership checking
avoids a rebuilt held-key union, but native eviction still scans past protected
oldest keys; LRU additionally pops and reinserts skipped heap entries. Retain this
cost as a scale limitation. The plan has no CPU latency cutoff; Stage 2 guardrails
must still measure the whole serving cost at its stated scale.


## Post-response delivery candidate

`pr-post-response` in the runtime patch manifest adds one EngineCore utility method.
It locates the existing offload manager, enters that manager's lock, and calls the
primary policy's `publish_envelope`. It does not alter request scheduling, keys,
readiness, transfer, or eviction. The patched file hash is
`d33ae4c099f20bcb6b3e1dca4cbde2038988519ab5bc6f5ef50ec26b5b103ddb`; the original is
`0fc5c97fdf9082b77501492d244d7c4dd0c8848532ce8438d2f082bbf73757e8`.
Both runtime environments have identical dependency versions. A3p sends the same
post-response envelopes to AdviceOffARC; A4p uses PostResponseARC.

The closed envelope uses native KvHintsEnvelope v1 fields and 1-256 unique actions,
with `wait_hint.post_response` v1 carrying the closed policy action. Action IDs,
not message IDs, establish idempotency. Unknown fields/versions, repeated IDs
within a message and inconsistent inner/outer IDs refuse the complete envelope
before publication. The engine records one monotonic acceptance timestamp for
its actions. Duplicate actions replay their original decision without renewing
the deadline. Request-TTL actions must match their own request registration.

`artifacts/runs/pr-post-feasibility-01` exercises the actual utility dispatcher and
MessagePack argument encoding with fixture scheduler state. A3p and A3 eviction
sequences match. Acceptance-origin deadlines, late readiness, cancellation and
invalid-message refusal pass. Incremental CPU dispatch/policy cost for 1,000 fresh
disabled messages was p50 62 us / p99 123 us. Socket/interprocess transport and
serving overhead are outside that CPU measurement and remain to be qualified.

## W1 admission recipe

`pr-admit-w1` validates the eight pinned parquet hashes, inspects all source rows,
deduplicates each task by the lowest SHA-256 trajectory ID (with deterministic
ID/shard/row tie-breaks), and partitions by the repository hash. It renders the
original text, tool calls and observations with each pinned native tokenizer;
trace-only duplicate thought/action fields are omitted. Tool arguments become JSON
objects, tool-result IDs are retained, and no missing tool schema is invented.
Thinking is disabled consistently. A post-response signal uses only the current
recorded assistant's tool-call names; exactly `submit` is terminal.

Each request is the chronological recorded prefix. Output length is the rendered
completed transcript's token suffix after its LCP with the request rendering,
capped at 256; generation-prefix differences are recorded. The first context
overflow ends the episode, as does the eight-turn cap; fewer than two turns is
ineligible. Selected tasks are the first 2,048 eligible trajectory IDs per model
and partition, ordered by SHA-256. SQLite retains all eligible tokenized episodes;
the manifest identifies the admitted rows, exclusions, source success mix,
repository counts and stratum inclusion shares. Confirmation repositories may be
processed for content admission but remain held out from performance development.
The admitted content is frozen by [the compact identity](../benchmarks/pr-evidence/w1-freeze.json).
The immutable manifest SHA-256 is
`d993b516214bd534e198c7122fe470407b011f974c080a42bb8cc650cd0d40da`.
`pr-freeze-w1` verifies the admission before deriving compact counts and explicit
excluded task IDs. Pilot pools contain D8 1,691, HM 1,689 and HS 1,562 tasks from
28 repositories. Only 30, 27 and 17 respectively end naturally; all other ends are
context/turn truncations. This is a major limit on whole-route/abandoned-hold claims.
No source-success filter was applied. Performance confirmation remains held out.

## Streaming measurement candidate

The common `pr_evidence` endpoint plugin accepts closed, bounded token-ID requests,
uses native delta output, and exposes authenticated generation, hints, reset and
cache-group metadata routes. It passes registrations and request TTL through native
AsyncLLM arguments; post-response uses only the hashed utility method. It performs
no chat rerendering and aborts its own unfinished request on disconnect. This is a
benchmark injection interface, not Stage 4 public-interface qualification.

The client timestamps actual token-bearing frames separately from acceptance/empty
metadata. It retains exact output IDs, chunk sizes, partial streams, submission
lateness and both submission- and schedule-origin TTFT. Native generation timing
provides mean per-token ITL; client per-token and per-chunk intervals are separate
measurements, since several tokens may arrive together. Engine early completion is
a treatment failure. Invalid frame identities, clock regression or contradictory
instrument counts are capture failures. Independent async requests have no client
semaphore. Real-socket fault tests and instrument overhead remain pending.

Native CPU probe `pr-policy-feasibility-08` qualifies endpoint registration, exact
token passthrough, request TTL delivery, metadata/token separation and cancellation.
It also reproduces the Qwen lookup component boundary: a ready full-attention plus
recurrent checkpoint yields 512 external tokens from zero local hits, but zero
external tokens when supplied the later 512-token local boundary. The core scheduler
then reconciles diverged local groups without repeating external lookup at the lower
boundary (source inspection). This caps Qwen interpretation; it does not establish
full GPU behavior. Public bug filing is deferred because Stage 2b posting is not
authorized.
