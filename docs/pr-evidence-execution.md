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
