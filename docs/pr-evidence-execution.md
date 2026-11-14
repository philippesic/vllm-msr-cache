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

Both routes pass the CPU feasibility gate. The original public-callback-only stop
interpretation was too narrow: the accepted plan permits pinned private state reads.
The qualified policy recovers complete keys/group geometry and releases returned
sessions before eviction through read-only scheduler inspection. CPU lifecycle,
identity, geometry, tiering and bookkeeping evidence is described below. The
[Stage 1 exit and screening procedure](pr-evidence-screen.md) freezes hold-only and
admits GPU model qualification; it does not yet qualify any GPU model.

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
The Stage 1 exit now freezes these limits and route behavior before GPU acquisition.

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
semaphore. The real loopback qualification now covers scheduled/submitted timestamps, ignored
empty metadata, coalesced tokens, a 40 ms injected delay, clock/count corruption,
truncation, cancellation, timeout and connection death. A blocked 32 MiB reader does
not stop its neighboring request. A second loopback uses the actual authenticated
FastAPI endpoint and native SamplingParams with a CPU fixture engine. Raw frames
regenerate the reported metrics during verification.

Native CPU probe `pr-policy-feasibility-08` qualifies endpoint registration, exact
token passthrough, request TTL delivery, metadata/token separation and cancellation.
It also reproduces the Qwen lookup component boundary: a ready full-attention plus
recurrent checkpoint yields 512 external tokens from zero local hits, but zero
external tokens when supplied the later 512-token local boundary. The core scheduler
then reconciles diverged local groups without repeating external lookup at the lower
boundary (source inspection). This caps Qwen interpretation; it does not establish
full GPU behavior. Public bug filing is deferred because Stage 2b posting is not
authorized.


## Common measurement observer

`EvidenceConnector` delegates to the frozen native OffloadingConnector;
`ObservedCPUOffloadingSpec` obtains the native CPU manager and exposes a delegating
observer. Both are common benchmark instruments for every arm. They do not supply
hold lifecycle callbacks, mutate scheduler state or change policy decisions. A4t's
mechanism remains the independently tested CachePolicy plugin. The observer logs
actual offered/stored/evicted keys, readiness and failure at manager operations,
per-group restore ranges at allocation, native scheduled prompt boundaries,
preemptions and worker-reported transfer bytes/time. Lookup's proposed GPU boundary
is recorded separately from the scheduled boundary; the former can be reconciled
by the hybrid scheduler and must never substitute for an actual GPU hit count.

Each process has an exclusive line-flushed journal with sequence numbers and a
monotonic clock. The reader rejects dropped/reordered/truncated records. It keeps
incomplete journals from dead servers as failure evidence; a missing close alone
is not permission to discard an arm failure. Policy snapshots record acceptance
outcomes, physical held-key unions, per-session leases, yields and metadata only
when state changes. No observer calls policy maintenance merely to sample it.
Process-tree sampling adds proportional set size (shared pages divided among their
users), per-process page faults and swap, host available memory/swap counters and
outside-process CPU counters. Sampling misses peaks between samples; native worker
transfer timings are separate from end-to-end request delay.

`pr-feasibility --instrument` snapshots and qualifies this CPU/network path.
Attempt `pr-instrument-01` exposed an incorrect fixture assumption that a no-op lock
property returns the same object on each access; its failure is retained. Attempt
02 passes six policy variants with identical observed/unobserved eviction sequences,
40 failed stores and 160 loads per 200-request trace. Attempt 03 adds the actual HTTP
endpoint. Its native run passed; the first reader rejected the new endpoint field,
then the corrected version-aware reader verified the unchanged record. The measured
median small-lifecycle capture increment is roughly 70–140 us, with raw samples
retained; these fixed-order CPU timings are descriptive, not a serving overhead
bound. GPU transfer correctness and live scheduler accounting still need admission.

## CPU compatibility inspection

Frozen-main AutoAWQConfig and MarlinLinearKernel declare minimum CUDA capability
7.5. This source check and the wheel's compiled architecture list do not replace
sm_75 execution. The prior Gemma head-size-512 Turing tile/stage adjustment still
matches the frozen source uniquely. `gemma-turing-main` hashes that adaptation:
original `1c68d8aedfa8907b55f71f9ca87c0dd18a3905b06c33559c390325e7bdebf015`,
patched `ad963a2c366a00b529979704ab11d6b758242e6569ce4695511bcccefaf66b1c`.
It is installed identically in both evidence runtimes, applies only to head size
512 on sm_75, and is a common platform patch separate from post-response delivery.
No performance data were used to select it.

LMCache 0.5.6 at `3e11e10812b448142c6cdb177a3058652bd826fa` fails CPU dependency
admission with this vLLM: its OpenTelemetry API maximum is 1.40.0, while the frozen
vLLM wheel requires at least 1.45.1. The isolated resolver failure is retained in
`artifacts/runs/pr-evidence-runtime-02/lmcache-install-01.log`; neither native runtime
was changed to force compatibility. A dry-run against an existing environment had
proposed dependency downgrades without jointly resolving vLLM, so it is not evidence
of compatibility. No LMCache comparison is in the Stage 2 core screen; a replacement
compatible revision/tuple must be verified and frozen before any later comparison.


## Paired client schedule

The W1 planner verifies the committed manifest and token-database hashes, then orders
admitted partition tasks by SHA-256 of `[seed,"task",task_id]`. It cycles only if
more sessions than distinct tasks are needed, preserving task occurrence counts.
Independent seeded streams own Poisson session arrivals, lognormal gaps and Poisson
one-shots. Eight of sixteen sessions are preassigned adopters by a separate hash;
arm order uses its own hash. Changing an arrival rate cannot redraw gaps, tasks,
adoption or the independently fixed one-shot stream. Confirmation partitions and
seeds outside 12001–12003 / 12101–12108 refuse in the Stage 2 client.

The client runs every session independently with unlimited HTTP connections. A
return is scheduled at response completion plus its predrawn gap. Post-response
publication is a separate task: a slow acknowledgement cannot postpone the return.
TTL attaches to all adopting requests, including final turns. Post eligibility uses
only the current recorded tool action, also including eligible truncated final turns.
A3/A4t/A5 carry matched TTL messages; A3p/A4p carry matched post-response messages.
One-shots receive no hints. All planned one-shots before the common fixed horizon
belong to the harm cohort. Even an arm that finishes its sixteen sessions early
continues through that horizon.

Requests have a 120-second deadline from their scheduled submission, additionally
bounded by the cell end. Sessions have the same absolute cell-end deadline (at most
900 seconds including drain). A request failure terminates later turns, whose IDs
remain explicitly recorded as unoffered after failure/deadline. All sixteen sessions
stay in the denominator. Hint delivery has a ten-second bound, also capped by cell
end; refused/ineffective hints remain outcomes. Capture failures remain distinct
from engine/transport treatment failures. These are finite-horizon screen settings;
arrival rates, one-shot rates and the common horizon still require baseline-only
calibration under a frozen algorithm before decision runs.

`pr-instrument-05` exercises all seven client arms over real loopback HTTP with two
injected failed sessions, two one-shots and an acknowledgement barrier that cannot
release until all fourteen surviving sessions return. Every arm offers 32 requests,
retains two failed sessions and accounts for two unoffered later turns. TTL arms
send 14 messages and post arms send 12, including eligible final turns. All reach
the common horizon. The receipt hashes every request/report and journal; retained
streams independently regenerate their metrics. This qualifies transport/scheduling
logic on a CPU fixture, not model execution. Native `pr-policy-feasibility-09` adds
the full 36-case matrix of ARC/LRU/post policies, capacities 1/2/16/1024 and hold
shares 0/0.25/0.5, including shared unions and pressure/failed-store behavior.


## Restore probe and event reconciliation

The admission-only worker proxy snapshots actual source regions, delegates native
copies and compares every destination byte. Before each restore it fills the owned
GPU destination with a known byte pattern, then verifies both the copy and its
provenance from an observed successful store. It covers every native group, tensor,
CPU slot and partial chunk. A missing copy cannot pass because of stale matching
GPU contents. This synchronous probe changes transfer timing and is disabled in
performance cells; it currently admits only the direct TP1 layout.

CPU instrument attempts 06/07 exercise exact copies, an omitted region and CPU
corruption. They qualify the detector, not CUDA execution. Policy attempt 10 failed
because the new native request-construction fixture omitted its required null
block-hasher argument; attempt 11 fixes that fixture and passes. All attempts remain.
The endpoint carries a separate client request ID in transfer metadata; vLLM keeps
its normal internal-ID randomization. The native request conversion test verifies
that session registration, TTL envelope and client identity survive this boundary.

`pr-reconcile-events` checks the scheduler's CPU-slot lifecycle, readiness, load
references, eviction capacity and the worker's TP1 job acknowledgements. It retains
store refusals, failed transfers, pending jobs and stream-close state. Token-source
accounting uses actual scheduled compute intervals and successful allocated CPU
loads. Later computation takes precedence over an earlier restore; repeated prompt
computation is reported separately. Failed or unfinished requests retain unobserved
tokens rather than assigning them to GPU reuse. Per-group restored ranges accompany
the reconciled global prefix counts. Hand-verifiable fixtures inject wrong slots,
unready stores and duplicate acknowledgements; each is detected. Live model/event
reconciliation remains part of GPU admission.


## Bounded job and cell measurement

The job runner hashes all model files, all 201 installed distributions and eleven
native source/patch targets before launch. It uses a clean source snapshot and an
exclusive new attempt directory. One fixed campaign directory owns the 3,600-second
admission and 28,800-second screen accounting. Every launch reserves its maximum
wall time plus 60 seconds for cleanup against the remaining ceiling; actual startup
through process cleanup is charged, including unsuccessful attempts. A fresh
`nvidia-smi` observation must show the sole local RTX 2080 Ti with no compute
processes. Busy observations retain a zero-exposure deferral. There is no reservation
system. Missing terminal receipts block another launch until reconciled.

The runner owns process identities and sessions, samples process-tree PSS, faults,
CPU counters, swap and available RAM, and samples GPU processes and device state.
It stops on its deadline, server exit, available RAM below 2 GiB or additional swap.
Failed-resource observations remain outcomes. SIGINT/SIGTERM enters owned cleanup;
a CPU test exercises actual parent/descendant termination using process descriptors.
A receipt verifier checks hashes and time accounting independently of the model
correctness verdict. Abrupt machine loss still requires explicit orphan/receipt
reconciliation; it is not permission to reset the budget.

All arms use FP16 activations, unquantized FP16 attention cache and explicit FP16
convolution/recurrent state, eager execution, 0.85 utilization, 6,144 context and
sixteen concurrent sequences. Batched-token limits are 2,048 for D8/HM and 512 for
HS. Native argument validation covers all 21 combinations in pr-instrument-08.
No W1 GPU-block override is allowed. The same observer and endpoint serve every arm;
post arms alone use the hashed delivery patch.

The cell reader keeps every planned request, including unoffered later turns and
failed one-shots. It joins external IDs to native randomized IDs and derives reuse
from scheduled computation and successful copies. Eligibility is the exact earlier
request LCP, capped below the current prompt's last token, rounded down to the native
common group boundary (hash boundary when partial-tail reuse is supported). Lost
alignment suffixes remain explicit. It counts current submitted prefixes' reachable
native keys, respecting window/recurrent geometry, as logical working-set demand
independent of readiness; no future turn contributes before submission. Shared
keys count once while their sessions overlap. CPU allocator chunk bytes supply the
common unit for comparison with physical pool budgets. Cache residency is reported
separately. pr-instrument-09 joins the full seven-arm HTTP failure fixture to hand
traces and detects an injected 28-token recomputation regression, preserving both
failed sessions and both unoffered returns.


Admission attempts D8/A2-01 and -02 stopped at the unchanged swap guard during
Python startup, before model execution. Charged time was 16.9305 and 6.1462 seconds.
Both had over 12 GiB available RAM and zero owned swapped pages; global swap-out
increased by 31 and 170 pages. Preflight hashing had promoted about 6 GiB of model
weight pages into file cache. An advisory release of those exact read-only file
pages increased free RAM from about 3 GiB to 9 GiB without changing file bytes,
MemAvailable, the swap guard or OS settings. The checksum reader now releases each
safetensors file's clean cache pages after hashing. This is startup input hygiene,
not a policy/threshold change. The two failed attempts remain immutable; a new
attempt must establish whether this removes the interference.
