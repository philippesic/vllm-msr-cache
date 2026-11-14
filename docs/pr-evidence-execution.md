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
lifecycle contract, especially for tiering contexts without that scheduler.
Changing to REQUEST_LEVEL offers GPU-hit chunks again but changes storage behavior,
is controlled by the manager rather than the policy, and still lacks arrival
notification. Releasing at request finish is too late; sending a cancellation at
return changes A4t's request-only delivery and adds behavior beyond A4p's
post-response-only patch. Monkey-patching the manager changes the core contract.

Both routes therefore need a revised interface under the accepted route boundary:
A4t cannot meet the lifecycle using CachePolicy callbacks alone; A4p's delivery-only
patch does not fix the same arrival gap. This triggers the Stage 1 stop before GPU.
It is not an efficacy STOP. No mechanism, hold limits, W1 manifest, delivery patch
or A3p identity was frozen for acquisition. No dependent GPU run is admitted.

Recommended revision: a manager-to-policy request-start notification and a complete,
group-described key snapshot at finish, preserving native storage decisions. Apply
any shared core change consistently to controls and treatments and explicitly
retire the pure-plugin label. A private-scheduler integration is an alternative
with greater version coupling and a separate CPU cost/lifecycle qualification.
Philip must choose the revised route contract before reopening this phase.
