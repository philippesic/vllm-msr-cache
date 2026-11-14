# PR evidence on frozen vLLM main

## Goal
Complete the authorized Stage 1 and 2080 Ti admission/screen under the accepted plan,
then close the results for Philip's 5080 decision.

## Authorized
Philip authorized Stage 1 in full and Stage 2 on 2026-11-13: 1 GPU hour admission,
8 GPU hours screen maximum, local 2080 Ti only. Check idle before every GPU job.
Freeze dependent inputs before GPU; follow paired-block priority and re-quote after
the first. Retain all attempts. No Stage 2b posting, Stage 3/pilot, 5080/rental,
WSL memory increase, changed decision rules or check exceptions. Push milestones.
When truly finished and GPU free, send Philip's exact two-task authorization message
from this conversation to thread 4dd4323d-196a-4973-a4cb-3e985a47c982. Not sent yet.

## Done when
Stage 1 and Stage 2 complete or stop under an actual plan rule or resource ceiling,
with close-results, per-route/model verdicts, harms, opportunity and not-run coverage,
updated docs/findings and pushed commits. Then send the GPU handoff.

## State
Active owner: current agent on main. GPU exposure: zero. Isolated exact upstream
wheel installed at artifacts/venvs/pr-evidence; checkout at
artifacts/checkouts/vllm-pr-evidence, pin e9281bd939d4ade90d6aacf4b2319aa89ea46c39.
Final callback diagnostic artifacts/runs/pr-feasibility-08 passes; actual callbacks
omit request arrival and GPU-hit keys. The earlier stop interpretation was incorrect:
private ReqContext._scheduler and _offload_key_positions are available, and the plan
does not restrict the plugin to public callbacks. A result correction records this;
immutable observations/attempts remain. The private read-only probe passed at artifacts/runs/pr-private-feasibility-01:
complete group keys and newer-return detection are available. The hold prototype now uses closed session/action schemas and private read-only
scheduler state. Native geometry/tiering checks passed in
artifacts/runs/pr-policy-feasibility-05: partial-tail readiness across all groups,
shared anchors and distinct forks, selected-boundary sliding windows, primary-tier
finish/late-store/return/reset. Attempt 04 failed from an incomplete test fixture;
retained. Old runtime remains compatible (28 tests). The full 18-cell CPU cost
matrix passed in artifacts/runs/pr-bookkeeping-01; at 1M chunks/1k held sessions,
adversarial oldest-key holds cost p99 ~14.9 ms ARC / 81.5 ms LRU per one-chunk eviction,
with ~18.2 MB owned Python metadata. This is a limitation, not a GPU efficacy result.
Post-response delivery now has a separate runtime at artifacts/venvs/pr-evidence-post,
with the same installed versions as the unpatched runtime and a hashed 13-line
EngineCore utility method. Patched core SHA-256 d33ae4c099f20bcb6b3e1dca4cbde2038988519ab5bc6f5ef50ec26b5b103ddb;
original remains 0fc5c97fdf9082b77501492d244d7c4dd0c8848532ce8438d2f082bbf73757e8.
Native utility dispatch/MessagePack test pr-post-feasibility-01 passes A3p/A3
identity, acceptance-origin deadline, late readiness, replay, refusal and cancel.
Its 1,000 fresh disabled messages cost p50 62 us / p99 123 us; no network/process
transport included yet. Streaming endpoint and client now preserve token IDs, ignore metadata for TTFT,
separate token/chunk ITL, retain partial failures and abort disconnected requests.
Native CPU endpoint/lookup tests pass in pr-policy-feasibility-08; old-runtime CPU
lane passes 28 tests. New streaming tests pass. Real loopback and native-manager observer qualification passes at pr-instrument-03
with the corrected reader (new schema field omission fixed without altering raw run).
Attempt 01 failed an incorrect no-op lock identity assertion; 02 passes lower scope.
Observer p50 small-lifecycle overhead ~70–140 us; raw samples retained. Common event
journals preserve actual store/load/eviction keys, scheduled boundaries and worker
transfer stats. Full acceptance passes (112 portable tests plus rule proofs); old-runtime CPU lane
passes 28 tests.
Live scheduler reconciliation and transfer correctness remain admission work. Qwen diverged-local-hit component lookup reproduces:
external hits are zero at the supplied GPU boundary despite a ready lower checkpoint;
full scheduler reconciliation is source-inspected, not GPU-qualified. No public filing.
W1 admission completed and verified in artifacts/runs/pr-w1-admission-01 (2071 s CPU).
Its healthcheck was deleted. All eight source shards, 24,100 rows/10,637 tasks/129
repositories are retained with both source successes and failures. Frozen manifest
SHA-256 d993b516214bd534e198c7122fe470407b011f974c080a42bb8cc650cd0d40da;
compact identity/recipe/counts in benchmarks/pr-evidence/w1-freeze.json, explicit
excluded IDs in artifacts/runs/pr-w1-freeze-01/exclusions.json. Pilot admitted tasks:
D8 1691, HM 1689, HS 1562. Roughly 98% are benchmark-truncated rather than natural
ends; this bounds route/abandoned-hold interpretations. Stage 1 gate remains open.

## Next
Implement frozen open-loop schedules/acquisition and offline group accounting;
complete Stage 1 remaining tests and protocol. Kernel source inspection passes:
AutoAWQ/Marlin min capability75; gemma-turing-main applied identically to both runtimes.
LMCache0.5.6 fails joint resolution against frozen vLLM from incompatible OpenTelemetry
requirements, retained log; no competitor comparison in Stage2. Finish Stage 1
freezes/instrument before GPU admission.
