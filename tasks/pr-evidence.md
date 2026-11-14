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
transport included yet. Endpoint/streaming and instrument remain pending.
All eight W1 tool shards are downloaded and SHA-256 verified. Source row inspection
found 24,100 rows, 10,637 tasks and 129 repositories; both success/failure retained.
W1 admission is actively running (CPU only) in artifacts/runs/pr-w1-admission-01,
using a source snapshot, three pinned model tokenizers and the closed 6k/256/2-8-turn
recipe. Log reports 3,000 tasks processed; 12 GiB available, swap unchanged at116 MiB.
Healthcheck every10 min: scheduled-task:dddd647a-0134-4c9f-9bc2-5ac0a4f6a03b, bound
here; delete it when admission ends. No manifest is frozen yet. New W1 tests pass;
post-change fast checks pass, full/runtime checks due before commit.

## Next
Finish W1 admission/verification and freeze its manifest; implement streaming
endpoint/load generator and per-group instrument; complete Qwen lookup/kernel checks;
then finish Stage 1 freezes/instrument before GPU admission.
