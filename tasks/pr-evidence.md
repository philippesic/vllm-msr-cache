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
complete group keys and newer-return detection are available. The prototype uses
fixture session/generation parameters, not the final closed wire schema. Old runtime CPU lane: 28 passed; portable: 87 tests,
105 rule proofs. No hold implementation, W1 admission or GPU instrument yet.

## Next
Implement hold-only shared machinery, bounded metadata/action state and
deterministic lifecycle tests; measure private scheduler polling cost. Then finish Stage 1 freezes/instrument before GPU admission.
