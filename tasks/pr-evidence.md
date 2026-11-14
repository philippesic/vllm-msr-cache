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
105 rule proofs. Hold-only prototype and pure ledger tests now exist. Native lifecycle probe
artifacts/runs/pr-policy-feasibility-03 passed, including advice-off stock eviction
identity, shared GPU-only keys, cancellation/expiry, yield and failed-store cleanup.
Post-response policy-method injection is not the delivery patch. W1 first parquet
shard is downloaded under artifacts/assets/pr-evidence-w1; pyarrow 26.0.1 installed
in the isolated new runtime for admission. Full W1 admission/instrument pending.

## Next
Complete native geometry/tiering/fault matrix and bookkeeping/polling microbenchmark;
implement W1 admission and streaming instrument; build post-response delivery patch;
then finish Stage 1 freezes/instrument before GPU admission.
