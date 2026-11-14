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
Active owner: current agent on main. GPU exposure: zero. Exact wheel
0.34.0rc1.dev465+ge9281bd93 / torch2.14+cu130 in artifacts/venvs/pr-evidence;
matched post runtime in pr-evidence-post. Clean frozen checkout in
artifacts/checkouts/vllm-pr-evidence. Supported old runtime remains unchanged.

The original public-callback-only stop interpretation was wrong and explicitly
retracted in results/pr-evidence-feasibility. Private read-only scheduler access is
allowed by the plan and works. Native policy probe pr-policy-feasibility-09 passes:
ready/late-ready/shared/GPU-only prefixes, return/expiry/cancel/reset, failed-transfer
and failed-eviction atomicity, yield before store failure, all-group partial-tail and
window geometry, forks and primary-tier lifecycle. New 36-case native matrix covers
1/2/16/1024 chunks and 0/0.25/0.5 shares for ARC/LRU/post policies. Neither route is
infeasible. Hold-only remains the CPU-qualified candidate; final Stage1 freeze pending.

The 18-cell production-size bookkeeping matrix pr-bookkeeping-01 found adversarial
1M-chunk/1k-session eviction p99 14.9ms ARC /81.5ms LRU, ~18.2MB owned metadata.
Retained limitation, not GPU efficacy. Post utility patch SHA
 d33ae4c099f20bcb6b3e1dca4cbde2038988519ab5bc6f5ef50ec26b5b103ddb
passes native MessagePack dispatch/control identity in pr-post-feasibility-01;
p50/p99 fresh disabled dispatch 62/123us excluding network/process transport.

W1 admission pr-w1-admission-01 completed/verified (2071s CPU); healthcheck deleted.
Frozen manifest d993b516214bd534e198c7122fe470407b011f974c080a42bb8cc650cd0d40da,
tracked compact recipe/identities/counts in benchmarks/pr-evidence/w1-freeze.json.
Pilot D8/HM/HS tasks 1691/1689/1562 across28 repos; natural ends30/27/17, remaining
ends truncated. No success filter; source 24,100 rows/10,637 tasks/129repos. Confirmation
performance remains held out. All assets/exclusions/failed attempts retained.

Common delegating manager/connector observer records actual per-group store/load/
eviction keys, scheduled boundaries and native transfer stats; process-tree PSS,
faults, swap/available RAM and outside CPU counters implemented. pr-instrument-05
passes all seven arms through real HTTP with deadline/failure accounting, independent
arrivals and post-hint ack barriers; both failure sessions and unoffered later turns
retained. Raw capture metrics regenerate independently; files/journals are hashed.
Earlier instrument failures:01 incorrect no-op-lock fixture identity;03 reader
omitted newly added endpoint field, corrected without changing raw successful run.
Capture p50 lifecycle overhead ~70–140us in fixture; serving overhead unqualified.

Qwen diverged-local-hit component behavior persists; core reconciliation inspected,
full GPU scheduler behavior pending admission. Public filing deferred (unauthorized).
AutoAWQ/Marlin source minimum capability75; Gemma Turing adaptation applied identically
to both new runtimes as gemma-turing-main (hashes in manifest). LMCache0.5.6 joint
resolution fails due OpenTelemetry<=1.40 vs vLLM>=1.45.1; retained isolated install log,
no native env dependency changes, no competitor lane in Stage2. Replacement must be
verified/frozen before a future comparison.

The paired W1 client, full capacity matrix, closed hint acknowledgements and deeper
instrument qualification pass full acceptance (114 portable tests plus105 rule
proofs) and the old-runtime lane (28 tests). Test inventories added cases only.
The real-corpus CPU schedule is retained at pr-evidence-runtime-02/schedule-cpu-01.json;
its illustrative rates are not performance-calibrated settings.

The admission-only byte detector passes CPU exact/missing-copy/corruption fixtures
in pr-instrument-06/07. Native request-ID randomization is preserved with an explicit
client mapping; pr-policy-feasibility-11 passes that conversion. Attempt10 had a
fixture-constructor error and remains retained. Offline event reconciliation now
checks slot/readiness/load/eviction lifecycle and successful transfer attribution;
hand traces catch corrupted accounting. Live GPU qualification is still pending.

## Next
Finish cell metrics and server/admission runner with restored
bytes and same-boundary continuation, budget/receipt/interruption lifecycle,
calibration/statistical edge-case protocol. Freeze and commit Stage1 exit inputs,
then idle-check and execute authorized 2080 Ti admission/screen. No GPU runs yet.
