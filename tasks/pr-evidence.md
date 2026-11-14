# PR evidence on frozen vLLM main

## Goal
Complete the authorized Stage 1 and 2080 Ti admission/screen under the accepted plan,
then close the results for Philip’s 5080 decision.

## Authorized
Philip authorized Stage 1 and Stage 2: at most 1 GPU hour admission and 8 GPU hours
screen, local 2080 Ti only. Check idle before every job. Freeze dependent inputs
before GPU; run complete paired blocks in priority order and re-quote after the
first. Retain every attempt. No Stage 2b posting, Stage 3/pilot, 5080/rental, WSL
memory increase, changed decision rules or check exceptions. Push milestones.
When genuinely finished and GPU free, send Philip’s exact two-task authorization
message from this conversation to thread 4dd4323d-196a-4973-a4cb-3e985a47c982.
The message has not been sent.

## Done when
Stage 1 and Stage 2 complete or stop under an actual plan rule or resource ceiling,
with close-results, per-route/model verdicts, harms, opportunity and not-run coverage,
updated docs/findings and pushed commits. Then send the GPU handoff.

## State
Active owner: current agent on main. Stage 1 is committed/pushed at 0b41d6a4;
checksum cache cleanup is pushed at 087a1d8d. Both routes are CPU-feasible, using the
same frozen hold-only mechanism. The original public-callback-only stop was wrong
and is explicitly corrected in results/pr-evidence-feasibility. Read-only private
scheduler access is allowed and passes native lifecycle, geometry and identity tests.
No GPU benefit numbers have been observed.

Exact runtime wheel 0.34.0rc1.dev465+ge9281bd93 / torch 2.14+cu130 lives in
artifacts/venvs/pr-evidence; matched post-response runtime is pr-evidence-post.
The clean upstream checkout is artifacts/checkouts/vllm-pr-evidence. The original
supported runtime is unchanged. Both environments have the common Gemma Turing
patch; only A3p/A4p have the delivery patch. Exact hashes, model revisions, route
limits, dependency tuple and protocol hash are in benchmarks/pr-evidence/stage1-freeze.json.

CPU receipts retained under artifacts/runs: pr-policy-feasibility-11,
pr-post-feasibility-01/02, pr-bookkeeping-01 and pr-instrument-10. Instrument 10
covers all 21 model/arm parser configurations, poisoned-destination byte fixtures,
real HTTP failures, paired schedules, metric regeneration and working-set accounting.
Live GPU correctness remains unqualified. Production-size bookkeeping p99 at 1M
chunks/1k holds is 14.9 ms ARC / 81.5 ms LRU with 18.2 MB owned metadata; this is a
retained limitation, not GPU efficacy. LMCache 0.5.6 fails joint dependency resolution
and has no Stage 2 lane. Qwen’s diverged-local-hit component observation remains;
its full scheduler probe has not run.

W1 admission pr-w1-admission-01 is complete and verified (2,071 CPU seconds).
The full manifest SHA is d993b516214bd534e198c7122fe470407b011f974c080a42bb8cc650cd0d40da;
benchmarks/pr-evidence/w1-freeze.json retains the compact source/tokenizer identities.
The semi-synthetic corpus has no success filter. D8/HM/HS pilot sets contain
1,691/1,689/1,562 tasks across 28 repositories, mostly chronologically truncated.
Confirmation performance remains held out. Raw attempts, assets and exclusions stay.

Five admission attempts stopped during startup on additional host swap-out:
D8/A2-01/02/03, HM/A2-01 and HS/A2-01. Owned process swap stayed zero and minimum
available RAM was 9.26–13.31 GiB. No model reached generation. All five receipts
verify; charged admission time totals 186.461502051 seconds (0.051795 hours),
leaving 3,413.538497949 seconds. Screen time is zero. The ledger is
artifacts/runs/pr-evidence-2080ti. Logs/CPU checks are in pr-evidence-runtime-02.
The checksum cache release raised free RAM but did not remove host-level swap-out.
The GPU is idle and admission healthcheck 6b7e19e4-2836-413e-b106-6119e6418aed was deleted.

A user decision remains pending through the question tool: permission to temporarily
set machine-wide vm.swappiness from 60 to 0 and restore 60 after acquisition, keeping
the WSL memory limit, 2 GiB headroom and every no-swap/decision rule. No OS setting
has changed. Do not repeat identical GPU attempts while this condition is unresolved.

CPU reporting work now implements pooled post-stratified BCa and analytic clustered
sensitivity intervals, the paired whole-route gap, calibration arithmetic and the
per-route/model decision rules. Tests preserve censoring, reject duplicate seed
pairs, exercise threshold disagreement and retain complete-block budget reserves.
These calculations do not certify supplied validity/attribution or resource gates.
Full acceptance passes: 134 portable tests and 105 rule proofs
(pr-evidence-runtime-02/stats-full-02.log). Test inventory changes add ten tests
and remove none. The earlier old-runtime lane passed 28 CPU tests; this reporting
checkpoint changes no adapter behavior.

## Next
Await Philip’s machine-setting answer; useful independent CPU reporting work is
complete. If approved, back up the
original setting, record/freeze the environment change before dependent runs, retry
admission within the remaining budget, and restore the setting after acquisition.
Recreate a healthcheck for resumed long acquisition. If refused, preserve the common
environment blockage and resolve the authorized phase under the plan’s admission
rules; do not call it a failure of holding or silently weaken the swap guard.
