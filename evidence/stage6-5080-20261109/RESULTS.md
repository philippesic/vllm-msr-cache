# RTX 5080 Stage 6: execution complete; full protocol failed

The sm_120 LMCache rebuild works. Every planned phase and final audit has finished.
Functional qualification passes, but the full throughput protocol **does not pass**:
three original B06 candidate processes lack required CPU-restore activity. All 524
confirmation identities were executed; 521 are valid, yielding 207/210 complete
matched blocks. The three excluded blocks were never replaced, imputed or relabeled.
There is no missing execution awaiting a favorable retry. The frozen full-completion
and throughput flags remain false. [Final audit](completion-audit-01.json) binds
the actual completion flags, exact missing identities and source/ledger hashes.

The headline candidate gains on Qwen3-8B B11 are unresolved: candidate/ARC is
**1.0726 [0.9478,1.2139]** and candidate/Continuum is
**1.0689 [0.9304,1.2280]** (eight matched process blocks; 95% confidence intervals).
On B09, candidate/ARC is **0.9733 [0.9617,0.9850]**, about 2.7% slower, and
candidate/Continuum is **0.6826 [0.6558,0.7106]**, about 31.7% slower. These adverse
intervals exclude parity. Qwen3-4B B08 also favors ARC: 0.9107 [0.8740,0.9489],
about 8.9% slower. No overall benefit, answer-quality equivalence or production
qualification is established.

## What ran and qualified

The linked new-host campaign follows the 2080 Ti handoff sequence: runtime_setup,
transfer_setup/import, bench-tests/check-light, preflight, screen, qualify,
activate-import, confirm, overhead, operational and final audits. Native ARC,
the native wait-hint candidate, Continuum and configured upstream vLLM PR #59714
retain their frozen sources, models, workload order, geometry and repeat counts.
External comparisons cover the configured anchor subset, not every model/cell.
The models are Qwen3-8B-AWQ, Ministral3-3B and Qwen3-4B-Instruct-2507, with
TP1/FP16/eager and the original dense full-attention GQA support scope.

Cells cover chat(B01), independent QA(B02), document summaries(B03), hot/scan
traffic(B04), sequential repository tools(B05), repository pressure(B06), roomy
repository control(B07), full-CPU-cache document tools(B08), half-CPU-cache
document tools(B09), shared-team tools(B10), mixed traffic(B11), and repository
tools with one tool worker(B12). These are fixture definitions; actual observed
pressure/restore evidence is reported separately. The anchor repository fixtures
use their frozen 8-session limit; secondary repository fixtures retain 16 sessions.

| Phase | Executed result |
| --- | --- |
| Direct kernel and live Continuum smoke | Exact FP16 CPU/GPU transfer; four serving requests pass |
| Preflight | Four configurations, 223 sealed files; native correctness/fresh/reuse/drain/namespace guards pass |
| Screen | All 54 planned configurations /22 complete blocks; all five Continuum screens pass |
| Qualify / activate-import | Functional qualification true; parent order/repeats/precision unchanged |
| Confirm | All 524 identities; 537 retained attempts, 56,362 sealed files; 521 valid identities /207 complete blocks |
| Overhead | All 38 controls once; 2,150 sealed files; control checks complete |
| Operational | All six cases once; 2,448 sealed files; all bounded operational requirements pass |
| Final integrity and audits | All 641 retained attempts /67,120 sealed files verify, including inner aggregates; source/cap/recovery/portability complete |

[Preflight](preflight.json), [qualification](qualification.json),
[activation](activation.json), [confirmation inventory](confirmation-attempts-complete.json),
[overhead](overhead-boundary-01.json) and [operational](operational-boundary-01.json)
retain exact identities, request/tool/unit coverage, cleanup and seals. Retained
attempt states are 603 verified, 35 failed and three interrupted; raw ledger state
is distinct from the frozen activity-validity rules. Setup failures and superseded
partial blocks are preserved, with credit only for valid whole matched blocks.

The four-request one-session Continuum smoke proves warmup, GPU KV stores,
pinned geometry and clean serving. Its original policy verdict remains
inconclusive because it produces no scheduler pins, and its timing is excluded.
The unchanged full Continuum screens separately prove policy activation.

The genuine exclusions are Qwen3-8B B06 repeat 0 and Ministral B06 repeats 0/3.
The anchor candidate completed 32 requests/24 tools cleanly but restored zero CPU
bytes and had eight scheduler preemptions, so the zero-preemption GPU-resident
exception cannot apply. Ministral's two candidate processes each completed 64
requests/48 tools but restored zero bytes despite 15 eviction-affected returning
requests. Its 16-session workload is outside the anchor 8-session exception.
These are workload-purpose failures, not missing kernels or harness errors.
[Anchor failure](confirmation-b06-inactive-restore-01.json) and
[Ministral failures](confirmation-ministral-b06-inactive-restore-01.json) retain
the actual evidence. No activity retry, forced flag or repeat reduction was used.

## Throughput and latency

The [complete 42-comparison table](final-results-01/tables.md),
[machine-readable data](final-results-01/comparisons.json) and
[interactive chart](final-results-01/chart.html) retain all native and
external comparisons, pair counts, per-process ratios, output lengths, caps,
serving times and latency figures. Every comparison row exactly reproduces the
earlier terminal 42-row extraction; all 32 anchor rows previously reproduced exactly.

Ratios below are candidate divided by baseline: above 1 means faster. Each is a
geometric mean of paired whole-process throughput ratios with a Student-t 95% CI
on log ratios, conditional on the frozen cohort/host and without multiplicity
adjustment. Requests within a process are not independent repetitions. Throughput
counts completed offered graph units per second and excludes startup; it is not
correctness-adjusted useful-answer throughput. No across-workload aggregate is introduced.

| Qwen3-8B cell | Valid / required blocks | Candidate / ARC [95% CI] | Candidate / Continuum [95% CI] | Candidate / upstream [95% CI] |
| --- | ---: | --- | --- | --- |
| B06 | 7/8 | 0.9781 [0.9546, 1.0021] | 0.9477 [0.9246, 0.9714] | 1.0015 [0.9716, 1.0323] |
| B08 | 12/12 | 0.9857 [0.9646, 1.0072] | 0.6851 [0.6397, 0.7336] | 0.9003 [0.8690, 0.9327] |
| B09 | 12/12 | 0.9733 [0.9617, 0.9850] | 0.6826 [0.6558, 0.7106] | 0.9395 [0.9277, 0.9514] |
| B11 | 8/8 | 1.0726 [0.9478, 1.2139] | 1.0689 [0.9304, 1.2280] | 1.0014 [0.8952, 1.1203] |
| B12 | 12/12 | 0.9759 [0.9227, 1.0321] | 0.9248 [0.8867, 0.9645] | 0.9869 [0.9504, 1.0248] |

Against Continuum, B06/B08/B09/B12 are slower with intervals below 1; B11 remains
unresolved. Native B09 has observed CPU pressure and active CPU restores in all 12
processes per arm. Anchor B06 pressure is inconclusive; it cannot substantiate a
CPU-pressure gain. External native counters are unavailable, not zero. The linked
tables retain every native cell, including adverse and control results.

For B09, mean serving seconds are 43.643 candidate, 42.484 ARC, 29.840 Continuum
and 41.008 upstream. The following latency values are milliseconds. Each is the
arithmetic mean of the named per-process statistic; mean process p95 is not a
pooled request percentile or a population-tail guarantee. Full latency figures
for all 42 comparisons are in the linked table and JSON.
TTFT is end-user time to first token; completion latency covers the whole request.

| B09 arm | Mean TTFT ms | Mean process p95 TTFT ms | Mean completion ms | Mean process p95 completion ms |
| --- | ---: | ---: | ---: | ---: |
| Candidate | 5881.409 | 13087.611 | 8820.174 | 15664.495 |
| ARC | 5492.068 | 12631.278 | 8321.630 | 15723.632 |
| Continuum | 2708.248 | 15290.478 | 4770.513 | 18386.012 |
| Upstream PR #59714 | 5085.238 | 13605.931 | 7797.725 | 16533.650 |

Each B09 arm completed 768 requests. Output tokens differ: 84,360 candidate,
84,384 ARC, 80,055 Continuum and 83,828 upstream; Continuum has 25 length caps,
the other arms zero. Live outputs, quality, runtime/backend/chunking and hint
interfaces differ. This comparison cannot isolate GPU architecture or establish
answer-quality equivalence.

| Secondary model / cell | Candidate / ARC [95% CI] | Valid / required pairs |
| --- | --- | ---: |
| ministral3_3b / B02 | 0.9613 [0.8980, 1.0290] | 6/6 |
| ministral3_3b / B06 | 0.9905 [0.8376, 1.1711] | 4/6 |
| ministral3_3b / B08 | 1.0053 [0.9705, 1.0413] | 6/6 |
| ministral3_3b / B09 | 1.0043 [0.9926, 1.0161] | 6/6 |
| ministral3_3b / B11 | 1.0257 [0.9567, 1.0997] | 6/6 |
| qwen3_4b_2507 / B02 | 1.0005 [0.9782, 1.0233] | 10/10 |
| qwen3_4b_2507 / B06 | 0.9657 [0.9321, 1.0004] | 10/10 |
| qwen3_4b_2507 / B08 | 0.9107 [0.8740, 0.9489] | 10/10 |
| qwen3_4b_2507 / B09 | 0.9919 [0.9773, 1.0068] | 10/10 |
| qwen3_4b_2507 / B11 | 1.0175 [0.9904, 1.0453] | 10/10 |

Every Ministral interval includes parity. Every Qwen3-4B interval except B08
includes parity, including B06 whose upper bound is 1.0004. Qwen3-4B B08 has
observed pressure and active CPU restores in all ten processes per arm, with no
caps. Mean serving time is 48.050s candidate /43.685s ARC; mean TTFT is 6894.475ms
/6366.195ms and mean completion is 8608.781ms /7905.869ms. Its adverse result is retained.

## Overhead and bounded operation

Overhead ran six balanced process blocks for each anchor B01/B02 cell with O0
native ARC/plugin unloaded, O1 plugin loaded but actual off-switch/budget0, and
O2 enabled candidate connector without hint production. Requests are identical
and tool-free within each cell. Two secondary O2 smoke cases bring the total to 38.
O3 is the existing B06 confirmation reference; it is never paired with B01/B02 O0.

| Cell | Ratio | Geometric throughput ratio [95% CI], n=6 | Declared 3% loss bound |
| --- | --- | --- | --- |
| B01 | O1/O0 | 1.0010 [0.9941, 1.0079] | bounded |
| B01 | O2/O0 | 0.9924 [0.9775, 1.0075] | bounded |
| B01 | O2/O1 | 0.9914 [0.9827, 1.0002] | bounded |
| B02 | O1/O0 | 0.9313 [0.8650, 1.0026] | inconclusive |
| B02 | O2/O0 | 0.9906 [0.9405, 1.0434] | inconclusive |
| B02 | O2/O1 | 1.0637 [0.9844, 1.1494] | bounded |

Chat bounds pass within this scope. Short-QA O1/O0 and O2/O0 remain inconclusive
for the 3% loss bound; inconclusive is not equivalence. No additional favorable
repetition was requested or substituted.

Operational used 16 sessions and 64 normal requests per wave: six waves per anchor
arm and two per secondary arm, totaling 1,280 normal requests /320 graph units
across ARC/candidate. Persistent cancellation, drain/recovery and actual native
restore pass. Registry receipts show no outstanding speculative jobs or reserved
GPU blocks. Duplicate advice is idempotent; unknown sessions, future information,
oversize identifiers, delayed expiry and abandoned advice all satisfy the frozen
checks. These are bounded sequences, not production or leak-freedom certification.

## Audit coverage and retained failures

[Final integrity](post-acquisition-audit-01.json) verifies all 67,120 sealed files
and inner aggregates, the unchanged 2,192-file acquisition source, both 4,606-file
execution/import bundles, rebuilt wheel/installed extension hashes, and the
November 8 campaign's41 attempts/3,142 files and original LMCache binary.
The actual interrupted screen process has sealed cleanup and whole-block recovery
proof; partial processes were never continued for comparison credit.

The [terminal primary source review](confirm-review-validation-19.json) covers
10,520 objective finals, 5,983 automatic final failures and 291 selected capped
intermediates. All 7,011 selected outputs /849 full contexts are judged, with all 516
archived instruction/source records verified. Exact reviewed records were
[published](confirm-quality-published-19.json) and frozen finish passes. Selected
duplicate-inclusive findings are 1,514 source failures, 4,029 instruction failures
and 185 not useful; these are not population error rates or quality-equivalence tests.

The separate [all-attempt cap audit](caps-all-review-index-final-01.json) covers
every ordinary cap across valid, failed, interrupted and unmatched attempts:
516 responses /186 contexts (132 screen, 348 confirm, 18 overhead, 18 operational;
447 intermediates and 69 finals). All prior 480 occurrences remain. Five new
occurrences in four novel full contexts were explicitly read/judged; no pending
response or partial stream fragment remains. 137 archived instructions/raw bindings
independently verify, and [reproduction](cap-review-validation-final-01.json)
passes. Counts 348 source failures, 515 instruction failures and 101 not useful
include repeated contexts. Supported useful fragments receive separate credit
from false source/instruction claims. Suggested model commands were never executed.

[Final portability proof](portability-final-03.json) physically relocates the
exported bundle/evidence, hides the original campaign reversibly, verifies all 641
attempts/67,120 files and reproduces all 42 comparison records offline. Guards
reject original model weights/blob symlinks, source/assets and GPU/model imports.
The original is restored. Rebinding/import/dry compilation from exported source
uses independently prepared runtime02, retains parent schedule/precision/source
and creates a distinct campaign whose confirmation remains blocked until hardware
preflight. This proof does not execute imported models or grant another campaign.

## Declared setup deviations and reproducibility

[Build details](BUILD.md), [wheel/kernel receipt](lmcache-build.json) and
[live smoke](continuum-smoke.json) record the approved same-source LMCache 0.3.6
revision cad4db10592e8487d1e0bbd02d1a6f92c37d0581 rebuilt for `12.0+PTX`, torch 2.8.0+cu128,
CUDA 12.8.93, GCC 14 and an isolated compatible header sysroot. Five sm_120 ELF
images and PTX 8.7 are present. All 178 release Python files and kernel/algorithm
sources are unchanged. Exact bidirectional FP16 transfer covers three layers,
256 permuted slots and 393,216 pinned CPU bytes. The separate Continuum environment
uses the same pinned package versions; the original November 8 environment is intact.

Frozen workers bind the literal pinned serving interpreter. Sealed artifacts are
materialized on the project ext4 image because DrvFS symlinks are unsuitable.
After timed acquisition, the existing image was safely grown 20→28 GiB without
replacement/format/unmount ([proof](volume-growth-01.json)). Export initially
assumed a native asset-cache layout unavailable to imported campaigns; its
[repair](portability-repair-02.json) reuses/hash-validates sealed imported assets.
A later fast filesystem copy silently zeroed ranges in one 44MB archive; its
failed proof/log and corrupt sample are preserved. [Buffered verified-copy repair](portability-copy-repair-03.json)
repairs/reuses only a matching disposable tree and avoids a second 12GB allocation.
The successful derived transfer source changes only campaign_verify.py and
campaign_portability.py; the exact acquisition closure is retained separately.
19 focused transfer regressions and 245 CPU tests in that exact derived source pass.
Earlier maintained-suite 260 tests, original 240 benchmark tests and 31 light checks
also pass. No model, serving policy, measurement estimator, validity rule, source
rating, study order or repetition changed for these repairs.

November 8 and all 2080 Ti history remain unchanged. Raw campaign, all failed setup
attempts, source review packets, reusable runtimes/models/builds and relocation
proofs remain locally preserved; compact evidence and authored tools are tracked.
There was no model training, other-machine/project work or Stage 7 writing.

Owned servers/processes are stopped. The RTX 5080 and both GPU locks were
verified released after the results/audits were pushed ([release proof](gpu-release-final-01.json)).
The [completion record](completion.json) tracks pushed checkpoints, requested
waiting-thread delivery and monitoring shutdown. Useful artifacts remain preserved.
The requested completion notification was delivered after the pushed release
proof; its stable delivery receipt is retained. Own hourly monitoring is disabled
with no next run. All authorized work is closed.
