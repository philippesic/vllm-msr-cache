# Current work

Last reconciled: 2026-11-08. This is a handoff summary, not a live process monitor.
Update the affected entry at a meaningful handoff, change of direction, or finish.
Replace old state; move completed outcomes to [history](PROJECT_HISTORY.md).
Verify stale ownership before using shared resources. Keep detailed records in
linked evidence/run directories, not a second diary here.

Latest completed handoff: [Stage 6 findings](../evidence/stage6-2080ti-20261106/RESULTS.md) and [gated 5080 replication](../evidence/stage6-2080ti-20261106/HANDOFF_5080.md). Workload-dependent gains and adverse results remain qualified; the missed repository pressure comparison is now separately measured. No 5080 jobs or production qualification.

Plan and completed local evidence: [wait-hint validation and release plan](WAIT_HINT_PRODUCT_PLAN.md).
The original four stages cover causal validation, competitive comparison, repeatability
across realistic conditions, and compatibility across models/environments. The
user authorized staged execution and routine repairs; the retained protocol
declares owner-selected cases, thresholds and bounded budgets. Completion of
these selected runs does not establish a universal performance or quality claim.

## Stage 6 RTX 5080 rebuild — active

The same pinned LMCache source was rebuilt for sm_120 plus PTX in a separate
Continuum environment; exact KV round trip and live serving passed. All 54 screen
configurations / 22 matched blocks and functional qualification pass, including
all five Continuum fixtures and every model's native pressure gate. The imported
524-configuration / 210-block confirmation is running under the durable
coordinator and 15-minute health schedule. The first chat/QA/document cells have complete
paired estimates without a resolved throughput difference; the full matrix remains
pending. The final screen source audit covers
1,248 selected outputs / 522 full contexts. Partial confirmation review now covers
635 selected outputs / 74 contexts, including sixteen fully read document sets;
final confirmation review and later phases remain pending. No final throughput
or quality-equivalence claim. The November 8
failure is preserved. [Current results](../evidence/stage6-5080-20261109/RESULTS.md),
[continuation](../evidence/stage6-5080-20261109/CONTINUE.md) and
[latest integrity snapshot](../evidence/stage6-5080-20261109/confirmation-integrity-checkpoint-02.json)
retain actual ownership/progress and verified evidence.

## Stage 6 RTX 5080 replication — closed on capability failure; GPU released

The pinned Continuum/LMCache GPU KV-transfer extension failed all five anchor
screen warmups with no compatible kernel image. Binary inspection confirms no
sm_120 kernel or PTX fallback. Native preflight passed, but qualification is
false; activation, confirmation, overhead and operational phases were not run.
The authorized capability-failure stop preserves all 41 attempts / 3,142 verified
files and unchanged transfer seals. No confirmed 5080 speed or quality claim.
Owned GPU clients/servers are gone, both locks are free, both health schedules
are disabled, and useful raw evidence/caches/the ext4 image remain preserved.
[Results and limitations](../evidence/stage6-5080-20261108/RESULTS.md),
[failure receipts](../evidence/stage6-5080-20261108/capability-failure.json), and
[closed continuation](../evidence/stage6-5080-20261108/CONTINUE.md) retain the proof.
All 2080 Ti seals and historical study state remain unchanged; no addendum opened.

## Repository pressure recovery — complete; GPU released

All 18 configurations and integrity/source/all-cap audits passed acquisition
requirements. Candidate/ARC throughput is 1.0210 [0.9885, 1.0546], an unresolved
gain; quality equivalence remains unestablished. Runtime fix and results are
pushed. Owned processes exited, both locks are free, and the health schedule is
disabled with no next run. [Results and evidence](../evidence/stage6-2080ti-20261106/repository-recovery-03/README.md)
and [brief history](PROJECT_HISTORY.md) retain the completed outcome.

## Wait-hint release extension — Stage 6 complete within bounded scope

- **Stage 6 acquisition and audits:** all 524 primary configurations / 210 whole
  matched blocks completed. The separate bounded addendum completed 24 QA pairs
  and four three-arm advice blocks (60 configurations / 28 blocks). Its repository
  gate failed before serving under the test interpreter; the failed attempt is
  preserved and the section is closed under the no-replacement rule. Repeated
  four-system repository performance under CPU pressure remains unmeasured.
- **Measured outcome:** performance depends on workload/model. Anchor mixed
  traffic candidate/ARC throughput is 1.1244 [1.1090, 1.1401], while Continuum is
  faster on documents and secondary Qwen3-4B documents favor ARC. All intervals
  use independent matched processes, not request-level repetitions. Anchor B06
  is a GPU-reuse control with zero CPU pressure; all completed secondary native
  B06/B08/B09/B11 processes have actual pressure receipts. Configured external
  systems differ in runtime/backend/chunking/hint interface. See
  [results and limitations](../evidence/stage6-2080ti-20261106/RESULTS.md).
- **Adverse findings:** fresh enabled/no-hints QA throughput is 0.9667
  [0.9539, 0.9797], leaving the declared 3% loss bound unresolved. Misleading hints
  raise unhinted mean first-token latency to 1.1456 [1.1066, 1.1860] versus ARC.
  All four stress blocks recorded CPU pressure and false-positive recency touches.
  These findings are retained; the selected policy, original acquisition and
  frozen schedules were not tuned or replaced for favorable results.
- **Quality/integrity:** objective checks cover 10,616 primary and 3,504 addendum
  final outputs. Explicit source review is audited for 6,998 primary items / 638
  full contexts and 2,467 addendum items / 96 contexts, plus all 699 archived
  capped intermediate/final outputs, including retained failed attempts. Review
  retains unsupported claims, strict-format failures and truncation; it does not
  establish answer-quality equivalence or unbiased population accuracy. Primary
  verification covers 73,849 files; addendum covers 6,009. See
  [audit checkpoint](../evidence/stage6-2080ti-20261106/post-acquisition-audit-checkpoint.json).
- **Transfer:** addendum relocation and GPU-free source/input import against
  independently rebuilt runtime02 passed. Primary relocation/offline verification
  and runtime02 import also passed from the attested pre-addendum analysis closure;
  original source 21 and the four-module derived allowlist are unchanged. The
  [5080 handoff](../evidence/stage6-2080ti-20261106/HANDOFF_5080.md) records exact
  separately gated commands. No 5080 jobs, new base check or SAE integration.
- **Resources:** GPU acquisition has ended and both reservations are free. The
  existing scheduled task is disabled with no next run; no new schedule was created.
  Fast worker supervision is unchanged. [Unattended recovery](STAGE6_UNATTENDED.md)
  and the saved queue retain exact identities. Preserve raw evidence, reusable
  caches and the paused AWQ work; check current ownership before new GPU work.
- **Stage 7 remains pending:** integration/release writing, actual new-hardware
  qualification and any separately authorized closure of competitive-evidence
  gaps. Stage 6 supports scoped preliminary PR evidence, not production or broad
  architecture/performance claims. Earlier Stage 5 optimization remains complete;
  [history](PROJECT_HISTORY.md) and [its findings](../evidence/stage5-optimization-20261105/README.md)
  preserve that outcome.

## Wait-hint staged qualification — selected local execution completed; GPU released

- **Authority:** user authorized all four stages, routine diagnosis and continuation while unavailable. Plain ARC is primary; upstream/Continuum secondary. Only abandon if the mechanism is genuinely unworthy.
- **Outcome:** selected execution across all four stages is complete. Batch time fell 7.51% versus ARC in the main comparison, 8.27% in the fresh repository reserve, 18.12% in the fresh full-budget document reserve, and 8.87% with timed CUDA graphs. These are conditional workload results; tight documents and several other regimes remain unresolved. Original-workload upstream confirmation is competitive within 3%, without a resolved lead; upstream/Continuum win in some regimes. Answer-quality and production gates remain unmet.
- **Validation:** eight functional cells, four performance cells and two client examples completed. Functional checks cover actual pinned/shared buffers, restore versus fresh computation, greedy equality, and native abort/drain/recovery. All 46 retained coordinator attempts verify, including original failures and repairs. Focused runtime, benchmark-runner and portable checks pass; the isolated source/environment also passes native integration and 60 focused checks. Its dependency differences are disclosed. The finished archive verifies all 38,842 files and 93 task/source links.
- **Scope/resources:** pinned runtime on the local RTX 2080 Ti/WSL2 host. The newer published runtime is API-incompatible; other hardware/native Linux remain untested. Owned GPU processes are reaped and both reservations are available. Disposable task caches were cleaned; useful evidence, environments, recovery backup, other work and paused AWQ are preserved. No deployment; no commit/push at its original handoff. Stage 5 subsequently checkpointed the saved completed foundation separately.
- **Handoff:** [outcomes](../evidence/wait-qualification-20261105/README.md), [support matrix](../evidence/wait-qualification-20261105/SUPPORT_MATRIX.md), [adverse findings and diagnoses](../evidence/wait-qualification-20261105/REGRESSIONS.md), [verified archive](../output/wait-qualification-20261105/release/wait-hint-qualification-20261105.tar.xz). Raw attempts and recovery backup remain in `output/wait-qualification-20261105/`; final delivery and storage receipts are adjacent to the archive.

## Four-system comparison — completed; GPU released

- **Outcome:** primary configurations completed 256 EOS responses / 192 tools.
  Wait ARC 101.628 s, corrected upstream 102.711 s, ARC 108.153 s and corrected
  Continuum 109.108 s. Different outputs, paging/runtime/backend differences,
  one pass and explicit orchestration/correction gaps leave no established winner.
- **Correction/quality:** Continuum's supported decode-publication setting is
  enabled in a predeclared primary replacement; its original false row remains
  diagnostic and unchanged. All 35 decode saves match completed stores. Strict
  status formatting passed only 2/16 local/upstream and 6/16 Continuum; normalized
  test outcomes were correct throughout, reasoning ungraded.
- **Handoff:** [results, controls, two sealed source sets and provenance](../evidence/fair-four-20261104/README.md).
  Initial four cases / 516 files and corrected case / 139 files verify; 124
  benchmark and 31 portable checks pass. Owned processes reaped, GPU idle and
  both locks available. Other work and paused AWQ preserved; no qualification,
  deployment, commit or push.

## Wait-hint opportunity evaluation — completed; GPU released

- **Outcome:** the real-tool pressure profile completed three rotated ARC/wait/semantic comparisons: 576 responses / 432 tools. Early CPU wait retention improved workflow throughput 7.0%, token rate 6.2%, and average first-token latency 0.43 s; semantic retention added no gain over wait ARC.
- **Handoff:** [results, controls, limits and reproduction](../evidence/wait-opportunity-20261104/README.md). All nine retained run sets verify; 453 affected tests and 31 portable checks passed. Owned servers/tools are reaped, both locks available, prior work preserved; no deployment. Paused AWQ work remains paused.

## Real repository-agent benchmark — completed; GPU released

- **Outcome:** `make bench-agent` executes live repository searches/tests and
  four generated requests per session. Four 7B ARC/wait cases passed: 128 model
  responses, 96 real tools, active CPU advice and eviction-affected returns.
  All native/endpoint accounting checks and offline verification passed.
- **Finding:** wait arms restored more tokens and had slightly shorter workflows;
  different live outputs and only two pairs prevent a causal speedup claim.
  Use this tier for policy development; broader answer quality remains ungraded.
- **Resources:** owned tools/servers reaped, GPU idle and both locks available.
  Prior work preserved; no deployment; paused AWQ loading remains paused.
- **Handoff:** [commands, metrics, measurements and limits](../evidence/real-agent-eval-20261104/README.md).
  Raw evidence and recovery backup: `output/real-agent-eval-20261104/`.

## Semantic-guided KV SVD factorial — completed; limited workload evidence

- **Outcome:** 54 real-serving cases / 972 completed responses; no repeatable
  gain over wait-hint ARC. Rank-4 compression failed factual retrieval checks;
  rank 8 passed the narrow fixture but retained cost and public-quality limits.
  Smaller solver batches and prefetch leases repaired workspace/tail failures.
- **Resources:** owned serving processes reaped, GPU released. Default-off
  experimental implementation retained; no deployment. Prior qualification and
  paused AWQ work remain preserved.
- **Handoff:** [results, refinements and stopping decision](../evidence/semantic-svd-serving-20261104/README.md).
  Raw runs and frozen versions: `output/semantic-svd-serving-20261104/`.
  Subsequent review found scripted public waits gave no prefetch head start;
  those runs establish overhead/quality observations, not a representative policy verdict.

## MSR hybrid qualification — evidence phase finished; qualification incomplete

- **Goal:** qualify coordinated GPU retention and queued-prefix prefetch over CPU
  wait-hint ARC, separating MSR predictive value, transfer opportunity and net cost.
- **Owner/resources:** resumed qualification task finished its bounded evidence
  phase. GPU released; no compute clients or owned serving/capture processes
  remained and both locks were available. Paused AWQ loading remains paused.
- **Current:** retained smoke/pressure artifacts verified; new longer-history ARC
  and wait-hint ARC diagnostic completed 54 requests and 13 restores each. Relevant
  activity exists; usable causal headroom and victim costs remain inconclusive.
  Source audits resolved the stalled nested-Parquet inspection and exposed
  missing/shifted/reconstructed timing. Original LMCache source supports a limited
  returning-only pilot, not complete absolute arrival/completion observations.
- **Finding:** 711 examples with task-disjoint splits; none of the primary one-
  second scalar/distribution/QK gains was established. Exploratory quarter-second
  richer Q/K prediction and simulated retained-value gains survived more flexible cheap
  metadata controls. This is a promising separate representation hypothesis,
  not original scalar MSR success or real-cache-value qualification.
- **Stopping boundary:** original real-cache-value, usable opportunity and native
  cost gates remain unmet. No live hybrid/four-arm implementation or performance
  claim. Further qualification needs representative causal inputs/timing/session/
  observation-horizon semantics, observed cache competition and native costs.
  No user source question remains pending. 170 focused and 31 portable tests pass;
  useful sources/results retained, empty failed preparation cleaned.
- **Handoff:** [results, limitations and remaining work](../evidence/msr-hybrid-qualification-20261103/README.md).
  [Unchanged protocol](../evidence/msr-hybrid-qualification-20261103/PROTOCOL.md).
  Raw runs: `output/msr-hybrid-qualification-20261103/` and
  `output/msr-hybrid-qualification-20261104-resume/`.

## Lower-VRAM sanity check — completed; GPU released

- **Goal:** test whether the qualified continuation profile retains comparable
  throughput/latency at smaller GPU KV capacity; compare native ARC at each size.
- **Owner:** live optimization thread linked below. GPU custody released;
  owned processes are reaped, GPU has no compute clients and port 19372 is closed.
- **Plan:** `output/gpu-cache-frontier-20261103/screen/plan.json`; ten natural-EOS
  invocations, 1,536/1,280/1,024/768/512 blocks, equal CPU capacity and priority.
  This reuses frozen qualified source/runtime and public inputs. Screening alone
  cannot establish a production or latency equivalence claim.
- **Current:** screen and balanced 36-run confirmation completed (3,312 natural
  EOS responses). The 1,280-block candidate saved 224 MiB serving VRAM with
  +6.13% token throughput, -347 ms mean and -740 ms aggregate p95 first-token
  latency. Startup peak stayed unchanged; 23 preemptions and an uncertain
  positive-wait return tail require narrower qualification. A first opt-in AWQ
  loading prototype worsened startup memory and is rejected. Its revision is
  stopped at the user's scope correction after five complete cases and one
  intentionally interrupted case. No follow-on pressure byte audits or persistent
  checks were run. Experimental loading code remains opt-in, default off and
  unqualified. This completes the requested serving-memory sanity check.
- **Retained for revisit:** user requested preserving the loading revision,
  tests, frozen runtime stage and positive/negative results. Its two completed
  comparison pairs showed 476 MiB lower sampled startup peak, at about two
  seconds more model-loading time. This is preliminary, not qualification.
  See `output/gpu-cache-frontier-20261103/bounded-loading-v2/partial-progress-readonly.json`
  and `output/gpu-cache-frontier-20261103/checkpoint.json`. Remain stopped until
  explicitly resumed; revisit the remaining live comparisons and pressure checks.
- **Result:** [comparison and limitations](../evidence/vram-frontier-2080ti-20261103/README.md).
- **Evidence:** raw runs retain the `live-nov3-vram-` prefix in the prior campaign;
  coordinator, plans and reports are in `output/gpu-cache-frontier-20261103/`.

## Live serving optimization — completed; GPU released

- **Outcome:** lead and independent Astra support the exact measured profile for
  controlled asynchronous, human-reviewed continuation drafts. Detailed results
  and limitations are in [history](PROJECT_HISTORY.md) and the
  [final assessment](../evidence/live-optimization-2080ti-20261103/operational-final-root-assessment.json).
- **Owner/record:** <t3-thread://v2/25702ae9-40a1-4a63-bca0-1330b2322fe1/486ba4e6-64f0-4742-9761-2a6ccbabcc1c>.
- **Resources:** campaign GPU custody is released. All owned processes were
  reaped; GPU idle, lock available and serving port closed at final handback.
  Recheck current processes before another task launches work.
- **Boundary:** qualification binds the frozen source and installed-wheel/overlay
  [profile](../evidence/live-optimization-2080ti-20261103/qualified-live-profile.json).
  Subsequent concurrent benchmark-telemetry edits and the new vendored launcher
  are outside this qualification.
- **Retention:** useful results, failures and frozen dependencies remain in
  `output/prefetch-native-20261102/`; no further GPU run is needed for this scoped
  objective. No deployment or latency-SLA acceptance is implied.

## Repository workflow cleanup — completed

The isolated runtime installation, shared GPU reservation, reusable research
validators, lightweight CI and local checkpoints are complete. Full runtime tests
and a fresh-environment GPU smoke passed. No GPU work is active in this task.
See [verification](../evidence/workflow-finalization-20261103/README.md).
The paused AWQ research work above remains preserved and paused.
