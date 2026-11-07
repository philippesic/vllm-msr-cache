# Current work

Last reconciled: 2026-11-06. This is a handoff summary, not a live process monitor.
Update the affected entry at a meaningful handoff, change of direction, or finish.
Replace old state; move completed outcomes to [history](PROJECT_HISTORY.md).
Verify stale ownership before using shared resources. Keep detailed records in
linked evidence/run directories, not a second diary here.

Latest completed handoff: [Stage 5 optimization](../evidence/stage5-optimization-20261105/README.md). Keep the existing wait-hint behavior; no new serving variant qualified. The eviction CPU lead was reproduced on actual states but accounted for less than 0.1% of workflow time. Final functional checks passed; GPU released.

Plan and completed local evidence: [wait-hint validation and release plan](WAIT_HINT_PRODUCT_PLAN.md).
The original four stages cover causal validation, competitive comparison, repeatability
across realistic conditions, and compatibility across models/environments. The
user authorized staged execution and routine repairs; the retained protocol
declares owner-selected cases, thresholds and bounded budgets. Completion of
these selected runs does not establish a universal performance or quality claim.

## Wait-hint release extension — Stage 5 complete; Stage 6 in progress

- **Outcome:** the optimization pass is complete under the user's narrowed final
  scope: 144 valid GPU cases and two retained invalid settings. Keep the existing
  wait-hint rule; optional experiments remain disabled. Actual-state eviction
  speedups were real but too small in total serving cost to establish a practical
  gain. See [findings and validation](../evidence/stage5-optimization-20261105/README.md)
  and the brief [history](PROJECT_HISTORY.md).
- **Stage 6 — implementation and preparation in progress:** the
  [agent execution plan](STAGE6_EXECUTION_PLAN.md) specifies a durable campaign
  layer over the maintained serving suite, vLLM only, and Qwen3-8B-AWQ,
  Ministral 3-3B Instruct, and Qwen3-4B-Instruct-2507. Complete the 2080 Ti matrix
  and portable handoff first; 5080 execution and release writing follow. Other
  engines and hybrid models are deferred. The
  [selected profile](../evidence/stage5-optimization-20261105/selected-candidate.json)
  remains frozen. Execution authorized 2026-11-06; durable orchestration and pinned
  public inputs are implemented. Independently rebuilt runtimes passed all three
  native model correctness fixtures and both external 64-request activation checks.
  Recovery/relocation CPU checks pass. Resource screens and the complete acquisition
  remain in progress; failed development attempts are retained. No Stage 6
  confirmation or performance claim yet. See [execution record](../evidence/stage6-2080ti-20261106/README.md).
- **Stage 6 active resources:** `development-15` completed all 22 development
  cells / 54 configurations with no request failures. All three native model
  fixtures, both external-system activation checks, the registry probe, all
  three pressure checks and the matched roomy-capacity check passed. The
  66-case pilot owns the GPU and runs immutable source 19; timed serving source
  matches native/screen source 15. Confirmation has not started. See
  [complete screen receipts](../evidence/stage6-2080ti-20261106/development-complete-15.json).
  Candidate behavior stays frozen. The anchor repository cells use eight
  complete sessions at the unchanged 2.6016 GiB base CPU cache; secondary models
  and operational waves retain sixteen. The 4.5 GiB roomy control reduced ARC
  eviction-affected returns from two to zero; pressure is modest, not a speed
  claim. B10 remains a GPU-resident team control under versioned validity;
  original inconclusive raw treatment result is preserved.
  Actual interruption and whole-block replacement passed. A separate full-cohort
  operational development repeat passed 128 requests per arm, cancellation,
  drain/recovery and all advice probes with zero speculative staging. Two public
  runtime rebuilds match; 219 benchmark checks, 31 portable checks and a standalone
  wheel build pass. Model source review retains format failures, unsupported
  claims and capped answers; confirmation quality and the archive-wide capped
  response audit remain due. After pilots: fix repetitions, freeze, run controls,
  operations and comparisons, complete quality and final relocation proof.
  Check live ownership before starting other GPU work.
- **Stage 7 — compile release, pending:** verified installation and minimal
  integration, reviewed evidence-backed README/docs/post drafts, accessible
  reproduction artifacts and a focused vLLM proposal. A formal paper is not the
  target. Throughput remains primary; latency is secondary and memory a constraint.
- **Resources:** Stage 5 serving processes are reaped and GPU custody is released.
  Raw runs and source freezes remain in `output/stage5-optimization-20261105/`.
  The paused AWQ work is preserved. [Plan and completion criteria](WAIT_HINT_PRODUCT_PLAN.md#stage-5-optimize--completed).

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
