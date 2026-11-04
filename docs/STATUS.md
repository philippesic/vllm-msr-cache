# Current work

Last reconciled: 2026-11-03. This is a handoff summary, not a live process monitor.
Update the affected entry at a meaningful handoff, change of direction, or finish.
Replace old state; move completed outcomes to [history](PROJECT_HISTORY.md).
Verify stale ownership before using shared resources. Keep detailed records in
linked evidence/run directories, not a second diary here.

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
