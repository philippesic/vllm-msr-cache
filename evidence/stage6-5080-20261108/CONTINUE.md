# Stage 6 RTX 5080 — closed on capability failure

2026-11-08 Pacific closeout: **do not resume this job**. The pinned
Continuum/LMCache extension failed all five anchor screen warmups with
`CUDA error: no kernel image is available for execution on the device` in
`lmc_ops.multi_layer_kv_transfer`. Binary inspection confirms no `sm_120`
kernel and no PTX fallback. This meets the authorized capability-failure stop.
See [results](RESULTS.md), [failure receipts](capability-failure.json), and
[supervision takeover](supervision-takeover.json).

The recorded coordinator was interrupted through its normal owned cleanup.
All 41 attempts / 3,142 files verify; both original transfers, execution bundle
and audit packet verify. Qualification is false. Activation, confirmation,
overhead and operational phases were not run. Both health schedules are
disabled. Owned GPU clients/servers are gone and both locks can be acquired.
Keep the mounted ext4 image, retained attempts and useful caches. Never format
the image or reset this campaign. Any future runtime repair or study requires
separate scope; this job supplies no confirmed 5080 performance result.

Current supervision thread:
<t3-thread://v2/92296584-2a2c-4998-8f7d-4ecb7d7f5ff2/mcp:launch:88f1283af00bb5a99a0c27b25e8461a3417935a8903dc08e845bf2bdb84e2f19>

## Historical continuation instructions (superseded by failure closeout)

This is the same authorized job, not a new study request. Continue through
all primary phases and final audits; a scheduled check is not task completion.
The original user prompt authorizes a stop on a real capability, resource
geometry or workload-purpose failure, with a preserved/pushed receipt.

Work only in `/home/philip/vllm-semantic-cache-ext`. Branch:
`stage6-5080-20261108`, user-owned origin `philippesic/vllm-msr-cache`.
Read the original user prompt and the 2080 Ti RESULTS/HANDOFF before claims.
Do not work on Binary Eagle/shaderlod, modify copied seals, invent an
addendum, reset old studies, freeze/replan or substitute models/workloads.

Health schedule: `scheduled-task:fbd7ddce-7c9a-45b1-b122-09c729422576`,
every 15 minutes, bound to this thread. Keep enabled while active. Its first
due time is `2026-11-09T05:05:42.986Z`. During confirm the existing bounded
`benchmarks.research_checks.stage6_healthcheck` is suitable; before activation
inspect process/phase receipts directly because that checker requires a lock.
Do not keep an agent turn polling unchanged setup/serving logs.

Paths:

```bash
REPO=/home/philip/vllm-semantic-cache-ext
BASE=$REPO/output/stage6-5080-20261108
BUNDLE=$BASE/primary-execution-bundle-01
RUNTIMES=/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01
PY=$RUNTIMES/local-env/bin/python
CACHE=/home/philip/.cache/semantic-offload/stage6-5080-assets
HOST=$BASE/host-replica.private.json
OUT=$BASE/linux-volume-01/primary-01
export PATH=/usr/lib/wsl/lib:$PATH
# Intended new Windows model-cache location; create only this task's directory.
export HF_HOME=/mnt/c/vllm-semantic-cache-ext-stage6-5080/huggingface
```

`runtime_setup` completed successfully in session 72310, with final
`published-identities.json` and all three GPU-free imports. Retain this cache.
Pinned runtime logs are under `$RUNTIMES`; summary is `$BASE/runtime-setup.log`.

Model reacquisition completed successfully in session 74455 from the original
primary export. All three pinned model contents verified; the initial private
host file is `$BASE/host.private.json`. Log: `$BASE/transfer-setup-03.log`.

Offline binding/import completed in session 71664. All host/cell interpreters
explicitly bind `$PY`; all four transfer/execution seals verify. See
`import.json` here and the raw `transfer-bind-replica-01.log` / `import-01.log`.

Native preflight completed in session 39853: all four cases passed, including
restore/fresh-reference correctness for every model and the Qwen3-8B registry
probe. All four attempt seals and 223 files verify, with clean GPU release.
See `preflight.json` here and `$BASE/preflight-summary.json` for full proofs.
Do not repeat preflight or call the confirmation-specific health checker
before protocol activation.

Storage placement completed successfully in session 53061, log
`$BASE/storage-placement.log`, with the maintained
`benchmarks.research_checks.stage6_5080_storage` helper. All 4,934 file contents
are preserved, the same four attempts / 223 files verify after placement, and
real Windows writer-lock contention passed. `storage.json` here is the compact
proof; `$BASE/storage-placement.private.json` retains full file hashes.
The Windows path below is the preserved first-screen tree. Live `OUT` is now
the Linux volume defined above; read the repair note before any recovery.

`/mnt/c/Users/philip/AppData/Local/vllm-semantic-cache-ext-stage6-5080/campaigns/primary-01`

The old duplicate was removed only after old/copy/alias content and seals
verified. Private host files remain mode 600 in
`$BASE/private-bindings/primary-01`, linked into the Windows campaign.

Screen session 50824 stopped on a retained filesystem
sealing error after its first ARC worker passed. Read [repair](FILESYSTEM_REPAIR.md)
before any serving/resume. Native preflight remains passed. Recovery completed
in session 56322 through `stage6_5080_linux_storage`, log
`$BASE/linux-storage-recovery.log`. All four preflight attempts and the one
interrupted screen attempt verify (254 files). The Linux volume is mounted and
the exact pinned launcher left no staging in its GPU-free lifecycle probe.
The source/policy, models, geometry, frozen schedule/repeats and campaign identity
are unchanged. See `filesystem-repair.json` here for compact proof.

The exported verifier confirmed recovery and both original/execution seals.
**Active phase:** whole screen resumed in session 98188 using
`$PY -m benchmarks.serving_suite.campaign resume --out "$OUT" --phase screen`
from `$BUNDLE/source`, log `$BASE/screen-02.log`. Current owner/birth/mount proof
is `$BASE/screen-owner.private.json`; the first launch owner is preserved as
`screen-01-owner.private.json`. Do not duplicate or repeat preflight.
The affected B01 block is replaced in full; original unmatched ARC gets no
measurement credit. Require `findmnt` to prove the ext4 image is mounted before
resume; see the repair note for the existing image/remount command. Never reformat
it or create a new campaign in an unmounted directory.

Screen is 54 configurations in
22 required cells. A completed native GPU-resident control can have an original
inconclusive restore status; use the existing attested `measurement_validity`
to distinguish a complete control from an actual execution/correctness failure.
Do not turn such a control into CPU-pressure evidence. Stop on real capability,
resource or workload-purpose failure. `campaign run` can return normally while
retaining failed cases; inspect the ledger/results, not only its exit code.
After all screens finish, run `qualify` and inspect
`all_required_correctness_passed`; do not activate an import with a failed gate.

The previous primary retained 14,182,204,461 bytes, exceeding the WSL root's
remaining space. The Linux image now provides about 19.5 GiB usable output
capacity, reserving 20 GiB on the Windows backing disk. At repair, WSL root had
about 7.3 GiB free and Windows about 8.6 GiB free. Monitor **all three** volumes
and verify the ext4 mount is present at every check/resume. Preserve the image
and its mounted results; its exact backing/mount recipe is in the repair note.
Use the current canonical Linux `OUT` for process matching. Private bindings
stay in WSL. No campaign identity, source, matrix, order, repeats or capacities
changed; the whole interrupted B01 block is the only declared replacement.
Earlier setup attempts/logs are retained (one verifier rejection of generated
caches; one stopped cwd correction). All original seals verify again.

The separate `$BUNDLE` is a sealed copy derived from the original primary:
its coordinator is from checkpoint 1319f09, adding health-check source
attestation, with a new source identity. No other transfer/protocol fields
changed. Read `source-revision.json` here and the full raw receipt in `$BASE`.
It is not an addendum. `bench-tests` and `check-light` passed from this new
execution source too: 240 tests, 31 tests and compileall (`*-02.log`).

Recheck disk before continuing. Initial WSL free space was 23,943,786,496
bytes; the Windows filesystem had 60,804,325,376 bytes free. Sealed model
weights total 29,540,039,872 bytes. Existing uv cache permits hardlink reuse;
runtime reconstruction leaves about 7.3 GiB in WSL; the later Linux image
reservation leaves about 8.6 GiB on Windows, plus about 19.5 GiB in the new
output filesystem. Do not delete unrelated data or fill a volume.

The commands below retain the entire authorized sequence. Offline rebinding
and import/preflight completed. The first screen stopped and storage recovery
completed; the repaired whole screen is active as described above.
Use the pinned interpreter throughout:

```bash
cd "$BUNDLE/source"
export PYTHONDONTWRITEBYTECODE=1
"$PY" -m benchmarks.serving_suite.transfer_setup --bundle "$BUNDLE" --runtimes "$RUNTIMES" --cache "$CACHE" --out "$HOST" --id rtx5080 --offline
"$PY" -m benchmarks.serving_suite.campaign import --bundle "$BUNDLE" --host "$HOST" --out "$OUT"
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase preflight
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase screen
"$PY" -m benchmarks.serving_suite.campaign qualify --out "$OUT"
"$PY" -m benchmarks.serving_suite.campaign activate-import --out "$OUT"
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase confirm
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase overhead
"$PY" -m benchmarks.serving_suite.campaign run --out "$OUT" --phase operational
```

Run serving phases in persistent execution sessions with receipts and the
coordinator's built-in `gpu_reserved` protection. Verify each phase's receipts,
not only CLI exit (run_phase can retain failed cases yet return normally).
Stop on capability/resource/purpose failure. Do not run later dependent phases
after a failed gate. The primary exported coordinator explicitly launches with
cell profile `runtime_python`, falling back to host `runtime_python`; it does
not inherit the test interpreter. Imported cells must bind `$PY` explicitly.
The handoff's repaired-addendum warning must not cause a substitution of the
addendum source for primary. Preserve exact primary serving inputs. Document
the new imported acquisition identity and any necessary coordinator/source
adaptation separately, never patching the sealed transfer in place.

Avoid writing generated cache/package files into any sealed bundle. For
additional validation use external `PYTHONPYCACHEPREFIX` and pytest
`-o cache_dir=...`; `compileall` writes bytecode despite
`PYTHONDONTWRITEBYTECODE`. Verify seals after any tool runs there.

After acquisition, verify campaign integrity, produce source and capped-output
audits as required by the original prompt, retain adverse/unresolved outcomes,
write 5080 RESULTS and compact evidence, update only the affected STATUS entry,
commit and push the branch. Preserve private host files as private. Release GPU
reservations and owned serving processes; preserve useful caches/raw evidence.

Only when actually complete or failed for good: disable this saved schedule,
POST JSON to the following webhook with this thread and outcome (plus headline
results or failure reason), retain the response receipt, and stop this job:

Read the endpoint from the original user prompt or the mode-600 ignored file
`output/stage6-5080-20261108/completion-endpoint.private.json`. Keep the URL
out of tracked files. It appeared in earlier runbook commits on this public
branch; removing it from the current file does not remove that history.

Thread link:
<t3-thread://v2/92296584-2a2c-4998-8f7d-4ecb7d7f5ff2/mcp:launch:4ba24d5cb6f0895afb9f1420c78fb52741f61c910c692734f5dd8fa4aba7b875>
