# RTX 5080 rebuild and full Stage 6 pass

Authorized 2026-11-09: rebuild the same pinned LMCache 0.3.6 source for
`12.0+PTX`, declare the hardware-specific setup deviation, prove Continuum
serving, then rerun the complete primary campaign with unchanged design.
Preserve the closed November 8 campaign and every 2080 Ti artifact.

Repository: `/home/philip/vllm-semantic-cache-ext`; branch
`stage6-5080-20261109`; push only its established user-owned origin.

Current phase: corrected fresh import running in persistent session 92431;
log `$BASE/import-02.log`. Offline transfer setup completed for all three models.
The first new import was stopped before any acquisition after a binding edit
used the wrong working directory; its partial tree/log are retained separately.
The current host binds the literal original pinned local interpreter.
LMCache release tag `v0.3.6` matches the installed wheel's embedded revision.
Build source is on the existing mounted ext4 volume. Use CUDA 12.8 tools to
match Continuum's pinned torch 2.8.0+cu128, not the host CUDA 13.1 compiler.
Prepare a separate Continuum environment; preserve the original one.

Paths:

```bash
REPO=/home/philip/vllm-semantic-cache-ext
BASE=$REPO/output/stage6-5080-20261109
OLD=$REPO/output/stage6-5080-20261108
VOLUME=$OLD/linux-volume-01
BUNDLE=$OLD/primary-execution-bundle-01
RUNTIMES=/home/philip/.cache/semantic-offload/stage6-5080-runtimes-02
PY=/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python
OUT=$VOLUME/primary-20261109
export PATH=/usr/lib/wsl/lib:$PATH
export HF_HOME=/mnt/c/vllm-semantic-cache-ext-stage6-5080/huggingface
```

Verify `findmnt --target "$VOLUME"` before writing; never reformat the image.
Monitor root, Windows and ext4 free space. The old FILESYSTEM_REPAIR.md records
the existing image/remount recipe. Model caches, local/upstream runtimes and
the separately sealed execution bundle are reusable. Explicitly bind `PY` as
host and cell `runtime_python` before importing. Materialize all sealed files.

Health schedule: `scheduled-task:d1de183b-4ef5-498a-aafe-ba015a2805ae`, every
15 minutes in this thread. Inspect owned process/birth state before resuming;
never duplicate a live job. Repair setup/harness/storage failures, retain failed
attempts, resume complete matched blocks, and advance finished phases. Follow
HANDOFF_5080.md: setup/import, bench-tests/check-light, smoke and preflight,
screen, qualify, activate-import, confirm, overhead, operational, integrity and
campaign/quality/source/capped-output audits, report with geometric mean ratios
and 95% CIs plus latency, commit/push. Do not freeze/replan or alter policy,
models, workloads, geometry, order or repeats. Gate failures require diagnosis.
Disable this schedule only at a pushed completed outcome or irreparable genuine
failure. Release all owned servers and GPU locks. Never track credentials.

Build success: `$VOLUME/lmcache-build-20261109-05/wheels/` contains the wheel;
new Continuum environment already has it installed. `lmcache-build.json` here
records hashes and source/toolchain details. The original environment is intact.
CUDA 12.8 uses GCC 14 and isolated Ubuntu 24.04 glibc/kernel headers because
the host GCC 15/glibc 2.42 are too new. Do not switch back to CUDA 13.1.
All 178 release Python files match the original installed release.

`transfer_setup --offline --id rtx5080` uses the new runtime manifest in
`$RUNTIMES`, reuses the pinned weights/local/upstream runtimes and source.
After it completes, explicitly replace the unimported new host runtime_python
with `$PY`, then import from `$BUNDLE/source` into the fresh `$OUT`. Do not
modify November 8 bindings. PYTHONDONTWRITEBYTECODE=1 is mandatory when running
from sealed source; direct compileall created only generated caches, now moved
outside the bundle and all 4,606 files verified unchanged. CPU checks already
passed; use PYTHONPYCACHEPREFIX and a separate pytest cache for future checks.

Run the separate short serving smoke from the mutable project with `$PY -m
benchmarks.research_checks.stage6_5080_smoke serve --campaign "$OUT" --out
"$VOLUME/continuum-smoke-20261109-01"`. Its changed one-session/16-token
parameters are diagnostic only, never campaign measurements. Then run preflight
from the sealed exported source with the pinned interpreter and original design.

Old history reverified after rebuild: all 41 attempts / 3,142 files, 4,606
execution-bundle files and the original LMCache binary are unchanged; see
[history integrity](history-integrity.json). Both environments have identical
package versions. Runtime preparation/build evidence is pushed.
