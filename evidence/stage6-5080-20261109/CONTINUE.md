# RTX 5080 rebuild and full Stage 6 pass

Authorized 2026-11-09: rebuild the same pinned LMCache 0.3.6 source for
`12.0+PTX`, declare the hardware-specific setup deviation, prove Continuum
serving, then rerun the complete primary campaign with unchanged design.
Preserve the closed November 8 campaign and every 2080 Ti artifact.

Repository: `/home/philip/vllm-semantic-cache-ext`; branch
`stage6-5080-20261109`; push only its established user-owned origin.

Current phase: runtime rebuild preparation. No new campaign acquisition yet.
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
