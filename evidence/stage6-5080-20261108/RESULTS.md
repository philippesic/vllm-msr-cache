# Stage 6 RTX 5080 replication — capability failure

The authorized replication stopped during screening on 2026-11-08 Pacific.
The pinned Continuum/LMCache runtime cannot execute its GPU KV-transfer
operation on this RTX 5080. All five anchor screens (B06/B08/B09/B11/B12)
failed warmup with `CUDA error: no kernel image is available for execution on
the device`, at `lmc_ops.multi_layer_kv_transfer`. Each completed zero measured
requests and verified cleanup. Offline `cuobjdump` inspection of the installed
LMCache extension finds kernels for sm_70/75/80/86/89/90/100, no sm_120 kernel,
and no PTX fallback. [Exact attempts, seals, logs and binary identity](capability-failure.json)
preserve the failure. Torch's earlier sm_120 import check did not qualify this
separate compiled extension.

This is the original authorized campaign, closed under its genuine capability
failure rule. No runtime/model/workload/policy substitution, training, new study
or addendum was performed. The 2080 Ti seals and historical results are unchanged.
Qualification reports `all_required_correctness_passed: false`; import activation,
confirmation, overhead and operational phases were not run. There are **no
confirmed 5080 performance results**, and no speed or quality equivalence claim.

The supervision takeover found the ext4 volume mounted and the recorded
coordinator alive, advancing past Qwen3-8B B12 into Ministral B02. Inspecting
terminal receipts exposed the earlier Continuum failures that the coordinator
had continued past. It was stopped with SIGINT after verifying its birth identity;
the existing worker/server cleanup ran and the current attempt was sealed as
interrupted. The [takeover receipt](supervision-takeover.json) records both schedules
and the new thread. The replacement 15-minute schedule was created as requested;
both old and replacement schedules are now disabled because acquisition has ended.

Retained evidence includes four passed native preflight attempts and 37 screen
attempts: 29 verified, six raw failed, and two interrupted. Five failed attempts
are the genuine Continuum failures; the sixth is an attested restore-inactive
GPU-resident control, preserved with its original status. The two interruptions
are the historical DrvFS sealing failure and the capability-stop interruption.
Seven of 22 screen blocks are complete. Integrity verification passes for all
41 attempts / 3,142 files, the execution bundle (4,606 files), original primary
(4,606), original addendum (8,362), and completion packet (17). All 41 measurement
source receipts use the execution identity; its 2,192 acquisition-source files
verify. Integrity and source identity do not establish successful acquisition.

The screen objective packet covers 298 finals from complete blocks, with 186
mechanical failures and 271 selected source-review items. It remains pending
answer/source judgment. The all-attempt inventory retains 1,322 screen request
rows, including 603 finals and 39 capped responses (five finals and 34
intermediates). Incomplete and interrupted attempts are included in this inventory.
Content review of those caps was not completed after the capability stop; no
answer-quality certification follows. Full texts and pointers remain in the
private raw closeout and generated review packet, linked from the compact receipt.

Owned coordinator/workers/servers have exited, no GPU compute clients remain,
both reservations were acquired and released, and owned server ports are closed.
At closeout, the ext4 output volume had about 18.7 GiB free, WSL root 7.2 GiB,
and Windows backing 8.6 GiB. Resource exhaustion did not cause this failure.
The existing image, mount, all raw attempts and useful caches are preserved.

Preparation remains valid within its scope: [transfer](transfer.json),
[separately sealed execution source](source-revision.json), [import](import.json),
and [four native hardware preflights](preflight.json). Exported CPU checks passed
240 benchmark and 31 lightweight tests plus compilation. The initial DrvFS screen
sealing error and verified ext4 recovery remain in [the storage repair](FILESYSTEM_REPAIR.md)
and [repair receipt](filesystem-repair.json); they did not repair the external
runtime's GPU capability. Repeated fixed inputs, host/runtime/filesystem
differences and partial screening preclude GPU-only attribution.

Raw logs and full private receipts remain in `output/stage6-5080-20261108/`.
The closed [continuation record](CONTINUE.md) preserves the historical runbook.
