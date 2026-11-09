# Stage 6 RTX 5080 replication

Authorized 2026-11-08 at about 20:47 Pacific, on DESKTOP-E78LJSI.
Model reacquisition is active; pinned runtime reconstruction succeeded.
There are no 5080 serving measurements or hardware
qualification results yet. No performance or answer-quality claim follows.

Both sealed transfers and the completion audit packet were copied over the
configured Tailscale SSH alias and verified with the exported verifier before
use: 4,606 primary files, 8,362 addendum files and 17 audit-packet files.
The repository checkpoint was copied as a full Git bundle and verified.
[Transfer receipt](transfer.json) retains identities and local paths.
The 2080 Ti seals, comparisons and failed-gate artifacts are unchanged.

The primary exported source passed `make bench-tests` (240 tests) and
`make check-light` (31 tests and compilation) using the reconstructed local
pinned interpreter. All three pinned runtimes reconstructed successfully and
passed GPU-free imports. Local/upstream Torch report `sm_120` support;
this does not establish native engine/kernel correctness on the 5080.
The exported primary coordinator already explicitly selects cell/host
`runtime_python` for workers; the historical addendum interpreter defect
does not occur in this primary worker-launch path. A separately sealed
[5080 execution source revision](source-revision.json) takes the current
checkpoint's coordinator into a copy of the primary export and additionally
attests its already-exported bounded health checker. All non-source transfer
fields and frozen order/repeats are unchanged. Both CPU check targets passed
again from this execution source (240 and 31 tests plus compilation), and
both its seal and the original primary seal verified afterward.

The first post-setup verification caught generated Python/test/package caches
inside the original transfer. These were moved out, preserving their inventory,
and all 4,606 original files verified again. An initial setup retry used the
checkout as its working directory; it was stopped before host binding/import,
and setup restarted from the primary export, retaining downloaded cache files.
These were setup-only attempts, with no serving or measurements.

Initial telemetry identifies an RTX 5080 with 16 GiB VRAM and no compute
clients. Both GPU reservation files could be acquired and released.
This is preparation telemetry, not the campaign's native hardware preflight.

Storage is tight: initially about 22.3 GiB available in WSL and 56.6 GiB on
Windows. Pinned model weights require 27.5 GiB before ancillary files.
Native runtimes now leave about 7.4 GiB free in WSL; reacquired models use
a dedicated Windows cache. Recheck free space before downloads/serving;
stop with a resource receipt if this layout cannot support the frozen job.
No unrelated data has been removed.

The 15-minute health schedule is enabled, first due 2026-11-08 21:05 Pacific.
The whole authorized primary sequence remains required. No addendum is
authorized, and no frozen order, repetitions, models, workloads or policy
may be changed. See [continuation instructions](CONTINUE.md).

Raw setup logs, verification manifests, runtime plan and private process/
resource receipts are retained in `output/stage6-5080-20261108/`.
