# Stage 6 RTX 5080 replication

Authorized 2026-11-08 at about 20:47 Pacific, on DESKTOP-E78LJSI.
All three pinned models were reacquired and content-verified; pinned runtime
reconstruction and campaign import succeeded. [Native hardware preflight](preflight.json)
passed all four cases: actual pinned CPU geometry and restore/fresh-reference
correctness for all three models, plus the registry probe. All four attempt
seals / 223 files verify, cleanup passed, and GPUs were released. Screens and
external capability/purpose qualification remain required. There are no 5080
confirmed performance measurements yet. The first screen stopped on a retained
[filesystem sealing error](FILESYSTEM_REPAIR.md); [storage repair](filesystem-repair.json)
is verified and the whole screen has resumed. The unmatched ARC worker passed
all six requests, but its incomplete
block has no measurement credit. No speed or answer-quality equivalence claim
follows.

The [replacement B01 block](repair-runtime-validation.json) subsequently
completed both ARC and candidate with verified strict outer seals, successful
cleanup and no residual native staging. This validates the storage repair with
actual serving; screen has advanced beyond B01. The original failed attempt
and its answers remain preserved and separately identified.

[Import proof](import.json) confirms a new linked hardware acquisition identity,
all host/cell serving interpreters explicitly bound, and the unchanged 524
confirmation configurations / 210 blocks, 38 overhead and six operational
configurations. Confirmation remains blocked until preflight/screens qualify
and the imported protocol is activated.

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
Native runtimes leave about 7.3 GiB free in WSL; reacquired models use
a dedicated Windows cache. The later 20 GiB Linux image reservation leaves
about 8.6 GiB free on Windows and about 19.5 GiB usable within the output
filesystem. Recheck all three volumes; stop with a resource receipt if this
layout cannot support the frozen job.
The previous primary campaign retained about 14.2 GB, so the complete run's
outputs need Windows storage. A dedicated campaign directory with a restricted
Windows ACL is prepared; [verified placement](storage.json) preserved all 4,934
file contents and the campaign identity, and proved Windows writer exclusion.
Private host files remain mode 600 in WSL, linked into the campaign. That initial
storage proof missed native staging cleanup; direct DrvFS output left symlinks
and failed the first outer attempt seal. The complete original Windows tree is
preserved; the campaign is now on a project-owned ext4 image backed by
Windows storage. Its native staging lifecycle probe passed, and exported
verification covers all four preflight attempts plus the interrupted attempt
(254 files). Resumed valid blocks
will use that Linux filesystem consistently across arms. These remain
host-conditional measurements; differences from 2080 Ti cannot isolate GPU
hardware from filesystem/runtime/environment effects. No unrelated data has
been removed.

The 15-minute health schedule is enabled, first due 2026-11-08 21:05 Pacific.
The whole authorized primary sequence remains required. No addendum is
authorized, and no frozen order, repetitions, models, workloads or policy
may be changed. See [continuation instructions](CONTINUE.md).

Raw setup logs, verification manifests, runtime plan and private process/
resource receipts are retained in `output/stage6-5080-20261108/`.
