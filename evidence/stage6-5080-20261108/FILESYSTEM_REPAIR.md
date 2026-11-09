# Screen storage repair

The first B01 ARC worker completed all six requests with EOS and passed its
evaluation/cleanup checks. The outer coordinator nevertheless stopped while
sealing: 347 native runtime-staging symlinks remained on DrvFS. The inner
measurement verifier excludes this ephemeral staging; the campaign's outer
seal correctly rejects symlinks. The ledger was left running because the
error handler's sealing attempt rejected the same links.

The Windows copy/lock tests did not exercise this lifecycle. Native preflight
had run successfully on WSL ext4. This is a retained runner/storage failure,
not an adverse policy result or evidence of a GPU capability failure. No
paired screen estimate is available from this unmatched ARC process.
Its six answers, timing, source/environment receipts and error remain retained.

The repair uses a project-owned 20 GiB ext4 file under the restricted Windows
campaign directory, mounted at
`output/stage6-5080-20261108/linux-volume-01`. This gives the native launcher
Linux filesystem semantics while keeping raw results outside the full WSL
root volume. A GPU-free invocation of the exact pinned launcher on this volume
completed with an empty staging parent. No system mount configuration, model,
policy, GPU/CPU geometry, source closure or schedule was changed.

`benchmarks/research_checks/stage6_5080_linux_storage.py` copies and verifies
the complete old tree, including symlink identities, before switching the
original project alias. The complete old Windows campaign remains preserved.
Only the new copy's unsealed ephemeral staging is quarantined outside its
attempt. Its sealed measurement files/manifest are verified unchanged, then
the existing ledger recovery seals the interrupted attempt without measurement
credit. Ordinary screen resume replaces the entire B01 matched block. This
does not authorize tuning, a new import, a new study, or a favorable-result retry.

Raw diagnosis, symlink inventory and lifecycle proof:

- `output/stage6-5080-20261108/filesystem-repair-01.private.json`
- `output/stage6-5080-20261108/linux-storage.private.json`
- `output/stage6-5080-20261108/linux-stage-probe.log`
- `output/stage6-5080-20261108/linux-storage-recovery.log`
- `output/stage6-5080-20261108/linux-storage-recovery.private.json` once complete

The image was created exclusively as a new regular file, sized to 20 GiB, then
formatted with `mkfs.ext4 -F -m 0 -L stage6-5080 -E
nodiscard,lazy_itable_init=1,lazy_journal_init=1`. The existing image now contains
retained work and must never be formatted again. Its Windows parent ACL allows
only Philip, SYSTEM and Administrators; the mounted filesystem root is owned
by Philip with mode 700. Host-private bindings remain in WSL with mode 600.

If WSL restarts, first ensure no coordinator/worker is running. Inspect
`linux-storage.private.json` and remount the **existing** image if needed:

```bash
IMAGE=/mnt/c/Users/philip/AppData/Local/vllm-semantic-cache-ext-stage6-5080/campaigns/stage6-5080-outputs-01.ext4
MOUNT=/home/philip/vllm-semantic-cache-ext/output/stage6-5080-20261108/linux-volume-01
/mnt/c/Windows/System32/wsl.exe -d Ubuntu -u root -- mount -t ext4 -o loop,noatime "$IMAGE" "$MOUNT"
findmnt --target "$MOUNT"
```

Verify the filesystem is actually mounted before reading/resuming the campaign;
do not create a replacement campaign in the unmounted directory. Keep the image,
mount and useful raw results while acquisition/audits are active. Monitor free
space on the ext4 volume, WSL root and Windows backing volume. The image reserves
about 20 GiB on Windows, leaving about 8.6 GiB there at preparation; the ext4
volume initially has about 19.5 GiB usable. Host/filesystem/runtime differences
still prevent attribution of cross-host changes solely to GPU hardware.
