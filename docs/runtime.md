# Runtime assumptions, not an installed serving environment

The CPU foundation pins CPython 3.12.3, uv 0.13.6, Ruff 0.14.4, mypy 1.18.2,
Import Linter 2.7 and pytest 8.4.2. `uv.lock` hashes the complete development
closure. The CPU image uses the same Python and hash-pinned uv bootstrap.
No torch, vLLM, native compiler or GPU runtime is installed by this project.

## Historical dependency closures to preserve separately

| Closure | vLLM/source | torch / CUDA basis | transformers / Triton |
| --- | --- | --- | --- |
| Native ARC / candidate, Stage 6 | 0.26.1rc1.dev75+gdc1be7903 with local patches | 2.13.0 / toolkit distribution 13.0.3.0 | **5.5.3** / 3.7.1 |
| Old root development requirements | Same native base | 2.13.0 / toolkit distribution 13.0.3.0 | **5.22.1** / 3.7.1 |
| Continuum | vLLM 0.10.2 binary base; LMCache 0.3.6 | 2.8.0+cu128 / 12.8 | 4.55.2 / 3.4.0 |
| Upstream retention PR #59714 | 0.30.1rc1.dev530+gccda9098c plus mapped-host-pointer patch | 2.13.0 / CUDA 13 runtime | 5.22.1 / 3.7.1 |

These are historical records, not freshly resolved compatible pins. Native base,
patch order and exact upstream/Continuum revisions are linked in
[the runtime provenance decision](decisions/003-runtime.md). Do not collapse the
root and Stage 6 transformers versions into one claimed environment.
The Stage 6 local requirements name nvcc package 13.4.92 and runtime 13.0.96;
upstream names nvcc 13.5.37. A toolkit metapackage is not a compiler identity.
Verify actual nvcc, host compiler, ABI, glibc/sysroot and loaded binary/source
hashes in future manifests. Old root metadata did not prove a host nvcc on PATH.

## Two GPUs, separate qualification

RTX 2080 Ti (DESKTOP-T4JKGJB) targets **sm_75**; RTX 5080 (DESKTOP-E78LJSI)
targets **sm_120**. Historical native serving used TP1, FP16, TRITON_ATTN and
dense full-attention GQA. This does not qualify BF16 on Turing, hybrid/sliding
attention, MHA/MQA, multi-GPU, or an arbitrary upstream runtime.

Keep CUDA-13/torch-2.13 and CUDA-12.8/torch-2.8 builds in distinct environments.
Future recipes must target both architectures (or attest two same-source wheels),
inspect every relevant shared object's native kernels/PTX and preserve licenses.
A torch architecture check cannot qualify LMCache's separate transfer binary.
See the [5080 failure and rebuild evidence](../FINDINGS.md#5080-binary-capability).
The successful historical rebuild used CUDA nvcc 12.8.93 and GCC 14.3.0, with
isolated compatible headers; it is not a license to mutate a host toolchain.

## Gates before serving

The current native group explicitly skips without native source. Adding owned
C++/CUDA/CMake material fails closed until real recipes and diagnostic specimens
are qualified. [The dispatch manifest](../tools/native.json) records the intended
clang-format, clang-tidy, CMake and binary-inspection commands for sm_75/sm_120.
Native tool versions will be pinned with actual ABI-specific build recipes;
listing untested versions here would imply qualification that has not occurred.
CMake compilation is CPU work; GPU byte equality and sanitizers are later gates.

Future deterministic builds must reject wrong wheel/interpreter/API/layout,
missing symbols or architecture coverage, and verify CPU imports without CUDA
initialization. Then each host separately needs actual FP16 CPU/GPU round trips,
completion/failure fences, restore-versus-fresh generation, and cancellation,
drain/recovery. A byte round trip is not full lifecycle or quality qualification.
No old runtime source, patches, wheel assets or execution trees were copied here.

## Explicit runtime verification entry

```sh
uv run --locked python -m tools.runtime_verify
```

This stub exits **2** and prints `NOT IMPLEMENTED: runtime verification is not
implemented until source port.` It never launches GPU work or returns success.
The ordinary validator verifies the stub's exit/message contract and counts
runtime as **skipped**, outside passed groups. This is preparation, not evidence
of byte equality, lifecycle correctness, working kernels, performance or quality.
A source port must replace the stub and its mutation proofs in a separately
reviewed change before any runtime pass is possible.
