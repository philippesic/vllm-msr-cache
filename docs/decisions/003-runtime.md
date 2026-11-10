# 003: Separate runtime provenance

Active amendment, 10 November 2026: One locked native runtime is active for this port. Continuum and the distinct historical development/Stage 6 closures remain provenance, not install requirements.

The earlier clauses below are historical where superseded by this amendment.

Accepted for later runtime work. Replace the source/wheel overlay with an
attested installed patched runtime, or explicitly attest a temporary parity
overlay. Imports against an arbitrary wheel do not qualify it. Keep separate
native/upstream torch-2.13/CUDA-13 and Continuum torch-2.8/CUDA-12.8 closures.
Pin actual sources, patches, compilers, ABIs and loaded binary hashes later.

Historical native base and patch relationships:
[vendor/VLLM.md](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/vendor/VLLM.md).
Executed local dependency basis:
[Stage 6 local.txt](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/benchmarks/serving_suite/runtime_requirements/local.txt).
Distinct root record:
[requirements-runtime.txt](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/requirements-runtime.txt).
External revisions/binary bases:
[published.py](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/benchmarks/serving_suite/published.py).
The upstream compatibility change is
[mapped-host-pointer.patch](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/upstream-pin-fix-20261104/mapped-host-pointer.patch).

Do not combine the alternative native-prefix patch with CPU advice. Qualification
and graph metrics patches are distinct/default-off where not required. No patch
or runtime dependency is imported in this foundation.
