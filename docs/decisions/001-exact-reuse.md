# 001: Exact reuse and the first candidate

Active amendment, 10 November 2026: Namespace and scope: wait_hint_offload is the active namespace. Port CPU recency advice only; semantic scoring and prefetch remain historical. The matched control shares correctness guards and disables advice.

The earlier clauses below are historical where superseded by this amendment.

Accepted for design; no runtime implemented. Keep the `semantic_offload`
namespace. First reconstruct exact CPU offload and the unchanged CPU wait-hint
recency profile beside native ARC, with matched scheduling. Semantic scoring,
lossless transfer, speculative staging, advice and priority need separate switches
and evidence. Scores never authorize approximate KV or substitute answers.

The selected candidate is frozen in the old
[selected-candidate.json](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/stage5-optimization-20261105/selected-candidate.json).
It is not the helper's general semantic/prefetch default. SVD/lossy transport,
MTP, affinity scheduling and unqualified AWQ/ARC experiments stay historical.
This avoids treating optional feature presence as evidence of benefit.
