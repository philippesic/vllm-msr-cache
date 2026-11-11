# Lower serving memory retained gains but increased preemptions

2026-11-10 · historical · mixed

Reducing the candidate to 1,280 GPU KV blocks saved 224 MiB versus native ARC at 1,536 blocks while improving measured throughput. Startup memory did not improve, and the candidate incurred 23 preemptions.

![Candidate token-throughput gain (%)](chart.svg)

Whole matched-workflow nominal 95% intervals on three fixed traces; higher is better.

## Scope

Thirty-six invocations: twelve matched triples across three previously used public traces, 2,592 natural-EOS responses; equal 1.75 GiB CPU cache.

## Limits

Positive-wait return-tail improvement was unresolved. No host-RAM or universal tail claim follows. Separate startup-loading prototypes were rejected or interrupted; their partial work remains paused in the old repository.

## Decision

Retain serving-memory savings as a scoped result. Do not import the paused loading prototype or claim reduced startup memory.

## Sources

- [evidence/vram-frontier-2080ti-20261103/README.md](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/vram-frontier-2080ti-20261103/README.md)
- [evidence/vram-frontier-2080ti-20261103/confirmation.json](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/vram-frontier-2080ti-20261103/confirmation.json)

Exact source hashes and figure inputs: [result.json](result.json).
