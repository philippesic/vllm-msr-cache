# Lower serving VRAM sanity check

The candidate with 1,280 GPU KV blocks used **224 MiB less serving VRAM** than
native ARC with 1,536 blocks. CPU cache remained 2,048 blocks (1.75 GiB).
Both used the same model, request traces, context, eight active sequences,
priority policy and frozen runtime/source. Generation continued to natural EOS;
these were fresh model invocations, with actual generated history on return turns.

The balanced confirmation comprised 36 invocations, 12 matched triples on three
fixed previously used public-input traces, and 2,592 natural-EOS responses.
All invocations passed HTTP, history and priority validation. Astra independently
recomputed the primary arithmetic from raw results and obtained matching values.

| Candidate 1,280 vs native 1,536 | Paired estimate | Nominal 95% interval |
| --- | ---: | ---: |
| Serving CUDA memory | 224 MiB less | Exact capacity difference |
| Token throughput | +6.13% | +2.81% to +9.56% |
| Workflow throughput | +6.33% | +3.26% to +9.49% |
| Tokens per serving GiB | +9.31% | +5.89% to +12.85% |
| Mean first-token latency | 347 ms faster | 182 to 513 ms faster |
| Overall p95 first-token latency | 740 ms faster | 94 to 1,385 ms faster |

Intervals use whole matched workflows, conditional on these three traces, without
multiplicity correction. They do not establish general workload performance.
Against native ARC at the same 1,280-block capacity, candidate throughput was
+4.19% (interval +1.40% to +7.06%). Simply shrinking native ARC's cache did not
show a reliable throughput increase.

Startup peak stayed unchanged. Candidate preemptions increased to 23, versus zero
for native 1,536 and one for native 1,280. All requests completed, but additional
recomputation can hurt other workloads. For return requests after positive waits,
p95 first-token latency was 463 ms worse, with an inconclusive interval from
520 ms faster to 1,446 ms worse. Host RAM saving, uniform tail improvement,
interactive latency guarantees and a new production qualification are not claimed.

Primary statistics, source identities, exact counts and tail breakdowns are in
[confirmation.json](confirmation.json). Full plans, frozen inputs, independent
recomputation, reports and raw evidence remain under
`output/gpu-cache-frontier-20261103/confirm/` and the prior campaign's
`output/prefetch-native-20261102/runs/live-nov3-vram-confirm-*` directories.

Additional startup-memory work exceeded the requested sanity-check scope and was
stopped at the user's clarification. A first loading prototype worsened startup
memory and was rejected. A revision completed five of nine planned cases before
the sixth was deliberately interrupted and cleaned up. Those partial results
are retained without a startup claim; no targeted operational followthrough ran.
The opt-in loading edits remain default off and unqualified. No wheel was modified,
and no system was deployed. All owned serving processes stopped and GPU custody
was released. The bounded retained inventory is in
`output/gpu-cache-frontier-20261103/retained-inventory.json`.
