# Final report evidence review

Bounded read-only comparison of REPORT.md with long/combined-short results, quality assessment, gather/hint screen reports, native transport qualification source/results, and final source verification. No benchmarks, GPU work, project imports, or tests were run.

**Required clarification:** identify the standalone native transport test as **synthetic packed pages matching the 28-layer layout, without a model forward or native adoption inside that test**. `qualify_native_bytes.py` initializes random int8 pages and drives the installed handlers directly. It does establish byte equality on those pages, including reordered/duplicate sources and post-fence reuse. It does not establish live serving KV equality. The linked result records this limitation; the report should make it equally explicit.

Otherwise, no material report correction found:

- Tables, throughput/CPU deltas, memory savings and pair ratios match the structured results. Long has six pairs above 99% of large ARC; short has two equal-budget failures despite a passing pooled ratio. Return-latency regressions remain explicit.
- Confirmation counts are 21 long + 21 short runs. Candidate adoption totals are 36 long jobs/13,860 blocks and 36 short jobs/3,492 blocks: 72/17,352 overall.
- Every candidate run applies numerical prefix scores. Long applies 32 scored candidates across 36 adoptions; the prose does not assert every adopted prefix was scored.
- Confirmations have zero constrained choices. Separate diagnostics provide five comparable constrained choices out of nine actual multi-candidate decisions, with 2/5 ordering agreement. The report distinguishes arithmetic equivalence from policy quality and excludes diagnostic performance.
- The ARC outlier remains disclosed and unresolved. No production or general statistical-equivalence qualification is claimed. The gather optimization remains private according to final source verification.
