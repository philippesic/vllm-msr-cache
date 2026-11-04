# Maintained serving suite qualification — November 3, 2026

The four maintained tiers passed bounded real serving checks on the RTX 2080 Ti
WSL host using the repository launcher and pinned vLLM wheel. This qualifies the
benchmark infrastructure for the recorded configurations; it is not a new
cache-policy win or a production/answer-quality certification.

Use the [benchmarking guide](../../docs/BENCHMARKING.md). The compact
[qualification data](qualification.json) contains plans, per-case measurements,
activity, cleanup, source identities, artifact-manifest hashes, and offline
verification. Full logs, requests, token streams, model/runtime inventories and
failed attempts remain locally in ignored `output/suite-qualification/`.

## Measured turnaround and coverage

End-to-end wall times include hashing/preflight, server startup, workload,
telemetry settlement, and shutdown. These are observations on the existing cached
host, not time limits or cold-install estimates. All four default-size runs below
used the explicitly synthetic pressure fixture and Qwen2.5-0.5B-Instruct, FP16,
TRITON_ATTN, eager execution, 128 GPU blocks, and 256 MiB CPU offload space.

- **Smoke:** 45.55 seconds; one server, eight requests, deterministic repeated
  outputs matched, store and restore observed.
- **Screen:** 79.72 seconds; two servers, 24 requests, matched ARC/candidate,
  candidate scoring plus store/restore observed. No automatic rejection rule.
- **Confirm:** 274.83 seconds; six servers, 216 requests, three disjoint paired
  session groups, alternating order, candidate scoring plus store/restore.
- **Operational:** 169.29 seconds; two persistent servers, 288 normal requests
  over four waves per arm. All reached natural EOS. Both isolated cancellation
  probes produced one exact scheduler abort receipt and recovery generation passed.

Additional qualified configurations:

- **7B AWQ smoke:** 63.20 seconds, eight requests, eager mode and 0.85 GPU memory
  utilization. This caught and then verified the fix for missing wheel support
  files in the original source-only launcher staging.
- **CPU wait-hint screen:** 76.84 seconds, 16 requests. The candidate refreshed
  228 ready CPU keys across four hints. This is CPU recency advice, not speculative
  GPU admission or proof of benefit. The full tested configuration is in the plan.
- **Public-input graph serving, 0.5B:** 89.05 seconds, 32 normal requests, two
  overlapping waves per arm, natural EOS, abort receipt and recovery. Four sessions,
  two turns, concurrency two per wave, 512 GPU blocks, 8,192 context tokens.
- **Public-input graph serving, 7B AWQ:** 137.33 seconds, 16 normal requests,
  two sessions and two overlapping waves per arm with the same context/cache
  geometry, 0.85 GPU memory utilization and a 180-second request timeout.
  Natural EOS, abort receipt, and recovery passed.

The selected successful runs account for **608 normal workload completions across
18 server launches**, plus six deliberate disconnect probes and 24 separate
warmup/recovery completions. All selected server cleanups verified. The public
corpus preserves retained repository-issue user text; tools are not executed, and
these inputs are not a fresh held-out policy evaluation. Only store activity was
required in the larger-context public-input cases; restore/scoring pressure was
checked separately by the synthetic cases.

Each run records its executed-source hash and checks for changes during execution.
Optional hint-protocol and overlap-drain reporting fixes were exercised by the
later optional-path runs; the record keeps the actual per-run source identities.
Do not replace these identities with a blanket claim about a future revision.

## Failure handling and data validation

Real startup timeout, workload deadline, and SIGTERM-during-startup controls all
returned failure, retained diagnostics, and verified owned-process cleanup. Their
wall times were 8.01, 39.84, and 19.66 seconds respectively. SIGTERM also left the
owned staging directory empty. A held-lock control rejected concurrent suite use.

Offline verification passed for all eleven selected successful/failure-control
results, checking 879 sealed artifacts, request identity coverage, and recomputed
request/timing aggregates. An integrity check passing does not change a failed
measurement into a successful one.

Automated tests additionally cover malformed/truncated/empty token streams, HTTP
failure and request timeout without hidden retry, missing/reset counters,
inactive-mechanism handling, source staging precedence, termination cleanup,
detached/early-exit children, cross-process locking, invalid input, corpus
conversion, full hint identity/binding, wrong acknowledgement handling, CPU advice
versus GPU admission, content-check failure, exact abort matching, missing/mutated
artifacts, duplicate identities and incorrect aggregates. Final regression results
and cleanup accounting are recorded in [validation](validation.json).

Development failures remain retained: absent initial scoring telemetry was treated
as unavailable; the frontend abort counter proved unsuitable for disconnects;
quantized startup exposed missing wheel assets; wait hints exposed explicit plugin,
initial observed-prefix and full identity requirements. Fixes are maintained source
and versioned native instrumentation, not run-local engine method replacements.

## Limits

The small fixture establishes mechanism activity and harness behavior, not useful
semantic ranking or representative CPU eviction competition. Native scalar scoring
and CPU-recency receipts are not quality scores. Exact KV bytes are not inspected;
smoke compares repeated deterministic outputs. Optional content assertions are
simple substring checks, not a general evaluator. Sampled whole-device memory and
finite persistent waves do not prove allocator peaks or absence of long-term leaks.

The old research campaign's operating profile is separately frozen and is not
requalified by these runs. A clean-environment installation, other hardware/model
families, speculative-copy attribution and real-user deployment remain outside
this record. Upgrade/runtime changes require the documented qualification checks;
no finite test run can guarantee that future dependency versions need no work.
