# Stage 5 optimization

Stage 5 is complete under the user's narrowed final scope. **Keep the existing
wait-hint behavior; no new serving optimization qualified for promotion.** The
campaign completed **144 valid GPU cases**, with two invalid-configuration
attempts retained separately. Experimental switches remain off by default.

The search covers queue choice and ordering, prefix selection, recency strength,
refresh timing, pressure gates, advice budgets, publication plumbing, eviction,
audit cost, scheduling and storage. All optional experiments remain disabled by
default. The user narrowed the final breadth and asked for a deeper explanation
of the promising eviction CPU result; that lead was followed into actual serving
and exact-state replay.

## Consequential results

The fixed pressure-gate repeat panel measured existing wait hints at **6.15% more
workflow throughput than plain ARC**, with a conditional 95% interval of
**4.99–7.32%**. The gate itself was effectively unchanged versus existing wait
(−0.01%; interval −2.15 to +2.17%). A separate fixed timing panel measured tool-only
publication at +1.21% (−3.87 to +6.55%) and a timing/grace/eviction combination at
+1.54% (−3.64 to +6.99%). Neither new candidate met the declared 2% point threshold
and an interval excluding no improvement. These are three-repetition local
development comparisons, not fresh or selection-adjusted confirmation.

The eviction lead is real but small in absolute serving time. Larger actual-state
calls were about 5–7× faster in the primary/tight replay and up to about 23×
faster in the larger-batch document samples. Native eviction nevertheless used
less than 0.05% of workflow time in the primary/tight diagnostics. A deliberately
larger prefill batch on real documents produced up to 250-victim calls and cut
body time from 159 ms to 18 ms, while native eviction still occupied only 0.09%
of a roughly three-minute workflow. The synthetic CPU speedup did not establish
a material end-to-end speedup. See the [eviction diagnosis](EVICTION.md).

Other results constrain tempting shortcuts: truncating prefixes or quotas loses
reuse; reducing refresh frequency can increase recomputation; a GPU-prefix bridge
had no extra-key activation even after a targeted follow-up; and live publication
overlap/coalescing did not establish a workflow improvement. The activity-free
bridge result is unresolved, not a general rejection. The broader audit and
uneven-wait repetition panels were deferred under the user's reduced scope.

[Category decisions and remaining ideas](EXHAUSTION.md) retain measured losses,
inactive mechanisms, uncertain results and the stopping rationale.
[Protocol and declared selection rules](PROTOCOL.md) preserve the scope changes.
[Repeated comparisons](selection-checkpoint.json) and [chart](overview.html)
separate repeat evidence from the selecting screens.

## What was implemented

- Default-off linear ARC eviction with exact native decisions and atomic failure.
- Default-off queue, prefix, bounded-boost, cooldown, occupancy and budget variants.
- Default-off CPU nomination grace; no extension of GPU staging leases or widening
  of the public wait declaration enum.
- Optional client publication schedules, duplicate suppression and owned/joined
  RPC overlap, including cancellation cleanup.
- A bounded serving profiler with exact-state replay, finite experiment recipes,
  cohort completion analysis and preflight checks for unsupported settings.

The advice variants only reorder eligible residents. They do not change queue
membership, ghost history, block ownership or ARC's adaptive target. Repeated
identical-token cases check offered work; they do not prove KV byte integrity.
The separate unforced functional probe passed all 48 assertions across nine
waves: actual CPU-restored continuations agreed with full fresh recomputation;
malformed advice and operational recovery checks passed. Actual pinned CPU
backing, shared by loads and stores, was independently checked from allocation
receipts. This is observed generation agreement, not a KV byte-equality claim.

The [selected candidate](selected-candidate.json) freezes the unchanged profile,
client settings and serving-source identity. [All case results](results.json),
[validation](validation.json) and [actual allocations](actual-allocation.json)
retain the measurements and their limits. Focused patched-runtime tests passed
100 checks, runner tests 149, and portable checks 31. Each measured tranche has
an offline artifact/aggregate verification record. The final three functional
cases each passed backend abort, drain and recovery checks.

The final storage pass and scheduling combination nominated no new change.
Prompt-only storage/advice lost reuse; the final scheduling combination's point
gain was only 1.04%, below the useful margin. Matched audit-off controls passed,
but audit-off is an existing observer-cost option, not a new retention algorithm.

## Validation and reproduction

The benchmark baseline is native no-hint ARC in the same patched runtime, with
matching model, resource budget and offered work. It is not a claim about an
unmodified upstream server with different storage or instrumentation settings.
The development host is the RTX 2080 Ti under WSL2, with the pinned 7B AWQ model,
FP16 and the supported attention backend. Historical corpora were explicitly
reused as development inputs. Live outputs and forced-token replays are reported
separately. Diagnostic profiling and unequal-resource references cannot qualify
a policy winner. Fresh workloads, broader quality checks and other hardware
remain Stage 6 work.

Fast CPU and runner checks use the patched runtime:

```sh
make test PY=/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python
make bench-tests PY=/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python
make check-light PY=/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python
```

New serving work should use the maintained four-tier commands in
[benchmarking](../../docs/BENCHMARKING.md). The retained campaign can be
reconstructed with [stage5_optimization.py](../../benchmarks/research_checks/stage5_optimization.py),
[stage5_followups.py](../../benchmarks/research_checks/stage5_followups.py) and
[stage5_closeout.py](../../benchmarks/research_checks/stage5_closeout.py).
Those development recipes require the declared historical local artifacts; they
are not a self-contained public workload package. Every measured tranche retains
its complete spec, source freeze, external inputs and offline verification under
`output/stage5-optimization-20261105/`. Negative and invalid attempts are retained.

For another iteration, first check the complete measurement and mechanism activity,
then compare changed cache work, CPU cost and workflow time against both controls.
Follow a credible mechanism with fixed repeats; do not pool the selecting screen
or keep repeating until a favorable result appears. Keep defaults when the gain
or neutral/adverse guards are unresolved. The handoff freezes that decision for fresh
Stage 6 testing without asserting universal optimality.


## Handoff and retention

Serving processes were reaped and both GPU reservations were acquired/released
successfully after the last run. Cleanup removed 146 empty task-owned runtime
containers; all measured inputs, negative results and recovery material remain.
Approximately 2.23 GB of substantial evidence is retained. WSL free bytes changed
from 883,118,055,424 to 883,118,628,864 (observed +573,440 bytes); concurrent activity
can affect that delta, and this is not a Windows VHDX reclamation claim.
The [retained inventory](../../output/stage5-optimization-20261105/retention-inventory.json)
is outside Git; compact cleanup findings are in validation.json. The implemented
source and compact results are checkpointed in the project's Git repository.
