# Benchmark evidence contract

Benchmarks support development decisions about exact CPU KV reuse and causal wait
advice. They must distinguish a broken measurement, a valid unfavorable outcome,
correctness/reliability qualification, and evidence of useful benefit. No suite
completion badge establishes all four.

## Questions and comparisons

The comparison systems are upstream GPU prefix caching, native CPU ARC, current
wait-hint advice, and LMCache CPU caching. Keep model/runtime/resource configurations
explicit. Matched nominal DRAM capacity does not imply equal staging memory or RSS.
Report total resource cost. Whole-system comparisons do not isolate a policy change
unless the runtime and all other configuration are matched.

The primary mechanism outcomes are recomputed prefix tokens and time to first token
when a waiting session returns. Report protected and unhinted returning cohorts
separately, plus failed stores, transfer bytes, readiness, hint outcomes and latency.
End-to-end coding time and original task usefulness remain application outcomes;
long decode time can make substantial prefill savings nearly invisible in wall time.
Do not demand a universal five-percent application speedup from that geometry.

## Coverage that can distinguish retention policies

A real pressure experiment must keep several sessions alive across tool waits and
interleave their requests without fully clearing the cache between each application.
The GPU cache must be smaller than the admitted working set, and the CPU pool must
hold more than one conversation while still facing measured competition. Report
actual block/chunk sizes and observed working-set ratios, not names such as pressure.

Publish one causal hint per measured tool wait, as the real proxy does. Include an
unhinted returning cohort and abandoned advice. Do not warm or return the target
prefix before competition merely to prove a restore: that can promote native ARC
membership and remove the mechanism under investigation. Check transfer feasibility
in a separate phase or session. Natural low opportunity remains a useful observation.

Controlled traffic explains causality; captured coding traffic checks whether its
assumptions occur in practice. A CPU trace replay can compare policy directions in
seconds using a declared concurrent-session schedule, but is a simulation. It must
retain native policy/source identity, exact prefix-key assumptions, store/touch/ready
ordering, expiry, capacity and failed-store accounting. Token counts alone do not
establish shared-prefix identities or hybrid cache-group bytes. Philip approved
replacing the recency hint with promotion or bounded protection if it consistently
wins. Two promote-hold rounds with rules fixed before engine results did not adopt it
([first](../results/promote-hold-advice/result.md),
[settle](../results/promote-hold-settle/result.md)); reachable-hold passed the same
rule on both models and replaced recency ([decide](../results/reachable-hold-decide/result.md)). Report
unhinted harm and abandoned-hint cost.

## Correctness, execution and usefulness

Fresh, GPU-resident and CPU-restored continuations need explicit checks. Retain exact
token IDs and diagnostic logprobs when they differ; do not waive a mismatch as noise
without separating numeric execution differences from incorrect restored state.
Initial probes do not qualify long-running hybrid/window behavior. Failed or early
operating runs contribute only their actual exposure, not their allocated duration.

Use original task graders and retain trajectories, patches and grader receipts.
A resolved patch after a deadline is distinct from complete timed acquisition.
Timeouts are censored measurements and remain in denominators. Do not compute speed
ratios from censored pairs. Report usefulness and per-task paired outcomes rather
than presenting the small selected task collection as a general coding score.

Admission freezes source, models, inputs, task caps, resource settings and exposure
history. A failed or unfavorable attempt is immutable. Resume may execute only
never-started work, under current authorization, with original budgets preserved.
A stopped campaign remains partial; it does not become complete by removing cells.
Fresh confirmation cannot use already exposed source groups without that limitation.

## Workflows and limits

Functional checks exercise protocol/cache invariants and failure cleanup. The quick
comparison is advisory: one repetition, at most 45 minutes. Release acquisition has
two repetitions and a 30-hour ceiling. More repetition cannot repair advice-blind
traffic or insufficient mechanism activity. Preparation and bounded diagnosis are
reported separately. Do not launch a full matrix to test a runner edit.

The old release-01 schedule is stopped by Philip's audit decision. Its retained data
can support observed overhead, individual outcomes and completed correctness checks;
it cannot establish release eligibility, competitive superiority or broad benefit.
A full pressure run needs Philip's decision; its projected cost is in the result
above. Maintained
commands are in [the benchmark guide](benchmarks.md); durable conclusions follow
[the result profile](results.md).


The frozen-main PR plan has its own CPU admission receipts, external policy and
post-response patch/control. It does not reuse the old release qualification.
For W1, retain the exact source-row and tokenizer identities, success/failure
selection, repository-held-out partitions, chronological truncation, output caps,
actual token LCP and template-induced lost suffix. Its declared tool gaps make it
semi-synthetic, and transcript replay supports serving-latency claims only. Request
TTL deliberately includes final turns; post-response eligibility is determined
only from the current recorded nonterminal tool action. The route comparison
therefore measures their whole delivery/information difference. Stage 1 CPU policy
and dispatch tests do not qualify GPU restores. The separate CPU loopback instrument
qualification covers network capture with a fixture engine; live scheduler/GPU
accounting remains an admission requirement. All arms use the same delegating event
observer. Read actual scheduled boundaries and per-group loads, rather than treating
an offered hybrid GPU boundary or aggregate cached tokens as a cache-source count.
Recompute stream summaries from retained raw frames. A dead server's partial records
remain treatment evidence; they cannot be discarded merely for lacking a clean close.
