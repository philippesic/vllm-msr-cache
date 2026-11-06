# Bounded admission design investigation

Design only, authorized by the latest user discussion. No admission policy, forced GPU eviction, or new admission benchmark is implemented. Existing capture optimization is a separate experiment. Prior threshold0/2 results establish admission starvation versus churn, not an optimal budget.

## Derive budgets from state, then measure their payback

Use three independent limits, charged before a copy is queued and rechecked atomically:

1. **DRAM residency:** capacity minus protected ready blocks, incomplete stores and reservations. Track per-session existing plus pending use, and charge shared physical copies once under a documented sharing rule. Give competing paused sessions a fair share; unused shares may be borrowed, but do not churn occupied useful prefixes merely to fill a quota. Free host memory is an expansion constraint, not a second deduction from an already allocated tier.
2. **Write traffic/service:** a global token bucket plus per-session burst cap. Fit queued work and a conservative observed completion estimate inside the currently advertised hint lifetime, with demand-copy headroom. Native completion envelopes can help estimate service but are not isolated DMA bandwidth. Hint250ms in the prior trace was a policy estimate, not knowledge of actual future waits. Do not use private simulated wait durations or future resume identities as policy inputs. Outstanding writes consume tokens/reservations; retry a refused batch only after relevant capacity state changes. Deduplicate unchanged native keys.
3. **GPU staging:** actual allocatable native blocks after running/demand reservations and a demand headroom allowance, not a percentage of total preallocated pool. Existing staging and in-flight allocations count. Recheck before loading and consuming; demand has priority. A DRAM preservation budget does not grant a GPU staging budget.

Numerical illustration, not tuned settings: this tier holds199blocks, each2.25MiB/16tokens. With no reserve subtracted, equal shares are199blocks for one paused session,99each for two,66each for three,33each for six. Six-way sharing gives only74.25MiB/528tokens each. The actual prior per-request eligible-size distribution and arithmetic are in admission-budget-arithmetic.json. It may be impossible to preserve every complete context concurrently within this tier; choose useful partial prefixes or admit fewer sessions rather than promise full preservation.

Choose write-rate/headroom settings using observed demand latency, useful restores per byte and capacity pressure, then small matched experiments if later authorized. Current data do not establish sustainable background bandwidth or an optimal fraction. No unexplained fixed percentage is claimed safe. If performance evidence is unavailable, start with a tightly bounded single-segment opportunity or skip, rather than treating absence of a budget as unlimited permission.

## Which blocks

Identify a completed request's exact ordered native KV chain under matching model/tokenizer/adapter/layout/generation. Subtract ready/in-flight exact DRAM copies. Preserve a consecutive segment that extends an already usable prefix, normally from the earliest missing position. If leading GPU coverage is relied on, it must remain valid/protected through consumption; an unprotected cached GPU prefix is not durable coverage. The threshold0 experiment selected positions56..111 but lost its leading GPU prefix by return. Otherwise preserve from position0, and recompute/restore the rest on demand.

Within scarce capacity, compare session-level consecutive-prefix candidates using past observed reuse, current tool-wait evidence, predicted recomputation avoided and observed copy cost. Do not choose isolated semantically high-scoring blocks that cannot be installed by the native contiguous-prefix path. A longer prefix can have greater value but also consume the entire tier. The first implementation candidate should be deterministic fair-share/recency, with semantic selection evaluated separately against that control; no ranking advantage has yet been demonstrated.

The whole-chain admission helper is not a ready solution for growing conversations: identity includes full-chain digest and requires exact same bounded tuple on a distinct request. Prefix credit must instead be explicitly proven with native predecessor/key/position and compatibility namespace; removing the digest without an equivalent proof would weaken correctness.

## When GPU blocks can be reclaimed

A successful store makes a DRAM copy. It does not decrement source references. After copy completion/fence, only native blocks with no active users can be made available for eviction/reuse. Respect shared prefix ownership, in-flight stores/loads and reserved staging. Do not call free on blocks still used by an active request, and do not manufacture refcount changes from a tool hint. Existing completed requests may already have returned blocks to the native free queue while retaining cached hashes; those are reclaimable without an extra manual free.

First evaluate preservation plus ordinary pressure-driven native eviction. An explicit reclaim-on-wait policy would be a separate change requiring source ownership, fence and native cache-manager support, with a robust resume fallback. Never promise GPU capacity from D2H bytes alone. Measure independently which blocks become newly allocatable, useful prefix availability at resume, additional supported sessions at fixed latency, and peak/time-weighted pressure.

## Follow-up acceptance criteria, not work started

A candidate should reduce threshold0's repeated writes/evictions while preserving more useful contiguous context than threshold2. Compare matched ARC demand and prefetch with the same admission rule. Track write amplification, useful restore fraction, latency/background interference, demand headroom and ownership/exact-output invariants. Test growing/diverging/shared prefixes, shortened/cancelled waits, duplicate offers, partial/failed copies, fair-share changes and zero headroom. Offline trace arithmetic cannot establish end-to-end policy or concurrency benefit.
