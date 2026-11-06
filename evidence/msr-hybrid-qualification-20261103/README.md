# MSR hybrid qualification — gates remain unmet

Resumed November 4, 2026 under the unchanged [qualification protocol](PROTOCOL.md).
No live hybrid was built, no policy/default changed, and no serving performance
or VRAM win is established. The primary semantic pilot did not establish added
value. An exploratory quarter-second result supports further investigation of
richer query/key representations, separately from the original scalar MSR idea.
This phase stops at the evidence boundary; qualification remains unfinished.

## What completed

Default-off native observations now expose request arrival, asynchronous restore
issue/receipt, admission and bounded waiting/reclaimable-capacity samples through
`--qualification-trace`. The CPU analyzer separates queue time, restore residence
and post-restore scheduling. The interrupted thread's smoke/pressure artifacts
were independently verified; its 30 restores per pressure arm were real activity,
not proof of useful prefetch gain.

A new longer-history diagnostic used Qwen2.5-1.5B FP16, 2,048 GPU KV blocks,
1.75 GiB CPU capacity and six untruncated 8–9K-token recorded training contexts.
Authored summary requests, capped responses and scripted waits make this a
synthetic systems diagnostic. ARC and wait-hint ARC each completed 54 requests
and 13 restores, with optimistic overlap for all 13. Both passed maintained-suite
integrity checks: 143 files, two cases, 110 request records including health checks.
Their aggregate async-load residence was 1.278 and 1.075 seconds. Residence
includes scheduling/polling; it is not exposed transfer stall or a throughput
bound. CPU readiness, bandwidth headroom, displaced-history value and causal
savings remain unmeasured. See [opportunity results](long-history-opportunity.json).
The single diagnostic pair does not qualify a policy advantage.

## Source boundary and causal pilot

Bounded row-group inspection replaced the stalled whole-table nested-Parquet
inspection. Audits of 1,329 Exgentic sessions and 36 original transcripts found
missing/overlapping observations, shifted retry timing and documented timestamp
reconstruction. Structural prefix matches alone cannot certify exact model inputs
or usable timing. [Source findings](source-followup.json) preserve those limits.

The original [LMCache agentic dataset](https://huggingface.co/datasets/sammshen/lmcache-agentic-traces/blob/6e043b9e89865df3aec19fd5679286b683bfd70e/README.md)
provides cumulative input messages and source-documented measured response-to-next-
request gaps, unlike reconstructed absolute timestamps. All five pinned shards
were retained: 24,880 rows across 767 sessions. It still lacks absolute cross-
session arrival/completion times, terminal observation horizons and external tool
schemas. The dataset was selected for long, continuing agent sessions; results
cannot describe abandonment or an unfiltered user population.

One completed-prefix decision per complete-start session was selected at/after
turn three, without using its future gap. No truncation: 55 oversized sessions
were excluded. The 711 selected examples split into 453 training, 126 validation
and 132 test examples; all model/run variants of a task stay together. The test
partition contains 104 tasks. Only text known at the completed response becomes
features; subsequent tool/user text and final session length are excluded.

Local Qwen2.5-1.5B teacher-forced re-encoding captured a middle-layer final query
and own-prefix keys. Separate hypotheses were a scalar query/mean-key score,
seven block-score summaries, and a fixed 32-dimensional Q/K projection. These
are input-message representation hypotheses, not reproduction of native original-
model MSR. The cheap predictor includes known timing, lifecycle, sizes, cohort,
source model and the already-issued tool name. Training-only transforms and
validation-selected ridge strength precede test evaluation.
[Protocol and frozen-source references](feature-pilot-protocol.json).

## Results and interpretation

At the frozen primary one-second deadline, every task-paired 95% improvement
interval included zero. Brier score measures probability error; lower is better.
The useful-token exercise assumes simultaneous pauses and a common budget of
25% of total prefix tokens, without real cache competition or shared-prefix
deduplication. Its results are explicitly hypothetical.

| Added feature | Test Brier score | Brier improvement, task-paired 95% interval | Simulated useful-token improvement / budget, 95% interval |
| --- | ---: | ---: | ---: |
| Cheap metadata | 0.05313 | reference | reference |
| Scalar | 0.04885 | −0.00712 to +0.01819 | −1.37% to +1.58% |
| Distribution | 0.05167 | −0.01133 to +0.01259 | −6.76% to +1.36% |
| Q/K representation | 0.04399 | −0.00780 to +0.02267 | −1.42% to +1.57% |

[Full primary result](feature-pilot-result.json). The cheap selector already filled
98.9% of the hypothetical budget with timely useful tokens, so this deadline
leaves little room to improve that exercise. It cannot reject MSR generally.

After the primary result, exploratory sensitivity froze five deadlines:
0.05, 0.1, 0.25, 1 and 5 seconds. A second robustness check added numeric pair
products, numeric/category interactions and joint categorical states to the cheap
baseline. Each analysis used the same frozen splits/features, validation selection
and paired task bootstrap with 10,000 draws. Its 30-comparison Bonferroni intervals
have 99.833% nominal coverage; extreme bootstrap tails are coarsely estimated.
This controls the declared family within each exploration, not all sequential
research, and does not make these independent confirmations.

At 0.25 seconds, Q/K representations improved prediction and simulated retained
value under both baselines. Against the more flexible metadata baseline, the Brier
improvement interval was +0.01482 to +0.10501. The expanded cheap model itself had worse test Brier
error (0.11601 versus 0.10955 for main effects); this is robustness to additional
controls, not proof of an optimal nonsemantic comparator. The observed simulated useful-token
gain was 108,163 tokens, or 30.35% of the common budget; its corrected interval
was +0.48% to +53.90%. The interval is wide. At 0.1 seconds prediction improved,
but retained-value uncertainty included zero. No other tested deadline established
both gains; scalar and distribution variants never established positive retained
value. Scalar prediction at 0.05 seconds improved with the original baseline and
worsened with interactions. All results, including negative and uncertain cases,
are in [deadline sensitivity and baseline robustness](deadline-sensitivity.json).

The quarter-second result is a promising richer-representation lead. It is not
an original scalar MSR win, actual cache-value evidence, or a serving speedup. It also does not identify a semantic causal mechanism.
Local re-encoding took about 1,045 seconds of forward/hook time over the 711
examples. The frozen legacy timer excluded final CPU summary/projection work;
future code fixes its scope. Neither timing is incremental native serving
capture/scoring overhead, and its model memory is not a serving VRAM comparison.

## Stopping boundary and remaining work

The protocol requires positive held-out **real cache value**, relevant system
opportunity and native costs below attainable gain before live policy engineering.
These gates are not met. Further scripted benchmark repetitions cannot supply the
missing observation. Qualification needs representative causal model inputs with
actual arrival/completion/session and observation-horizon semantics, observed
cache competition/victim costs, and an independent test of the promising feature.
Native feature fidelity, capture/scoring/displacement/interference costs and then
the prescribed four live arms remain unfinished. No new source request is pending
with the user; their lack of a known source is recorded, not treated as consent to
fabricate timing or as a rejection of the hypothesis.

## Validation, cleanup and retention

The final focused pinned-runtime CPU checks passed 170 tests; 14 existing PyTorch
warnings remain. Portable `make check-light` passed 31 tests and compilation.
The longer-history suite passed offline artifact verification. GPU capture/server
processes exited, no compute clients remained, and both GPU locks were acquired
and released for the cleanup check. No lock was removed. Paused AWQ work stayed
paused. Changes remain uncommitted and nothing was pushed.

Raw sources, frozen inputs/code, positive/negative results, logs and receipts stay
in ignored `output/msr-hybrid-qualification-20261104-resume/`; original runs stay
in `output/msr-hybrid-qualification-20261103/`. New retained output is about
2.51 GB, dominated by the 2.37 GB reusable pinned dataset. Cleanup removed only
the empty failed input-preparation artifact. WSL free space measured
909,404,295,168 bytes before and 909,404,299,264 after: +4,096 bytes observed,
with no Windows-volume reclaim claim. The retained inventory and exact custody
check are linked by the [checkpoint](qualification-checkpoint.json).
