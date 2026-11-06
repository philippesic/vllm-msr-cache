"""Present completed Stage3 outcomes; preserve the detailed investigation record."""
from pathlib import Path
import json
b=Path(__file__).resolve().parent;e=b.parents[1]/'evidence'/b.name;p=e/'STAGE3.md';backup=b/'stage3-before-final.md'
if not backup.exists():backup.write_bytes(p.read_bytes())
def table(cells):
 lines=['| Cell and evidence role | Wait batch time saved versus ARC | Conditional95% interval |','| --- | ---: | ---: |']
 for name,label in cells:
  s=json.loads((b/(name+'-summary.json')).read_text());c=s['comparisons']['wait_vs_arc'];v=c['candidate_batch_time_reduction_pct'];ci=c['candidate_batch_time_reduction_95ci_pct'];interval=f'{ci[0]:+.2f} to {ci[1]:+.2f}%' if ci else 'Single exposure; no independent-run interval'
  lines.append(f'| [{label}](../../output/{b.name}/{name}-summary.json) | {v:+.2f}% | {interval} |')
 return '\n'.join(lines)
s='''# Stage3: operating range and independent workloads

All selected Stage3 execution is complete. Clear cache-pressure gains coexist
with neutral, adverse and inconclusive cells. This is a bounded curated study on
the pinned7B AWQ/FP16/2080Ti/WSL2 configuration, not population-wide qualification.
Positive percentages below mean less whole-workflow batch time. Intervals are
conditional95% intervals over paired whole runs, not independent requests or a
multiple-comparison-adjusted deployment claim. No arbitrary workload average is used.

## Independent families and instruction validation

The new families investigate public more-itertools code/tests and Python PEP
documents using real searches, Python/pytest and parsing tools. Tasks are authored
fixed tool pipelines, not production traffic or autonomous-agent trajectories.
Public revisions, source hashes, instructions and references are frozen in the
retained inputs. The [operating scope](OPERATING_SCOPE.md) records the intended
use and coverage limits.

Original-instruction outcomes remain development evidence after source review
found unsupported coverage/limitation claims. Bounded instruction calibration used
already-seen tasks, not serving-speed outcomes; the adopted literal template was
then frozen before fresh16-task reserves. Both document memory cells used the same
fresh tasks and were selected before either outcome; they are not independent
workload populations. No reserve outcome selected a new capacity or instruction.

'''
s+=table([('stage3-repository','Original repository, full cache: development'),('stage3-repository-neutral-confirm','Repository, separate fixed-six neutral confirmation'),('stage3-repository-cohost','Original repository, half-byte cohost: development'),('stage3-repository-reserve','Fresh repository reserve, half-byte cohost'),('stage3-documents','Original documents, full cache: development'),('stage3-documents-cohost','Original documents, half-byte cohost: development'),('stage3-documents-admission-diagnostic','Documents, four-sequence mitigation: development diagnostic'),('stage3-documents-reserve','Fresh document reserve, half-byte cohost'),('stage3-documents-reserve-full','Fresh document reserve, full cache')])
s+='''

Fresh constrained repository tasks retain8.27%less time than ARC and8.79%less
than the published upstream retention system. Fresh full-budget documents retain
18.12%less than ARC; their upstream difference is unresolved. The fresh tight
document result is below the2%useful point target and narrowly fails the2%neutral
bound. Original and confirmation cohorts are never pooled. All full-budget and
neutral outcomes remain alongside the favorable constrained cells.

The tight original document cell includes an ARC-loss pair of5.83% and a second,
fresh reserve pair also loses slightly. The retained inactive-hint diagnostic
cannot resolve a retention effect. GPU admission, restores, preemptions and decode
tails interact; transfer volume alone is insufficient to explain the adverse case.
A preselected max-sequences8→4 diagnostic improves9.29%, but is development evidence
without fresh four-sequence or upstream qualification. Candidate preemptions fall
but are not eliminated. No universal fix is claimed and no outcome-driven search
or further confirmation follows. [Inactive control](../../output/wait-qualification-20261105/stage3-documents-pressure-control-summary.json),
[regression diagnosis](../../output/wait-qualification-20261105/document-cohost-regression-diagnosis.json).

The author source-audited448first-repetition final answers across28cases, including
250exact text/source reuses. This is not448distinct tasks or timing repetitions.
Correct subject, metadata and test status are narrower than useful analysis.
Unsupported facts and missing concrete limitations remain in every system;
no general equal-quality task-success claim follows. The [quality report](QUALITY.md),
[frozen rubric](INDEPENDENT_QUALITY_RUBRIC.md) and
[integrity check](../../output/wait-qualification-20261105/manual-review-integrity.json)
retain those failures without changing timing eligibility.

## Original-workload operating conditions

'''
s+=table([('stage3-ample','Ample cache, initial checkpoint'),('stage3-ample-neutral-confirm','Ample cache, separate fixed-six confirmation'),('stage3-cpu-pressure','Reduced CPU cache'),('stage3-concurrency','Twelve active sequences'),('stage3-serialized-tools','One real tool worker'),('stage3-mixed','Half hinted, half unhinted'),('stage3-soak','Finite six-wave churn exposure')])
s+='''

The ample-cache confirmation meets the2%neutral bound without a useful gain.
Initial and confirmation ARC-loss pairs remain alarmed; the initial cell has no
eviction/preemption opportunity. Reduced CPU cache retains a useful ARC benefit;
its upstream difference is unresolved. Twelve active sequences have all positive
pair points, but neither useful gain nor the2%neutral bound is established.

The serialized-tool cell uses3.65%less time than ARC, with unresolved gain
precision. The upstream system uses5.59%less time than wait and Continuum10.32%less.
These secondary losses remain visible. Continuum was selected for this regime
before outcomes because its returning latency was strongest on the original
workload. Different runtimes, schedulers and outputs prevent isolated policy ranking.

Mixed traffic completes all144offered workflows/576responses. Hinted and unhinted
group means both satisfy the predeclared5%regression bound. Hinted returning p95
is nevertheless5.03%worse on the descriptive geometric comparison; no broad
latency-tail guarantee follows. Upstream uses4.74%less batch time than wait.
No unhinted request receives either local advice or published native-retention
metadata, while common arrival-time continuation priority remains shared.
[Group checks](../../output/wait-qualification-20261105/stage3-mixed-stability.json).

The finite churn exposure completes1,152responses/864tools,96workflows per system
and zero failed requests. Its six waves are dependent repetitions within one
server exposure, not six independent runs. Wait uses5.82%less batch time than ARC
in that exposure; upstream uses3.71%less than wait. First/final-wave summed RSS
changes are ARC−167.75MiB, wait−380.32MiB and upstream+3.11MiB, within the declared
512MiB endpoint-growth bound. Wait's absolute RSS is higher. Shared mappings make
summed RSS an imperfect allocation measure; this does not prove absence of leaks.
This cell uses the agent tier and has no explicit operational abort/drain/recovery
receipt; those checks belong to Stage4. [Exposure decision](../../output/wait-qualification-20261105/soak-decision.json).

## Disposition and retained evidence

Every slower completed ARC/wait pair is retained in a `*-pair-alarms.json` file,
with native work, output, CPU/paging and error covariates where available. Those
covariates guide diagnosis; they do not establish an exclusive cause. A noisy or
configuration-specific loss did not justify abandoning a mechanism with causal
and fresh-task gains. The [regression report](REGRESSIONS.md) states the remaining
limits rather than averaging them away. Owner-selected numerical margins are
predeclared protocol decisions, not requirements supplied by the user.

All original/calibration/diagnostic/confirmation attempts, frozen sources and
negative results remain available. The full chronological investigation is
preserved as `output/wait-qualification-20261105/stage3-before-final.md`; this report
leads with the final outcomes. [Protocol](PROTOCOL.md),
[all comparisons](../../output/wait-qualification-20261105/all-wait-arc-comparisons.csv),
[all-attempt verification](../../output/wait-qualification-20261105/all-retained-verification.json).
'''
p.write_text(s);print('Consolidated Stage3 with all selected primary timing cells and adverse findings')
