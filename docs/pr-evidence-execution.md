# Frozen-main evidence execution

The accepted [plan](pr-evidence-plan.md) owns the decision rules. Philip authorized
Stage 1 and the local 2080 Ti admission/screen on 2026-11-13, with a 1-hour admission
and 8-hour screen ceiling. Later stages and public posting remain unauthorized.

## Implementation estimate before building

These are engineering estimates, not measured runtime or commitments of GPU time.
The minimum end-to-end screen requires, in dependency order:

| Work | Estimated engineering time | Gate |
| --- | --- | --- |
| Frozen interface inspection and CPU feasibility reproductions | 2–6 hours | At least one feasible route |
| Isolated runtime build, hold plugin, delivery patch and deterministic lifecycle tests | 12–24 hours | Identity, completeness and readiness |
| W1 admission, streaming client, group events and instrument qualification | 12–24 hours | Frozen inputs and trustworthy measurement |
| Admission, calibration, paired acquisition and verification | 4–8 hours plus authorized GPU exposure | Hardware and budget |
| Statistical report, source retention and closeout | 3–6 hours | Independently recomputable verdicts |

Minimum screen path: approximately 33–68 engineering hours, conditional on the
interface and build gates. This excludes Stage 3 and competitor comparisons.
A simulator port would add an estimated 8–16 hours and cannot replace required
hardware evidence before rank agreement; do not build it for this screen.
Competitor integrations/full-matrix work would add at least several engineering
days plus separately authorized compute and are outside this phase.

Stage 1 failures are classified per route. If both routes require an interface
revision, stop before GPU and retain the precise missing contract; do not silently
expand the delivery patch or revive the old connector as a pure policy plugin.
