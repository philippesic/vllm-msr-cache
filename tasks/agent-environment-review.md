# Agent environment review

## Goal
Review and reshape the repository's agent environment with Philip: instructions,
checks, rules and document conventions, before benchmark design and agent operations.

## Authorized
CPU work and local commits. No GPU, no remote. Approved so far: porting-doc purge,
new AGENTS.md, enabled hook, document convention, skills allowed, no GPU reservation
system, removal of speculative benchmark/plugin rules, proofs moved to `--self-test`,
and the core fixes listed under Next.

## Done when
Philip has reviewed each environment component and the agreed changes are committed
with the check and rule proofs passing.

## State
Branch `main`, local only. Documentation, rules, proof split and this convention are
committed. Comment ban kept everywhere. All sections reviewed. Decided: runtime lane
defaults its output to the attempts folder and prints a one-line verdict; run-record
tooling is revisited with benchmark design; CI waits for the GitHub repository change;
agent operations are listed in `tasks/agent-operations.md`. Open: the runtime binding
redesign (replace the overlay and fingerprints with a vLLM general plugin and one pin)
and a project-agnostic efficacy test.

## Next
Apply the approved core fixes: a new generation per turn and a new incarnation
replaces the old session; exact replays return the original answer even after the
deadline; the observation store owns all its mutations; validation in constructors;
typed outcomes with refusal reasons; named limits; a deliberate memory bound; remove
dead settings and "absent" feature labels.
