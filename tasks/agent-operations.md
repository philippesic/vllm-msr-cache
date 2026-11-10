# Agent operations

## Goal
Add repository-specific agent operations once the environment overhaul is settled:
skills for repeated procedures and support for long-running work.

## Authorized
Nothing yet. Philip will revisit this; do not implement any item below until then.

## Done when
Philip has chosen which items to build, and each chosen item is implemented, used once
on a real task, and documented.

## State
Candidate list only:
- Skill: run the runtime CPU lane and report its one-line verdict and record location.
- Skill: GPU qualification run: confirm authorization, check the GPU is idle and disk
  headroom, run the hardware lane, record the result as evidence and a finding.
- Skill: record a finding: write the evidence record, add the FINDINGS entry with scope.
- Skill: task handoff: create, update and close `tasks/` files correctly.
- Skill: vLLM repin: the approved procedure for moving to a new vLLM version and
  requalifying both advice modes.
- Session start: surface open `tasks/` files and recent FINDINGS automatically.
- Completion notification for long GPU runs, tested end to end before relying on it.
- A recurring fresh-agent evaluation of this repository (see the efficacy test design).

## Next
Philip reviews the candidate list.
