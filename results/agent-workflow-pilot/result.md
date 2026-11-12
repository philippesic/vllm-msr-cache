# Earlier agent trials did not establish an effectiveness improvement

2026-11-12 · historical · unresolved

The earlier 28-trial agent-environment pilot observed useful behavior and adverse continuation errors, but inherited instructions contaminated every trial and the intended handoff interruptions did not occur. It cannot establish that the new environment improved agent effectiveness.

## Scope

Local 2026-11-10 BEFORE/AFTER development fixtures: seven tasks, two repetitions per arm, plus four continuation agents. The report, frozen prompts/rubric, grades, claim audit and available handoff diffs are retained. No new agent experiment was run for this closeout.

## Limits

Unblinded judgment, inherited instructions, a drift fixture that did not isolate environment drift, and completed rather than interrupted AFTER handoffs prevent causal comparison. Observed timings and task scores are descriptive; they do not qualify current global guidance or prove general agent reliability. Raw trial checkouts and full conversations may be pruned after this evidence is verified.

## Decision

Use the documented failure modes as development context and observe behavior during normal work. Do not repeat the pilot or claim improvement from its contaminated comparisons. Keep current workflow guidance in current docs rather than reviving completed overhaul plans.

## Sources

- [efficacy/20261110/REPORT.md](../sources/aee60d032f697acea2a27d4ef8519da7305cabdd699f295edb2ad6c122353669.json)
- [efficacy/20261110/grades.json](../sources/22e7af02a99b57d20c0ebfac6ef0e8be874fc6ace23197331ab4f80b52309ff8.json)
- [efficacy/20261110/integrity.json](../sources/f26306dc3dfa20ebae2e6bc2d592aa8896ed098472323008dc22ff5b899e5b63.json)
- [efficacy/20261110/claim-audit.json](../sources/42b3f9e1ee5c7268365d0a002d7f83bd35052aaffd425a5f07295d5fe30bb711.json)
- [efficacy/20261110/audit-actions.json](../sources/0f2b50caf7730ab647edf3e361c66919105e8d3dee65818b53c5363cc7b41a5c.json)
- [efficacy/20261110/suite/protocol.json](../sources/196fe0b9856418df95cec3d4c398f16ea0c18997e1256ebfb745d4b2a445af68.json)
- [efficacy/20261110/suite/tasks.json](../sources/de1a27e463dc2f35eeae5b19bb1557165c30869fd40c8390cf4c01375ff3a6fc.json)
- [efficacy/20261110/suite/rubric.md](../sources/36776af641220c506deb653709a9f1af981409ef1146c2d7f8e17e1e27c0b8f4.json)
- [efficacy/20261110/trials/handoff-after-1/diff.patch](../sources/2dcac83b1bf7bd1cfcbe64d6f2f62b1c8e4012146f2b62166179458cd55afc11.json)
- [efficacy/20261110/trials/handoff-after-1/handoff-A-diff.patch](../sources/94ec6508d515fb22b077cadab7f61c813f3d93f6dcacbd75190decd2dc7e30bf.json)
- [efficacy/20261110/trials/handoff-after-2/diff.patch](../sources/04f4b3bf6ab2724d28c8ae0fc6d8e7f49a202852602d48633780aa615b041eb0.json)
- [efficacy/20261110/trials/handoff-after-2/handoff-A-diff.patch](../sources/160613b7339cbc15050c5bff2d78d78c16cf7d33cec7b7655b111a9711ea4a9b.json)
- [efficacy/20261110/trials/handoff-before-1/diff.patch](../sources/4cee872c65eb06c447444c5c450b11c31a30423fefbd72b5ac45e1220f7afc4a.json)
- [efficacy/20261110/trials/handoff-before-2/handoff-A-diff.patch](../sources/94e03c18de63a8c0fcfc74c7d45822c3fa109f1657c1f9996f902d0962855748.json)
- [efficacy/20261110/trials/handoff-before-2/diff.patch](../sources/492a94afd913d5140c504976d7414006368f0f4c8eb202e6ae1dd2547885356e.json)

Exact source hashes and figure inputs: [result.json](result.json).
