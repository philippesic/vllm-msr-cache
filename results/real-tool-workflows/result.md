# Real tool waits exposed recency benefits and output-dependent comparisons

2026-11-10 · historical · supported

An initial two-pair repository workflow was diagnostic. A later three-repeat pressure study measured higher workflow throughput from early CPU wait hints; adding semantic retention provided no incremental gain.

![Workflow throughput change (%)](chart.svg)

Three whole-case paired repetitions on fixed inputs; original 95% intervals. Higher is faster.

## Scope

Pressure study: Qwen2.5-7B-AWQ, RTX 2080 Ti, decode graphs, nine cases, 576 responses and 432 actual tools. Initial study: four cases and 128 responses.

## Limits

Later live prompts/outputs differ. These are closed-loop workflow results, not identical-work policy attribution. Narrow test-status correctness does not establish general answer quality.

## Decision

Retain actual tool waits and demonstrated cache competition as useful experimental ingredients, without fixing the product to these workloads.

## Sources

- [evidence/real-agent-eval-20261104/README.md](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/real-agent-eval-20261104/README.md)
- [evidence/wait-opportunity-20261104/README.md](https://github.com/philippesic/vllm-msr-cache/blob/1319f09bafac46bab89df465dd433c67441f4b26/evidence/wait-opportunity-20261104/README.md)

Exact source hashes and figure inputs: [result.json](result.json).
