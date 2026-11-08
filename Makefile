PY ?= python3
RUNTIME = "$(PY)" tools/vllm_runtime.py "$(PY)"
TESTS ?= tests/test_config.py tests/test_wait_hint_cpu_recency.py tests/test_native_child_runtime.py tests/test_arc_optimization.py tests/test_arc_serving_profile.py

.PHONY: help runtime-check test test-harness benchmark-help bench-tests bench-smoke bench-screen bench-confirm bench-operational bench-agent bench-agent-pressure bench-verify
help:
	@echo 'Use PY=/path/to/pinned/environment/bin/python'
	@echo 'runtime-check: verify repository source and native extension imports (no model)'
	@echo 'test: focused CPU checks; override TESTS for affected tests'
	@echo 'test-harness: parser and preservation checks (no model)'
	@echo 'benchmark-help: show maintained serving suite options (no model)'
	@echo 'bench-smoke / bench-screen / bench-confirm / bench-operational: MODEL=local/snapshot OUT=new/directory [BENCH_ARGS="..."]'
	@echo 'bench-tests: suite contracts, transport, lifecycle, and native telemetry tests'
	@echo 'bench-agent-pressure: real repository tools, server queues, CPU wait retention; MODEL=local/snapshot OUT=new/directory'
	@echo 'bench-verify: verify retained artifacts offline; OUT=existing/directory'
runtime-check:
	$(RUNTIME) tools/check_runtime.py
test:
	$(RUNTIME) -m pytest -q $(TESTS)
test-harness:
	$(RUNTIME) -m pytest -q tests/test_latency_validation.py tests/test_preservation_mapping.py
benchmark-help:
	"$(PY)" -m benchmarks.serving_suite.runner --help
bench-tests:
	$(RUNTIME) -m pytest -q tests/test_serving_suite.py tests/test_benchmark_metrics.py tests/test_repo_workload.py tests/test_wait_opportunity_report.py tests/test_published_benchmark.py tests/test_stage5_optimization.py tests/test_campaign_contracts.py tests/test_campaign_resume.py tests/test_campaign_reporting.py tests/test_campaign_workflows.py tests/test_campaign_execution.py tests/test_campaign_quality.py tests/test_campaign_transfer.py tests/test_stage6_capped_review.py tests/test_stage6_addendum.py
bench-smoke bench-screen bench-confirm bench-operational bench-agent:
	@test -n "$(MODEL)" -a -n "$(OUT)" || { echo 'Set MODEL to a local snapshot and OUT to a new artifact directory'; exit 2; }
	"$(PY)" -m benchmarks.serving_suite.runner $(patsubst bench-%,%,$@) --model "$(MODEL)" --output "$(OUT)" $(BENCH_ARGS)
bench-agent-pressure:
	@test -n "$(MODEL)" -a -n "$(OUT)" || { echo 'Set MODEL to a local snapshot and OUT to a new artifact directory'; exit 2; }
	"$(PY)" -m benchmarks.serving_suite.runner agent --model "$(MODEL)" --output "$(OUT)" --task-file benchmarks/serving_suite/repo_pressure_tasks.json --sessions 16 --concurrency 8 --client-concurrency 0 --tool-concurrency 4 --gpu-blocks 2560 --cpu-bytes 2818572288 --continuation-credit-ms 8000 --hint-timing before-tool --hint-through-queue --compact-events --stable-tool-output --execution graphs --request-timeout-s 180 --candidate-config benchmarks/research_checks/fixtures/wait_retention_ungated.json $(BENCH_ARGS)
.PHONY: bench-agent-five
bench-agent-five:
	@test -n "$(MODEL)" -a -n "$(OUT)" -a -n "$(RUNTIMES)" || { echo 'Set MODEL, new OUT, and RUNTIMES (published runtime manifest)'; exit 2; }
	"$(PY)" -m benchmarks.serving_suite.runner agent --model "$(MODEL)" --output "$(OUT)" --task-file benchmarks/serving_suite/repo_pressure_tasks.json --sessions 16 --concurrency 8 --client-concurrency 0 --tool-concurrency 4 --gpu-blocks 2560 --cpu-bytes 2818572288 --continuation-credit-ms 8000 --hint-timing before-tool --hint-through-queue --compact-events --stable-tool-output --execution eager --request-timeout-s 180 --repeats 1 --candidate-config benchmarks/research_checks/fixtures/wait_retention_ungated.json --semantic-config benchmarks/research_checks/fixtures/semantic_wait_prefix.json --published-runtimes "$(RUNTIMES)" $(BENCH_ARGS)
bench-verify:
	@test -n "$(OUT)" || { echo 'Set OUT to an existing suite result directory'; exit 2; }
	"$(PY)" -m benchmarks.serving_suite.verify "$(OUT)"

.PHONY: check-light
check-light:
	"$(PY)" -m pytest -q tests/test_workflow_contracts.py tests/test_latency_validation.py tests/test_mtp_roundtrip.py
	"$(PY)" -m compileall -q semantic_offload harness benchmarks tools
