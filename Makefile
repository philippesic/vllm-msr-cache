PY ?= python3
RUNTIME = "$(PY)" tools/vllm_runtime.py "$(PY)"
TESTS ?= tests/test_config.py tests/test_wait_hint_cpu_recency.py tests/test_native_child_runtime.py

.PHONY: help runtime-check test test-harness benchmark-help bench-tests bench-smoke bench-screen bench-confirm bench-operational bench-verify
help:
	@echo 'Use PY=/path/to/pinned/environment/bin/python'
	@echo 'runtime-check: verify repository source and native extension imports (no model)'
	@echo 'test: focused CPU checks; override TESTS for affected tests'
	@echo 'test-harness: parser and preservation checks (no model)'
	@echo 'benchmark-help: show maintained serving suite options (no model)'
	@echo 'bench-smoke / bench-screen / bench-confirm / bench-operational: MODEL=local/snapshot OUT=new/directory [BENCH_ARGS="..."]'
	@echo 'bench-tests: suite contracts, transport, lifecycle, and native telemetry tests'
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
	$(RUNTIME) -m pytest -q tests/test_serving_suite.py tests/test_benchmark_metrics.py
bench-smoke bench-screen bench-confirm bench-operational:
	@test -n "$(MODEL)" -a -n "$(OUT)" || { echo 'Set MODEL to a local snapshot and OUT to a new artifact directory'; exit 2; }
	"$(PY)" -m benchmarks.serving_suite.runner $(patsubst bench-%,%,$@) --model "$(MODEL)" --output "$(OUT)" $(BENCH_ARGS)
bench-verify:
	@test -n "$(OUT)" || { echo 'Set OUT to an existing suite result directory'; exit 2; }
	"$(PY)" -m benchmarks.serving_suite.verify "$(OUT)"

.PHONY: check-light
check-light:
	"$(PY)" -m pytest -q tests/test_workflow_contracts.py tests/test_latency_validation.py tests/test_mtp_roundtrip.py
	"$(PY)" -m compileall -q semantic_offload harness benchmarks tools
