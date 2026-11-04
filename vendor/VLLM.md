# Vendored vLLM runtime

`vllm/` is vLLM source at upstream revision
`dc1be79031d948d7a18c37600881e45ca708d913`, with
`patches/vllm-2080ti-cpu-advice-runtime.patch` and
`patches/vllm-benchmark-abort-receipt.patch` applied. The upstream license is
in `vllm/LICENSE`. The smaller native-prefix patch is a historical alternative;
do not combine that alternative with the CPU-advice patch. The abort-receipt
patch is additive.

For a new Linux x86-64 environment, use Python 3.12 and the exact wheel URL
and dependency versions in `requirements-runtime.txt` (paths below are examples):

```bash
uv venv --python 3.12 .venv-runtime
uv pip install --python .venv-runtime/bin/python -r requirements-runtime.txt
uv pip install --python .venv-runtime/bin/python --no-deps -e .
```

Keep that environment outside Git. The requirements file captures the installed
vLLM/pytest dependency closure, not unrelated tools from the host environment.
The root `uv.lock` describes the generic extension environment; `uv sync` is not
the setup command for this pinned runtime. The broad package extra remains a
compatibility declaration, not an experiment pin.

On the existing 2080 Ti host, run commands with the matching installed wheel
and the repository source using:

```bash
PY=/home/philip/.local/share/venvs/vllm-semantic-cache-ext/bin/python
$PY tools/vllm_runtime.py $PY -m pytest -q tests/test_config.py
```

The launcher requires the child to use the same Python executable and verifies
the wheel version, creates a temporary copy of the source
package, fills missing distribution-owned support files and compiled libraries from
the exact wheel, and removes that copy on exit or handled interruption. Repository
source takes precedence; wheel-generated `_version.py` supplies the build identity.
It does not modify the installed wheel. It is intended for development
checks, not a build or distribution method. Existing frozen experiments use
their original overlays; do not change them during a run. GPU serving has now
passed bounded 0.5B and 7B AWQ checks through this launcher;
see the [suite qualification record](../evidence/benchmark-suite-20261103/README.md).


Validation on this host: 60 focused config, CPU-recency, and native-runtime tests
passed through the launcher. A module-resolution check selected the temporary
repository source, and a different child Python was rejected. A no-op launch
including the original source staging took 0.58 seconds (one historical
observation, before supplemental support assets were added). The pinned requirements dry run matched
all 168 packages in the existing environment without changes. A clean Python 3.12 environment has now installed these requirements and passed
runtime imports, the full test suite, and a real GPU smoke; see
[clean-install verification](../evidence/workflow-finalization-20261103/README.md). The new suite records the actual supplemental
wheel asset hashes as well as
repository-source identity. The original frozen research qualification remains
bound to its own overlay; suite qualification is a separate result.


The benchmark receipt patch emits a structured record only for tagged cancellation
probes when `VLLM_BENCHMARK_ABORT_RECEIPTS=1`. It records the actual scheduler abort
after native cleanup, without replacing a runtime method. The ordinary frontend
request-success metric does not count client-disconnect aborts on this pin; it
cannot qualify cancellation by itself.

The retained, default-off AWQ loading experiment is recorded separately in
`patches/vllm-experimental-awq-loading.patch`. It is present in this checkout but
remains paused and unqualified; the clean-install smoke used its default-off path.
Do not infer endorsement from retaining its code and tests.
