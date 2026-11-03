# Vendored vLLM runtime

`vllm/` is vLLM source at upstream revision
`dc1be79031d948d7a18c37600881e45ca708d913`, with
`patches/vllm-2080ti-cpu-advice-runtime.patch` applied. The upstream license is
in `vllm/LICENSE`. The smaller native-prefix patch is a historical alternative;
do not apply both patches.

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
package, links the wheel's compiled extensions into it, and removes that copy
on exit. It does not modify the installed wheel. It is intended for development
checks, not a build or distribution method. Existing frozen experiments use
their original overlays; do not change them during a run. GPU serving using
this launcher has not yet been qualified.


Validation on this host: 60 focused config, CPU-recency, and native-runtime tests
passed through the launcher. A module-resolution check selected the temporary
repository source, and a different child Python was rejected. A no-op launch
including source staging took 0.58 seconds (one observation); no persistent
staging cache is warranted by that check. The pinned requirements dry run matched
all 168 packages in the existing environment without changes. A clean environment
installation has not been verified. GPU serving validation is pending because
an existing frozen experiment was actively using the GPU; it was left untouched.
