"""Verify the staged source and binary imports without allocating a model."""
from pathlib import Path
import importlib.metadata
import vllm
import torch  # Load libtorch before the pinned wheel extension.
import vllm._C_stable_libtorch

from vllm_runtime import EXPECTED, SOURCE

package = Path(vllm.__file__).resolve().parent
if not package.parent.name.startswith('semantic-vllm-'):
    raise SystemExit('vLLM was not loaded through tools/vllm_runtime.py')
for relative in ('__init__.py', 'v1/core/block_pool.py'):
    if (package / relative).read_bytes() != (SOURCE / relative).read_bytes():
        raise SystemExit(f'Staged source differs from repository: {relative}')
if importlib.metadata.version('vllm') != EXPECTED:
    raise SystemExit('Unexpected vLLM distribution version')
print(f'Runtime imports passed: vLLM {EXPECTED}; repository source and native extension.')
print('No model was loaded; GPU serving and transfer correctness remain separate checks.')
