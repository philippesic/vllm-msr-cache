#!/usr/bin/env python3
"""Run a command with the repository vLLM source and matching wheel binaries."""
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'vendor' / 'vllm' / 'vllm'
EXPECTED = '0.26.1rc1.dev75+gdc1be7903'

def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python tools/vllm_runtime.py COMMAND [ARGS...]')
    command = sys.argv[1:]
    # Compare environment paths without resolving venv symlinks to system Python.
    executable = shutil.which(command[0])
    if executable is None or os.path.abspath(executable) != os.path.abspath(sys.executable):
        raise SystemExit('Run the child with the same Python as the launcher: '
                         f'{sys.executable} tools/vllm_runtime.py {sys.executable} -m MODULE ...')
    dist = importlib.metadata.distribution('vllm')
    if dist.version != EXPECTED:
        raise SystemExit(f'Expected vLLM wheel {EXPECTED}; found {dist.version}')
    wheel = Path(dist.locate_file('vllm')).resolve()
    with tempfile.TemporaryDirectory(prefix='semantic-vllm-') as directory:
        package = Path(directory) / 'vllm'
        shutil.copytree(SOURCE, package, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for binary in wheel.rglob('*.so'):
            target = package / binary.relative_to(wheel)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(binary)
        version = wheel / '_version.py'
        if version.exists():
            shutil.copy2(version, package / '_version.py')
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join([directory, str(ROOT), env.get('PYTHONPATH', '')])
        env.setdefault('VLLM_WSL2_ENABLE_PIN_MEMORY', '1')
        return subprocess.call(command, env=env)

if __name__ == '__main__':
    raise SystemExit(main())
