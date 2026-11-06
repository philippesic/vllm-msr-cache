"""Execute exact changed production methods and new tests without torch/vLLM.

This is method-level CPU validation, not a replacement for native integration.
AST extraction avoids importing the unavailable Mac torch/vLLM dependencies.
"""
from __future__ import annotations
import ast
import __future__
from pathlib import Path
import sys
import os
import types
import time
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[2]
connector = types.ModuleType('semantic_offload.connector')
connector.__dict__.update(time=time, record_wait_hint_event=lambda e: None)
package = types.ModuleType('semantic_offload')
package.connector = connector
sys.modules['semantic_offload'] = package
sys.modules['semantic_offload.connector'] = connector


def execute(nodes, filename, namespace):
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(filename), 'exec', flags=__future__.annotations.compiler_flag), namespace)


source = Path(os.environ.get('CANDIDATE_SOURCE', ROOT / 'semantic_offload/connector.py'))
tree = ast.parse(source.read_text())
klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SemanticOffloadingConnectorScheduler')
methods = [n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name in {'_retry_wait_prefetches', '_try_publish_wait_prefix'}]
klass.bases = []
klass.body = methods
execute([klass] + [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_observed_prefix_digest'], source, connector.__dict__)
compat = ROOT / 'semantic_offload/_vllm_compat.py'
execute([n for n in ast.parse(compat.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in {'config_blocks_per_chunk', 'group_config_tokens_per_block', 'group_config_tokens_per_chunk'}], compat, connector.__dict__)
SemanticOffloadingConnectorScheduler = connector.SemanticOffloadingConnectorScheduler
names = {'test_wait_nomination_without_candidates_still_expires_hints', 'test_native_adoption_digest_reads_only_prefix', 'test_wait_nomination_nonempty_keeps_decision_and_attempt'}
tests = ROOT / 'tests/test_step_1_5_prefetch.py'
execute([n for n in ast.parse(tests.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names], tests, globals())
