import sys,importlib.abc,importlib.util
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[1];sys.path[:0]=[str(H/'sources/nov3-advice-promotion'),str(R)]
class Overlay(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='vllm.v1.core.block_pool':return importlib.util.spec_from_file_location(fullname,H/'runtime-evictable-warm/vllm/v1/core/block_pool.py')
sys.meta_path.insert(0,Overlay())
import semantic_offload.manager as m
assert Path(m.__file__).is_relative_to(H/'sources/nov3-advice-promotion')
import pytest
sys.exit(pytest.main(['-q','--tb=short','-p','no:cacheprovider','--junitxml='+str(H/'live-nov3-advice-promotion-tests.xml'),str(H/'test_advice_promotion_nov3.py'),'tests/test_completed_transcript.py','tests/test_step_1_5_prefetch.py','tests/test_cpu_physical_refs.py','tests/test_score_residency_cleanup.py']))
