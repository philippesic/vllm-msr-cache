import importlib.abc,importlib.util,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
class Overlay(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='vllm.v1.core.block_pool':
   return importlib.util.spec_from_file_location(fullname,ROOT/'output/prefetch-native-20261102/runtime-evictable-warm/vllm/v1/core/block_pool.py')
sys.meta_path.insert(0,Overlay())
import pytest
sys.exit(pytest.main(['-q','--tb=short','-p','no:cacheprovider','--junitxml='+str(ROOT/'output/prefetch-native-20261102/live-nov3-integrated-gpu-gate-tests.xml'),
 'tests/test_gpu_gated_cpu_retention.py','tests/test_preservation_audit_mode.py','tests/test_wait_hint_cpu_recency.py','tests/test_completed_transcript.py','tests/test_step_1_5_prefetch.py','tests/test_cpu_physical_refs.py','tests/test_preservation_mapping.py','tests/test_summary_offload_overlap.py','tests/test_batched_summaries.py','tests/test_scoring_schedule.py','tests/test_mean_only_summaries.py','tests/test_full_prefix_nomination.py','tests/test_score_residency_cleanup.py','tests/test_async_scoring.py']))
