import ast, math, time, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import torch
from semantic_offload import index
from semantic_offload.profiling import PROFILE, profiled
from semantic_offload.scoring_search import ClusterSearch, SearchConfig
ns = dict(vars(index), ClusterSearch=ClusterSearch, SearchConfig=SearchConfig, torch=torch, PROFILE=PROFILE, profiled=profiled, time=time, math=math,
          _TIMING=False, debug_print=lambda *a, **k:None, _METHOD_FIELDS={"mean":("mean",)},
          _EMA_ALPHA=.3, _EMA_RANK_POWER=1., make_offload_key=lambda value, group:value)
def execute(nodes):
    module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0)]+nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),"<actual-source-ast>","exec"),ns)
worker=ast.parse(Path("semantic_offload/worker.py").read_text())
cls=next(n for n in worker.body if isinstance(n,ast.ClassDef) and n.name=="SemanticOffloadingWorker")
cls.bases=[]
execute([cls])
manager=ast.parse(Path("semantic_offload/manager.py").read_text())
execute([n for n in manager.body if isinstance(n,ast.FunctionDef) and n.name in ("_rank_ema_weight","validate_ema_rank_weight_floor","compose_relevance_updates")])
helpers=ast.parse(Path("tests/test_step_1_3_scoring.py").read_text())
execute([n for n in helpers.body if isinstance(n,ast.FunctionDef) and n.name in ("_make_worker","_summary","to_key","_seed_durable_summaries")])
tests=ast.parse(Path("tests/test_scoring_schedule.py").read_text())
execute([n for n in tests.body if isinstance(n,ast.FunctionDef)])
for name, function in list(ns.items()):
    if name.startswith("test_") and name in ("test_ordinary_capture_does_not_score_even_with_resident_blocks", "test_decision_scores_only_eligible_rows_preserves_metric_and_feature_time", "test_stale_or_missing_snapshot_cannot_be_refreshed_by_scoring", "test_snapshots_are_detached_owned_and_bounded_and_do_not_require_dram", "test_index_unvisited_rows_are_absent_from_score_updates"):
        function()
        print(name,"PASS")
print("CUDA initialized:",torch.cuda.is_initialized())
