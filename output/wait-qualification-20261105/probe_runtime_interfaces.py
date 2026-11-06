"""Read-only import/CPU-allocation boundary; no GPU model or port implied."""
import importlib,importlib.metadata,inspect,json,sys,traceback
report=dict(schema='wait-hint-runtime-interface-probe-v1',python=sys.executable,checks=[],scope='Imports, native method availability and four-entry CPU manager construction only; no GPU serving or compatibility qualification.')
try:
 import vllm
 report.update(vllm_file=vllm.__file__,distribution_version=importlib.metadata.version('vllm'))
except Exception as exc:
 report['runtime_import_error']=repr(exc)
for module,name,methods in [
 ('vllm.v1.core.block_pool','BlockPool',['publish_completed_observed_keys','publish_completed_prefix','publish_completed_suffix']),
 ('vllm.v1.core.kv_cache_manager','KVCacheManager',['get_computed_blocks_for_connector']),
 ('vllm.v1.core.sched.scheduler','Scheduler',['publish_kv_connector_wait_hint']),
 ('vllm.v1.kv_offload.cpu.manager','CPUOffloadingManager',['_get_num_free_blocks','_allocate_blocks','_free_block']),
 ('vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker','OffloadingConnectorWorker',[])]:
 entry=dict(module=module,symbol=name)
 try:
  cls=getattr(importlib.import_module(module),name);entry.update(imported=True,constructor=str(inspect.signature(cls)),required_methods={m:hasattr(cls,m) for m in methods})
 except Exception as exc:entry.update(imported=False,error=repr(exc),traceback=traceback.format_exc())
 report['checks'].append(entry)
for module in ['semantic_offload.manager','semantic_offload.spec','semantic_offload.worker','semantic_offload.connector']:
 entry=dict(module=module)
 try:importlib.import_module(module);entry['imported']=True
 except Exception as exc:entry.update(imported=False,error=repr(exc),traceback=traceback.format_exc())
 report['checks'].append(entry)
entry=dict(check='current_profile_four_entry_ARC_manager')
try:
 from semantic_offload.manager import PreservationOffloadingManager
 manager=PreservationOffloadingManager(num_blocks=4,cache_policy='arc',enable_events=True,store_threshold=0)
 entry.update(passed=True,manager_class=type(manager).__name__)
except Exception as exc:entry.update(passed=False,error=repr(exc),traceback=traceback.format_exc())
report['checks'].append(entry)
report['required_interfaces_present']=all(e.get('imported',True) and all(e.get('required_methods',{}).values()) and e.get('passed',True) for e in report['checks'])
if len(sys.argv)>1:
 from pathlib import Path
 Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
 print('Wrote runtime interface report:',sys.argv[1])
else:print(json.dumps(report,indent=2))
