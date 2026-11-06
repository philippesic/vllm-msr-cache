import fcntl,gzip,importlib.util,json,sys
from collections import Counter
from pathlib import Path
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('validator',H.parents[1]/'benchmarks/preservation.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
with open('/tmp/vllm-semantic-cache-gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 d=H/'runs/cpu-recency-v3-byte-qualified-c24-b2048-candidate-s1103';raw=json.load(gzip.open(d/'raw.json.gz'));s=json.loads((d/'summary.json').read_text());terms={t['external_request_id']:t for t in raw['terminals']};bad={tuple(q['identity']) for q in s['proofs']['failures']};rows=[]
 for req in raw['requests']:
  if (req['session_id'],req['turn']) not in bad:continue
  t=terms[req['payload']['request_id']];calls=[]
  def trace(f,e,a):
   if e=='return' and f.f_code.co_filename.endswith('preservation.py') and (a=='physical_block_metadata_unavailable' or isinstance(a,tuple) and 'physical_block_metadata_unavailable' in a):calls.append({'function':f.f_code.co_name,'line':f.f_lineno})
   return trace
  sys.settrace(trace);ans=m._map_one_preservation_event(t);sys.settrace(None)
  loaded=t['physical_block_ids']['loaded'];external=t['physical_block_ids']['external'];lr={(g,b) for _,g,b in loaded};er={(g,b) for _,g,b in external};dups=[list(k)+[v] for k,v in Counter((g,b) for _,g,b in loaded).items() if v>1]
  rows.append({'identity':[req['session_id'],req['turn']],'reason':ans[1],'physical_complete_flag':t['physical_block_identity_complete'],'preempted':t['preempted'],'preemption_count':t['preemption_count'],'admission_count':t['admission_count'],'duplicate_loaded_physical_count':len(dups),'duplicate_loaded_refs':dups,'external_without_load_refs':sorted(er-lr),'load_without_external_refs':sorted(lr-er),'validator_failure_locations':calls})
 out={'source':'unchanged strictvalidator, no waivers','audit_all_component_assertions_completed':raw['status']=='complete','strict_passed':s['proofs']['passed'],'failed_requests':rows,'disposition':'Rejected. Optimize audit readbacks and rerun all72; no performance inference from intrusive audit.'};(H/'qualification/cpu-recency-v3-audit-failure.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='failed_requests'},indent=2));print([(r['identity'],r['preemption_count'],r['duplicate_loaded_physical_count'],len(r['external_without_load_refs'])) for r in rows])
 del raw
 notes=[]
 for arm in ['arc','candidate']:
  name=f'cpu-recency-v3-common-curve-b1792-{arm}-s1103';r=json.load(gzip.open(H/'runs'/name/'raw.json.gz'));terms=r['terminals'];fields=['actual_prompt_tokens_locally_executed','loaded_tokens','admitted_local_tokens','admitted_external_tokens','recomputed_tokens','reused_tokens','preemption_count']
  notes.append({'name':name,**{k:sum(t[k] for t in terms) for k in fields}});del r
 (H/'qualification/cpu-recency-v3-lower-work.json').write_text(json.dumps(notes,indent=2)+'\n');print(json.dumps(notes,indent=2))
