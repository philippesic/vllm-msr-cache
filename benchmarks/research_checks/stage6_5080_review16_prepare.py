"""Prepare the completed Ministral B06 inventory and isolated review 16 once."""
from pathlib import Path
import sys
repo=Path(__file__).resolve().parents[2]
source=repo/'output/stage6-5080-20261108/primary-execution-bundle-01/source'
assert Path.cwd()==source.resolve()
assert sys.executable=='/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python'
assert sys.dont_write_bytecode,'Set PYTHONDONTWRITEBYTECODE=1'
sys.path[:]=[str(source)]+[v for v in sys.path if Path(v).resolve()!=Path(__file__).resolve().parent]

def main():
 
 from pathlib import Path
 from unittest.mock import patch
 import hashlib,json
 from benchmarks.serving_suite.campaign_contracts import load,digest,atomic_json,local_path,Ledger,verify_tree
 from benchmarks.serving_suite.campaign_validity import measurement_validity
 from benchmarks.serving_suite.campaign_report import completed_blocks
 from benchmarks.serving_suite import campaign_quality as q
 repo=Path('/home/philip/vllm-semantic-cache-ext')
 out=repo/'output/stage6-5080-20261108/linux-volume-01/primary-20261109'
 base=repo/'output/stage6-5080-20261109';e=repo/'evidence/stage6-5080-20261109'
 workspace=base/'quality-confirm-workspace-16';dest=base/'quality-confirm-checkpoint-16'
 assert not workspace.exists()and not dest.exists()
 live=out/'analysis/quality'
 def inventory(p):
  return {str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest()for f in p.rglob('*')if f.is_file()}if p.exists()else{}
 before=inventory(live)
 lock=load(out/'protocol.lock.json');ledger=Ledger(out)
 def cell_entries():
  return [v for v in ledger.index().values()if(v['details']['identity']['phase'],v['details']['identity']['model'],v['details']['identity']['cell'])==('confirm','ministral3_3b','B06')]
 events=cell_entries();event_digest=digest(events)
 expected={(v['repeat'],s)for v in lock['schedule']if(v['model'],v['cell'])==('ministral3_3b','B06')for s in v['systems']}
 assert len(expected)==len(events)==12 and {(v['details']['identity']['repeat'],v['details']['identity']['system'])for v in events}==expected
 assert all(v['state']in('verified','failed','interrupted')for v in events)
 selected=completed_blocks(out,'confirm')
 cell_blocks={k:v for k,v in selected.items()if k[:2]==('ministral3_3b','B06')}
 assert set(k[2]for k in cell_blocks)=={1,2,4,5}
 selected_attempts={x['event']['attempt_id']for v in cell_blocks.values()for x in v['systems'].values()}
 known={x['path']for path in e.glob('confirmation-ministral-b06-inactive-restore-*.json')for x in load(path)['attempts']}
 rows=[];invalid=set()
 for event in events:
  details=event['details'];identity=details['identity'];d=out/details['path'];case=load(d/'measurement/result.json')['cases'][0];a=case['campaign'];corpus=load(d/'measurement/corpus.json');validity=measurement_validity(d,event)
  assert identity['source']==lock['source']['digest']and case['summary']['completed']==case['expected_requests']==64 and case['summary']['failed']==0 and case['cleanup']['verified']and not a['measurement_errors']
  assert a['tools']==sum(n['kind']=='tool'for s in corpus['sessions']for n in s['graph']['nodes'])==48 and a['completed_units']==a['offered_units']==len(corpus['sessions'])==16
  requests=[json.loads(line)for line in(d/'measurement'/case['name']/'requests.jsonl').read_text().splitlines()]
  keys=[(x['wave'],x['source_id'],x['node_id'])for x in requests];ek={(0,s['id'],n['id'])for s in corpus['sessions']for n in s['graph']['nodes']if n['kind']=='request'}
  assert len(keys)==len(set(keys))==64 and set(keys)==ek and all(x['success']and x['terminal_complete']for x in requests)
  if not validity['valid']:
   invalid.add(details['path']);assert details['path']in known
  rows.append(dict(attempt=event['attempt_id'],identity={k:identity[k]for k in('model','cell','repeat','system')},path=details['path'],state=event['state'],block_id=details['block_id'],seal=details['seal'],verified_files=verify_tree(d,details['seal']),validity=validity,completed_requests=case['summary']['completed'],output_tokens=case['summary']['output_tokens'],length_caps=case['summary']['finish_reasons'].get('length',0),tools=a['tools'],completed_units=a['completed_units'],request_graph_coverage_verified=True,cleanup_verified=case['cleanup']['verified'],activity=case['activity'],native_cache=a['native_cache'],selected_for_matched_comparison=event['attempt_id']in selected_attempts))
 assert invalid==known and len(selected_attempts)==8 and sum(x['validity']['valid']for x in rows)==10
 assert digest(cell_entries())==event_digest
 finding=dict(schema='stage6-5080-terminal-cell-inventory-v1',model='ministral3_3b',cell='B06',execution_complete=True,acquisition_complete=False,planned_configurations=12,terminal_logical_configurations=12,retained_attempts=12,valid_measurements=10,required_matched_blocks=6,complete_matched_blocks=4,missing_repeats=[0,3],requests=sum(x['completed_requests']for x in rows),tools=sum(x['tools']for x in rows),output_caps=sum(x['length_caps']for x in rows),verified_files=sum(x['verified_files']for x in rows),all_request_tool_unit_coverage_and_cleanup_verified=True,activity_retry=False,source_digest=lock['source']['digest'],protocol_digest=digest(lock),ledger_model_cell_digest=event_digest,inventory_stable=True,executed_helper='benchmarks/research_checks/stage6_5080_review16_prepare.py',executed_helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),attempts=rows,scope='All six original Ministral B06 repetitions ran exactly once; all twelve outer seals, request graphs, tools, workload units and cleanup verify. Candidate repeats0 and3 remain activity-invalid, with their valid ARC peers unmatched. Only four complete frozen blocks receive comparison credit. No rule/capacity change, imputation, repeat reduction or favorable-activity retry. Every capped output remains subject to all-attempt review. Other cells/phases and final audits continue.')
 target=e/'confirmation-ministral-b06-attempts-complete.json'
 if target.exists():assert load(target)['ledger_model_cell_digest']==event_digest
 else:atomic_json(target,finding)
 print(json.dumps({k:v for k,v in finding.items()if k not in('attempts',)},indent=2))
 with patch.object(q,'completed_blocks',lambda ignored,phase:selected if phase=='confirm'else None),patch.object(q,'local_path',lambda ignored,relative:local_path(out,relative)):
  summary=q.prepare_review(workspace,'confirm')
 (workspace/'analysis/quality').rename(dest)
 (workspace/'analysis').rmdir();workspace.rmdir()
 assert inventory(live)==before
 idxs=[load(e/'screen-review-index-07.json')]+[load(e/f'confirm-review-index-{i:02}.json')for i in range(1,16)]
 fields=idxs[0]['signature_fields'];known={x['signature']for idx in idxs for x in idx['contexts']}
 packet=load(dest/'blind-packet.json');novels=[];seen=set()
 for x in packet['items']:
  sig=digest({k:x[k]for k in fields})
  if sig not in known and sig not in seen:novels.append(x)
  seen.add(sig)
 atomic_json(dest/'novel-contexts.json',novels)
 source=Path(q.__file__).resolve();assert source.is_relative_to(Path.cwd().resolve())
 atomic_json(dest/'acquisition-scope.json',dict(phase='confirm',acquisition_complete=False,status='partial ongoing acquisition',objective_outputs=summary['objective_outputs'],selected_items=summary['blind_review_items'],matched_blocks=len(selected),novel_full_contexts=len(novels),live_quality_directory_unchanged=True,quality_module_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_io_routing='Frozen completed_blocks snapshot and original sealed attempt paths; only derived JSON writes isolated and materialized at this new snapshot.',scope='All six Ministral B06 repetitions executed exactly once; four of six pairs selected, repeats0 and3 activity-invalid. All earlier Qwen3-8B blocks retain frozen selection and first-B06 exclusion. Secondary acquisition continues. New contexts unread/unjudged until explicit review. No source/protocol/measurement/live-quality changes.'))
 print(json.dumps(dict(summary={k:summary[k]for k in ['objective_outputs','blind_review_items','automatic_failures','capped_intermediate_outputs']},distinct_contexts=len(seen),novel_contexts=len(novels),blocks=len(selected),novel_kinds=[x['reference'].get('kind','investigation')for x in novels]),indent=2))
 

if __name__=='__main__':
 main()

