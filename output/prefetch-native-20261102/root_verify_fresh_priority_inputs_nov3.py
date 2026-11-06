from pathlib import Path
import json,gzip,hashlib,datetime,collections
H=Path(__file__).resolve().parent;R=H.parents[1];D=H/'fresh-agent-priority-nov3';E=R/'evidence/live-optimization-2080ti-20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((D/'plan.json').read_text());v=json.loads((D/'validation.json').read_text());iv=json.loads((D/'independent-validation.json').read_text());s=json.loads((D/'path-selection.json').read_text());a=json.loads((D/'acquisition.json').read_text());c=json.loads((D/'session-corpus.json').read_text());api=json.loads((D/'dataset-api.json').read_text())
for n,h in plan['file_pins'].items():assert sha(R/n)==h,n
for n,h in v['files'].items():assert sha(H/n)==h,n
assert iv['validation_sha256']==sha(D/'validation.json') and iv['corpus_sha256']==sha(D/'session-corpus.json') and iv['plan_sha256']==sha(D/'plan.json')
assert v['model_generation_performed'] is False and iv['no_model_generation'] is True and plan['no_model_generation'] is True and a['model_generation_performed'] is False
assert api['sha']==s['revision']==plan['revision']=='773748a7c1222e8a642a7059821498e14293562a'
paths=sorted([x['rfilename'] for x in api['siblings'] if x['rfilename'].startswith('swebench_verified_raw/gpt-5-mini/') and x['rfilename'].endswith('.traj.json')],key=lambda x:hashlib.sha256(x.encode()).hexdigest());excluded=set(s['all_excluded_paths']);ids=set(s['all_prior_source_ids']);assert len(excluded)==288 and len(ids)==360
expected=[p for p in paths if p not in excluded and Path(p).name.removesuffix('.traj.json') not in ids][:96];assert expected==s['candidate_paths'] and len(expected)==96
normalized=[json.loads((H/r['file']).read_text()) for r in a['records']];assert [x['path'] for x in normalized]==expected and all(x.get('reject') is None for x in a['records'])
for x in normalized:
 assert x['source_id'] not in ids and x['path'] not in excluded and 0<x['response_bytes']<=2000000 and x['http_status']==200
 assert hashlib.sha256(x['turns'][0]['user_text'].encode()).hexdigest() not in s['prior_normalized_issue_hashes']
 assert set(x)<=set(['path','url','response_sha256','response_bytes','http_status','revision_header','source_id','turns'])
assert len(c['sessions'])==72 and [x['source_id'] for x in c['sessions']]==[x['source_id'] for x in normalized[:72]]
assert collections.Counter(x['seed_partition'] for x in c['sessions'])=={14009:24,15013:24,16001:24}
for x in c['sessions']:
 assert len(x['turns'])>=3 and sum(len(t['user_token_ids']) for t in x['turns'])<=6000
 for i,t in enumerate(x['turns']):assert 0<len(t['user_token_ids'])<=(3072 if i==0 else 2048) and t['retained_user_tokens']==len(t['user_token_ids'])
seen=set();prior=0;corpus_path=(D/'session-corpus.json').resolve()
for p in (H/'runs').glob('live-nov3*/raw.json*'):
 if p.name not in ('raw.json','raw.json.gz'):continue
 b=p.read_bytes();r=json.loads(gzip.decompress(b) if p.suffix=='.gz' else b);prior+=1
 used=r.get('args',{}).get('corpus_path')
 if used:assert (R/used).resolve()!=corpus_path,str(p)
 seen.update(r.get('workload',{}).get('source_ids',[]))
assert not seen.intersection(x['source_id'] for x in c['sessions'])
out={'schema':'root-independent-fresh-priority-input-verification-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'corpus_sha256':sha(D/'session-corpus.json'),'input_plan_sha256':sha(D/'plan.json'),'input_plan_pins_checked':len(plan['file_pins']),'validation_pins_checked':len(v['files']),'prior_campaign_raw_records_scanned':prior,'exact_next96_order_independently_reconstructed':True,'all96priorIDs_and_normalized_issues_disjoint':True,'first72_selected_with_exact_partitions':True,'retained_integer_token_windows_and_budgets_verified':True,'independent_fast_tokenizer_receipt_bound':True,'selected_IDs_disjoint_from_all_prior_campaign_actual_runs':True,'new_corpus_not_used_by_prior_actual_runs':True,'no_generation_or_tools_receipts_bound':True,'limits':['ROOT rechecked retained token windows/budgets, but did not invoke tokenizer; two separately implemented upstream tokenizer validations are hash-bound.','Public repository overlap and semantic near-duplicates remain possible.','No model output or service quality qualification has occurred on fresh inputs.']};(E/'fresh-priority-inputs-root-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
