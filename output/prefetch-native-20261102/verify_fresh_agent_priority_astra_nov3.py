"""Independent input inventory/token-window checks, no model or network calls."""
import hashlib,json,re
from pathlib import Path
from tokenizers import Tokenizer
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'fresh-agent-priority-nov3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def th(s):return hashlib.sha256(s.encode()).hexdigest()
def text(v):return v if isinstance(v,str) else '\n'.join(str(x.get('text','')) for x in v if isinstance(x,dict)) if isinstance(v,list) else ''
def issue(v):
 s=re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b','[EMAIL REDACTED]',text(v));s=re.sub(r'(?<!\w)(?:\+\d[\d ()-]{10,}\d)(?!\w)','[PHONE REDACTED]',s).strip();m=re.search(r'<pr_description>(.*?)</pr_description>',s,re.S);assert m;body=m[1].strip().removeprefix('Consider the following PR description:').strip();return 'Source repository issue (quoted evidence):\n<repository_issue>\n'+body+'\n</repository_issue>'
def main():
 plan=json.loads((D/'plan.json').read_text());sel=json.loads((D/'path-selection.json').read_text());acq=json.loads((D/'acquisition.json').read_text());built=json.loads((D/'session-corpus.json').read_text());validation=json.loads((D/'validation.json').read_text())
 assert all((ROOT/p).is_file() and sha(ROOT/p)==v for p,v in plan['file_pins'].items());assert all(sha(H/p)==v for p,v in validation['files'].items())
 meta=json.loads((D/'dataset-api.json').read_text());assert meta['sha']==plan['revision'] and sha(D/'dataset-api.json')==sel['api_sha256'];universe=sorted([x['rfilename'] for x in meta['siblings'] if x['rfilename'].startswith('swebench_verified_raw/gpt-5-mini/') and x['rfilename'].endswith('.traj.json')],key=th)
 oldpaths=set(sel['original96_excluded'])|set(sel['prior_fresh96_excluded'])|set(sel['latest_gate96_excluded']);assert len(oldpaths)==288;expected=[p for p in universe if p not in oldpaths and Path(p).name.removesuffix('.traj.json') not in set(sel['all_prior_source_ids'])][:96];assert expected==sel['candidate_paths']
 ids=set();issues=set();oldroot=H.parent/'quantized-realworld-20261101';manifest=json.loads((oldroot/'data/swe-source-manifest.json').read_text())
 for row in manifest['files']:
  raw=json.loads((oldroot/row['local_path']).read_text());ids.add(raw['instance_id']);initial=next(m for m in raw['messages'] if m.get('role')=='user');issues.add(issue(initial['content']))
 oldacq=json.loads((H/'fresh-agent-nov3/acquisition.json').read_text())
 for row in oldacq['records']:
  old=json.loads((H/row['file']).read_text());ids.add(old['source_id']);issues.add(old['turns'][0]['user_text'])
 latestacq=json.loads((H/'fresh-agent-gate-nov3/acquisition.json').read_text())
 for row in latestacq['records']:
  old=json.loads((H/row['file']).read_text());ids.add(old['source_id']);issues.add(old['turns'][0]['user_text'])
 prior_tokens=set()
 for p in [H/'prepared/session-corpus.json',H/'prepared/triage-corpus-v2.json',H/'fresh-agent-nov3/session-corpus.json',H/'fresh-agent-gate-nov3/session-corpus.json']:
  old=json.loads(p.read_text())
  for s in old['sessions']:
   ids.add(s['source_id']);prior_tokens.add(tuple(s['turns'][0]['user_token_ids']))
   if old.get('normalization_profile')=='repository_issue_evidence_v2':issues.add(s['turns'][0]['user_text'])
 tok=Tokenizer.from_file(str(Path(plan['tokenizer'])/'tokenizer.json'));prior_tokens.update(tuple(tok.encode(x,add_special_tokens=False).ids[:3072]) for x in issues);oldhash={th(x) for x in issues};normalized=[]
 for i,row in enumerate(acq['records']):
  p=H/row['file'];assert sha(p)==row['sha256'];r=json.loads(p.read_text());assert row['reject'] is None and r.get('reject') is None;assert r['path']==expected[i] and r['source_id']==Path(r['path']).name.removesuffix('.traj.json');assert r['source_id'] not in ids and th(r['turns'][0]['user_text']) not in oldhash;assert 0<r['response_bytes']<=2000000 and r['http_status']==200;assert r['revision_header'] in [None,plan['revision']]
  assert set(r)<=set(['path','url','source_id','response_sha256','response_bytes','http_status','revision_header','turns']);assert r['turns'][0]['content_kind']=='recorded_repository_issue' and all(t['content_kind']=='external_recorded_tool_observation' for t in r['turns'][1:]);assert len(r['turns']) in [3,4];assert [t['source_message_index'] for t in r['turns']]==sorted(t['source_message_index'] for t in r['turns']);normalized.append(r)
 assert len(normalized)==len({r['source_id'] for r in normalized})==len({th(r['turns'][0]['user_text']) for r in normalized})==96
 assert [s['source_id'] for s in built['sessions']]==[r['source_id'] for r in normalized[:72]];seen=set()
 for i,s in enumerate(built['sessions']):
  assert s['seed_partition']==[14009,15013,16001][i%3];remaining=6000
  for j,t in enumerate(s['turns']):
   expected_ids=tok.encode(normalized[i]['turns'][j]['user_text'],add_special_tokens=False).ids[:min(3072 if j==0 else 2048,remaining)];assert expected_ids==t['user_token_ids'];remaining-=len(expected_ids)
  key=tuple(s['turns'][0]['user_token_ids']);assert key not in prior_tokens and key not in seen;seen.add(key);assert len(s['turns'])>=3
 out={'schema':'astra-independent-fresh-priority-input-verification-v1','plan_sha256':sha(D/'plan.json'),'source_selection_before_response_verified':True,'fixed96candidate_order_verified':True,'responses_within_bound':96,'source_IDs_and_normalized_issues_disjoint':True,'all288_prior_acquired_paths_excluded':True,'prior_source_ID_count':len(ids),'prior_distinct_normalized_issue_count':len(issues),'prior_retained_first_token_variant_count':len(prior_tokens),'selected72_are_first72_eligible':True,'retained_first_inputs_disjoint':True,'independent_fast_tokenizer_window_checks_passed':True,'no_historical_assistant_message_fields_retained':True,'no_model_generation':True,'no_tool_execution':True,'partitions':{str(seed):sum(s['seed_partition']==seed for s in built['sessions']) for seed in built['seeds']},'validation_sha256':sha(D/'validation.json'),'corpus_sha256':sha(D/'session-corpus.json'),'limits':['Exact normalized/token input disjointness does not rule out semantically similar issues or repository overlap.','Response hashes bind fetched pinned-URL bytes; full raw trajectories intentionally not retained.','No output-based input selection or quality/task-success validation.']}
 (D/'independent-validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
