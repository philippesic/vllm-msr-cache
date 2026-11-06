"""Fixed-work integrity judge; separate from the unchanged KV proof validator."""
import copy,json
from pathlib import Path

def validate(raw,trace,output_cap=96,clients=None,turns=3):
 selected=[r for r in trace['requests'] if clients is None or (r['index']//4 < clients and r['turn'] < turns)]
 expected={(r['session_id'],r['turn']):r for r in selected}
 rows=raw['requests'];seen=set()
 assert len(rows)==len(expected), 'incomplete request set'
 for row in rows:
  key=(row['session_id'],row['turn']);assert key not in seen and key in expected,'duplicate/foreign request'
  seen.add(key);reference=expected[key];body=row['payload']
  assert row['success'] is True,'request failure'
  assert body['prompt']==reference['prompt'],'different prompt token IDs'
  assert body['max_tokens']==output_cap and body['ignore_eos'] is True,'different decode budget'
  assert len(row['output_token_ids'])==output_cap,'incomplete decode work'
 assert seen==set(expected)
 return dict(passed=True,requests=len(rows),prompt_tokens=sum(len(r['payload']['prompt']) for r in rows),decode_tokens=len(rows)*output_cap)

def reject_invalid_controls(raw,trace,**kwargs):
 for kind in ['prompt','decode','duplicate']:
  wrong=copy.deepcopy(raw)
  if kind=='prompt':wrong['requests'][0]['payload']['prompt'][0]+=1
  elif kind=='decode':wrong['requests'][0]['output_token_ids'].pop()
  else:wrong['requests'][-1]=copy.deepcopy(wrong['requests'][0])
  try:validate(wrong,trace,**kwargs)
  except AssertionError:continue
  raise AssertionError('known-wrong '+kind+' control accepted')
 return True
