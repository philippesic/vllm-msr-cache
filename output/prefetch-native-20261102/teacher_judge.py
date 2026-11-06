"""Exact inputs, recorded output IDs and complete identities, no exemptions."""
import copy
def validate(raw,trace,**unused):
 expected={(r['session_id'],r['turn']):r for r in trace['requests']}
 assert len(expected)==len(trace['requests'])==24 and len(raw['requests'])==24
 seen=set()
 for r in raw['requests']:
  key=(r['session_id'],r['turn']);assert key in expected and key not in seen;seen.add(key)
  ref=expected[key];body=r['payload'];ids=ref['reference_output_token_ids']
  assert r['success'] is True and ids and len(ids)<=96
  assert body['prompt']==ref['prompt']
  assert body['max_tokens']==len(ids) and body['ignore_eos'] is True
  assert body['vllm_xargs']['replay_token_ids']==ids
  assert r['output_token_ids']==ids
 assert seen==set(expected)
 return dict(passed=True,requests=24,prompt_tokens=sum(len(r['payload']['prompt']) for r in raw['requests']),decode_tokens=sum(len(r['output_token_ids']) for r in raw['requests']),exact_recorded_output_ids=True,ordinary_generation=False)
def reject_invalid_controls(raw,trace,**unused):
 for kind in ['prompt','output','short','duplicate']:
  wrong=copy.deepcopy(raw)
  if kind=='prompt':wrong['requests'][0]['payload']['prompt'][0]+=1
  if kind=='output':wrong['requests'][0]['output_token_ids'][0]+=1
  if kind=='short':wrong['requests'][0]['output_token_ids'].pop()
  if kind=='duplicate':wrong['requests'][-1]=copy.deepcopy(wrong['requests'][0])
  try:validate(wrong,trace)
  except AssertionError:continue
  raise AssertionError('known-wrong teacher replay '+kind+' accepted')
 return True
