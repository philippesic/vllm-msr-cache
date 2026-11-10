"""Bind and consolidate the explicitly reviewed, partial all-attempt cap snapshot07.

Run from the project root with the pinned interpreter and bytecode disabled.
Use --check to independently reproduce the retained derived evidence.
"""
import sys
from pathlib import Path
from collections import defaultdict,Counter
import json,copy,re
from benchmarks.serving_suite.campaign_contracts import load,digest,atomic_json,Ledger,local_path
from benchmarks.serving_suite.campaign_quality import RUBRIC,objective
from benchmarks.serving_suite.campaign_capped_review import instruction_source
repo=Path('/home/philip/vllm-semantic-cache-ext');e=repo/'evidence/stage6-5080-20261109';base=repo/'output/stage6-5080-20261109';b=base/'caps-all-checkpoint-07';out=repo/'output/stage6-5080-20261108/linux-volume-01/primary-20261109'
assert Path.cwd()==repo and sys.dont_write_bytecode
assert sys.executable=='/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python'
assert sys.argv[1:]in([],['--check'])
check=bool(sys.argv[1:])
for filename in ('caps-all-source-judgments-07.json','caps-instruction-bindings-07.json','caps-all-review-index-07.json'):
 assert (e/filename).exists()==check,'Use fresh destinations for creation, or --check for retained evidence'
packet=load(b/'blind-packet.json');summary=load(b/'summary.json');pd=digest(packet['items']);assert pd==summary['packet_digest']and not summary['complete_inventory']
prior=load(e/'caps-all-review-index-06.json');fields=prior['quality_signature_fields'];extended=prior['signature_fields'];oldbind=load(e/'caps-instruction-bindings-06.json');system=oldbind['executed_system_instruction_digest'];sha=oldbind['executed_instruction_source_sha256']
quality={};qprovenance={}
for stem,no in [('screen','07')]+[('confirm',f'{i:02}')for i in range(1,17)]:
 idx=load(e/f'{stem}-review-index-{no}.json');by={r['review_id']:r for r in load(e/f'{stem}-source-judgments-{no}.json')['ratings']};assert idx['executed_system_instruction_digest']==system and idx['executed_instruction_source_sha256']==sha
 for ctx in idx['contexts']:
  r=by[ctx['canonical_review_id']]
  if ctx['signature']in quality:assert all(quality[ctx['signature']][k]==r[k]for k in ('source_supported','instruction_followed','useful'))
  quality[ctx['signature']]=r;qprovenance[ctx['signature']]=f'{stem}-review-index-{no}.json'
oldpacket=load(base/'caps-all-checkpoint-06/blind-packet.json');assert digest(oldpacket['items'])==prior['packet_digest'];oldby={r['review_id']:r for r in load(e/prior['ratings'])['ratings']};oldext={digest({k:x[k]for k in extended}):oldby[x['review_id']]for x in oldpacket['items']};assert set(oldext)=={c['signature']for c in prior['contexts']}

novels=load(b/'novel-quality-contexts.json');assert len(novels)==24
manual={}
for no in range(1,7):
 part=load(e/f'caps-new-judgments-07-part-{no:02}.json')
 assert part['packet_digest']==pd and part['signature_fields']==extended and part['quality_signature_fields']==fields
 for r in part['ratings']:
  i=r['index'];assert i not in manual;manual[i]=r
assert set(manual)==set(range(len(novels)))
newquality={}
for i,x in enumerate(novels):
 r=manual[i];q=digest({k:x[k]for k in fields})
 assert r['review_id']==x['review_id']and r['quality_signature']==q and r['extended_signature']==digest({k:x[k]for k in extended})
 assert r['whitespace_words']==len(x['answer'].split())and q not in quality and q not in newquality
 assert all(type(r[k])is bool for k in ('source_supported','instruction_followed','useful'))
 assert isinstance(r['unsupported_claims'],list)and r['notes']
 newquality[q]=r
components=load(e/'caps-review-components-07.json')
assert components['packet_digest']==pd and len(components['contexts'])==24
inputs={'cap06':oldpacket['items']}
for no in range(2,17):inputs[f'confirm{no:02}']=load(base/f'quality-confirm-checkpoint-{no:02}/novel-contexts.json')
assert components['old_input_packet_digests']=={k:digest(v)for k,v in inputs.items()}
for i,(x,c)in enumerate(zip(novels,components['contexts'])):
 assert c['index']==i and c['review_id']==x['review_id']and set(c['components'])==set(extended)-{'answer'}
 def prior_item(p):
  if p['snapshot']=='cap07':
   assert p['index']<i;return novels[p['index']]
  return inputs[p['snapshot']][p['index']]
 for k,v in c['components'].items():
  assert v['digest']==digest(x[k])
  if v['prior_explicitly_read']:
   p=v['prior_explicitly_read'];assert p['field']==k and prior_item(p)[k]==x[k]
 if not c['components']['actual_dependencies']['prior_explicitly_read']:
  assert len(c['dependency_components'])==len(x['actual_dependencies'])
  for j,v in enumerate(c['dependency_components']):
   assert v['dependency']==j and v['digest']==digest(x['actual_dependencies'][j])
   if v['prior_explicitly_read']:
    p=v['prior_explicitly_read'];assert prior_item(p)['actual_dependencies'][p['dependency']]==x['actual_dependencies'][j]

labels=load(b/'private-labels.json');events=Ledger(out).index();cache={};bindings=defaultdict(set)
for x in packet['items']:
 label=labels[x['review_id']];a=label['attempt']
 if a not in cache:
  ev=events[a];d=local_path(out,ev['details']['path']);m=d/'measurement';snippet,actual=instruction_source(m);assert actual==sha and digest(snippet)==system
  corpus=load(m/'corpus.json');sessions={s['id']:(slot,s)for slot,s in enumerate(corpus['sessions'])};rows=[]
  for f in m.glob('*/requests.jsonl'):
   rows.extend((json.loads(line),f.parent)for line in f.read_text().splitlines()if line.strip())
  cache[a]=(snippet,sessions,rows)
 snippet,sessions,raw=cache[a];slot,session=sessions[x['source_id']];node=next(n for n in session['graph']['nodes']if n['id']==x['node_id']);final=x['node_id']in session['graph']['outputs']
 assert x['executed_system_instruction_source']==snippet and x['task_uses_tools']==bool(session.get('tools'))and x['instructions']==session['turns']and x['source_context']==session.get('context','')and x['current_instruction']==node['prompt']and x['response_role']==('final'if final else'capped_intermediate')
 ref=dict(session['quality_reference'])
 if label['cell']=='B10':ref['kind']='team'
 if not final:ref['kind']='intermediate'
 assert x['reference']==ref and x['objective']==objective(x['answer'],ref,capped=True)
 matches=[(j,v,case)for j,(v,case)in enumerate(raw)if v.get('payload',{}).get('request_id')==label['request_id']];assert len(matches)==1;j,actual,case=matches[0]
 assert actual['generated_text']==x['answer']and actual['finish_reason']=='length'and actual['node_id']==x['node_id']and actual['source_id']==x['source_id']
 assert x['instruction_context_digest']==digest(dict(prompt=actual['payload']['prompt'],workload_source=sha))
 nodes={n['id']:n for n in session['graph']['nodes']};ancestors=set();pending=list(node.get('depends_on',[]))
 while pending:
  ancestor=pending.pop()
  if ancestor not in ancestors:ancestors.add(ancestor);pending.extend(nodes[ancestor].get('depends_on',[]))
 deps=[v['generated_text']for v,_ in raw[:j]if v.get('source_id')==session['id']and v.get('wave',0)==actual.get('wave',0)and v.get('node_id')in ancestors]
 assert x['actual_dependencies']==deps
 tools=[dict(output=re.sub(r'\bin \d+(?:\.\d+)?s\b','in [elapsed time omitted]',p.read_text(errors='replace')))for p in sorted((case/'tools').glob(f'{actual.get("wave",0)}-{slot}-*.txt'))if p.stem.split('-',2)[2]in ancestors]
 assert x['tools']==tools
 bindings[digest({k:x[k]for k in fields})].add(x['instruction_context_digest'])
rows=[];groups={};sources=Counter()
for x in packet['items']:
 sig=digest({k:x[k]for k in extended});q=digest({k:x[k]for k in fields})
 if sig in oldext:r=oldext[sig];source='caps-all-review-index-06.json'
 elif q in quality:r=quality[q];source=qprovenance[q]
 else:r=newquality[q];source='caps-new-judgments-07-parts-01-through-06'
 v={k:copy.deepcopy(r[k])for k in ('source_supported','instruction_followed','useful','unsupported_claims','notes')};v['review_id']=x['review_id'];v['notes']+=' All-cap snapshot07 exact reuse from '+source+'. Archived instruction, sealed corpus/node/reference/role/tool flag, actual preceding dependencies/tools, raw answer and prompt/workload digest independently rechecked. Tokens were bound, not individually decoded.'
 assert all(type(v[k])is bool for k in ('source_supported','instruction_followed','useful'))and isinstance(v['unsupported_claims'],list)and v['notes']
 rows.append(v);groups.setdefault(sig,[]).append(x['review_id']);sources[source]+=1
assert len(rows)==summary['capped_responses']==418 and len(groups)==summary['unique_contexts']==175 and len({x['review_id']for x in rows})==418
ratings=dict(schema='stage6-source-audit-v1',reviewer='Codex exact extended-context or previously explicit full-quality-context reuse; archived instructions and sealed actual dependencies/tools/corpus/node/prompt bindings independently checked; 24 novel contexts explicitly read and manually rated; no automatic novel rating',rubric_digest=digest(RUBRIC),ratings=rows)
findings=dict(source_unsupported=sum(not r['source_supported']for r in rows),instruction_failures=sum(not r['instruction_followed']for r in rows),not_useful=sum(not r['useful']for r in rows))
binding_record=dict(schema='stage6-5080-cap-instruction-bindings-v1',packet_digest=pd,executed_instruction_source_sha256=sha,executed_system_instruction_digest=system,archived_instruction_sources_verified=len(cache),instruction_wrapper_reuse='Same complete archived workload source and previously inspected SYSTEM/system_instruction/request wrapper as cap06. Each task branch, corpus/node and raw prompt/workload binding rechecked.',checks=dict(all_system_sources_match=True,all_tool_branch_flags_match_sealed_corpus=True,all_source_reference_instruction_role_node_bindings_match_sealed_corpus=True,all_objective_bindings_match=True,all_actual_dependencies_match_sealed_preceding_requests=True,all_actual_tools_match_sealed_ancestor_outputs=True,all_prompt_source_binding_digests_recomputed_from_raw_requests=True,quality_contexts=len(bindings),maximum_prompt_binding_digests_per_quality_context=max(map(len,bindings.values()))),scope='Partial all-attempt cap inventory includes ordinary failed/interrupted/unmatched arms. Token prompts bound, not individually decoded. Acquisition remains active; not final inventory.')
index=dict(schema='stage6-5080-partial-all-cap-review-v1',acquisition_complete=False,complete_inventory=False,source_review_complete_for_snapshot=True,raw_packet=str((b/'blind-packet.json').resolve()),packet_digest=pd,signature_fields=extended,quality_signature_fields=fields,executed_system_instruction_digest=system,contexts=[dict(signature=sig,canonical_review_id=ids[0],review_ids=ids)for sig,ids in groups.items()],ratings='caps-all-source-judgments-07.json',instruction_bindings='caps-instruction-bindings-07.json',reviewed_outputs=418,distinct_extended_contexts=len(groups),distinct_quality_contexts=len(bindings),novel_quality_contexts=24,reuse_sources=dict(sources),review_findings=findings,phases=summary['phases'],roles=summary['roles'],scope='418 sealed ordinary capped responses:132 screen/286 confirm,370 intermediates/48 finals;175 extended contexts. Prior cap06 exact extended-context reuse and screen07/confirm01-through16 quality union reuse were checked; all24 novel full contexts (19 intermediates/5 finals) explicitly read and manually rated. Every cap-bearing archived instruction, sealed corpus/node/reference/current role/tool branch, actual preceding dependencies/tools, objective and raw prompt/workload digest independently checked. Includes ordinary failed/interrupted/unmatched arms, including genuine invalid Ministral B06 attempts. Counts include duplicates and are not population rates. One attempt pending; final inventory must be regenerated after all phases are terminal.')
for name,value in [('caps-all-source-judgments-07.json',ratings),('caps-instruction-bindings-07.json',binding_record),('caps-all-review-index-07.json',index)]:
 target=e/name
 if check:assert load(target)==value,name
 else:atomic_json(target,value)
print(json.dumps(dict(novel_findings=dict(source_unsupported=sum(not r['source_supported']for r in newquality.values()),instruction_failures=sum(not r['instruction_followed']for r in newquality.values()),not_useful=sum(not r['useful']for r in newquality.values())),reviewed=418,contexts=len(groups),new_quality_contexts=24,findings=findings,reuse_sources=dict(sources),archived_instruction_sources_verified=len(cache),max_prompt_bindings=max(map(len,bindings.values())),complete_inventory=False),indent=2))
