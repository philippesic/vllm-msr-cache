"""Consolidate/reproduce partial all-attempt cap review08 with no novel ratings."""
from pathlib import Path
import sys
repo_root=Path(__file__).resolve().parents[2]
frozen_source=repo_root/'output/stage6-5080-20261108/primary-execution-bundle-01/source'
assert Path.cwd()==frozen_source.resolve()and sys.dont_write_bytecode
assert sys.executable=='/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python'
assert sys.argv[1:]in([],['--check'])
check=bool(sys.argv[1:])
sys.path[:]=[str(frozen_source)]+[v for v in sys.path if Path(v).resolve()!=Path(__file__).resolve().parent]

from pathlib import Path
from collections import defaultdict,Counter
import json,copy,re
from benchmarks.serving_suite.campaign_contracts import load,digest,atomic_json,Ledger,local_path
from benchmarks.serving_suite.campaign_quality import RUBRIC,objective
from benchmarks.serving_suite.campaign_capped_review import instruction_source
repo=Path('/home/philip/vllm-semantic-cache-ext');e=repo/'evidence/stage6-5080-20261109';base=repo/'output/stage6-5080-20261109';b=base/'caps-all-checkpoint-08';out=repo/'output/stage6-5080-20261108/linux-volume-01/primary-20261109'
assert out.parent.is_mount()
for name in ('caps-all-source-judgments-08.json','caps-instruction-bindings-08.json','caps-all-review-index-08.json'):
 assert (e/name).exists()==check,'Use new outputs for creation, or --check for retained evidence'
assert all(Path(f.__code__.co_filename).resolve().is_relative_to(frozen_source.resolve())for f in(load,objective,instruction_source))
packet=load(b/'blind-packet.json');summary=load(b/'summary.json');pd=digest(packet['items']);assert pd==summary['packet_digest']and not summary['complete_inventory']and len(summary['pending'])==1 and not summary['partial_stream_fragments']and not load(b/'novel-quality-contexts.json')
prior=load(e/'caps-all-review-index-07.json');fields=prior['quality_signature_fields'];extended=prior['signature_fields'];oldbind=load(e/'caps-instruction-bindings-07.json');system=oldbind['executed_system_instruction_digest'];sha=oldbind['executed_instruction_source_sha256']
quality={};qprovenance={}
for stem,no in [('screen','07')]+[('confirm',f'{i:02}')for i in range(1,18)]:
 idx=load(e/f'{stem}-review-index-{no}.json');by={r['review_id']:r for r in load(e/f'{stem}-source-judgments-{no}.json')['ratings']};assert idx['executed_system_instruction_digest']==system and idx['executed_instruction_source_sha256']==sha
 for ctx in idx['contexts']:
  r=by[ctx['canonical_review_id']]
  if ctx['signature']in quality:assert all(quality[ctx['signature']][k]==r[k]for k in ('source_supported','instruction_followed','useful'))
  quality[ctx['signature']]=r;qprovenance[ctx['signature']]=f'{stem}-review-index-{no}.json'
oldpacket=load(base/'caps-all-checkpoint-07/blind-packet.json');assert digest(oldpacket['items'])==prior['packet_digest'];oldby={r['review_id']:r for r in load(e/prior['ratings'])['ratings']};oldext={digest({k:x[k]for k in extended}):oldby[x['review_id']]for x in oldpacket['items']};assert set(oldext)=={c['signature']for c in prior['contexts']}

for x in oldpacket['items']:
 q=digest({k:x[k]for k in fields});r=oldby[x['review_id']]
 if q in quality:assert all(quality[q][k]==r[k]for k in ('source_supported','instruction_followed','useful'))
 else:quality[q]=r;qprovenance[q]='caps-all-review-index-07.json (quality fields)'
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
 if sig in oldext:r=oldext[sig];source='caps-all-review-index-07.json'
 else:r=quality[q];source=qprovenance[q]
 v={k:copy.deepcopy(r[k])for k in ('source_supported','instruction_followed','useful','unsupported_claims','notes')};v['review_id']=x['review_id'];v['notes']+=' All-cap snapshot08 exact reuse from '+source+'. Archived instruction, sealed corpus/node/reference/role/tool flag, actual preceding dependencies/tools, raw answer and prompt/workload digest independently rechecked. Tokens were bound, not individually decoded.'
 assert all(type(v[k])is bool for k in ('source_supported','instruction_followed','useful'))and isinstance(v['unsupported_claims'],list)and v['notes']
 rows.append(v);groups.setdefault(sig,[]).append(x['review_id']);sources[source]+=1
assert len(rows)==summary['capped_responses']==480 and len(groups)==summary['unique_contexts']==182 and len({x['review_id']for x in rows})==480
ratings=dict(schema='stage6-source-audit-v1',reviewer='Codex exact extended-context or previously explicit full-quality-context reuse; archived instructions and sealed actual dependencies/tools/corpus/node/prompt bindings independently checked; no novel automatic rating',rubric_digest=digest(RUBRIC),ratings=rows)
findings=dict(source_unsupported=sum(not r['source_supported']for r in rows),instruction_failures=sum(not r['instruction_followed']for r in rows),not_useful=sum(not r['useful']for r in rows))
binding_record=dict(schema='stage6-5080-cap-instruction-bindings-v1',packet_digest=pd,executed_instruction_source_sha256=sha,executed_system_instruction_digest=system,archived_instruction_sources_verified=len(cache),instruction_wrapper_reuse='Same complete archived workload source and previously inspected SYSTEM/system_instruction/request wrapper as cap07. Each task branch, corpus/node and raw prompt/workload binding rechecked.',checks=dict(all_system_sources_match=True,all_tool_branch_flags_match_sealed_corpus=True,all_source_reference_instruction_role_node_bindings_match_sealed_corpus=True,all_objective_bindings_match=True,all_actual_dependencies_match_sealed_preceding_requests=True,all_actual_tools_match_sealed_ancestor_outputs=True,all_prompt_source_binding_digests_recomputed_from_raw_requests=True,quality_contexts=len(bindings),maximum_prompt_binding_digests_per_quality_context=max(map(len,bindings.values()))),scope='Partial all-attempt cap inventory includes ordinary failed/interrupted/unmatched arms. Token prompts bound, not individually decoded. Acquisition remains active; not final inventory.')
index=dict(schema='stage6-5080-partial-all-cap-review-v1',acquisition_complete=False,complete_inventory=False,source_review_complete_for_snapshot=True,raw_packet=str((b/'blind-packet.json').resolve()),packet_digest=pd,signature_fields=extended,quality_signature_fields=fields,executed_system_instruction_digest=system,contexts=[dict(signature=sig,canonical_review_id=ids[0],review_ids=ids)for sig,ids in groups.items()],ratings='caps-all-source-judgments-08.json',instruction_bindings='caps-instruction-bindings-08.json',reviewed_outputs=480,distinct_extended_contexts=len(groups),distinct_quality_contexts=len(bindings),novel_quality_contexts=0,reuse_sources=dict(sources),review_findings=findings,phases=summary['phases'],roles=summary['roles'],scope='480 sealed ordinary capped responses:132 screen/348 confirm,418 intermediates/62 finals. All418 earlier cap07 occurrences remain included;473 current occurrences exactly match prior extended contexts, including55 added repetitions. Seven new extended contexts/occurrences exactly match previously explicitly reviewed full quality contexts in the screen07/confirm01-through17/cap07 union. Zero novel full quality context or automatic novel judgment. Every cap-bearing archived instruction, sealed corpus/node/reference/current role/tool branch, actual preceding dependencies/tools, objective and raw prompt/workload digest independently checked. Ordinary failed/interrupted/unmatched arms included without timing/objective changes. Counts include duplicates and are not population rates. One attempt pending; regenerate final inventory after every acquisition phase is terminal.')
for name,value in [('caps-all-source-judgments-08.json',ratings),('caps-instruction-bindings-08.json',binding_record),('caps-all-review-index-08.json',index)]:
 target=e/name
 if check:assert load(target)==value,name
 else:atomic_json(target,value)
print(json.dumps(dict(reviewed=480,contexts=len(groups),new_quality_contexts=0,findings=findings,reuse_sources=dict(sources),archived_instruction_sources_verified=len(cache),max_prompt_bindings=max(map(len,bindings.values())),complete_inventory=False),indent=2))
