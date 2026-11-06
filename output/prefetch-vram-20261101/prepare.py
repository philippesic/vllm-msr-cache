import hashlib,json,shutil
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1];OLD=ROOT/'output/scoring-overhead-20261101';B=458752
H.joinpath('prepared').mkdir(exist_ok=True)
for seed in [1103,2207,3301]:
 old=json.loads((OLD/'prepared'/f'multi-{seed}.json').read_text())
 # Reuse already-tokenized, disjoint synthetic contexts; no model download.
 prefixes=[s['initial_tokens'] for s in old['sessions']]
 rounds=[]
 for i in range(3):
  # Extend pressure contexts with already observed trace tokens; distinct leading prefixes.
  def context(si,n):
   tokens=prefixes[si]+sum(old['sessions'][si]['turn_append_tokens'],[])
   while len(tokens)<n:tokens+=prefixes[si]
   return tokens[:n]
  rounds.append({'source':context(i,2048),'pressure_a':context((i+2)%6,3072),'pressure_b':context((i+4)%6,3072),'append':old['sessions'][i]['turn_append_tokens'][1][:16]})
 p={'model':old['model'],'namespace':old['namespace'],'trace':{'seed':seed},'rounds':rounds,'geometry':{'gpu_blocks':640,'gpu_kv_bytes':640*B,'cpu_blocks':1024,'cpu_kv_bytes':1024*B,'block_bytes':B,'max_model_len':3328,'max_num_seqs':1,'max_num_batched_tokens':1024}}
 for workload in ['qualify','causal']:(H/'prepared'/f'{workload}-{seed}.json').write_text(json.dumps(p,indent=2)+'\n')
frozen={'model':old['model'],'revision':old['namespace']['model_revision'],'seeds':[1103,2207,3301],'source_pins':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in (H/'sources/original/semantic_offload').glob('*.py')},'prepared_pins':{str(p.relative_to(H)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (H/'prepared').glob('*.json')},'purpose':'Proactive offload and actual full-prefix prefetch/adoption, physical KV pool headroom at near ARC throughput','qualify':'Technical cold-prefix dispatch after pressure; privileged timing, not prediction','causal':'Pending peer-tool phase starts after source completion, fixed-estimate refresh; no actual future return time or future tokens in producer','cpu_tier_bytes':1024*B,'reference_gpu_blocks':640,'no_commit_push':True}
(H/'frozen.json').write_text(json.dumps(frozen,indent=2)+'\n')
print('Prepared six fixed traces')
