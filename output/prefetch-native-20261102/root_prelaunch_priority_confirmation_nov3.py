from pathlib import Path
import json,hashlib,gzip,collections,fcntl,subprocess,datetime
H=Path(__file__).resolve().parent;R=H.parents[1];E=R/'evidence/live-optimization-2080ti-20261103';p=E/'priority-confirmation-frozen.json';m=json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in m['file_pins'].items():assert sha(R/n)==h,n
assert len(m['pairs'])==12 and len(m['execution_order'])==36
orders=collections.Counter(tuple(g['order']) for g in m['pairs']);assert len(orders)==6 and set(orders.values())=={2}
for seed in [14009,15013,16001]:assert sum(g['order'].index('native8')<g['order'].index('candidate') for g in m['pairs'] if g['partition']==seed)==2
corpus=H/'fresh-agent-priority-nov3/session-corpus.json';prior=0
for plan in m['execution_order']:
 assert not (H/'runs'/plan['name']).exists();a=plan['args'];assert '--audit' not in a and ('--no-audit' in a)==plan['gate_enabled'] and ('--gpu-gated-cpu-retention' in a)==plan['gate_enabled']
 assert int(a[a.index('--return-priority-ms')+1])==plan['priority_credit_ms'] and (R/a[a.index('--corpus-path')+1]).resolve()==corpus.resolve()
for p0 in (H/'runs').glob('live-nov3*/raw.json*'):
 if p0.name not in ('raw.json','raw.json.gz'):continue
 b=p0.read_bytes();r=json.loads(gzip.decompress(b) if p0.suffix=='.gz' else b);prior+=1;used=r.get('args',{}).get('corpus_path')
 if used:assert (R/used).resolve()!=corpus.resolve(),str(p0)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
with open('/tmp/vllm-semantic-cache-gpu.lock','a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
out={'schema':'root-three-arm-priority-confirmation-prelaunch-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':sha(p),'all_pins_verified':len(m['file_pins']),'names_absent':36,'six_orders_each_twice_verified':True,'two_each_matched_primary_pair_orders_per_partition':True,'candidate_bounded_mode_and_gate_explicit':True,'intrusive_audit_disabled_all36':True,'prior_raw_records_scanned':prior,'fresh_corpus_never_used_by_prior_actual_runs':True,'GPU_idle_and_lock_available':True,'custody_transfer':'ROOT to ASTRA for exact36frozeninvocations only'}
q=E/'priority-confirmation-root-prelaunch.json';assert not q.exists();q.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
