from pathlib import Path
import hashlib,json,gzip,datetime,subprocess
R=Path(__file__).resolve().parents[2];H=R/'output/prefetch-native-20261102';E=R/'evidence/live-optimization-2080ti-20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=E/'gpu-gate-confirmation-frozen.json';m=json.loads(p.read_text());c=H/'fresh-agent-gate-nov3/session-corpus.json'
for n,h in m['file_pins'].items(): assert sha(R/n)==h,n
assert len(m['pairs'])==12 and len(m['execution_order'])==24
for run in m['execution_order']:
 assert not (H/'runs'/run['name']).exists(),run['name']
 a=run['args'];assert '--audit' not in a
 assert a[a.index('--corpus-path')+1]==str(c.relative_to(R))
 assert ('--gpu-gated-cpu-retention' in a)==(run['arm']=='candidate')
 assert ('--no-audit' in a)==(run['arm']=='candidate')
scanned=0
for raw in (H/'runs').glob('live-nov3*/raw.json*'):
 if raw.name not in ('raw.json','raw.json.gz'):continue
 data=raw.read_bytes();x=json.loads(gzip.decompress(data) if raw.suffix=='.gz' else data);scanned+=1
 used=x.get('args',{}).get('corpus_path')
 if used:assert (R/used).resolve()!=c.resolve(),str(raw)
v=json.loads((c.parent/'validation.json').read_text());iv=json.loads((c.parent/'independent-validation.json').read_text())
assert v['model_generation_performed'] is False and iv['no_model_generation'] is True
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
b=E/'integrated-gate-bytes-root-verification.json'
out={'schema':'root-gpu-gate-confirmation-prelaunch-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':sha(p),'byte_receipt_sha256':sha(b),'pins_verified':len(m['file_pins']),'planned_names_absent':24,'prior_raw_records_scanned':scanned,'new_corpus_never_used_by_prior_campaign_runs':True,'fresh_acquisition_no_generation_receipts_verified':True,'candidate_bounded_mode_and_only_gate_enabled':True,'performance_intrusive_audit_disabled_both_arms':True,'gpu_idle':True,'gpu_custody_transfer':'ROOT to Astra after this receipt; exact frozen24 invocations only'}
q=E/'gpu-gate-confirmation-root-prelaunch.json';assert not q.exists();q.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
