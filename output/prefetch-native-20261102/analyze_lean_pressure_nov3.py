import collections,gzip,json,re,sys
from pathlib import Path
from summarize_finite_discovery_nov3 import summarize
H=Path(__file__).resolve().parent

def counter(text,name,source=None):
 values=[]
 for line in text.splitlines():
  if not (line.startswith(name+'{') or line.startswith(name+' ')):continue
  if source is not None and f'source="{source}"' not in line:continue
  values.append(float(line.rsplit(' ',1)[1]))
 return sum(values) if values else None

def read(name):
 path=H/'runs'/name/'raw.json.gz';r=json.load(gzip.open(path));s=summarize(path)
 before=r.get('metrics_initial','');after=r.get('metrics_final','')
 def delta(name,source=None):
  a=counter(before,name,source);b=counter(after,name,source)
  return b-a if a is not None and b is not None else None
 attr={source:delta('vllm:prompt_tokens_by_source_total',source) for source in ['local_compute','local_cache_hit','external_kv_transfer']}
 logical=sum(len(q['payload']['prompt']) for q in r['requests'])
 s['first_prefill_attribution']={'tokens':attr,'logical_prompt_tokens':logical,'reconciles':all(v is not None for v in attr.values()) and sum(attr.values())==logical,'preemptions':delta('vllm:num_preemptions_total'),'definition':'First-admission attribution, not cumulative physical executed work. Warmup subtracted.'}
 s['measurement_configuration']=r.get('postload_export',{}).get('scheduler',{}).get('measurement_configuration')
 return s

def main():
 names=sys.argv[1:] or ['live-nov3-lean-pressure-native','live-nov3-lean-pressure-candidate-h200','live-nov3-lean-pressure-candidate-h1000']
 rows={n:read(n) for n in names if (H/'runs'/n/'raw.json.gz').exists()}
 (H/'live-nov3-lean-pressure-analysis.json').write_text(json.dumps({'scope':'descriptive_discovery_only','rows':rows},indent=2)+'\n')
 for name,row in rows.items():
  print(json.dumps({k:row.get(k) for k in ['name','status','natural_eos','load_s','tokens','tokens_s','peak_nvml_mib','ttft_ms','first_prefill_attribution','load_transfer_bytes','measurement_configuration']}))
if __name__=='__main__':main()
