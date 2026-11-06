"""Deterministic 4x context screen derived from immutable competing datasets."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
for seed in (1103,2207,3301):
    base=H/'prepared'/f'competing-{seed}.json';p=json.loads(base.read_text())
    for r in p['rounds']:
        for s in r['sessions']:s['source']*=4
        for n in ('pressure_a','pressure_b'):r[n]*=4
    p['geometry'].update(gpu_blocks=2560,gpu_kv_bytes=2560*458752,cpu_blocks=4096,cpu_kv_bytes=4096*458752,max_model_len=12544)
    p['dataset_contract']={'schema':'causal-competing-peer-tools-long-v1','derivation':'repeat each pinned context four times, preserve unique leading tokens and appends','base_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'preparer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_tokens':6144,'peer_tokens':12288,'requests':18,'output_tokens':768,'minimum_concurrent_return_blocks_including_null':783,'candidate_blocks':784,'reference_blocks':2560,'not_production_trace':True}
    assert 2*((6144+32+16+64+15)//16)+1==783
    assert max(len(r['pressure_a'])+32 for r in p['rounds'])<=p['geometry']['max_model_len']
    dest=H/'prepared'/f'competing_long-{seed}.json';s=json.dumps(p,indent=2)+'\n'
    if dest.exists():assert dest.read_text()==s
    else:dest.write_text(s)
    print(dest)
