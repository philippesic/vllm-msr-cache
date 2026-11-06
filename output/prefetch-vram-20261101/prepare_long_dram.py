"""Isolate DRAM capacity from GPU pool changes, preserving exact pinned inputs."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
for seed in (1103,2207,3301):
    src=H/'prepared'/f'competing_long-{seed}.json';p=json.loads(src.read_text())
    p['geometry'].update(cpu_blocks=8192,cpu_kv_bytes=8192*458752)
    p['dataset_contract'].update(dram_followup='8192 CPU blocks in every arm; input arrays unchanged; avoid observed current-prefix CPU evictions',base_long_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),preparer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out=H/'prepared'/f'competing_long_dram-{seed}.json';s=json.dumps(p,indent=2)+'\n'
    if out.exists():assert out.read_text()==s
    else:out.write_text(s)
    print(out)
