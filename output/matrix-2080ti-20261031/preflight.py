import json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from harness.server import launch_server
from harness.policies import kv_transfer_config
out=Path(__file__).parent/'preflight';out.mkdir(exist_ok=True)
handle=None
try:
    handle=launch_server('Qwen/Qwen2.5-0.5B-Instruct',19371,str(out),gpu_memory_utilization=.65,max_model_len=4096,num_gpu_blocks_override=256,
        kv_transfer_config=kv_transfer_config('arc',16*1024**2,{'preservation_events':True,'store_threshold':2}),
        extra_args=['--dtype','half','--enforce-eager','--attention-backend','TRITON_ATTN','--max-num-seqs','3','--max-num-batched-tokens','1024'],
        extra_env={'VLLM_WSL2_ENABLE_PIN_MEMORY':'1','VLLM_USE_V2_MODEL_RUNNER':'1','PYTHONPATH':str(ROOT),'HF_HUB_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1'},ready_timeout_s=180)
    import urllib.request
    body={'model':'Qwen/Qwen2.5-0.5B-Instruct','prompt':'Explain why cache correctness matters.','max_tokens':16,'temperature':0}
    with urllib.request.urlopen(urllib.request.Request(handle.base_url()+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=90) as response:
        result=json.load(response)
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print('PREFLIGHT PASS',flush=True)
finally:
    if handle:handle.shutdown()
