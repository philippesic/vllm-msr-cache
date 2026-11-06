import json,sys,subprocess
from pathlib import Path
ROOT=Path('/home/philip/vllm-semantic-cache-ext');sys.path.insert(0,str(ROOT))
from harness.gpu_lock import GPULock

def main():
 with GPULock():
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  from vllm import LLM,SamplingParams
  row=json.loads((ROOT/'output/fair-four-20261104/comparison-final/00-continuum/requests.jsonl').read_text().splitlines()[0]);ids=row['payload']['prompt'];ids=ids[:len(ids)//256*256-8]
  print('PROBE_PROMPT_TOKENS',len(ids),flush=True)
  llm=LLM(model=str(ROOT.parent/'.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots/b25037543e9394b818fdfca67ab2a00ecc7dd641'),dtype='half',max_model_len=8192,num_gpu_blocks_override=2560,max_num_seqs=1,max_num_batched_tokens=1024,enforce_eager=True,gpu_memory_utilization=.75,enable_prefix_caching=True,scheduling_policy='continuum',kv_transfer_config={'kv_connector':'LMCacheConnectorV1','kv_role':'kv_both'})
  try:
   r=llm.generate({'prompt_token_ids':ids},SamplingParams(temperature=0,max_tokens=32,ignore_eos=True,extra_args={'job_id':'decode-probe','is_last_step':True}),use_tqdm=False)[0]
   assert len(r.outputs[0].token_ids)==32
   print('DECODE_PROBE_COMPLETED',json.dumps(dict(prompt_tokens=len(ids),output_tokens=list(r.outputs[0].token_ids))),flush=True)
  finally: llm.llm_engine.engine_core.shutdown()
if __name__=='__main__':main()
