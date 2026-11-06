import os,json,sys
from pathlib import Path
sys.path.insert(0,'/home/philip/vllm-semantic-cache-ext')
from harness.gpu_lock import GPULock

def main():
 with GPULock():
  import subprocess
  state=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  assert not state,state
  import vllm
  from vllm import LLM,SamplingParams
  print('SOURCE',vllm.__file__,flush=True)
  llm=LLM(model='/home/philip/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots/b25037543e9394b818fdfca67ab2a00ecc7dd641',dtype='half',max_model_len=8192,num_gpu_blocks_override=1024,max_num_seqs=1,max_num_batched_tokens=1024,enforce_eager=True,gpu_memory_utilization=.75,enable_prefix_caching=True,kv_transfer_config={'kv_connector':'OffloadingConnector','kv_role':'kv_both','kv_connector_extra_config':{'cpu_bytes_to_use':512*1024**2}})
  try:
   corpus=json.loads(Path('output/five-way-20261104/comparison-final/corpus.json').read_text())
   # Real repository context from the retained corpus; force CPU return by
   # clearing only GPU prefixes between identical deterministic requests.
   sessions=corpus['sessions'] if 'sessions' in corpus else corpus['records']
   text=sessions[0]['context']+'\n'+sessions[0]['turns'][0]
   tok=llm.get_tokenizer(); ids=tok.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,return_dict=False)
   outputs=[]
   for i in range(3):
    if i: assert llm.reset_prefix_cache(reset_connector=False)
    result=llm.generate({'prompt_token_ids':ids},SamplingParams(temperature=0,max_tokens=32),use_tqdm=False)[0]
    outputs.append(result.outputs[0].token_ids)
    print(json.dumps({'request':i,'prompt_tokens':len(ids),'output_tokens':list(outputs[-1])}),flush=True)
   assert outputs[0]==outputs[1]==outputs[2]
   print('PROBE_PASSED_IDENTICAL_OUTPUTS',flush=True)
  finally:
   llm.llm_engine.engine_core.shutdown()
if __name__=='__main__': main()
