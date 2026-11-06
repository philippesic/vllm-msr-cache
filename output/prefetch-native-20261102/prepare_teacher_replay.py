"""Freeze a separate exact-output history control, not ordinary generation."""
import ast,hashlib,json,shutil
from pathlib import Path
H=Path(__file__).resolve().parent
hooks=H/'hooks-teacher-replay';shutil.copytree(H/'hooks-idle-drain',hooks)
f=hooks/'sitecustomize.py';s=f.read_text().replace("TARGETS=RUNTIME|{", "TARGETS=RUNTIME|{'vllm.v1.worker.gpu.sample.sampler',")
needle="    if n=='vllm.v1.kv_offload.cpu.gpu_worker'"
assert s.count(needle)==1;s=s.replace(needle,"    if n=='vllm.v1.worker.gpu.sample.sampler':\n        sys.path.insert(0,str(H))\n        import teacher_replay\n        teacher_replay.patch(m,emit)\n    elif n=='vllm.v1.kv_offload.cpu.gpu_worker'");ast.parse(s);f.write_text(s)
s=(H/'short_fixed_workload.py').read_text().replace("past=list(prompt);return", "past=list(prompt)+list(output[:-1]);return")
s=s.replace("'max_tokens':a.max_output,'ignore_eos':True", "'max_tokens':len(fixed_rows[(sid,turn)]['reference_output_token_ids']),'ignore_eos':True,'vllm_xargs':{'replay_token_ids':fixed_rows[(sid,turn)]['reference_output_token_ids']}")
s=s.replace("assert len(row['output_token_ids'])==a.max_output, 'fixed decode budget incomplete'", "assert row['output_token_ids']==fixed_rows[(sid,turn)]['reference_output_token_ids'], 'recorded output replay diverged'")
s=s.replace("history='fixed reference prompts; response independent replay'", "history='evaluation-only exact recorded prompt/completion replay; normal model forwards, recorded sampling outcomes'")
s=s.replace("'same_prompt_and_decode_budget':True", "'same_prompt_and_decode_budget':True,'exact_recorded_output_ids_required':True,'ordinary_generation':False")
(H/'teacher_workload.py').write_text(s)
s=(H/'run_demand_fallback.py').read_text().replace('from short_fixed_workload import','from teacher_workload import').replace("else 'hooks-idle-drain'", "else 'hooks-teacher-replay'")
s=s.replace("'performance_eligible':not (a.audit", "'evaluation_mode':'recorded_token_replay_not_unconstrained_generation','performance_eligible':not (a.audit")
s=s.replace("result['terminals']=", "result['teacher_forcing_requests']=parse(handle.log_path,'MATRIX_TEACHER_REQUEST') if handle else []\n  result['terminals']=") if "result['terminals']=" in s else s
ast.parse(s);(H/'run_teacher_replay.py').write_text(s)
print('Prepared exact-output replay hook/workload; no GPU qualification yet.')
