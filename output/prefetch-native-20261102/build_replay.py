from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'session_workload.py').read_text();s=s.replace('past=list(prompt)+list(output[:-1]);return past[:len(past)//16*16]', 'past=list(prompt);return past[:len(past)//16*16]')
needle=" result['planned_requests']=sum("
s=s.replace(needle," fixed=json.loads((H/'prepared'/f'fixed-agent-{a.seed}.json').read_text())\n fixed_rows={(r['session_id'],r['turn']):r for r in fixed['requests']}\n result['fixed_input_trace']={'schema':fixed['schema'],'path':str(H/'prepared'/f'fixed-agent-{a.seed}.json'),'sha256':hashlib.sha256((H/'prepared'/f'fixed-agent-{a.seed}.json').read_bytes()).hexdigest(),'same_prompt_and_decode_budget':True,'future_prompts_are_evaluation_only':True}\n"+needle)
s=s.replace(" result['prefix_contiguity']=[];", " result['workload'].update(history='fixed reference prompts; response independent replay',respect_eos=False,output_cap=a.max_output)\n result['prefix_contiguity']=[];")
s=s.replace("'max_tokens':a.max_output,'temperature':0.","'max_tokens':a.max_output,'ignore_eos':True,'temperature':0.")
old="""    messages=messages_for_turn(session,turn,generated);validate_history(session,turn,generated,messages)
    tokens=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
"""
new="""    # The returning prompt is disclosed only at release. The independent
    # producer sees the previously completed input prefix, never this record.
    tokens=list(fixed_rows[(sid,turn)]['prompt'])
"""
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace("    row['success']=True", "    assert len(row['output_token_ids'])==a.max_output, 'fixed decode budget incomplete'\n    row['success']=True")
(H/'fixed_workload.py').write_text(s)
s=(H/'run_frontier_defer.py').read_text().replace('from session_workload import validate_hint_contract','from fixed_workload import validate_hint_contract').replace('from session_workload import exercise','from fixed_workload import exercise')
s=s.replace("'VLLM_BATCH_INVARIANT':'0'", "'VLLM_BATCH_INVARIANT':'1' if a.batch_invariant else '0'")
s=s.replace("ap=argparse.ArgumentParser();", "ap=argparse.ArgumentParser();ap.add_argument('--batch-invariant',action='store_true');")
s=s.replace("H/'session_workload.py'", "H/'fixed_workload.py'")
(H/'run_replay.py').write_text(s)
