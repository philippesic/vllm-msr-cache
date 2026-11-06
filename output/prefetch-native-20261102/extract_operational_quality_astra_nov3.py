"""Fixed source-grounded spot-check selection; no output-based exclusions."""
import gzip,hashlib,json,re
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
M=ROOT/'evidence/live-optimization-2080ti-20261103/operational-frozen.json'

def digest(text):return hashlib.sha256(text.encode()).hexdigest()

def main():
    manifest=json.loads(M.read_text());selected=[];inventory=[]
    for case in manifest['execution_order']:
        path=H/'runs'/case['name']/'raw.json.gz'
        if not path.exists():continue
        raw=json.load(gzip.open(path,'rt'));corpus=json.loads((ROOT/raw['args']['corpus_path']).read_text())
        sessions={seed:[s for s in corpus['sessions'] if s['cohort']=='agent' and s['seed_partition']==seed] for seed in (14009,15013,16001)}
        waves=raw.get('waves',[]);pick={}
        for wave in waves:
            for q in wave.get('requests',[]):
                text=q.get('generated_text','');flags=[]
                if not text.startswith('Status:'):flags.append('missing_Status_prefix')
                if '```' in text:flags.append('code_fence')
                if len(re.split(r'\n\s*\n',text.strip()))>1:flags.append('multiple_blank_blocks')
                if re.search(r'\bI (?:have |had )?(?:modified|changed|ran|executed|updated|applied|fixed)\b',text,re.I):flags.append('claimed_execution_phrase')
                reason=[]
                if wave['wave_index'] in {0,len(waves)-1} and q['index']//4 in {0,6,12,18} and q['turn'] in {0,2}:reason.append('first_or_last_wave_preselected_slots')
                if flags:reason.extend(flags)
                if reason:pick[(wave['wave_index'],q['index'])]=reason
        longest=sorted([(len(q.get('output_token_ids',[])),w['wave_index'],q['index']) for w in waves for q in w.get('requests',[])],reverse=True)[:3]
        for _,wi,qi in longest:pick.setdefault((wi,qi),[]).append('longest_three_case')
        for wave in waves:
            byindex={q['index']:q for q in wave.get('requests',[])}
            for q in wave.get('requests',[]):
                key=(wave['wave_index'],q['index'])
                if key not in pick:continue
                source=sessions[wave['seed']][q['index']//4]
                selected.append({'case':case['name'],'phase':case['phase'],'arm':case['arm'],'wave':wave['wave_index'],'partition':wave['seed'],'slot':q['index']//4,'turn':q['turn'],'source_id':source['source_id'],'session_id':q['session_id'],'reasons':pick[key],
                                 'initial_issue':source['turns'][0].get('user_text'),'latest_observation':source['turns'][q['turn']].get('user_text'),
                                 'actual_earlier_generated':[byindex[q['index']-q['turn']+t].get('generated_text') for t in range(q['turn']) if q['index']-q['turn']+t in byindex],
                                 'generated_text':q.get('generated_text'),'generated_text_sha256':digest(q.get('generated_text','')),'tokens':len(q.get('output_token_ids',[]))})
        inventory.append({'case':case['name'],'requests':len(raw.get('requests',[])),'selected':len(pick)})
    out={'schema':'astra-operational-quality-selection-v1','selection':'All cases first/last wave, slots0/6/12/18 turns0/2; longestthree percase and all mechanicalformat/actionflags. Chosen before inspecting operational generated texts. Failures retained.','inventory':inventory,'selected':selected,'distinct_texts':len({(q['source_id'],q['turn'],q['generated_text_sha256']) for q in selected}),'scope':'Grounded spot checks only; no accuracy equivalence, executed tool success, or productivity claim.'}
    (H/'live-nov3-operational-quality-samples.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'selected':len(selected),'distinct_texts':out['distinct_texts']}))
if __name__=='__main__':main()
