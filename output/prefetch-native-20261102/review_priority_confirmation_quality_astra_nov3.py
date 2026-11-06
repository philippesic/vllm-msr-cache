"""Source-grounded fixed confirmation sample receipt; no outcome exclusions."""
import json,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent

def main():
 p=H/'live-nov3-priority-confirmation-quality-samples.json';q=json.loads(p.read_text());np=H/'live-nov3-priority-confirmation-quality-notes.json';notes=json.loads(np.read_text());a=json.loads((H/'live-nov3-priority-confirmation-independent-analysis.json').read_text());rows=[];unique=set()
 for run in q['runs']:
  for x in run.get('samples',[]):
   key=x['source_id']+':'+str(x['turn']);assert key in notes,key;unique.add((x['source_id'],x['turn'],x['generated_text']))
   rows.append({'run':run['name'],'arm':run['arm'],'partition':run['partition'],'index':x['index'],'source_id':x['source_id'],'turn':x['turn'],'selection':x['selection'],'format_flags':x['format_flags'],'text_sha256':hashlib.sha256(x['generated_text'].encode()).hexdigest(),'evidence_sha256':hashlib.sha256(json.dumps(x['source_evidence_turns'],sort_keys=True).encode()).hexdigest(),'finding':notes[key],'self_claimed_service_execution_or_edit_observed':False,'runaway_repetition_observed':False})
 formats={}
 for arm in a['arm_outcomes']:
  rr=[r for r in a['rows'].values() if r['arm']==arm];formats[arm]={k:sum(r['service_format'][k] for r in rr) for k in ['responses','starts_status','single_blank_line_block','with_code_fence']};formats[arm]['completed_action_phrase_flags']=sum(len(r['service_format']['completed_action_phrase_indices']) for r in rr);formats[arm]['degeneration_screen_flags']=sum(r['output_screen']['flagged'] for r in rr)
 out={'schema':'astra-three-arm-fresh-priority-source-grounded-quality-v1','sample_receipt_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'notes_sha256':hashlib.sha256(np.read_bytes()).hexdigest(),'reviewed_observations':len(rows),'distinct_generated_text_cases':len(unique),'distinct_source_issues':len({x['source_id'] for x in rows}),'prior_campaign_note_reuse':False,'rows':rows,'mechanical_all_outputs':formats,'assessment':'Shared factual/source-reading defects and formatviolations persist despite naturalEOS. No selected runaway or self-claimed service execution/edit. Human review is necessary; no accuracy equivalence, autonomous task success or measured productivity claim. Cache efficiency is judged separately from baseline reasoning errors.',
 'examples':['Sympy18698 incorrectly changes combined squarefree factor multiplicity3 to6.','Django15525 reverses condition that defaultDB data masks the nondefault loadingbug.','sklearn14141 wrongly infers joblibabsent from rootdirectorylisting; latestNone is notdependencyinventory.','Django14238 confuses customBigAutoFieldsubclass with templateBigAutoFieldsetting.','Pytest5631 fencedvariants infer codeexecution fromgrep and violate singleparagraph/no-code contract.'],
 'limits':['All55distinct selected texts read against retained initial/latest source observations; repeated identical outputs share explicit source-turn notes.','Sample101observations is fixedslots/longest3firsttripleperpartition plusallformatflags, not randomaccuracyestimate.','Only included source text was reviewed; linkedexternalpages/images were notretrieved. Actualearliergeneratedhistories retained in samplefile, nohistoricalassistanttext substituted.','Source may itself duplicate issue sections or include proposals/PRclaims; these are evidence to assess, not executedactions.','No generatedoutput or run excluded, no prompt/profile/source changes basedonquality.']}
 (H/'live-nov3-priority-confirmation-quality-review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['rows']},indent=2))
if __name__=='__main__':main()
