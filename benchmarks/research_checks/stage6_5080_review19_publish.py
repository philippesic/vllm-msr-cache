"""Publish the validated terminal confirmation source audit with a materialized backup."""
from pathlib import Path
import sys
r=Path(__file__).resolve().parents[2]
frozen=r/'output/stage6-5080-20261108/primary-execution-bundle-01/source'
assert Path.cwd()==frozen.resolve() and sys.dont_write_bytecode
assert sys.executable=='/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python'
sys.path[:]=[str(frozen)]+[p for p in sys.path if Path(p).resolve()!=Path(__file__).resolve().parent]
import os,shutil,time,hashlib,json
from benchmarks.serving_suite.campaign_contracts import load,digest,atomic_json,Ledger
from benchmarks.serving_suite.campaign_quality import finish_review
assert all(Path(f.__code__.co_filename).resolve().is_relative_to(frozen.resolve())for f in(load,finish_review))
e=r/'evidence/stage6-5080-20261109';base=r/'output/stage6-5080-20261109';out=r/'output/stage6-5080-20261108/linux-volume-01/primary-20261109'
b=base/'quality-confirm-checkpoint-19';live=out/'analysis/quality';backup=base/'quality-before-terminal19-01';target=e/'confirm-quality-published-19.json'
assert out.parent.is_mount() and not target.exists() and not backup.exists()
terminal=load(e/'confirmation-attempts-complete.json');validation=load(e/'confirm-review-validation-19.json');idx=load(e/'confirm-review-index-19.json')
assert terminal['execution_complete'] and terminal['terminal_logical_configurations']==524 and terminal['complete_matched_blocks']==207
assert validation['confirmation_execution_complete'] and validation['source_review_complete_for_snapshot'] and validation['reviewed_outputs']==7011
assert validation['packet_digest']==idx['packet_digest']==digest(load(b/'blind-packet.json')['items'])
ratings=e/'confirm-source-judgments-19.json';assert len(load(ratings)['ratings'])==7011
ledger=Ledger(out)
def confirmation_digest():
 return digest([v for v in ledger.index().values()if v['details']['identity']['phase']=='confirm'])
assert confirmation_digest()==terminal['ledger_confirmation_digest']
assert live.is_dir() and all(not p.is_symlink()for p in live.rglob('*'))
old={str(p.relative_to(live)):hashlib.sha256(p.read_bytes()).hexdigest()for p in live.rglob('*')if p.is_file()}
old_summary=load(live/'summary.json')
shutil.copytree(live,backup,symlinks=False)
assert {str(p.relative_to(backup)):hashlib.sha256(p.read_bytes()).hexdigest()for p in backup.rglob('*')if p.is_file()}==old
for name in ('blind-packet.json','summary.json','private-labels.json','objective.json','rubric.json'):
 source=b/name;temp=live/(name+'.terminal19.tmp');assert source.is_file()and not source.is_symlink()and not temp.exists()
 shutil.copy2(source,temp);os.replace(temp,live/name)
(live/'private-labels.json').chmod(0o600)
result=finish_review(out,ratings)
isolated=base/'quality-confirm-audit-checkpoint-19/analysis/quality'
assert result==load(isolated/'summary.json') and result['status']=='audited' and result['reviewed']==7011
for name in ('blind-packet.json','objective.json','rubric.json','private-labels.json','source-audit.json','summary.json'):
 assert load(live/name)==load(isolated/name)
assert confirmation_digest()==terminal['ledger_confirmation_digest']
receipt=dict(schema='stage6-5080-terminal-source-audit-publication-v1',recorded_unix=time.time(),confirmation_execution_complete=True,throughput_acquisition_complete=False,live_quality_audited=True,phase='confirm',objective_outputs=10520,selected_outputs=7011,distinct_full_contexts=849,packet_digest=idx['packet_digest'],ratings_digest=digest(load(ratings)),canonical_index='confirm-review-index-19.json',independent_validation='confirm-review-validation-19.json',prior_live_summary_scope={k:old_summary.get(k)for k in('status','phase','objective_outputs','blind_review_items','packet_digest')},prior_live_materialized_backup=str(backup),prior_live_files_sha256=old,all_six_live_quality_records_equal_isolated_audit=True,confirmation_ledger_unchanged=True,source_digest=terminal['source_digest'],helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Exact independently validated terminal selected-block packet/ratings published with frozen finish. Three original activity-invalid blocks stay excluded; all ordinary caps, including failed/unmatched arms and supplementary phases, require the separate final all-attempt audit. No acquisition, source, estimator, objective, validity or repetition change. Source review is not quality equivalence or a population accuracy estimate.')
atomic_json(target,receipt);print(json.dumps(receipt,indent=2))
