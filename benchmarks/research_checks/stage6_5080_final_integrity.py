"""Bind final CLI verification, source/cap audits and unchanged pre-rebuild history."""
from pathlib import Path
import sys
r=Path(__file__).resolve().parents[2]
frozen=r/'output/stage6-5080-20261108/primary-execution-bundle-01/source'
assert Path.cwd()==frozen.resolve()and sys.dont_write_bytecode
assert sys.executable=='/home/philip/.cache/semantic-offload/stage6-5080-runtimes-01/local-env/bin/python'
sys.path[:]=[str(frozen)]+[p for p in sys.path if Path(p).resolve()!=Path(__file__).resolve().parent]
import json,time,hashlib
from benchmarks.serving_suite.campaign_contracts import load,digest,atomic_json,Ledger,verify_tree,file_hash
from benchmarks.serving_suite.campaign_verify import verify_campaign
from benchmarks.serving_suite.campaign import source_snapshot
from benchmarks.serving_suite.campaign_completion import recovery
assert all(Path(f.__code__.co_filename).resolve().is_relative_to(frozen.resolve())for f in(load,verify_campaign,source_snapshot,recovery))
e=r/'evidence/stage6-5080-20261109';base=r/'output/stage6-5080-20261109';volume=r/'output/stage6-5080-20261108/linux-volume-01';out=volume/'primary-20261109'
target=e/'post-acquisition-audit-01.json';assert volume.is_mount()and not target.exists()
terminal=load(e/'confirmation-attempts-complete.json');supp={p:load(e/f'{p}-boundary-01.json')for p in('overhead','operational')}
assert terminal['execution_complete']and all(v['execution_complete']and v['frozen_phase_complete']for v in supp.values())
ledger=Ledger(out)
with ledger.writer():
 before=digest(ledger.events());entries=ledger.index();assert len(entries)==641 and all(v['state']in('verified','failed','interrupted')for v in entries.values())
 protocol=load(out/'protocol.lock.json');source=source_snapshot();assert source==protocol['source']and len(source['files'])==2192
 verification=load(base/'campaign-verification-final-01.log');assert verification['protocol_frozen']and verification['verified_files']==67120 and verification['attempts']=={'verified':603,'interrupted':3,'failed':35}
 assert sum(verification['attempts'].values())==len(entries)
 quality=load(out/'analysis/quality/summary.json');ratings=load(out/'analysis/quality/source-audit.json');packet=load(out/'analysis/quality/blind-packet.json')
 assert quality['status']=='audited'and quality['phase']=='confirm'and quality['reviewed']==7011 and quality['objective_outputs']==10520
 assert quality['packet_digest']==digest(packet['items'])and quality['audit_digest']==digest(ratings)
 published=load(e/'confirm-quality-published-19.json');assert published['packet_digest']==quality['packet_digest']and published['ratings_digest']==digest(ratings)
 cap=load(base/'caps-all-final-01/summary.json');ci=load(e/'caps-all-review-index-final-01.json');cr=load(e/'caps-all-source-judgments-final-01.json')
 assert cap['complete_inventory']and cap['status']=='audited'and cap['reviewed']==516 and not cap['pending']and not cap['partial_stream_fragments']
 assert cap['input_ledger_digests']=={'primary-01':before}and cap['audit_digest']==digest(cr)and cap['packet_digest']==ci['packet_digest']
 bundle_files=verify_tree(frozen.parent);assert bundle_files==4606
 imported_files=verify_tree(out/'imported');assert imported_files==4606
 old=verify_campaign(volume/'primary-01');history=load(e/'history-integrity.json');assert old==history['closed_campaign']
 build=load(e/'lmcache-build.json');original=Path(build['original']['site'])/'c_ops.cpython-312-x86_64-linux-gnu.so'
 assert file_hash(original)==build['original_extension_sha256']==history['original_lmcache_binary_sha256']
 assert file_hash(Path(build['wheel']))==build['wheel_sha256']
 assert file_hash(Path(build['kernel_smoke']['extension']))==build['kernel_smoke']['extension_sha256']
 interruptions=recovery(out);assert interruptions['complete']
 assert digest(ledger.events())==before and source_snapshot()==source
 receipt=dict(schema='stage6-5080-final-post-acquisition-audit-v1',recorded_unix=time.time(),execution_complete=True,
  full_throughput_acquisition_complete=False,verification=verification,verification_cli_log=str(base/'campaign-verification-final-01.log'),
  verification_cli_log_sha256=file_hash(base/'campaign-verification-final-01.log'),stable_ledger_digest=before,
  frozen_acquisition_source=source['digest'],frozen_acquisition_source_files_verified=2192,execution_bundle_files_verified=bundle_files,
  imported_bundle_files_verified=imported_files,source_equals_protocol=True,confirmation='confirmation-attempts-complete.json',
  supplementary_boundaries={p:dict(receipt=f'{p}-boundary-01.json',complete=True,configurations=v['planned_configurations'],verified_files=v['verified_files'])for p,v in supp.items()},
  primary_source_audit=dict(published='confirm-quality-published-19.json',objective_finals=10520,reviewed=7011,contexts=849,status='audited'),
  all_attempt_cap_audit=dict(index='caps-all-review-index-final-01.json',complete_inventory=True,status='audited',reviewed=516,contexts=186,ledger_unchanged=True),
  recovery=interruptions,november8_history=dict(verification=old,unchanged_original_lmcache_binary=True),
  rebuilt_wheel_and_installed_extension_match_build_receipt=True,volume_growth='volume-growth-01.json',
  missing_confirmation_blocks=terminal['missing_blocks'],portability_pending=True,resource_release_announced=False,
  scope='Every retained sealed attempt and inner aggregate verified by frozen CLI after all timed phases and online storage growth. Source, execution/import bundles, Nov8 campaign and original kernel unchanged; rebuilt artifacts still match recorded hashes. All phase execution and source/cap/recovery audits handled, with three genuine activity exclusions retained. Final offline relocation/import proof, final report/render/push and owned cleanup/notification remain required.')
 atomic_json(target,receipt)
 print(json.dumps({k:v for k,v in receipt.items()if k!='recovery'},indent=2))
