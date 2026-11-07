"""Resume exact frozen acquisition with an attested derived validity revision only."""
import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--analysis-receipt',type=Path,required=True)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args(); out=args.out.resolve()
    original=json.loads((out/'protocol.lock.json').read_text())['source']
    receipt=json.loads(args.analysis_receipt.read_text())
    relative='benchmarks/serving_suite/campaign_validity.py'
    project=Path(__file__).resolve().parents[2]
    predicate=project/relative
    expected=receipt['analysis_source']['files'][relative]
    if hashlib.sha256(predicate.read_bytes()).hexdigest()!=expected:
        raise ValueError('Derived predicate changed since its verified analysis receipt')
    host=json.loads((out/'host.private.json').read_text())
    frozen=Path(host['asset_cache'])/'source-snapshots'/original['digest']
    if not frozen.is_dir():raise ValueError('Frozen acquisition closure is missing')
    # Import the coordinator from its original closure, not the mutable checkout.
    for name in list(sys.modules):
        if name=='benchmarks' or name.startswith('benchmarks.') or name=='harness' or name.startswith('harness.'):
            del sys.modules[name]
    sys.path.insert(0,str(frozen))
    package=types.ModuleType('benchmarks')
    package.__path__=[str(frozen/'benchmarks')]
    sys.modules['benchmarks']=package
    from benchmarks.serving_suite import campaign,campaign_validity
    if Path(campaign.__file__).resolve()!=frozen/'benchmarks/serving_suite/campaign.py':
        raise ValueError('Coordinator did not bind to its original frozen closure')
    if campaign.source_snapshot()!=original:raise ValueError('Frozen acquisition source changed')
    spec=importlib.util.spec_from_file_location('benchmarks.serving_suite._attested_validity',predicate)
    revised=importlib.util.module_from_spec(spec);spec.loader.exec_module(revised)
    campaign_validity.measurement_validity=revised.measurement_validity
    campaign_validity.VERSION=revised.VERSION
    if args.check:
        from benchmarks.serving_suite.campaign_contracts import Ledger
        ledger=Ledger(out)
        reusable=[e for e in ledger.index().values() if e['state']=='failed' and
            e['details'].get('identity',{}).get('phase')=='confirm' and
            revised.measurement_validity(out/e['details']['path'],e)['valid'] and
            ledger.reusable(e['details']['identity'])]
        print(json.dumps(dict(frozen_coordinator=True,serving_files_unchanged=True,
            derived_validity_version=revised.VERSION,complete_activity_only_attempts_reusable=len(reusable))))
    else:
        print(json.dumps(campaign.run_phase(out,'confirm',resume=True)))


if __name__=='__main__':main()
