"""Freeze ROOT-reviewed nine-case operational block and separate GPU custody."""
import copy
import hashlib
import json
from pathlib import Path
import time
import root_review_operational_nov3 as review

ROOT, H, E = review.ROOT, review.H, review.E

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

if __name__ == '__main__':
    frozen = E / 'operational-frozen.json'
    authorization = E / 'operational-launch-root-authorization.json'
    assert not frozen.exists() and not authorization.exists()
    review.check()
    plan = json.loads((H / 'operational-review-plan-nov3.json').read_text())
    inventory = json.loads((H / 'live-nov3-operational-supervisor-inventory.json').read_text())
    manifest = copy.deepcopy(plan)
    manifest.update(schema='root-frozen-operational-block-v1', freeze_pending_root_review=False,
                    launch_authorized=False, root_review='passed', frozen_at_ns=time.time_ns())
    extra = dict(inventory['file_pins'])
    for path in [H/'operational-review-plan-nov3.json',
                 H/'live-nov3-operational-supervisor-inventory.json',
                 H/'root_review_operational_nov3.py', Path(__file__).resolve(),
                 E/'operational-harness-root-review.json']:
        extra[str(path)] = sha(path)
    all_pins = {}
    for group in [plan['existing_frozen_inputs'], plan['new_review_file_pins'], extra]:
        for name, expected in group.items():
            path = (ROOT / name).resolve()
            assert sha(path) == expected, name
            assert str(path) not in all_pins or all_pins[str(path)] == expected
            all_pins[str(path)] = expected
    manifest['file_pins'] = extra
    manifest['root_supervisor_review'] = {
        'controls_passed':21, 'root_rerun':True,
        'scope':'Exact argv freeze, explicit launch certificate, no replacement, pin checks per case, owned descendant cleanup',
        'probe_priority_limit':'Public helper bytes and actual body pinned; no independent exact clock timestamp receipt for abort/health probes',
        'all_unique_input_pins':len(all_pins), 'production_endorsement':False}
    frozen.write_text(json.dumps(manifest,indent=2)+'\n')
    certificate = {'schema':'root-operational-launch-authorization-v1',
                   'at_ns':time.time_ns(), 'launch_authorized':True, 'gpu_custody':'ASTRA',
                   'manifest_sha256':sha(frozen),
                   'supervisor_sha256':sha(H/'supervise_operational_astra_nov3.py'),
                   'authority':'Existing user sustained optimization request; ROOT reviewed exact finite operational block',
                   'scope':'Only these nine frozen cases, no retries or replacements; return custody and retain failures',
                   'performance_confirmation_replaced':False,'production_endorsement':False}
    authorization.write_text(json.dumps(certificate,indent=2)+'\n')
    print(json.dumps({'manifest':str(frozen),'authorization':str(authorization),
                      'files':len(all_pins),'launch_authorized':True,'gpu_custody':'ASTRA'}))
