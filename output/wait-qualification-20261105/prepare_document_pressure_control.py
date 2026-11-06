"""Predeclare one finite live pressure diagnostic after the paired ARC alarm."""
import copy,json,time,hashlib
from pathlib import Path
b=Path(__file__).parent;source=b/'stage3-documents-cohost-spec.json';dest=b/'stage3-documents-pressure-control-spec.json';receipt=b/'stage3-documents-pressure-control-premeasurement.json'
assert not dest.exists() and not receipt.exists()
s=copy.deepcopy(json.loads(source.read_text()));arc=copy.deepcopy(s['conditions']['arc']);wait=copy.deepcopy(s['conditions']['wait']);inactive=copy.deepcopy(wait)
# Existing tested Stage1 control validates ready prefixes, but skips only ARC queue mutation.
inactive.setdefault('candidate_config',copy.deepcopy(s['plan']['candidate_config']))['kv_connector_extra_config']['benchmark_wait_hint_noop']=True
s['conditions']={'arc':arc,'inactive':inactive,'wait':wait};s['order']=[[0,'arc'],[0,'inactive'],[0,'wait'],[1,'inactive'],[1,'wait'],[1,'arc'],[2,'wait'],[2,'arc'],[2,'inactive']]
s['plan'].update(repeats=3,scope_note='One finite live diagnosis after tight-document ARC regression alarm. Same public inputs, instructions, memory and scheduling. Inactive hint control changes only existing benchmark queue-mutation no-op. Development diagnostic, not fresh held-out validation, no pooling with original or unlimited extensions.')
dest.write_text(json.dumps(s,indent=2)+'\n');receipt.write_text(json.dumps(dict(prepared_unix=time.time(),trigger='Original tight-document repeat1 wait5.83pct slower than ARC; earlier repeat0 wait5.03pct faster. Third wait not yet measured.',hypothesis='Separate hint production/processing overhead from active queue mutation and its CPU restore/admission interactions.',finite_budget='Exactly3 rotating runs for each ARC/inactive/full wait, one campaign; no serving/capacity tuning.',input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),spec_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()),indent=2)+'\n');print(dest)
