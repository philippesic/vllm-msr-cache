"""After the single calibration, freeze both originally tested byte geometries."""
import copy,hashlib,json,time
from pathlib import Path
b=Path(__file__).parent;half=b/'stage3-documents-reserve-spec.json';original=b/'stage3-documents-spec.json';full=b/'stage3-documents-reserve-full-spec.json';receipt=b/'stage3-documents-reserve-range-premeasurement.json'
assert half.exists(),'First adopt the one calibrated instruction on untouched reserve tasks'
assert not full.exists() and not receipt.exists(),'Preserve prior reserve attempts'
s=copy.deepcopy(json.loads(half.read_text()));o=json.loads(original.read_text())
s['plan'].update(gpu_blocks=o['plan']['gpu_blocks'],cpu_bytes=o['plan']['cpu_bytes'],scope_note='Fresh sixteen-task document reserve, original full-byte geometry and three rotating ARC/wait/upstream cases. Full and half both selected and frozen before any reserve outcomes, same adopted instructions/source tasks. No pooling, capacity search or selection conditional on a reserve win.')
s['conditions']=copy.deepcopy(o['conditions'])
assert len(s['order'])==9 and len(s['corpus']['sessions'])==16
full.write_text(json.dumps(s,indent=2)+'\n');paths=[half,full,b/'documents-reserve-corpus.json',b/'stage3-documents-reserve-range-selection.json',Path(__file__)]
receipt.write_text(json.dumps(dict(prepared_before_any_reserve_serving_unix=time.time(),cells=['stage3-documents-reserve','stage3-documents-reserve-full'],mandatory_both=True,no_pooling=True,sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2)+'\n');print(full)
