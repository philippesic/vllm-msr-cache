from pathlib import Path
import shutil,json,hashlib
H=Path(__file__).resolve().parent;dst=H/'sources/reclaim-backoff';shutil.copytree(H/'sources/shared-lookup',dst)
f=dst/'semantic_offload/connector.py';s=f.read_text();needle='''        state.status = "retired"
        state.lease.expire()
        self._wait_prefetch.pop(event_id, None)
''';repl='''        if reason == "demand_reclaim" and released_blocks > 0:
            failures = self.__dict__.setdefault("_reclaimed_source_backoff", {})
            fingerprint = hashlib.sha256(repr(tuple(state.source.keys)).encode()).digest()
            failures.pop(fingerprint, None)
            failures[fingerprint] = time.monotonic_ns() + 5_000_000_000
            while len(failures) > 256:
                del failures[next(iter(failures))]
        state.status = "retired"
        state.lease.expire()
        self._wait_prefetch.pop(event_id, None)
''';assert s.count(needle)==1;s=s.replace(needle,repl)
needle='''        if not getattr(self, "_wait_hint_shared_local_prefix", False):
            return (bool(source_keys)''';repl='''        failures = getattr(self, "_reclaimed_source_backoff", {})
        if failures:
            fingerprint = hashlib.sha256(repr(tuple(source_keys)).encode()).digest()
            until = failures.get(fingerprint, 0)
            if time.monotonic_ns() < until:
                return False
            failures.pop(fingerprint, None)
        if not getattr(self, "_wait_hint_shared_local_prefix", False):
            return (bool(source_keys)''';assert s.count(needle)==1;s=s.replace(needle,repl);f.write_text(s)
pins=json.loads((dst/'pins.json').read_text());pins['source']='reclaim-backoff-private'
for n in pins['files']:
 p=dst/n if '/' in n else dst/'semantic_offload'/n
 if p.is_file():pins['files'][n]=hashlib.sha256(p.read_bytes()).hexdigest()
(dst/'pins.json').write_text(json.dumps(pins,indent=2)+'\n')
f=H/'run_backoff.py';f.write_text((H/'run_pressure.py').read_text().replace("'demand-headroom'","'reclaim-backoff'"))
p=json.loads((H/'PLAN-v3.json').read_text());template=p['runs'][0];r=dict(template);r['command']=list(template['command']);r['command'][2]=str(f);r['name']='backoff-b768-candidate-s1103';r['command'][r['command'].index('--name')+1]=r['name'];p['runs']=[r];p['progress_file']='backoff-progress.json'
p['pins']={str(f.relative_to(H)):hashlib.sha256(f.read_bytes()).hexdigest() for f in H.rglob('*') if f.is_file() and f.suffix in ['.py','.json'] and any(x in f.parts for x in ['sources','runtime','runtime-shared-lookup','hooks-defer','prepared'])}
for n in ['frontier.py','run_backoff.py','session_workload.py','runtime_support.py','campaign_metrics.py','readmission_observer.py','lookup_deferral_observer.py','byte_audit.py','model-selection.json']:p['pins'][n]=hashlib.sha256((H/n).read_bytes()).hexdigest()
p['note']='New narrower candidate after headroom suppressed all transfers: 5 second negative cache for exact source only after actual completed-copy demand reclaim, bounded to 256 fingerprints. No future information. Screen against fresh ARC/default/caps; not a production default.'
(H/'BACKOFF.json').write_text(json.dumps(p,indent=2)+'\n')
