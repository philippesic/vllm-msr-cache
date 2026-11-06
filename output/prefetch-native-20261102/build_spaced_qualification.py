import ast,copy,hashlib,json,shutil
from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'byte_audit.py').read_text()
s=s.replace("LOCAL_SNAPSHOTS[local_ids]=", "LOCAL_SNAPSHOTS[(job_id,local_ids)]=")
s=s.replace('def verify_local_prefix_on_adoption(prefix_ids,request_id,directory):','def verify_local_prefix_on_adoption(prefix_ids,request_id,directory,source_job_id):')
s=s.replace('entry=LOCAL_SNAPSHOTS.get(ids)','entry=LOCAL_SNAPSHOTS.pop((source_job_id,ids),None)')
assert 'pop((source_job_id,ids),None)' in s
(H/'byte_audit_qualified.py').write_text(s)
d=H/'hooks-byte-qualified';shutil.copytree(H/'hooks-defer',d)
p=d/'sitecustomize.py';s=p.read_text().replace('import byte_audit','import byte_audit_qualified as byte_audit')
needle="            original_adoption=cls._record_native_prefix_adoption"
replacement='''            original_publish=cls._try_publish_wait_prefix
            def qualified_publish(self,wait_state,**kwargs):
                job_id=wait_state.job_id
                result=original_publish(self,wait_state,**kwargs)
                if result:
                    published=self._published_wait_prefixes[wait_state.bound_request_id]
                    assert published['event_id']==wait_state.lease.hint.event_id
                    assert set(published['block_ids'][published.get('local_prefix_count',0):])==set(wait_state.gpu_block_ids)
                    published['_audit_transfer_job_id']=job_id
                return result
            cls._try_publish_wait_prefix=qualified_publish
            original_adoption=cls._record_native_prefix_adoption'''
assert s.count(needle)==1;s=s.replace(needle,replacement)
s=s.replace('expected[:count],request.request_id,DIRECTORY)','expected[:count],request.request_id,DIRECTORY,published[\'_audit_transfer_job_id\'])')
ast.parse(s);p.write_text(s)
s=(H/'run_spaced.py').read_text().replace("str(H/('quality_hooks_readmission' if a.quality else 'hooks-defer'))","str(H/'hooks-byte-qualified')")
(H/'run_spaced_qualification.py').write_text(s)
b=json.loads((H/'SPACED.json').read_text());p=copy.deepcopy(b)
r=copy.deepcopy(next(r for r in b['runs'] if r['arm']=='candidate' and r['gpu_blocks']==520));r['name']='spaced-byte-qualification-b520-candidate-s1103';c=r['command'];c[2]=str(H/'run_spaced_qualification.py');c[c.index('--name')+1]=r['name'];c.extend(['--audit','--audit-local-prefix-ids','1,2']);p['runs']=[r]
p['progress_file']='spaced-byte-qualification-progress.json';p['objective']='Intrusive same-model/path byte qualification only, not performance: actual D2H/H2D equality for every copied component and 1-byte wrong control; owned pre-H2D local prefix snapshots keyed by exact transfer job and checked at positive native adoption.'
for n in ['byte_audit_qualified.py','hooks-byte-qualified/sitecustomize.py','run_spaced_qualification.py','work_inventory.py']:
 p['pins'][n]=hashlib.sha256((H/n).read_bytes()).hexdigest()
(H/'SPACED-BYTE-QUALIFICATION.json').write_text(json.dumps(p,indent=2)+'\n')
print('Prepared intrusive qualification with exact transfer-job snapshot join')
