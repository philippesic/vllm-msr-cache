"""Review changes since the preserved initial workspace; never mutates project files."""
import difflib,hashlib,json,subprocess,tarfile
from pathlib import Path
b=Path(__file__).resolve().parent;root=b.parents[1];initial=json.loads((b/'preexisting-files.json').read_text());status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all','-z'],cwd=root).decode().split('\0');paths=set(initial);i=0
while i<len(status):
 item=status[i];i+=1
 if not item:continue
 paths.add(item[3:])
 if item[0] in 'RC' or item[1] in 'RC':
  paths.add(status[i]);i+=1
patch=[];rows=[]
with tarfile.open(b/'preexisting-work.tar.gz','r:gz') as archive:
 members={m.name.removeprefix('./'):m for m in archive.getmembers() if m.isfile()}
 for name in sorted(paths):
  p=root/name
  if p.exists() and not p.is_file():continue
  if name in initial:
   m=members.get(name)
   if m is None:raise RuntimeError('Initial archive member missing: '+name)
   before=archive.extractfile(m).read();origin='initial dirty workspace'
   if hashlib.sha256(before).hexdigest()!=initial[name]:raise RuntimeError('Initial recovery archive mismatch: '+name)
  else:
   run=subprocess.run(['git','show','HEAD:'+name],cwd=root,capture_output=True)
   before=run.stdout if run.returncode==0 else b'';origin='HEAD' if run.returncode==0 else 'new since snapshot'
  after=p.read_bytes() if p.is_file() else b''
  if before==after:continue
  row=dict(path=name,baseline=origin,before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(after).hexdigest(),before_bytes=len(before),after_bytes=len(after),present=p.is_file());rows.append(row)
  try:
   old=before.decode().splitlines(keepends=True);new=after.decode().splitlines(keepends=True)
   patch.extend(difflib.unified_diff(old,new,fromfile='initial/'+name,tofile='current/'+name))
  except UnicodeDecodeError:patch.append('Binary data differs: '+name+'\n')
(b/'changes-since-initial-snapshot.patch').write_text(''.join(patch))
(b/'changes-since-initial-snapshot.json').write_text(json.dumps(dict(files=rows,scope='Byte-level changes since the preserved initial workspace. Original dirty files are compared with recovery archive; previously clean/new files with HEAD/empty. This inventory does not by itself assign authorship if concurrent edits occurred.'),indent=2)+'\n')
print('Changed since initial snapshot:',len(rows),'files')
