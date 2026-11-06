"""Check task-evidence Markdown links against the finished archive members."""
import hashlib,json,posixpath,re,sys,tarfile,time
from pathlib import Path
from urllib.parse import unquote,urlsplit
archive=Path(sys.argv[1]);base=Path(__file__).resolve().parent.name
with tarfile.open(archive,'r:xz') as tar:
 names=set();documents={}
 for member in tar:
  names.add(member.name)
  if member.name=='README.md' or member.name.startswith('source/evidence/'+base+'/') and member.name.endswith('.md'):
   documents[member.name]=tar.extractfile(member).read().decode()
resolved=[];historical=[];missing=[]
for name,body in documents.items():
 for raw in re.findall(r'\[[^\]\n]*\]\(([^)\n]+)\)',body):
  target=raw.strip().strip('<>')
  parsed=urlsplit(target)
  if parsed.scheme or not parsed.path or parsed.path.startswith('/'):continue
  relative=unquote(parsed.path);path=posixpath.normpath(posixpath.join(posixpath.dirname(name),relative));entry=dict(document=name,target=target,archive_path=path)
  if path in names or any(x.startswith(path.rstrip('/')+'/') for x in names):resolved.append(entry)
  elif path.startswith('source/evidence/') and not path.startswith('source/evidence/'+base+'/') or path.startswith('source/output/') and not path.startswith('source/output/'+base+'/'):
   historical.append(entry)
  else:missing.append(entry)
r=dict(passed=not missing,checked_unix=time.time(),archive=str(archive.resolve()),documents=len(documents),resolved=len(resolved),resolved_links=resolved,historical_links_not_bundled=historical,missing_task_or_source_links=missing,scope='Finished-archive relative Markdown-link resolution for the final task evidence and root README. Prior-task evidence/output links are explicitly historical and outside this bundle; code/docs/current-task raw proof links must resolve. No HTTP availability or anchor validation.')
out=archive.with_suffix(archive.suffix+'.links.json');out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ('passed','documents','resolved','missing_task_or_source_links')}))
if missing:raise SystemExit(2)
