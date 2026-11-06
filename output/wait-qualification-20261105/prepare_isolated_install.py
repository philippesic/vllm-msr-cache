"""Create an owned source copy; execute later at a quiet measurement boundary."""
from pathlib import Path
import hashlib,json,shutil,subprocess,time
b=Path(__file__).resolve().parent;root=b.parents[1];target=b/'isolated-reproduction';checkout=target/'source'
if target.exists():raise FileExistsError('Preserve existing isolation attempt')
paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).decode().split('\0')
allowed={'semantic_offload','benchmarks','harness','tools','tests','examples','patches','docs'}
root_files={'Makefile','pyproject.toml','requirements-runtime.txt','requirements-checks.txt','AGENTS.md','.gitignore','README.md'}
selected=[]
for name in paths:
 if not name:continue
 p=Path(name)
 if p.parts[0] in allowed or name in root_files or name=='vendor/VLLM.md' or name=='vendor/vllm/LICENSE' or name.startswith('vendor/vllm/vllm/'):
  if any(x in p.parts for x in ('__pycache__','.pytest_cache','.git')) or p.suffix=='.pyc':continue
  selected.append(name)
checkout.mkdir(parents=True)
files={}
for name in sorted(set(selected)):
 source=root/name
 if not source.is_file():raise RuntimeError('Non-file source input: '+name)
 if source.is_symlink():raise RuntimeError('Review symlink before isolation: '+name)
 destination=checkout/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
 sha=hashlib.sha256(source.read_bytes()).hexdigest()
 if hashlib.sha256(destination.read_bytes()).hexdigest()!=sha:raise RuntimeError('Copy mismatch')
 files[name]=sha
(target/'source-copy.json').write_text(json.dumps(dict(created_unix=time.time(),author_run=True,independent_human=False,original=str(root),copy=str(checkout),files=files,scope='Explicit project source/test/docs roots and pinned runtime sources; no Git metadata, personal configuration, output artifacts, model weights or shared environment mutation.'),indent=2)+'\n')
print(checkout,len(files),'files')
