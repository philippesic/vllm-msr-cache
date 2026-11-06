import ast, difflib, hashlib, json, shutil, subprocess, tarfile
from pathlib import Path
H=Path(__file__).resolve().parent
BASE=Path('/home/philip/.local/share/venvs/vllm-semantic-cache-ext/lib/python3.12/site-packages')
names=['vllm/v1/engine/async_llm.py','vllm/v1/engine/core.py','vllm/v1/engine/core_client.py','vllm/v1/core/sched/scheduler.py']
records=[];diff=[]
with tarfile.open(H.parent/'opportunity-runtime-recovery/recovered-vllm-runtime.tar.gz') as t:
    for name in names:
        member=next(x for x in t.getmembers() if x.name.endswith('/'+name))
        data=t.extractfile(member).read();before=(BASE/name).read_bytes()
        target=H/'runtime'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        records.append({'module':name,'installed_sha256':hashlib.sha256(before).hexdigest(),'overlay_sha256':hashlib.sha256(data).hexdigest()})
        diff.extend(difflib.unified_diff(before.decode().splitlines(True),data.decode().splitlines(True),fromfile='installed/'+name,tofile='overlay/'+name))
(H/'runtime-overlay.patch').write_text(''.join(diff))
(H/'runtime-overlay.json').write_text(json.dumps(records,indent=2)+'\n')
name='vllm/v1/core/block_pool.py';target=H/'runtime'/name;before=(BASE/name).read_bytes();target.write_bytes(before)
patch=H.parents[1]/'patches/vllm-native-prefix-adoption.patch'
argv=['patch','--batch','-p1','-d',str(H/'runtime'),'-i',str(patch)]
subprocess.run([*argv,'--dry-run'],check=True,capture_output=True)
subprocess.run(argv,check=True,capture_output=True)
data=target.read_bytes();records.append({'module':name,'installed_sha256':hashlib.sha256(before).hexdigest(),'overlay_sha256':hashlib.sha256(data).hexdigest(),'native_adoption_disabled_in_fp16_runs':True})
diff.extend(difflib.unified_diff(before.decode().splitlines(True),data.decode().splitlines(True),fromfile='installed/'+name,tofile='overlay/'+name))
(H/'runtime-overlay.patch').write_text(''.join(diff));(H/'runtime-overlay.json').write_text(json.dumps(records,indent=2)+'\n')
source=(H.parent/'scoring-audit/round4-results/round4/run.py').read_text()
functions=[ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in {'digest','rid','payload','exercise'}]
text='import asyncio, hashlib, json, time\nfrom benchmarks.eviction_prefetch_campaign import consume_campaign_sse\nfrom pathlib import Path\n'
text+='def namespace(p): return p["namespace"]\n'
text+='def parse(path,marker):\n return [json.loads(x.split("SEMANTIC_PRESERVATION_EVENT ",1)[1]) for x in Path(path).read_text(errors="replace").splitlines() if "SEMANTIC_PRESERVATION_EVENT " in x]\n'
text+='\n\n'.join(functions)
text=text.replace("'model':'local-Qwen3-4B'","'model':'matrix-model'")
(H/'workload.py').write_text(text+'\n')
