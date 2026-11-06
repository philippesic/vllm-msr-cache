"""Exercise the actual pinned native logger and our reader at a quiet boundary."""
import json,tempfile
from pathlib import Path
from vllm.compilation.cuda_graph import CUDAGraphLogging,CUDAGraphStat
from vllm.config import CUDAGraphMode
from benchmarks.research_checks.qualification_graph_receipts import TABLE_ROW
b=Path(__file__).resolve().parent
logger=CUDAGraphLogging(CUDAGraphMode.FULL_DECODE_ONLY,[1,2,4])
for _ in range(3):logger.observe(CUDAGraphStat(2,4,2,str(CUDAGraphMode.FULL)))
logger.observe(CUDAGraphStat(17,17,0,str(CUDAGraphMode.NONE)))
texts=[];logger.log(log_fn=texts.append)
assert not logger.stats
logger.log(log_fn=texts.append)
assert len(texts)==1
rows=[dict(unpadded=int(m[1]),padded=int(m[2]),paddings=int(m[3]),mode=m[4].strip(),count=int(m[5])) for m in TABLE_ROW.finditer(texts[0])]
assert len(rows)==2 and sum(r['count'] for r in rows if r['mode'].rsplit('.',1)[-1]=='FULL')==3
assert sum(r['count'] for r in rows)==4
(b/'native-graph-logger-check.json').write_text(json.dumps(dict(passed=True,rows=rows,native_table=texts[0],scope='Actual pinned native CPU logger formatting, aggregation and disjoint-interval reset. No GPU graph execution claim.'),indent=2)+'\n')
print('Actual native graph logger: aggregation, parser and reset passed')
