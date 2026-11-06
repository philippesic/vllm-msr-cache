import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import analyze
H=Path(__file__).parent
rows=[]
for p in sorted((H/'runs').glob('*/raw.json.gz')):
 raw,s,requests,profiles=analyze.analyze(p)
 rows.append(s)
 if s['status']!='complete':
  print(s['name'],s['status'],s.get('error'));continue
 print(s['name'],s['status'],round(s['all']['output_tokens_s'],3),round(s['all']['release_ttft_ms']['mean'],1),round(s['resources']['cpu_s_sampled'],2),s['worker_capture_calls'],s['manager_counts'].get('relevance_scored_rows'),s['manager_counts'].get('semantic_evict_calls_with_scores'),s['transfer_failed_jobs'])
(H/'summaries.json').write_text(json.dumps(rows,indent=2))
