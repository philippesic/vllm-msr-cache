"""Restore ordinary numeric spacing in final prose without changing pins/paths."""
from pathlib import Path
import re
root=Path(__file__).resolve().parents[2];e=root/'evidence/wait-qualification-20261105'
files=[e/n for n in ('README.md','STAGE3.md','STAGE4.md','SUPPORT_MATRIX.md','REGRESSIONS.md','QUALITY.md','REPRODUCING.md')]+[root/'docs/WAIT_HINT_SUPPORT.md',root/'docs/WAIT_HINT_PRODUCT_PLAN.md']
words='is|has|to|A|in|short|long|unquantized|coverage|audited|default|The|the|All|all|Both|both|its|their|for|within|than|by|of|with|before|after|about|over|under|versus|retain|retains|uses|use|executes|generates|completes|completed|conditional|declared|selected|additional|primary|focused|benchmark|telemetry|portable|isolated|copies|copied|Stage|stage|across|and|pass|passed|cover|covers|exposure|budget|constrained|Only|only'
suffixes='context|tokens|finals|reuses|diagnostic|less|more|responses|response|tools|tool|workflows|workflow|requests|request|cases|case|cells|cell|pairs|pair|waves|wave|first|final|timed|dependencies|dependency|files|file|package|packages|focused|benchmark|qualification|checks|tests|source|tasks|task|neutral|useful|active|point|MiB|GiB|ARC|wait|under|overhead'
def part(s):
 s=re.sub(r'\b('+words+r')(?=\d)',r'\1 ',s,flags=re.I)
 s=re.sub(r'(\d)('+suffixes+r')\b',r'\1 \2',s)
 s=re.sub(r'%(?=[A-Za-z])','% ',s)
 s=re.sub(r';(?=\S)','; ',s)
 s=re.sub(r'(\d)(MiB|GiB)',r'\1 \2',s)
 s=re.sub(r'\b(MiB|GiB)(?=[a-z])',r'\1 ',s)
 return s
for p in files:
 text=p.read_text();lines=[];fenced=False
 for line in text.splitlines(keepends=True):
  if line.lstrip().startswith('```'):fenced=not fenced;lines.append(line);continue
  if fenced:lines.append(line);continue
  segments=re.split(r'(`[^`]*`|\[[^\]]*\]\([^)]*\))',line)
  lines.append(''.join(s if i%2 else part(s) for i,s in enumerate(segments)))
 new=''.join(lines);p.write_text(new)
print('Polished final narrative spacing; code blocks, inline code, links and version/hash tokens preserved')
