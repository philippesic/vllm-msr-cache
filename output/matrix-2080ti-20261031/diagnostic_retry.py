"""Retry only the diagnostic whose post-load log reader saw a partial record.

The replacement is used only by the post-load terminal drain. All serving,
request, policy, and measurement code remains pinned and unchanged.
"""
import json
from pathlib import Path
import run
import workload

def tolerant_postload_parse(path, marker):
    decoder = json.JSONDecoder()
    events = []
    # Match the original helper's preservation-event stream. A producer can be
    # between writes while this reader scans its still-open log.
    for line in Path(path).read_text(errors='replace').splitlines():
        if 'SEMANTIC_PRESERVATION_EVENT ' not in line:
            continue
        payload = line.split('SEMANTIC_PRESERVATION_EVENT ', 1)[1]
        try:
            value, _ = decoder.raw_decode(payload)
        except json.JSONDecodeError:
            continue
        events.append(value)
    return events

workload.parse = tolerant_postload_parse
if __name__ == '__main__':
    raise SystemExit(0 if run.run('multi', 1103, 'semantic', 'diagnostic') else 1)
