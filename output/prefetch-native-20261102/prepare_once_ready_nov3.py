"""Create separate, reviewable discovery harness; preserve all frozen harnesses."""
from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'live_service_workload_nov3.py').read_text()
s=s.replace('refresh_ms=200):','refresh_ms=200):',1)
needle="   if row['response'].get('status')=='accepted':state['binding']=row['response'].get('canonical_event_id') or event\n"
assert s.count(needle)==1
s=s.replace(needle,needle+"   ack=row['response']\n   ready=ack.get('ready_cpu_prefix_keys')\n   if (ack.get('status')=='no_action' and ack.get('reason')=='cpu_prefix_recency_refreshed' and isinstance(ready,int) and not isinstance(ready,bool) and ready>0):\n    row['producer_policy_stop']='first_positive_cpu_recency_ack'\n    return\n")
needle="'policy_refresh_ms':a.hint_interval_ms,"
assert s.count(needle)==1
s=s.replace(needle,needle+"'hint_emission_policy':'retry_while_pending_until_first_positive_cpu_recency_ack',")
p=H/'live_once_service_workload_nov3.py';assert not p.exists();p.write_text(s)
s=(H/'run_live_bridge_nov3.py').read_text().replace('live_service_workload_nov3','live_once_service_workload_nov3')
p=H/'run_live_once_ready_nov3.py';assert not p.exists();p.write_text(s)
print('Created separate workload and runner; frozen dependencies untouched')
