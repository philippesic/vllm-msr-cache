from pathlib import Path
import hashlib,json,shutil
H=Path(__file__).resolve().parent;D=H/'sources/nov3-advice-promotion/semantic_offload'
for name in ['manager.py','connector.py']:
 p=D/name;q=H/('promotion-pretelemetry-'+name+'.saved');assert not q.exists();shutil.copyfile(p,q)
p=D/'manager.py';s=p.read_text();needle='def refresh_wait_hint_cpu_recency(self, keys: Collection[OffloadKey]) -> int:';assert s.count(needle)==1
s=s.replace(needle,'def refresh_wait_hint_cpu_recency(\n        self, keys: Collection[OffloadKey], *, return_stats: bool = False,\n    ) -> int | tuple[int, dict[str, int]]:')
a=s.index('        """EXPERIMENT:');b=s.index('\n    def __init__(',a);part=s[a:b]
part=part.replace('        if not isinstance(policy, ARCCachePolicy):\n            return 0\n        refreshed = 0','        stats = {"cpu_advice_promoted_t1_keys": 0,\n                 "cpu_advice_refreshed_t2_keys": 0}\n        if not isinstance(policy, ARCCachePolicy):\n            return (0, stats) if return_stats else 0\n        refreshed = 0').replace('                policy.t2[key] = policy.t1.pop(key)','                policy.t2[key] = policy.t1.pop(key)\n                stats["cpu_advice_promoted_t1_keys"] += 1').replace('                policy.t2.move_to_end(key)','                policy.t2.move_to_end(key)\n                stats["cpu_advice_refreshed_t2_keys"] += 1').replace('        return refreshed','        return (refreshed, stats) if return_stats else refreshed')
s=s[:a]+part+s[b:];p.write_text(s)
p=D/'connector.py';s=p.read_text();needle='            ready_count = refresh(ready) if callable(refresh) else 0\n';assert s.count(needle)==1
s=s.replace(needle,'            ready_count, queue_details = (\n                refresh(ready, return_stats=True) if callable(refresh) else\n                (0, {"cpu_advice_promoted_t1_keys": 0,\n                     "cpu_advice_refreshed_t2_keys": 0})\n            )\n')
needle='                cpu_advice_queue_policy="experimental_ready_resident_T2_promotion",\n';assert s.count(needle)==1;s=s.replace(needle,needle+'                **queue_details,\n');p.write_text(s)
p=H/'live-nov3-advice-promotion-source.json';old=json.loads(p.read_text());old['files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.glob('*.py'))};old['per_call_telemetry']='Newly promoted T1 key touches versus already-T2 refreshed touches; local ephemeral counters, no manager history. Replayed ACK describes original action and must not be double counted.';p.write_text(json.dumps(old,indent=2)+'\n')
