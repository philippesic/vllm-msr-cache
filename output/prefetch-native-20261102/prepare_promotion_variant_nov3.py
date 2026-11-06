"""Isolated experimental source; never modifies checkout or frozen sources."""
from pathlib import Path
import shutil,hashlib,json
H=Path(__file__).resolve().parent;S=H/'sources/nov3-hardened/semantic_offload';D=H/'sources/nov3-advice-promotion/semantic_offload';D.mkdir(parents=True,exist_ok=False)
for p in S.glob('*.py'):shutil.copyfile(p,D/p.name)
p=D/'manager.py';s=p.read_text();a=s.index('        """Refresh ready ARC residents without promoting speculative accesses.');b=s.index('\n    def __init__(',a)
s=s[:a]+'''        """EXPERIMENT: promote ready advised residents into ARC's T2 queue.

        This is predicted reuse, not measured demand. It changes eviction
        competition while preserving residency, refs, ghost history and target.
        It can pollute T2; default checkout behavior is intentionally different.
        """
        from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
        policy = self._policy
        if not isinstance(policy, ARCCachePolicy):
            return 0
        refreshed = 0
        for key in reversed(list(keys)):
            block = policy.get(key)
            if block is None or not block.is_ready:
                continue
            if key in policy.t1:
                policy.t2[key] = policy.t1.pop(key)
            else:
                policy.t2.move_to_end(key)
            refreshed += 1
        return refreshed
''' +s[b:];p.write_text(s)
p=D/'connector.py';s=p.read_text();needle='expires_at_ns=expires_at_ns, ready_cpu_prefix_keys=ready_count,\n';assert s.count(needle)==1;s=s.replace(needle,needle+'                cpu_advice_queue_policy="experimental_ready_resident_T2_promotion",\n');p.write_text(s)
result={'schema':'isolated-predicted-reuse-promotion-source-v1','source':str(D.parent),'base_source':str(S.parent),'production_checkout_unchanged':True,'scope':'Experimental source selection is explicit opt-in. Ready CPU prefix only, ordinary native ARC eviction remains allowed. No measured-hit claim, no GPU refs, transfer or persistent state. T1/T2 competition changes and may regress.','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.glob('*.py'))},'changed_from_base':[p.name for p in D.glob('*.py') if p.read_bytes()!=(S/p.name).read_bytes()]}
(H/'live-nov3-advice-promotion-source.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'source':str(D.parent),'changed':result['changed_from_base']}))
