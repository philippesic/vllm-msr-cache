"""Actual per-call queue operation accounting; exclude cached replay from totals."""
from cadence_receipt_astra_nov3 import verify_cadence

def verify_promotion(raw,arm):
 mode='once_positive' if arm=='promotion_once_positive' else 'native' if arm=='native' else 'periodic'
 cadence=verify_cadence(raw,mode);expected=arm.startswith('promotion_');issues=[];acknowledgements=[];replayed=[]
 hints=raw.get('hints')
 if not isinstance(hints,list):return {'qualified':False,'issues':['missing_hint_rows'],'cadence':cadence}
 for h in hints:
  ack=h.get('response',{})
  if ack.get('reason')!='cpu_prefix_recency_refreshed':continue
  marked=ack.get('cpu_advice_queue_policy')=='experimental_ready_resident_T2_promotion'
  if marked!=expected:issues.append('wrong_queue_policy_activation')
  if not expected:continue
  values=[ack.get(k) for k in ['cpu_advice_promoted_t1_keys','cpu_advice_refreshed_t2_keys','ready_cpu_prefix_keys']]
  if not all(type(v) is int and v>=0 for v in values) or values[0]+values[1]!=values[2]:issues.append('invalid_or_unreconciled_queue_counts');continue
  row={'event_id':h['hint'].get('event_id'),'session_id':h['hint'].get('session_id'),'promoted_t1_keys':values[0],'refreshed_t2_keys':values[1],'ready_keys':values[2]}
  (replayed if ack.get('idempotent_replay') is True else acknowledgements).append(row)
 if expected and not acknowledgements:issues.append('no_actual_policy_acknowledgements')
 return {'qualified':cadence['qualified'] and not issues,'issues':issues,'cadence':cadence,'expected_experimental_policy':expected,'nonreplayed_action_acknowledgements':len(acknowledgements) if expected else None,'replayed_acknowledgements_excluded':len(replayed) if expected else None,'new_T1_to_T2_key_operations':sum(x['promoted_t1_keys'] for x in acknowledgements) if expected else None,'existing_T2_refresh_key_operations':sum(x['refreshed_t2_keys'] for x in acknowledgements) if expected else None,'total_ready_key_operations':sum(x['ready_keys'] for x in acknowledgements) if expected else None,'acks_with_actual_promotion':sum(x['promoted_t1_keys']>0 for x in acknowledgements) if expected else None,'all_action_acknowledgements':acknowledgements,'scope':'Actual manager per-call operation counts, not unique lifetime keys, demand hits, CPU residence guarantee or future restoration. Cached replay excluded; old control counters unavailable rather than zero.'}
