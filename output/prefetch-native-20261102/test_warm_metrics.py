import campaign_metrics_warm as metrics
def test_cache_warming_closes_reservation_before_later_retirement_and_reuse():
 events=[];completions=[]
 for eid,job,start,warm,end in [('a',1,100,110,200),('b',2,120,130,180)]:
  def event(kind,**kw):return dict(event=kind,event_id=eid,**kw)
  events += [event('staging_submitted',submitted_at_ns=start,job_id=job,reserved_block_ids=[4],submitted_bytes=16,bytes_per_key=16),
             event('staging_ready',ready_at_ns=warm,job_id=job,transfer_bytes=16),
             event('staging_cache_warmed',observed_at_ns=warm,released_reservation_blocks=1),
             event('staging_retired',retired_at_ns=end,released_bytes=0,reason='retention_expired')]
  completions.append(dict(event_id=eid,job_id=job,success=True,cancelled=False,completed_at_ns=warm,bytes=16))
 u=metrics.staging_utility(dict(wait_hint_events=events,speculative_transfer_completions=completions,terminals=[]),0,300)
 assert u['global_explicit_successful_completion_bytes_observed']==32
 assert u['global_named_demand_consumed_bytes']==0
 assert u['completed_unconsumed_retired_bytes_by_category']['expired']==32
 assert u['reservation_highwater']['known_closed_interval_peak_unique_blocks']==1
 assert not u['reservation_highwater']['physical_overlap_errors']
 assert u['highwater_complete_for_submitted_attempts']
 assert all(r['reservation_end_event']=='staging_cache_warmed' for r in u['event_rows'])
def test_pending_fence_retirement_is_not_a_reservation_release():
 events=[dict(event='staging_submitted',event_id='a',submitted_at_ns=100,job_id=1,reserved_block_ids=[4],submitted_bytes=16,bytes_per_key=16),
         dict(event='staging_retired_pending_fence',event_id='a',retired_at_ns=110),
         dict(event='staging_retired',event_id='a',retired_at_ns=150,released_bytes=16,reason='cancelled')]
 u=metrics.staging_utility(dict(wait_hint_events=events,speculative_transfer_completions=[],terminals=[]),0,300)
 assert u['event_rows'][0]['reservation_end_ns']==150
 assert u['event_rows'][0]['reservation_end_event']=='staging_retired'
