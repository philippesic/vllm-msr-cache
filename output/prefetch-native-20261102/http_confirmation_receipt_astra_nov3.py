"""Read-only HTTP accounting, including separately matched intentional hint stops."""
import collections

def verify_http(r):
 cfg=r.get('http_client_configuration');events=r.get('http_connection_events');requests=r.get('requests');hints=r.get('hints');issues=[]
 if not isinstance(cfg,dict) or not all(isinstance(x,list) for x in (events,requests,hints)):
  return {'available':False,'qualified':False,'issues':['missing_accounting'],'counts':None}
 for k,v in {'force_close':True,'automatic_POST_retry':False,'application_retries':0,'connection_trace':True}.items():
  if cfg.get(k)!=v:issues.append('configuration_'+k)
 expected={('generation',q['index']):q for q in requests}
 if len(expected)!=len(requests):issues.append('duplicate_generation_index')
 for q in hints:
  key=('hint',q.get('hint',{}).get('event_id'))
  if key in expected:issues.append('duplicate_hint_id')
  expected[key]=q
 groups=collections.defaultdict(list)
 for e in events:
  req=e.get('request') or {};key=(req.get('kind'),req.get('index') if req.get('kind')=='generation' else req.get('event_id'));groups[key].append(e)
 if set(groups)!=set(expected):issues.append('event_request_keys_mismatch')
 counts=collections.Counter(e.get('event') for e in events);bad=[];canceled=[];completed=collections.Counter();unexpected=[]
 full=['request_start','connection_create_start','connection_create_end','request_headers_received']
 for key,seq in groups.items():
  row=expected.get(key,{});labels=[e.get('event') for e in seq];valid=True
  stop=key[0]=='hint' and row.get('error')=='producer_stopped'
  exc=[e for e in seq if e.get('event')=='request_exception']
  if stop:
   # Cancellation can occur before headers or during response-body receipt.
   valid=(labels==full and not exc) or (len(exc)==1 and exc[0].get('error')=='CancelledError()' and labels[-1]=='request_exception' and labels[:-1]==full[:len(labels)-1] and 1<=len(labels)-1<=3)
   valid &= row.get('completed_ns',-1)>=seq[-1].get('at_ns',0)
   canceled.append({'event_id':key[1],'trace_exception':exc[0].get('error') if exc else None,'events':labels,'completed_ns':row.get('completed_ns')})
  else:
   valid=labels==full and row.get('http_status')==200 and not row.get('error')
   unexpected.extend({'kind':key[0],'id':key[1],**e} for e in exc)
   if valid:completed[key[0]]+=1
  valid &= all(e.get('status')==200 for e in seq if e.get('event')=='request_headers_received')
  valid &= all(isinstance(e.get('at_ns'),int) for e in seq) and [e['at_ns'] for e in seq]==sorted(e['at_ns'] for e in seq)
  if not valid:bad.append({'kind':key[0],'id':key[1],'events':labels,'raw_http_status':row.get('http_status'),'error':row.get('error')})
 if bad:issues.append('invalid_per_request_trace')
 if counts.get('connection_reuse',0):issues.append('connection_reuse_observed')
 if unexpected:issues.append('unexpected_transport_exception')
 if r.get('planned_requests')!=len(requests):issues.append('planned_generation_count_mismatch')
 return {'available':True,'qualified':not issues,'issues':issues,'configuration':cfg,'generation_rows':len(requests),'hint_rows':len(hints),'expected_total_HTTP_requests':len(expected),'counts':{k:counts.get(k,0) for k in full+['connection_reuse','request_exception']},'completed_HTTP200':dict(completed),'intentional_hint_cancellations':canceled,'unexpected_transport_exceptions':unexpected,'per_request_mismatches':bad,'scope':'Intentional hint stops are retained, event-ID matched and counted separately from completed HTTP requests. Headers are not SSE completion; natural EOS is checked separately.'}
