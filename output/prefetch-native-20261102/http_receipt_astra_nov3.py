"""Strict read-only verification of common fresh-connection HTTP accounting."""
import collections

def verify_http(r):
 cfg=r.get('http_client_configuration');events=r.get('http_connection_events');requests=r.get('requests');hints=r.get('hints');issues=[]
 if not isinstance(cfg,dict):issues.append('missing_configuration')
 if not isinstance(events,list):issues.append('missing_event_list')
 if not isinstance(requests,list):issues.append('missing_request_list')
 if not isinstance(hints,list):issues.append('missing_hint_list')
 if issues:return {'available':False,'qualified':False,'issues':issues,'counts':None}
 for k,v in {'force_close':True,'automatic_POST_retry':False,'application_retries':0,'connection_trace':True}.items():
  if k not in cfg or cfg[k]!=v:issues.append('configuration_'+k)
 expected={('generation',q['index']):q.get('http_status') for q in requests}
 if len(expected)!=len(requests):issues.append('duplicate_generation_indices')
 for q in hints:
  key=('hint',q.get('hint',{}).get('event_id'))
  if key in expected:issues.append('duplicate_hint_id')
  expected[key]=q.get('http_status')
 groups=collections.defaultdict(list)
 for e in events:
  req=e.get('request') or {};key=(req.get('kind'),req.get('index') if req.get('kind')=='generation' else req.get('event_id'));groups[key].append(e)
 counts=collections.Counter(e.get('event') for e in events)
 if set(groups)!=set(expected):issues.append('event_request_keys_do_not_match_raw_rows')
 required={'request_start':1,'connection_create_start':1,'connection_create_end':1,'request_headers_received':1}
 bad=[]
 for key,seq in groups.items():
  c=collections.Counter(e.get('event') for e in seq)
  if dict(c)!=required or expected.get(key)!=200 or any(e.get('status')!=200 for e in seq if e['event']=='request_headers_received') or [e.get('at_ns') for e in seq]!=sorted(e.get('at_ns') for e in seq):bad.append({'kind':key[0],'id':key[1],'events':dict(c),'raw_http_status':expected.get(key)})
 if bad:issues.append('incomplete_or_invalid_per_request_trace')
 if counts.get('connection_reuse',0):issues.append('connection_reuse_observed')
 if counts.get('request_exception',0):issues.append('request_exception_observed')
 if r.get('planned_requests')!=len(requests):issues.append('planned_generation_count_mismatch')
 return {'available':True,'qualified':not issues,'issues':issues,'configuration':cfg,'generation_rows':len(requests),'hint_rows':len(hints),'expected_total_HTTP_requests':len(expected),'counts':{k:counts.get(k,0) for k in ['request_start','connection_create_start','connection_create_end','request_headers_received','connection_reuse','request_exception']},'per_request_mismatches':bad,'scope':'Zero event counts require an existing complete event list with exact start/create/HTTP200-header accounting for every generation/hint row. Header-received trace is not SSE completion; naturalEOS separately verifies generation completion.'}
