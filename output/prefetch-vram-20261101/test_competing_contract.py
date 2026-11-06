"""Fast stdlib-only contracts: no vLLM, tokenizer, torch, network or GPU."""
import ast
import asyncio
import hashlib
import importlib.util
import json
import time
import types
import unittest
import sys
from pathlib import Path

H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parents[1]))
from semantic_offload.wait_hint import KnownPrefixWaitHint
assert 'torch' not in sys.modules
NAMESPACE={'model_revision':'frozen','tokenizer_revision':'frozen','adapter_namespace':'none','layout_namespace':'fp16-block16'}
spec=importlib.util.spec_from_file_location('competing_preparer_contract',H/'prepare_competing.py')
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
source=(H/'competing_workload.py').read_text();tree=ast.parse(source)
selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))
          and node.name in ('digest','rid','_publish_hint','_hint_policy','_completed_observed_prefix')]
module={'asyncio':asyncio,'hashlib':hashlib,'json':json,'time':time}
exec(compile(ast.Module(body=selected,type_ignores=[]),str(H/'competing_workload.py'),'exec'),module)


class CompletedTranscriptContracts(unittest.TestCase):
    def test_final_sampled_token_is_never_nominated(self):
        source=list(range(32));output=list(range(100,132));prefix=module['_completed_observed_prefix']
        self.assertEqual(prefix(source,output[:16]),tuple(source))
        self.assertEqual(prefix(source,output[:17]),tuple(source+output[:16]))
        self.assertEqual(prefix(source,output),tuple(source+output[:16]))
        # Known-wrong nomination at a block boundary would include uncomputed KV.
        wrong=tuple(source+output[:16])
        self.assertNotEqual(prefix(source,output[:16]),wrong)


class DatasetContracts(unittest.TestCase):
    def test_complete_dataset_geometry_and_source_pins(self):
        for seed in prepare.SEEDS:
            p=json.loads((H/'prepared'/f'competing-{seed}.json').read_text())
            prepare.validate(p)
            fresh = prepare.build(seed)
            pinned = p['dataset_contract']['workload_sha256']
            historical = [q for q in H.glob('harness-v*/competing_workload.py') if hashlib.sha256(q.read_bytes()).hexdigest()==pinned]
            self.assertTrue(historical, 'historical workload pin must remain recoverable')
            fresh['dataset_contract']['workload_sha256'] = pinned
            self.assertEqual(p,fresh)
            contract=p['dataset_contract']
            self.assertEqual(contract['requests'],18)
            self.assertEqual(contract['output_tokens'],768)
            self.assertEqual(contract['minimum_simultaneous_return_blocks_including_null'],207)
            self.assertEqual(contract['workload_sha256'],prepare.sha(historical[0]))
            self.assertEqual(contract['preparer_sha256'],prepare.sha(H/'prepare_competing.py'))

    def test_producers_have_no_driver_private_bindings(self):
        forbidden={'p','round_data','round_index','lead_ms','delays','peer_complete',
                   'common_release','warm','tokens','private'}
        for node in selected:
            if node.name not in ('_publish_hint','_hint_policy'):continue
            names={n.id for n in ast.walk(node) if isinstance(n,ast.Name)}
            self.assertFalse(names&forbidden,(node.name,names&forbidden))
        compile(source,str(H/'competing_workload.py'),'exec')


class Response:
    status=200
    def __init__(self,reply):self.reply=reply
    async def __aenter__(self):return self
    async def __aexit__(self,*args):return False
    async def json(self):return self.reply


class ProducerContracts(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_ack_uses_canonical_binding_and_only_observed_prefix(self):
        observed=(1,2,3,4);namespace=NAMESPACE
        state={'pending':True,'completed_ns':time.monotonic_ns(),'changed':asyncio.Event()}
        records=[];sent=[]
        class Client:
            def post(self,url,json):
                KnownPrefixWaitHint.from_mapping(json)
                sent.append((url,json))
                return Response({'status':'accepted','canonical_event_id':'canonical-owned-event'})
        await module['_publish_hint'](Client(),'http://unused',namespace,state,4,'agent',observed,0,records,250)
        await module['_publish_hint'](Client(),'http://unused',namespace,state,4,'agent',observed,1,records,250)
        self.assertEqual(state['binding'],'canonical-owned-event')
        self.assertNotEqual(sent[0][1]['event_id'],sent[1][1]['event_id'])
        self.assertEqual(len(records),2)
        for _,hint in sent:
            parsed=KnownPrefixWaitHint.from_mapping(hint)
            self.assertEqual(parsed.wait_provenance,'driver_declared_tool_wait_budget')
            wrong=dict(hint,wait_provenance='constant_driver_tool_wait_estimate')
            with self.assertRaisesRegex(ValueError,'provenance mismatch'):
                KnownPrefixWaitHint.from_mapping(wrong)
            self.assertEqual(hint['declared_wait_ms'],250)
            self.assertEqual(hint['prefix_boundary_tokens'],4)
            self.assertEqual(hint['observed_prefix_id'],module['digest']({'namespace':namespace,'token_ids':list(observed)}))
            self.assertNotIn('intended_release_ns',hint)
            self.assertNotIn('driver_private_return_delays_ms',hint)
            self.assertNotIn('future_input',hint)

    async def test_accepted_policy_continues_refresh_and_respects_public_gate(self):
        state={'pending':True,'completed_ns':time.monotonic_ns(),'changed':asyncio.Event()}
        records=[];sent=[];activation=asyncio.Event()
        class Client:
            def post(self,url,json):
                KnownPrefixWaitHint.from_mapping(json)
                sent.append(json)
                if len(sent)==2:state['pending']=False
                return Response({'status':'accepted','canonical_event_id':'same-canonical'})
        async def immediate_timeout(coroutine,timeout):
            coroutine.close()
            raise asyncio.TimeoutError
        original=module['asyncio']
        module['asyncio']=types.SimpleNamespace(wait_for=immediate_timeout,
            TimeoutError=asyncio.TimeoutError,CancelledError=asyncio.CancelledError)
        try:
            task=asyncio.create_task(module['_hint_policy'](Client(),'http://unused',NAMESPACE,state,4,
                'agent',(1,2,3),records,activation=activation))
            await asyncio.sleep(0)
            self.assertEqual(sent,[])
            activation.set()
            await task
        finally:module['asyncio']=original
        self.assertEqual(len(sent),2)  # Acceptance did not stop true pending work.
        self.assertEqual(state['binding'],'same-canonical')


if __name__=='__main__':unittest.main()
