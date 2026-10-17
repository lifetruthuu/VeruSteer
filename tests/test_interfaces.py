"""Behavior checks for the standalone selector and pipeline interfaces."""
import importlib.util
import json
import logging
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'pipelines/repair'))
from verusteer_selector import VeruSteerSelector
from run_pipeline import main as run_pipeline
from planner_prompt import PromptSelector
from eval_tree_prompt_selector import CachedPlanner

class Interfaces(unittest.TestCase):
    def test_history_uses_preceding_actions_and_candidate_mask(self):
        selector=object.__new__(VeruSteerSelector)
        selector.enabled=True; selector.model=object(); selector.torch=torch
        selector.device=torch.device('cpu'); selector.history=[]; selector.max_len=5
        selector.failure_types2id={'AssertFail':11,'PostCondFail':2}
        selector.logger=logging.getLogger('test')
        captured=[]
        def infer(seq):
            captured.append([x.fail_type_id for x in seq])
            logits=torch.zeros(36); logits[35]=100; logits[11]=2; logits[2]=1
            return logits
        selector._infer_logits=infer
        self.assertEqual(selector.select('c0','e0',['AssertFail','PostCondFail']),'AssertFail')
        self.assertEqual(selector.select('c1','e1',['PostCondFail']),'PostCondFail')
        self.assertEqual(selector.select('c2','e2',['AssertFail']),'AssertFail')
        self.assertEqual(captured,[[0],[0,11],[0,11,2]])
        selector.max_len=1
        selector.select('c3','e3',['AssertFail'])
        self.assertEqual(captured[-1],[11])

    def test_explicit_selector_load_failure(self):
        with self.assertRaises(RuntimeError):
            VeruSteerSelector(dict(use_verusteer=True,verusteer_model_path='/nonexistent/checkpoint.pth'),logging.getLogger('test'))

    def test_pipeline_invocations_and_temporary_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); sample=tmp/'input.rs'; sample.write_text('fn main() {}')
            for mode in ['collect','repair']:
                for pipeline in ['autoverus','verusage']:
                    inspected=[]
                    def fake_run(cmd,cwd,check):
                        cfg=Path(cmd[cmd.index('--config')+1])
                        config=json.loads(cfg.read_text()); inspected.append(cfg)
                        self.assertEqual(config['aoai_api_key'],['test-key'])
                        self.assertTrue(Path(config['example_path']).is_dir())
                        self.assertEqual(cmd[cmd.index('--repair')+1],'5')
                        self.assertTrue(Path(cmd[1]).is_file())
                    with (patch.dict('os.environ',dict(LLM_API_KEY='test-key',LLM_BASE_URL='https://example.invalid',VERUS_PATH='/tmp/verus')),
                          patch('run_pipeline.subprocess.run',side_effect=fake_run)):
                        run_pipeline(['--mode',mode,'--pipeline',pipeline,'--input',str(tmp if mode=='collect' else sample),'--output',str(tmp/'out')])
                    self.assertEqual(len(inspected),1)
                    self.assertFalse(inspected[0].exists())

    def test_prompt_candidate_routing_and_resume(self):
        prompts=PromptSelector()
        state=dict(code_text='fn f() { /* $CODE */ }',verus_output_full='assertion failed',score_tuple=[0,1])
        calls=[]
        def requester(messages):
            calls.append(messages)
            return {'raw_response':{'choices':[{'message':{'content':json.dumps(dict(root_cause_error_type='AssertFail',selected_repairer='AssertFailRepairer'))}}]}}
        with sqlite3.connect(':memory:') as db:
            db.execute('CREATE TABLE responses (key TEXT PRIMARY KEY,request TEXT,response TEXT)')
            planner=CachedPlanner(prompts,{},db,requester)
            for _ in range(2):
                result=planner.select('tree','state',state,['AssertFail','PostCondFail'])
                self.assertEqual(result['selected_name'],'AssertFail')
            self.assertEqual(len(calls),1)
            self.assertIn('$CODE',calls[0][1]['content'])
            invalid=prompts.parse('{"root_cause_error_type":"Other"}',['AssertFail','PostCondFail'])
            self.assertIn(invalid['selected_name'],['AssertFail','PostCondFail'])
            with self.assertRaises(ValueError):
                planner.select('tree','state',{**state,'code_text':'changed'},['AssertFail','PostCondFail'])

if __name__=='__main__': unittest.main()
