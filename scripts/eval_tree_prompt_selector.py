#!/usr/bin/env python3
"""Run the final prompt selector on VeruSAGE trees; dry runs make no API calls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
from planner_common import ROOT, audit, json_text, load_config, load_trees, make_requester, rollout
from planner_prompt import PromptSelector

class CachedPlanner:
    def __init__(self,prompts,config,db,requester=None):
        self.prompts,self.config,self.db,self.requester=prompts,config,db,requester
    def select(self,tree_id,sid,state,candidates,history=()):
        messages=self.prompts.messages(state,candidates,history)
        key=json_text([tree_id,sid])
        request={'messages':messages,'configuration':self.config}
        saved=self.db.execute('SELECT request,response FROM responses WHERE key=?',(key,)).fetchone()
        if saved:
            if json.loads(saved[0])!=request: raise ValueError('Cached request differs; choose a new output directory')
            response=json.loads(saved[1])
        else:
            if self.requester is None: self.requester=make_requester(self.config)
            response=self.requester(messages)
            with self.db:
                self.db.execute('INSERT INTO responses VALUES (?,?,?)',(key,json_text(request),json_text(response)))
        choices=response['raw_response'].get('choices') or []
        content=choices[0].get('message',{}).get('content') if choices else None
        return {**self.prompts.parse(content,candidates),'request_key':key}

def run(trees,prompts,config,output,resume=False,requester=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    dbpath=output/'requests.sqlite3'
    if dbpath.exists() and not resume: raise ValueError('Use --resume or a new output directory')
    manifest={'tree_ids':sorted(trees),'configuration':config,'selector':'evidence_stagnation',
              'repair_budget':5,'backend':'test_double' if requester else 'live_api'}
    with sqlite3.connect(dbpath) as db:
        db.executescript('CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY,request TEXT,response TEXT);'
                        'CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT);')
        old=db.execute("SELECT value FROM metadata WHERE key='run'").fetchone()
        if old and json.loads(old[0])!=manifest: raise ValueError('Run configuration or selected trees changed')
        with db: db.execute("INSERT OR IGNORE INTO metadata VALUES ('run',?)",(json_text(manifest),))
        planner=CachedPlanner(prompts,config,db,requester)
        temp=output/'verusage.log.tmp'
        with temp.open('w') as f:
            for i,tid in enumerate(sorted(trees),1):
                # Revisit cached decisions to compare actual requests on resume.
                record=rollout(trees[tid],planner)
                f.write(json_text(record)+'\n'); f.flush()
                if i%25==0 or i==len(trees): print(f'Completed {i}/{len(trees)}',flush=True)
        temp.replace(output/'verusage.log')
        manifest['requests']=db.execute('SELECT count(*) FROM responses').fetchone()[0]
    (output/'run.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT/'data/paths/test_verusage.jsonl')
    p.add_argument('--bank',type=Path,default=ROOT/'data/prompt_examples/train_bank.json')
    p.add_argument('--config',type=Path,default=ROOT/'scripts/prompts/config.example.yaml')
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/prompt_selector')
    p.add_argument('--limit-trees',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.limit_trees<0: p.error('limit-trees must be nonnegative')
    trees,info=load_trees(args.dataset); prompts=PromptSelector(args.bank)
    tasks={row['tree_key']['task_name'] for t in trees.values() for row in t.paths}
    if tasks & set(prompts.bank['support_tasks']): raise ValueError('Evaluation and example tasks overlap')
    allowed=set((ROOT/'data/splits/test_task_names.txt').read_text().splitlines())|set((ROOT/'data/splits/val_task_names.txt').read_text().splitlines())
    if not tasks<=allowed: raise ValueError('Use held-out validation or test trees')
    if args.limit_trees: trees={k:trees[k] for k in sorted(trees)[:args.limit_trees]}
    if args.dry_run:
        print(json.dumps({**info,**audit(trees,'path')},indent=2)); return
    run(trees,prompts,load_config(args.config),args.output_dir,args.resume)

if __name__=='__main__': main()
