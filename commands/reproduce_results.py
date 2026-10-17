#!/usr/bin/env python3
"""Recompute the paper's overall and Retention/Exploration results from stored paths."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from eval_sasrec_log_v3 import (classify_tree_id, classify_tree_space_class,
    is_model_success_on_space_class, is_passed, summarize_tree_passes)

ROOT = Path(__file__).resolve().parents[1]
OBJECTIVES = ('verification', 'shorter', 'score')
SPACES = ('Retention', 'Exploration')
RQ2 = {
    'full': ['rq1/verusage.log'],
    'random': [f'rq2/random_s{s}.log' for s in range(21, 26)],
    **{k:[f'rq2/{k}.log'] for k in ('static','searched_static','combination','prompt',
        'combination_mlp','state_mlp','mean_mlp','wo_bpr','wo_ce','wo_error_emb',
        'wo_pos_emb','wo_code','wo_verus','len1','len2','len3','len4')},
}

def load_paths(path):
    trees=defaultdict(list)
    index={}
    with Path(path).open() as f:
        for line in f:
            source=json.loads(line)
            row={k:source[k] for k in ('tree_id','path_id','default','label')}
            row['path_states']=[{k:source['path_states'][-1][k] for k in ('score_tuple','round_id')}]
            key=(row['tree_id'],row['path_id'])
            if key in index: raise ValueError(f'Duplicate path: {key}')
            trees[row['tree_id']].append(row); index[key]=row
    return dict(trees),index

def evaluate(trees,index,log):
    selected={}
    with Path(log).open() as f:
        for line in f:
            row=json.loads(line); tid=row['tree_id']
            if tid in selected or row.get('dead_end'):
                raise ValueError(f'Duplicate or incomplete rollout: {log}: {tid}')
            selected[tid]=index[tid,row['reached_path_id']]
    if set(selected)!=set(trees): raise ValueError(f'Incorrect tree coverage: {log}')
    counts=defaultdict(lambda:[0,0])
    paired=defaultdict(lambda:dict(trees=0,default_verified=0,selector_verified=0,gained=0,lost=0))
    for tid,paths in trees.items():
        default=next(p for p in paths if p['default'])
        model=selected[tid]; d,m=is_passed(default),is_passed(model)
        categories=('Total',classify_tree_id(tid))
        for category in categories:
            total=paired[category]; total['trees']+=1
            total['default_verified']+=d; total['selector_verified']+=m
            total['gained']+=m and not d; total['lost']+=d and not m
        alternative,default_pass,alternative_pass=summarize_tree_passes(paths)
        eligible=[]
        if alternative and default_pass: eligible.append(('verification','Retention',m))
        elif not default_pass and alternative_pass: eligible.append(('verification','Exploration',m))
        classification=classify_tree_space_class(paths)
        if classification and classification[1]!='class1':
            label,cls=classification
            eligible.append(({'class2':'shorter','class3':'score'}[cls],
                'Retention' if label=='默认空间' else 'Exploration',
                is_model_success_on_space_class(label,cls,default,model)))
        for objective,space,hit in eligible:
            for category in categories:
                cell=counts[objective,space,category]; cell[0]+=hit; cell[1]+=1
    return counts,dict(paired)

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/metrics')
    args=parser.parse_args(); rows=[]; overall=[]; seed_rows=[]
    expected={'autoverus':[(14,15),(3,4),(5,6),(2,8),(100,102),(9,38)],
              'verusage':[(12,14),(9,11),(7,10),(3,3),(76,79),(10,26)]}
    for pipeline in ('autoverus','verusage'):
        trees,index=load_paths(ROOT/f'data/paths/test_{pipeline}.jsonl')
        variants=RQ2 if pipeline=='verusage' else {'full':['rq1/autoverus.log']}
        for variant,logs in variants.items():
            runs=[evaluate(trees,index,ROOT/'results'/log) for log in logs]
            for key in sorted(runs[0][0]):
                numer=sum(run[0][key][0] for run in runs)/len(runs)
                denom=runs[0][0][key][1]
                if any(run[0][key][1]!=denom for run in runs): raise ValueError('Seed eligibility differs')
                objective,space,config=key
                rows.append(dict(pipeline=pipeline,variant=variant,objective=objective,space=space,
                    configuration=config,successes=numer,eligible=denom,
                    success_pct=100*numer/denom if denom else '',runs=len(runs)))
                if variant=='random':
                    for seed,(cells,_) in zip(range(21,26),runs):
                        seed_rows.append(dict(seed=seed,objective=objective,space=space,
                            configuration=config,successes=cells[key][0],eligible=denom))
            if variant=='full':
                cells,paired=runs[0]
                actual=[tuple(cells[o,s,'Total']) for o in OBJECTIVES for s in SPACES]
                if actual!=expected[pipeline]: raise ValueError(f'Paper metrics differ: {pipeline}: {actual}')
                for config,value in paired.items():
                    overall.append(dict(pipeline=pipeline,configuration=config,**value,
                        net=value['gained']-value['lost']))
                print(pipeline,paired['Total'],flush=True)
    totals=[r for r in overall if r['configuration']=='Total']
    pooled={k:sum(r[k] for r in totals) for k in ('trees','default_verified','selector_verified','gained','lost','net')}
    if pooled!=dict(trees=1270,default_verified=185,selector_verified=194,gained=12,lost=3,net=9):
        raise ValueError(f'Overall results differ from paper: {pooled}')
    overall.append(dict(pipeline='Both',configuration='Total',**pooled))
    write_csv(args.output_dir/'overall.csv',overall)
    write_csv(args.output_dir/'spaces.csv',rows)
    write_csv(args.output_dir/'random_seeds.csv',seed_rows)
    print('Paper totals and all six full-model spaces match.')

if __name__=='__main__': main()
