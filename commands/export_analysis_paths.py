#!/usr/bin/env python3
"""Export compact default/selected trajectories from the complete paths and final logs."""
import argparse
from collections import Counter
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/analysis')
    args=p.parse_args(); selected={}; defaults={}; reached={}; counts=Counter()
    for method in ['autoverus','verusage']:
        with (ROOT/f'results/full_corpus/{method}.log').open() as f:
            for line in f:
                row=json.loads(line)
                if row['tree_id'] in selected: raise ValueError('Duplicate rollout')
                selected[row['tree_id']]=row['reached_path_id']
    for path in sorted((ROOT/'data/paths').glob('*.jsonl')):
        split=path.name.split('_')[0]
        with path.open() as f:
            for line in f:
                row=json.loads(line); tid=row['tree_id']; counts[tid]+=1
                if row['default'] or selected.get(tid)==row['path_id']:
                    compact={k:row[k] for k in ('tree_id','tree_key','root_state_id','leaf_state_id','label','default','path_id')}
                    compact['split']=split
                    compact['path_nodes']=[{k:s.get(k) for k in ('state_id','round_id','score_tuple','failure_types','chosen_failure_type','is_default_chose')} for s in row['path_states']]
                    if row['default']: defaults[tid]=compact
                    if selected.get(tid)==row['path_id']: reached[tid]=compact
    if set(defaults)!=set(reached) or set(reached)!=set(selected): raise ValueError('Incomplete analysis path coverage')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,rows in [('default',defaults),('selector',reached)]:
        with (args.output_dir/f'{name}_paths.jsonl').open('w') as f:
            for tid in sorted(rows):
                row=rows[tid]
                row['tree_category']='同时包含默认路径和非默认路径' if counts[tid]>1 else '仅默认路径'
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print('Exported',len(defaults),'default and selected trajectories')

if __name__=='__main__': main()
