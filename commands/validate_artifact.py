#!/usr/bin/env python3
"""Validate data counts, IDs, mappings, and CSV encodings without file hashes."""
from __future__ import annotations
import argparse
import ast
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from encode_pairs import encode_path

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={'train':(342,1636,36028),'val':(69,318,3044),'test':(273,1270,17356)}

def csv_key(row,training):
    if training: return row['tree_id'],row['pos_path_id'],row['neg_path_id'],int(row['class_id'])
    return row['tree_id'],row['path_id'],int(row['class_id'])

def validate():
    split_tasks={s:set((ROOT/f'data/splits/{s}_task_names.txt').read_text().splitlines()) for s in EXPECTED}
    assert all(not split_tasks[a]&split_tasks[b] for a,b in [('train','val'),('train','test'),('val','test')])
    path_index={}; trees=defaultdict(list); summary={}; restored=0; observed_anomalies=set()
    tree_identities=defaultdict(set)
    def identity(row):
        return tuple(row[k] for k in ('method','task_name','llm_name','shot')) + (
            str(row['generation_candidate_id']).removeprefix('result_'),row['repair_llm_name'])
    known_anomalies={('verusage-VeriCoding-VeriCoding_VA0514_vericoded-qwen-coder-zero-shot-1-deepseek-chat',
                     'r4:root.b0.b1.b0.b0')}
    for split,(tasks,n_trees,n_paths) in EXPECTED.items():
        seen_tasks=set(); tree_ids=set(); count=0
        for pipeline in ['autoverus','verusage']:
            with (ROOT/f'data/paths/{split}_{pipeline}.jsonl').open() as f:
                for line in f:
                    row=json.loads(line); tid,pid=row['tree_id'],row['path_id']; task=row['tree_key']['task_name']
                    assert task in split_tasks[split] and row['tree_key']['method']==pipeline
                    tree_identities[pipeline].add(identity(row['tree_key']))
                    assert (tid,pid) not in path_index
                    path_index[tid,pid]=(split,task)
                    assert row['path_state_ids']==[s['state_id'] for s in row['path_states']]
                    assert row['root_state_id']==row['path_state_ids'][0] and row['leaf_state_id']==row['path_state_ids'][-1]
                    for state in row['path_states'][:-1]:
                        if state['chosen_failure_type'] not in state['failure_types']:
                            anomaly=(tid,state['state_id'])
                            assert anomaly in known_anomalies,anomaly
                            observed_anomalies.add(anomaly)
                    trees[tid].append(bool(row['default']))
                    restored+=bool(row.get('default_pruned_early_stop'))
                    seen_tasks.add(task); tree_ids.add(tid); count+=1
        assert seen_tasks==split_tasks[split]
        assert (len(seen_tasks),len(tree_ids),count)==(tasks,n_trees,n_paths)
        summary[split]=dict(tasks=tasks,trees=n_trees,paths=n_paths)
        print(split,summary[split],flush=True)
    assert len(trees)==3224 and len(path_index)==56428 and restored==24
    assert observed_anomalies==known_anomalies
    assert all(sum(v)==1 for v in trees.values())
    assert sum(len(v)>1 for v in trees.values())==718
    maps=[json.loads((ROOT/'selector/idmap'/f).read_text()) for f in ('code_text2id.json','verus_output2id.json','failure_types2id.json')]
    for forward,inverse in [(maps[0],'id2code_text.json'),(maps[1],'id2verus_output.json')]:
        inv=json.loads((ROOT/'selector/idmap'/inverse).read_text())
        assert all(inv[str(idx)]==text for text,idx in forward.items())
    dimensions=[]
    for name,mapping in zip(['code_embeddings.npy','verus_outputs_embeddings.npy','failure_types_embeddings.npy'],maps):
        table=np.load(ROOT/'selector/embedding_table'/name,mmap_mode='r')
        assert table.ndim==2 and table.shape[1]==1024 and max(mapping.values())<table.shape[0]
        assert set(mapping.values())==set(range(1,table.shape[0]))
        assert np.isfinite(table).all()
        dimensions.append(list(table.shape))
    for split,n in [('train',18885),('val',39),('test',139)]:
        training=split=='train'; stored={}; counts=Counter()
        with (ROOT/f'data/processed/{split}.csv').open() as f:
            for row in csv.DictReader(f):
                key=csv_key(row,training)
                parsed={k:ast.literal_eval(v) for k,v in row.items() if k.endswith('_ids')}
                stored.setdefault(key,parsed)
                assert stored[key]==parsed
                counts[key]+=1
        assert sum(counts.values())==n
        pairs=Counter()
        with (ROOT/f'data/pairs/{split}.jsonl').open() as f:
            for line in f:
                pair=json.loads(line); tid=pair['tree_id']; pid=pair['positive_path_id']
                assert path_index[tid,pid][0]==split
                pos=encode_path(pair['positive_path'],maps)
                if training:
                    nid=pair['negative_path_id']; assert path_index[tid,nid][0]==split
                    neg=encode_path(pair['negative_path'],maps)
                    expected=dict(pos_code_ids=pos[0],pos_verus_ids=pos[1],pos_fail_type_ids=pos[2],
                        pos_candidate_fail_type_ids=pos[3],neg_code_ids=neg[0],neg_verus_ids=neg[1],neg_fail_type_ids=neg[2])
                    key=(tid,pid,nid,pair['class_id'])
                else:
                    expected=dict(code_ids=pos[0],verus_ids=pos[1],fail_type_ids=pos[2],candidate_fail_type_ids=pos[3])
                    key=(tid,pid,pair['class_id'])
                assert stored[key]==expected,(split,key,'CSV differs from encoded path')
                for action,candidates in zip(pos[2],pos[3]): assert action in candidates
                pairs[key]+=1
        assert pairs==counts
        for pipeline in ['autoverus','verusage']:
            with (ROOT/f'data/processed/{split}_{pipeline}.csv').open() as f:
                actual=Counter(csv_key(row,training) for row in csv.DictReader(f))
            assert actual==Counter({k:v for k,v in counts.items() if k[0].startswith(pipeline+'-')})
        summary[split]['csv_rows']=n
        print(split,'CSV and pair encodings match',flush=True)
    edge_counts={}
    for pipeline in ['autoverus','verusage']:
        count=0; raw_identities=set()
        for path in (ROOT/f'data/trees/{pipeline}').glob('*.jsonl'):
            with path.open() as f:
                for line in f:
                    row=json.loads(line)
                    assert row['method']==pipeline
                    assert all(k in row for k in ['source_state_id','repaired_state_id','chosen_failure_type','source_code_text','repaired_code_text'])
                    raw_identities.add(identity(row))
                    count+=1
        edge_counts[pipeline]=count
        assert raw_identities==tree_identities[pipeline],pipeline
    return dict(splits=summary,embedding_shapes=dimensions,restored_default_paths=restored,
                known_raw_state_anomalies=[dict(tree_id=t,state_id=s) for t,s in sorted(observed_anomalies)],
                trees_with_alternatives=718,raw_edges=edge_counts,checks='counts, IDs, mappings, CSV encoding, JSON parsing')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/data_validation.json')
    args=p.parse_args(); result=validate()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Data validation passed.')

if __name__=='__main__': main()
