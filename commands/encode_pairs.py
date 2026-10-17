#!/usr/bin/env python3
"""Encode the published preference/positive paths using the fixed vocabulary."""
import argparse
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def encode_path(path,maps):
    states=path['path_states'][:-1]
    code,verus,types=maps
    return ([code[s['code_text']] for s in states],
            [verus[s['verus_output_full']] for s in states],
            [types[s['chosen_failure_type']] for s in states],
            [[types[t] for t in s['failure_types']] for s in states])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--idmap',type=Path,default=ROOT/'selector/idmap')
    parser.add_argument('--pipeline',choices=['autoverus','verusage'])
    args=parser.parse_args()
    maps=[json.loads((args.idmap/f).read_text()) for f in ('code_text2id.json','verus_output2id.json','failure_types2id.json')]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.input.open() as src,args.output.open('w',newline='') as dst:
        writer=None; count=0
        for line in src:
            pair=json.loads(line)
            if args.pipeline and not pair['tree_id'].startswith(args.pipeline+'-'): continue
            pos=encode_path(pair['positive_path'],maps)
            row={'tree_id':pair['tree_id']}
            if 'negative_path' in pair:
                neg=encode_path(pair['negative_path'],maps)
                row.update(pos_path_id=pair['positive_path_id'],pos_code_ids=pos[0],pos_verus_ids=pos[1],
                    pos_fail_type_ids=pos[2],pos_candidate_fail_type_ids=pos[3],neg_path_id=pair['negative_path_id'],
                    neg_code_ids=neg[0],neg_verus_ids=neg[1],neg_fail_type_ids=neg[2])
            else:
                row.update(path_id=pair['positive_path_id'],code_ids=pos[0],verus_ids=pos[1],
                           fail_type_ids=pos[2],candidate_fail_type_ids=pos[3])
            row['class_id']=pair['class_id']
            if writer is None: writer=csv.DictWriter(dst,fieldnames=list(row)); writer.writeheader()
            writer.writerow(row); count+=1
    print('Encoded',count,'rows')

if __name__=='__main__': main()
