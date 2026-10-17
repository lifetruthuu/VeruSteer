#!/usr/bin/env python3
"""Run tree collection or online repair using environment-based configuration."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def expand(value,root=ROOT):
    if isinstance(value,dict): return {k:expand(v,root) for k,v in value.items()}
    if isinstance(value,list): return [expand(v,root) for v in value]
    if not isinstance(value,str): return value
    def substitute(m):
        name=m[1]
        if name=='ARTIFACT_ROOT': return str(root)
        if name not in os.environ: raise ValueError(f'Set environment variable {name}')
        return os.environ[name]
    return re.sub(r'\$\{([A-Z_][A-Z_0-9]*)\}',substitute,value)

def build_command(args,config_file):
    mode_dir='tree_collection' if args.mode=='collect' else 'repair'
    cwd=ROOT/'pipelines'/mode_dir/args.pipeline
    target=args.output.resolve()
    if args.mode=='collect':
        cmd=[sys.executable,str(cwd/'run_refinement_dataset.py'),'--config',str(config_file),
             '--input-dir',str(args.input.resolve()),'--output-dir',str(target/'programs'),
             '--intermediate-root',str(target/'trees'),'--repair',str(args.repair),'--workers',str(args.workers)]
    else:
        cmd=[sys.executable,str(cwd/'main.py'),'--config',str(config_file),'--input',str(args.input.resolve()),
             '--output',str(target/'repaired.rs'),'--outdir',str(target/'intermediate'),'--repair',str(args.repair)]
        if args.pipeline=='autoverus': cmd+=['--mode','repair']
    return cmd,cwd

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['collect','repair'],required=True)
    parser.add_argument('--pipeline',choices=['autoverus','verusage'],required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--checkpoint',type=Path,help='Enable VeruSteer with this trained checkpoint')
    parser.add_argument('--bge-model',default='BAAI/bge-m3')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--repair',type=int,default=5)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--dry-run',action='store_true',help='Inspect invocation without credentials or execution')
    args=parser.parse_args(argv)
    if args.repair<1 or args.workers<1: parser.error('repair and workers must be positive')
    if not args.input.exists(): parser.error('Input does not exist')
    if args.mode=='collect' and (not args.input.is_dir() or args.checkpoint): parser.error('Collection requires an input directory and has no selector checkpoint')
    if args.mode=='repair' and not args.input.is_file(): parser.error('Repair requires one input file')
    if args.checkpoint and not args.checkpoint.is_file(): parser.error('Checkpoint does not exist')
    mode_dir='tree_collection' if args.mode=='collect' else 'repair'
    config_path=args.config or ROOT/'pipelines'/mode_dir/f'{args.pipeline}.example.json'
    template=json.loads(config_path.read_text())
    if args.dry_run:
        cmd,cwd=build_command(args,Path('<temporary-config>'))
        print(json.dumps({'cwd':str(cwd),'command':cmd,'config_template':str(config_path),
                          'selector_enabled':bool(args.checkpoint)},indent=2)); return
    config=expand(template)
    if args.checkpoint:
        config.update(use_verusteer=True,verusteer_model_dir=str(ROOT/'selector'),
            verusteer_model_path=str(args.checkpoint.resolve()),verusteer_bge_model_path=args.bge_model,
            verusteer_device=args.device,verusteer_state_size=5,verusteer_hidden_size=64)
    args.output.mkdir(parents=True,exist_ok=True)
    # A private temporary file is required by the upstream JSON-config API.
    with tempfile.TemporaryDirectory(prefix='verusteer-config-') as directory:
        config_file=Path(directory)/'config.json'
        config_file.write_text(json.dumps(config)); config_file.chmod(0o600)
        cmd,cwd=build_command(args,config_file)
        subprocess.run(cmd,cwd=cwd,check=True)

if __name__=='__main__': main()
