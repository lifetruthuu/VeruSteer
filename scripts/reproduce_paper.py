#!/usr/bin/env python3
"""Recompute the paper statistics and figures without model or verifier calls."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    for script in ['commands/motivation_study.py','commands/reproduce_results.py',
                   'commands/export_analysis_paths.py','commands/export_error_combinations.py','draw/plot_ablation.py',
                   'draw/plot_progress.py','draw/plot_choices.py']:
        subprocess.run([sys.executable,str(ROOT/script)],cwd=ROOT,check=True)

if __name__=='__main__': main()
