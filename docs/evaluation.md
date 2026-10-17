# Evaluation and figures

Run `python scripts/reproduce_paper.py` to recompute the motivation statistics, RQ1, RQ2 counts, and all five experimental figures. This workflow uses stored transitions and final selection logs; it makes no model or verifier calls.

| Paper content | Entry point | Output |
| --- | --- | --- |
| §3.2 motivation | `commands/motivation_study.py` | Console statistics |
| Corpus and split counts | `commands/validate_artifact.py` | `outputs/data_validation.json` |
| RQ1 overall and six spaces | `commands/reproduce_results.py` | `outputs/metrics/overall.csv`, `spaces.csv` |
| RQ2 selection, components, length | `draw/plot_ablation.py` | Three `outputs/figures/ablation_*.pdf` files |
| RQ3 progress across rounds | `commands/export_analysis_paths.py`, `draw/plot_progress.py` | `outputs/analysis/progress.csv`, `avg_score.pdf` |
| RQ4 candidate-set behavior | `commands/export_error_combinations.py`, `draw/plot_choices.py` | `outputs/analysis/error_combinations.csv`, `error_type_combination.pdf` |

RQ1 compares all 1,270 test trees: AutoVerus improves from 96 to 98 verified trees; VeruSAGE from 89 to 96. The six spaces are Retention and Exploration for Verification, Shorter path, and Higher score. Eligibility is determined separately for each objective, and configuration totals pool counts before division. RQ2 pools the four VeruSAGE generation configurations; random selection averages seeds 21–25. Its net gain is the Exploration rate minus the Retention drop, in percentage points, and is different from the overall net verification change in RQ1.

For an individual stored log:

```bash
python commands/eval_sasrec_log_v3.py --model-log results/rq1/verusage.log --merged-path data/paths/test_verusage.jsonl
python commands/eval_paired_outcomes.py --log VeruSteer=results/rq1/verusage.log --merged-path data/paths/test_verusage.jsonl --method verusage
```

RQ3 uses the 159/130 test trees with alternatives. A terminated path carries its final state into later rounds; negative error-count sentinels are omitted from the error averages. RQ4 uses the complete corpus, including training tasks, and counts unique states within a tree. Its selector-specific state populations differ; the differences are descriptive, not paired causal effects.

To replay a newly trained checkpoint, install `requirements-inference.txt` and obtain BGE-M3:

```bash
python scripts/eval_tree_better_path.py \
  --test-path-jsonl data/paths/test_verusage.jsonl --selector-mode model \
  --model-weights outputs/checkpoints/verusage.pth \
  --embedding-model /path/to/bge-m3 --encoder-backend flagembedding \
  --device cpu --save-selection-log outputs/verusage.log
```

The loader uses `selector/SModel.py`. Every replayed choice is restricted to available recorded actions. Random replay is available with `--selector-mode random --random-seed 21`. The fixed-rule and static-order search tools are `scripts/eval_tree_rule_baselines.py` and `scripts/search_static_priority.py`; pass the published `train_<pipeline>.jsonl` and `test_<pipeline>.jsonl` explicitly. Newly fitted rules use only the 342 training tasks; the recorded comparison logs retain their original results and their provenance limits are described in the implementation notes.

The final prompt selector uses DeepSeek-Flash, temperature 0.3, thinking disabled, one answer per multi-candidate decision, and up to six retrieved examples from training tasks. It includes actual path history, action-outcome evidence, and repeated-ineffective-repair information.

```bash
python scripts/eval_tree_prompt_selector.py --dry-run
export PLANNER_API_KEY='your-key'
python scripts/eval_tree_prompt_selector.py --config scripts/prompts/config.example.yaml --output-dir outputs/prompt_run
```

Set the provider URL/model in a local copy of the YAML if needed. `--resume` compares configuration, tree IDs, and actual cached request content before reusing responses. A fresh run samples new answers and need not reproduce the recorded 93 verifications. The prompt bank contains 121 states from 16 training tasks. No development variants or development requests are bundled.
