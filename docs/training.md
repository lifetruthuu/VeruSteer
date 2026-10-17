# Training

The only neural model and trainer sources are `selector/SModel.py` and `selector/Selector_trainer.py`. The model combines the current program and diagnostics with the preceding selected error type at each position, then applies causal self-attention. Training uses candidate-masked cross entropy plus a BPR term with default weight 0.2.

```bash
python selector/Selector_trainer.py \
  --pipeline verusage --data_dir data/processed \
  --emb_dir selector/embedding_table --output_dir outputs/checkpoints \
  --cuda 0 --seed 2022 --hidden_size 64 --max_seq_len 5 \
  --lr 0.0005 --l2 0.00001 --bsize 256 \
  --dropout 0.1 --bpr_weight 0.2 --mask_prob 0.1 \
  --max_epoch 50 --patience 10
```

The pooled `train.csv` supplies preferences from both pipelines. `--pipeline` selects `val_<pipeline>.csv` for early stopping and `test_<pipeline>.csv` for the final sequence evaluation. The score is `EM_verification + 0.5 * (EM_shorter + EM_score)`, using candidate-constrained exact matches on every valid recorded-prefix step. Test sequence scores do not select the epoch.

Default paths resolve from the package location, so the entrypoint also works from another working directory. Custom path arguments resolve from the caller's directory. `--cuda -1` uses CPU; a single GPU uses `--cuda 0`. The output directory is created automatically, with `<pipeline>.pth` holding the best validation checkpoint.

The inherited trainer's default dropout is 0.2; the command above explicitly uses the paper's 0.1. No hyperparameter search or full retraining is performed when checking the release. Record your chosen parameters and validation scores when training a new checkpoint. The existing paper outcomes are distributed as final rollout logs, not as claims about a new training run.
