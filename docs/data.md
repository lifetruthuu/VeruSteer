# Data

Run `python scripts/download_data.py` before using the data. The compressed packages and individual anonymous download links are listed on the [download page](../downloads/README.md).

All splits are by benchmark task. Every generation configuration and repair pipeline for a task belongs to the same split.

| Split | Tasks | Repair trees | Trajectories | CSV samples |
| --- | ---: | ---: | ---: | ---: |
| Train | 342 | 1,636 | 36,028 | 18,885 |
| Validation | 69 | 318 | 3,044 | 39 |
| Test | 273 | 1,270 | 17,356 | 139 |
| Total | 684 | 3,224 | 56,428 | — |

AutoVerus contributes 1,613 trees and VeruSAGE 1,611. Of all trees, 718 contain alternative trajectories. The test split has 635 trees per pipeline, including 159 AutoVerus and 130 VeruSAGE trees with alternatives. `splits/benchmark_task_names.txt` lists the original 762 benchmark tasks; the 78 without an exported tree cannot all be interpreted as successful initial generations.

`inputs/` holds collected initial programs. `trees/autoverus/` and `trees/verusage/` hold JSONL transitions. A transition contains task and configuration metadata, source/child state IDs, program text, full diagnostics, scores, selected error type, and default-selection flags. Local source-path fields are metadata; consumers use the embedded program text. Machine-specific path prefixes have been replaced consistently across the corpus, prompt examples, and text-to-ID maps with neutral `/recorded/host-*` prefixes. Distinct machines remain distinct, IDs and existing embedding arrays are unchanged, and all pair-to-CSV encoding checks pass. Re-encoding the normalized diagnostic strings with a text encoder can produce different vectors from the supplied arrays.

`paths/{train,val,test}_{autoverus,verusage}.jsonl` contains complete root-to-leaf paths, each stored once. `tree_id` identifies the task/configuration/pipeline combination; `path_id` identifies a path. `path_states` includes the final state, while training CSVs exclude that state because no next action is selected there. `label` is the stored verification outcome; `default` identifies the default trajectory; `round_id` counts consumed repair rounds. Score `[x,y]` prefers greater verified obligations `x`, then fewer errors `y`; `[-1,-1]` denotes unavailable verification counts after a compile failure.

The corpus includes 24 restored AutoVerus default trajectories with `default_pruned_early_stop`. They describe recorded default states at an early-pruned boundary. They must remain in corpus statistics and default comparisons. Static-priority search excludes these rows when building executable transitions, as the original search did.

One known training state has an `AssertFail` edge but an empty recorded candidate list: task `VeriCoding_VA0514_vericoded`, VeruSAGE/Qwen zero-shot, state `r4:root.b0.b1.b0.b0`. The validator reports this exact anomaly and rejects other candidate mismatches. The released records are preserved; the final prompt example bank excludes this ambiguous training tree.

`pairs/train.jsonl` contains positive/negative preference pairs. Classes 1, 2, and 3 respectively encode higher failed-path scores, verification versus failure, and shorter successful paths. `pairs/val.jsonl` and `pairs/test.jsonl` contain positive trajectories only. Their 39 and 139 rows support sequence exact-match evaluation and are not complete tree evaluation sets. Full repair outcomes use the complete path files.

`processed/` provides pooled and per-pipeline CSVs. Re-encode any supplied pair file using the fixed maps:

```bash
python commands/encode_pairs.py --input data/pairs/train.jsonl --output outputs/train.csv
python commands/encode_pairs.py --input data/pairs/val.jsonl --pipeline verusage --output outputs/val_verusage.csv
```

The three embedding arrays have shapes `(12329,1024)`, `(28968,1024)`, and `(36,1024)` for programs, diagnostics, and error types. ID zero is padding; the error vocabulary has 35 non-padding entries. Arrays and mappings are shared by training and online repair.

Run `python commands/validate_artifact.py` to check all split counts, JSON records, fixed-map encodings against every CSV row, per-pipeline CSV partitions, and array dimensions. `outputs/data_validation.json` records the checks and the known anomaly, without content hashes.
