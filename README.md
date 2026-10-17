# VeruSteer

VeruSteer learns which reported Verus error type to repair next from alternative repair trajectories. This research snapshot accompanies *VeruSteer: Learning What to Repair Next in Verus*. It includes the repair-tree corpus, fixed task splits, preference samples, the selector, collection and repair pipelines, and recorded outcomes used by the paper.

The data and embedding resources are compressed into 14 repository files, about 319 MB in total, with at most 24 MiB per file. They expand to about 6.60 GB. Download the [complete anonymous repository ZIP](https://anonymous.4open.science/api/repo/VeruSteer-6CA2/zip) or browse the [data download page](downloads/README.md). The packages are stored directly in this repository; they need no Release or Git LFS access.

Use Python 3.11 or newer. After extracting the repository ZIP or cloning the repository, run the examples from its root directory:

```bash
python scripts/download_data.py
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python commands/validate_artifact.py
python scripts/reproduce_paper.py
```

The first command restores the local shards and downloads any missing shards through the anonymous mirror. It needs only Python's standard library. The last two commands need no model API, Verus installation, or trained checkpoint. Validation checks parsed records, identifiers, split membership, CSV encodings, and embedding dimensions. Results are written under `outputs/`; the package does not generate a full-file hash inventory.

The recorded full-model outcomes reproduce 185 default verifications and 194 VeruSteer verifications on 1,270 test trees: 12 gained and three lost. Retention and Exploration are reported separately for verification, shorter successful paths, and higher final scores. `outputs/metrics/overall.csv` and `spaces.csv` contain the recomputed results. The five figures are written to `outputs/figures/`.

Train a selector with the supplied arrays:

```bash
python selector/Selector_trainer.py --pipeline autoverus --cuda 0 --dropout 0.1
python selector/Selector_trainer.py --pipeline verusage --cuda 0 --dropout 0.1
```

Use `--cuda -1` for CPU. Checkpoints are written to `outputs/checkpoints/<pipeline>.pth`. These commands train on the pooled training CSV and choose an epoch using the corresponding pipeline's validation CSV. Published checkpoints and other neural training implementations are not included. Recomputing recorded results and training a new model are separate workflows; new training is not guaranteed to reproduce the paper's selected checkpoints.

The directory contains:

- `data/`: initial programs, raw transitions, complete split paths, preference samples, CSVs, task lists, and training examples for the prompt selector.
- `selector/`: exactly `SModel.py`, `Selector_trainer.py`, `embedding_table/`, and `idmap/`; the resource folders are populated by the download command.
- `downloads/`: compressed data shards, their inventory, and anonymous download links.
- `pipelines/`: the multi-branch collection implementation and the single-trajectory repair implementation, each containing AutoVerus and VeruSAGE.
- `commands/`, `scripts/`, `draw/`: data utilities, execution entrypoints, evaluation, and figures.
- `results/`: final recorded outcomes and compact analysis inputs.
- `docs/`: [data](docs/data.md), [training](docs/training.md), [evaluation](docs/evaluation.md), [repair pipelines](docs/pipelines.md), [implementation notes](docs/implementation_notes.md), and [validation](docs/validation.md).

`manifest.json` lists files, purposes, and sizes for the expanded artifact. `downloads/manifest.json` lists the archive shards. This publication contains one source snapshot and no earlier development history. Verus, Rust/Cargo, BGE-M3, and API access are external dependencies for the workflows that need them; see the relevant guides. Upstream notices are retained in the pipeline directories and described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
