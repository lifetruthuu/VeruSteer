# Collecting trees and repairing programs

`pipelines/tree_collection/` branches on reported error types to collect alternative repair trajectories. `pipelines/repair/` follows one trajectory and can use the shared selector. Both preserve the underlying AutoVerus/VeruSAGE agents and acceptance checks. They are distinct implementations.

Install the Python requirements, Rust/Cargo, Z3 and Verus. Collection configuration records identify Verus `0.2025.09.25.04e8687`; using a different verifier or repair model can change transitions. Build the bundled Lynette tool for each implementation:

```bash
cargo build --manifest-path pipelines/tree_collection/utils/lynette/source/Cargo.toml
cargo build --manifest-path pipelines/repair/utils/lynette/source/Cargo.toml
```

The Cargo dependency declarations fetch Verus syntax libraries. The VeruSAGE utilities may also use an independently installed `verusfmt` executable. The wrapper runs each pipeline from its expected working directory and resolves shared package paths before invocation.

Provide configuration through environment variables:

```bash
export VERUS_PATH=/path/to/verus
export LLM_BASE_URL=https://api.deepseek.com/v1/
export LLM_API_KEY='your-key'
```

The JSON templates in each pipeline directory specify the repair model and generation parameters. Use `--config your.local.json` to select a modified template. The wrapper expands `${ARTIFACT_ROOT}` and environment placeholders, writes a private temporary configuration for the upstream interface, then removes it. Credentials are not printed or copied into the released results.

Collect a five-round tree corpus from a directory of initial programs:

```bash
python scripts/run_pipeline.py --mode collect --pipeline autoverus \
  --input data/inputs/few-shot/VerusBench/deepseek-chat \
  --output outputs/collection --repair 5 --workers 1
python commands/merge_results.py --input_dir outputs/collection/trees --output_file outputs/edges.jsonl
python commands/extract_root_to_leaf_paths.py --input outputs/edges.jsonl --output outputs/paths.jsonl
python commands/complete_pruned_default_paths.py --input outputs/paths.jsonl --check
```

Use `--pipeline verusage` for the other collector. Tree expansion consumes real model and verifier calls; its runtime and generated patches depend on the service and program. `--dry-run` on the wrapper prints the invocation without credentials, external calls, or output generation.

Run one repaired trajectory after training a checkpoint:

```bash
python -m pip install -r requirements-inference.txt
python scripts/run_pipeline.py --mode repair --pipeline verusage \
  --input /path/to/input.rs --output outputs/repair \
  --checkpoint outputs/checkpoints/verusage.pth \
  --bge-model /path/to/bge-m3 --device cpu --repair 5
```

Without `--checkpoint`, the repair implementation uses its default selection rule. With a checkpoint, loading or inference errors are explicit. Exact text IDs can use the supplied vectors; unseen states require BGE-M3. The selected type is always drawn from current diagnostics, and the history channel contains the preceding action at each state. Online runs resample patches and do not constitute a rerun of the paper's recorded-tree evaluation.
