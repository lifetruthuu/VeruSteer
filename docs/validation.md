# Snapshot validation

The complete data validator passed: 684 disjoint tasks, 3,224 trees, 56,428 paths, 24 restored default paths, and exact agreement between all supplied pair encodings and pooled/per-pipeline CSVs. The raw transitions cover the same 1,613 AutoVerus and 1,611 VeruSAGE tree identities as the path corpus. The documented malformed training state is reported explicitly.

The distribution workflow downloaded all 14 shards through a local HTTP test server into a fresh standalone copy, restored both packages, and compared all 4,685 data/resource files and directory notes against the expanded snapshot byte for byte. RQ1 and all six full-model spaces reproduced from that restored copy. Filesystem-prefix normalization also passed the complete data validator.

The snapshot was copied to a temporary directory outside the original project. All four collection/repair entrypoints imported successfully there. A CPU training run on 16 real preference pairs completed one epoch, evaluated the validation/test CSVs, and saved a checkpoint. Strict reloading succeeded. Inference agreed between the two model interfaces at history lengths one through five, and the online adapter selected only legal candidates. Temporary weights and the copied package were removed afterward.

Eight tests passed, covering preceding-action alignment, candidate restriction, explicit model-loading failures, both pipelines' command/configuration preparation, prompt response caching, split-archive restoration, corrupted gzip streams, archive path boundaries, and preservation of changed local files. Run them with:

```bash
python -m unittest discover -s tests -v
```

Before normalizing local filesystem prefixes for anonymous distribution, the final prompt implementation reconstructed all 479 recorded model requests exactly, using actual message-content comparisons and no new API calls. Published prompt examples now use neutral filesystem prefixes. Its complete test corpus passed structural validation. The new prompt runner also checks cached request equality when resuming.

The paper's motivation statistics, overall 185-to-194 verification result, and both pipelines' six spaces reproduced. RQ2 rows match the PDF as described in the implementation notes. RQ3's means match the reported rounding. All 194 RQ4 combination rows reproduce the recorded numerical data; equal-frequency action entries can be ordered differently in their textual display.

The five generated figures were inspected. File checks use inventories, sizes, parsing, IDs, dimensions, and field comparisons. The published source/configuration text was checked for credential patterns; templates contain environment-variable placeholders. No live repair API or Verus execution was performed during packaging, so the checks establish interface and replay portability rather than fresh online repair outcomes.
