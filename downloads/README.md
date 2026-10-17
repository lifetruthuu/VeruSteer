# Download the research data

The two packages contain the full corpus and the shared selector resources. All 14 shards are ordinary files in this repository. Each shard is at most 24 MiB. The total download is about 319 MB and expands to about 6.60 GB. Allow about 8 GB of free space for download and extraction.

After downloading and extracting the [complete anonymous repository ZIP](https://anonymous.4open.science/api/repo/VeruSteer-6CA2/zip), run this command from its root:

```bash
python scripts/download_data.py
```

The command uses local shards when present and downloads missing ones from the anonymous mirror. It requires only Python 3.11 or newer. Use `--package corpus` or `--package selector` to restore one package. Incomplete downloads are retried, and the downloaded shards can be reused. Gzip CRC, byte counts, and archive inventories are checked without a SHA-256 inventory.

For data only, download the standalone script and choose an output directory:

```bash
curl -fL 'https://anonymous.4open.science/api/repo/VeruSteer-6CA2/file/scripts/download_data.py' -o download_data.py
python download_data.py --destination artifact
```

| Package | Contents after extraction | Download | Shards |
| --- | --- | ---: | ---: |
| corpus | `data/`: inputs, trees, paths, pairs, CSVs, splits, prompt examples | 187.56 MB | 8 |
| selector | `selector/embedding_table/` and `selector/idmap/` | 131.91 MB | 6 |

The following links download individual shards directly through the anonymous site. Save all shards for a package under `downloads/`, alongside [manifest.json](manifest.json), before running the restoration command.

- [corpus-001.bin](https://anonymous.4open.science/api/repo/VeruSteer-6CA2/file/downloads/corpus-001.bin?download=1) (24.00 MiB)
- [corpus-002.bin](https://anonymous.4open.science/api/repo/VeruSteer-6CA2/file/downloads/corpus-002.bin?download=1) (24.00 MiB)
- [corpus-003.bin](https://anonymous.4open.science/api/repo/VeruSteer-6CA2/file/downloads/corpus-003.bin?download=1) (24.00 MiB)
- [corpus-004.bin](https://anonymous.4open.science/api/repo/VeruSteer-6CA2/file/downloads/corpus-004.bin?download=1) (24.00 MiB)
- [corpus-005.bin](https://anonymous.4open.science/api/repo/VeruSteer-6CA2/file/downloads/corpus-005.bin?download=1) (24.00 MiB)
- [corpus-006.bin](https://anonymous.4open.science/api/repo/VeruSteer-6CA2/file/downloads/corpus-006.bin?download=1) (24.00 MiB)
- [corpus-007.bin](https://anonymous.4open.science/api/repo/VeruSteer-6CA2/file/downloads/corpus-007.bin?download=1) (24.00 MiB)
- [corpus-008.bin](https://anonymous.4open.science/api/repo/VeruSteer-6CA2/file/downloads/corpus-008.bin?download=1) (10.87 MiB)
- [selector-001.bin](https://anonymous.4open.science/api/repo/VeruSteer-6CA2/file/downloads/selector-001.bin?download=1) (24.00 MiB)
- [selector-002.bin](https://anonymous.4open.science/api/repo/VeruSteer-6CA2/file/downloads/selector-002.bin?download=1) (24.00 MiB)
- [selector-003.bin](https://anonymous.4open.science/api/repo/VeruSteer-6CA2/file/downloads/selector-003.bin?download=1) (24.00 MiB)
- [selector-004.bin](https://anonymous.4open.science/api/repo/VeruSteer-6CA2/file/downloads/selector-004.bin?download=1) (24.00 MiB)
- [selector-005.bin](https://anonymous.4open.science/api/repo/VeruSteer-6CA2/file/downloads/selector-005.bin?download=1) (24.00 MiB)
- [selector-006.bin](https://anonymous.4open.science/api/repo/VeruSteer-6CA2/file/downloads/selector-006.bin?download=1) (5.80 MiB)

Existing identical data files are reused; different files cause a clear error. Pass `--overwrite` only to replace such files with the bundled data. The [data guide](../docs/data.md) documents the corpus and normalized filesystem prefixes.
