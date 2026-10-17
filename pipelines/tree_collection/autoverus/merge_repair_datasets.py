"""python merge_repair_datasets.py   --input-root /recorded/host-a/workspace/ErrorRanker/baseline_data/autoverus/VerusBench-gpt-4o-few-shot/refinement-batch-intermediate   --output /recorded/host-a/workspace/ErrorRanker/baseline_data/autoverus/VerusBench-gpt-4o-few-shot/merged_result.jsonl. See docs/pipelines.md for portable commands."""

import argparse
import json
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def progress_iter(iterable, total=None, desc=""):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge repair-dataset-*.jsonl files into one jsonl."
    )
    parser.add_argument(
        "--input-root",
        default="../results/refinement-batch-intermediate",
        help="Root directory that contains repair-dataset-*.jsonl files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output jsonl path. Default: ../results/repair-dataset-merged-<timestamp>.jsonl",
    )
    parser.add_argument(
        "--include-source-file",
        action="store_true",
        help="Append a field `source_dataset_file` to each merged record.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_root = (
        Path(args.input_root)
        if Path(args.input_root).is_absolute()
        else (script_dir / args.input_root)
    ).resolve()

    if args.output is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        output_path = (script_dir / f"../results/repair-dataset-merged-{ts}.jsonl").resolve()
    else:
        output_path = (
            Path(args.output)
            if Path(args.output).is_absolute()
            else (script_dir / args.output)
        ).resolve()

    if not input_root.is_dir():
        print(f"[ERROR] input root does not exist: {input_root}")
        return 1

    files = sorted(input_root.glob("**/repair-dataset-*.jsonl"))
    if len(files) == 0:
        print(f"[WARN] no files found under: {input_root}")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged_records = 0
    invalid_lines = 0

    with output_path.open("w", encoding="utf-8") as wf:
        for src in progress_iter(files, total=len(files), desc="Merging dataset files"):
            with src.open("r", encoding="utf-8") as rf:
                for raw_line in rf:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        invalid_lines += 1
                        continue
                    if args.include_source_file:
                        obj["source_dataset_file"] = str(src)
                    wf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    merged_records += 1

    print(f"[DONE] merged_files={len(files)}")
    print(f"[DONE] merged_records={merged_records}")
    print(f"[DONE] skipped_invalid_lines={invalid_lines}")
    print(f"[DONE] output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

