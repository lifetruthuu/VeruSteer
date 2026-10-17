"""Filter .rs files by shot tag in filename.. See docs/pipelines.md for portable commands."""

import argparse
import shutil
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
    parser = argparse.ArgumentParser(description="Filter .rs files by few-shot/zero-shot name.")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing .rs files.",
    )
    parser.add_argument(
        "--shot",
        choices=["few-shot", "zero-shot"],
        required=True,
        help="Which shot tag to keep.",
    )
    parser.add_argument(
        "--mode",
        choices=["list", "copy", "delete-others"],
        default="list",
        help="list: preview; copy: copy matches; delete-others: remove non-matching in-place.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Required when --mode copy.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively include subdirectories.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation for delete-others mode.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        print(f"[ERROR] input-dir does not exist: {input_dir}")
        return 1

    pattern = "**/*.rs" if args.recursive else "*.rs"
    all_rs = sorted([p for p in input_dir.glob(pattern) if p.is_file()])
    if not all_rs:
        print(f"[WARN] no .rs files found in {input_dir}")
        return 0

    matched = [p for p in all_rs if args.shot in p.name]
    non_matched = [p for p in all_rs if args.shot not in p.name]

    print(f"[INFO] input-dir={input_dir}")
    print(f"[INFO] total_rs={len(all_rs)}")
    print(f"[INFO] shot={args.shot}")
    print(f"[INFO] matched={len(matched)}")
    print(f"[INFO] non_matched={len(non_matched)}")

    if args.mode == "list":
        for p in matched:
            print(p)
        return 0

    if args.mode == "copy":
        if args.output_dir is None:
            print("[ERROR] --output-dir is required when --mode copy")
            return 1
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for src in progress_iter(matched, total=len(matched), desc="Copying matched .rs"):
            if args.recursive:
                rel = src.relative_to(input_dir)
                dst = output_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
            else:
                dst = output_dir / src.name
            shutil.copy2(src, dst)
            copied += 1
        print(f"[DONE] copied={copied}, output-dir={output_dir}")
        return 0

    # delete-others
    if not args.yes:
        print("[WARN] delete-others will permanently delete non-matching files.")
        confirm = input("Type 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("[INFO] cancelled.")
            return 0

    deleted = 0
    for p in progress_iter(non_matched, total=len(non_matched), desc="Deleting non-matching .rs"):
        p.unlink(missing_ok=True)
        deleted += 1
    print(f"[DONE] deleted={deleted}, kept={len(matched)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

