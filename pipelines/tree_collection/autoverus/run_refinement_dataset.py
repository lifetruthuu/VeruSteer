"""nohup python run_refinement_dataset.py   --config config-artifact-qwen-coder.json   --input-dir /recorded/host-a/workspace/ErrorRanker/repair_dataset_dedeplicate/DAFNY2VERUS-COLLECTION/deepseek-chat/few-shot   --output-dir /recorded/host-a/workspace/ErrorRanker/baseline_data/autoverus_test/DAFNY2VERUS-COLLECTION-deepseek-chat-few-shot/refinement-batch-output   --intermediate-root /recorded/host-a/workspace/ErrorRanker/baseline_data/autoverus_test/DAFNY2VERUS-COLLECTION-deepseek-chat-few-shot/refinement-batch-intermediate   --workers 30   --repair 5 > run.log 2>&1 &. See docs/pipelines.md for portable commands."""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def list_rs_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.rs" if recursive else "*.rs"
    return sorted([p for p in input_dir.glob(pattern) if p.is_file()])


def run_one(
    main_py: Path,
    config: Path,
    input_file: Path,
    output_file: Path,
    intermediate_dir: Path,
    log_file: Path,
    repair_steps: int,
    temp: float,
    extra_args: list[str],
) -> tuple[str, int, float]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(main_py),
        "--mode",
        "repair",
        "--config",
        str(config),
        "--input",
        str(input_file),
        "--output",
        str(output_file),
        "--intermediate-dir",
        str(intermediate_dir),
        "--repair",
        str(repair_steps),
        "--temp",
        str(temp),
    ]
    cmd.extend(extra_args)

    one_start = time.time()
    try:
        # Write subprocess logs directly to disk to avoid large IPC payloads.
        with log_file.open("w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
        return str(input_file), proc.returncode, (time.time() - one_start)
    except Exception as e:
        with log_file.open("a", encoding="utf-8") as lf:
            lf.write(f"\n[RUNNER_ERROR] {type(e).__name__}: {e}\n")
        return str(input_file), 99, (time.time() - one_start)


def sanitize_name(path: Path) -> str:
    # Keep relative path information to avoid name collisions across subfolders.
    return str(path.with_suffix("")).replace(os.sep, "__").replace(":", "_")


def safe_rel_to_parent(path: Path, parent: Path) -> Path:
    """Return path relative to parent if possible, else fallback to name."""
    try:
        return path.relative_to(parent)
    except ValueError:
        return Path(path.name)


def progress_iter(iterable, total=None, desc=""):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def main() -> int:
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Batch parallel runner for autoverus refinement mode"
    )
    parser.add_argument(
        "--config",
        default="config-artifact-deepseek.json",
        help="Path to autoverus config JSON",
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        default=None,
        help=(
            "Directory containing .rs files to repair. "
            "Repeat this flag to pass multiple input directories."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="../results/refinement-batch-output",
        help="Directory for repaired .rs outputs",
    )
    parser.add_argument(
        "--intermediate-root",
        default="../results/refinement-batch-intermediate",
        help="Root directory for per-file intermediate outputs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--executor",
        choices=["thread", "process"],
        default="thread",
        help=(
            "Parallel executor type. 'thread' is usually faster here because each "
            "task mainly launches and waits for a subprocess."
        ),
    )
    parser.add_argument("--repair", type=int, default=3, help="Repair steps")
    parser.add_argument("--temp", type=float, default=1.0, help="LLM temperature")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search input-dir for .rs files",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files if output .rs already exists",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra arg to pass through to main.py (repeatable), e.g. --extra-arg=--disable-safe",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    autoverus_dir_candidates = [
        # Running from verus-proof-synthesis/autoverus/run_refinement_dataset.py
        script_dir,
        # Running from repo-root level script variants
        script_dir / "verus-proof-synthesis" / "autoverus",
        # Running from commands/run_refinement_dataset.py
        script_dir.parent / "verus-proof-synthesis" / "autoverus",
    ]
    autoverus_dir = None
    for cand in autoverus_dir_candidates:
        if (cand / "main.py").is_file():
            autoverus_dir = cand
            break
    if autoverus_dir is None:
        print(
            "[ERROR] autoverus directory not found; checked: "
            + ", ".join(str(p) for p in autoverus_dir_candidates)
        )
        return 1

    main_py = autoverus_dir / "main.py"
    config = (
        Path(args.config)
        if Path(args.config).is_absolute()
        else (autoverus_dir / args.config)
    ).resolve()
    input_dir_args = args.input_dir or ["repair_dataset/DAFNY2VERUS-COLLECTION/deepseek-chat"]
    input_dirs = [
        (Path(p) if Path(p).is_absolute() else (autoverus_dir / p)).resolve()
        for p in input_dir_args
    ]
    output_dir = (
        Path(args.output_dir)
        if Path(args.output_dir).is_absolute()
        else (autoverus_dir / args.output_dir)
    ).resolve()
    intermediate_root = (
        Path(args.intermediate_root)
        if Path(args.intermediate_root).is_absolute()
        else (autoverus_dir / args.intermediate_root)
    ).resolve()

    if not main_py.is_file():
        print(f"[ERROR] main.py not found: {main_py}")
        return 1
    if not config.is_file():
        print(f"[ERROR] config not found: {config}")
        return 1
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            print(f"[ERROR] input-dir not found: {input_dir}")
            return 1

    rs_files_with_root: list[tuple[Path, Path]] = []
    for input_dir in input_dirs:
        rs_files = list_rs_files(input_dir, args.recursive)
        rs_files_with_root.extend((input_dir, f) for f in rs_files)

    if not rs_files_with_root:
        print(f"[WARN] no .rs files found in input-dirs: {input_dirs}")
        return 0

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_log_dir = intermediate_root / f"batch-logs-{timestamp}"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_root.mkdir(parents=True, exist_ok=True)

    # Build a stable namespace for each input dir so outputs/intermediates/logs
    # can be distinguished when multiple input dirs are mixed in one run.
    common_parent = Path(os.path.commonpath([str(d) for d in input_dirs]))
    input_ns_map: dict[Path, Path] = {}
    for d in input_dirs:
        ns = safe_rel_to_parent(d, common_parent)
        if str(ns) in {"", "."}:
            ns = Path(d.name)
        input_ns_map[d] = ns

    tasks = []
    for input_dir, f in rs_files_with_root:
        rel = f.relative_to(input_dir)
        ns = input_ns_map[input_dir]
        out_path = (output_dir / ns / rel).with_suffix(".rs")
        inter_dir = intermediate_root / sanitize_name(ns / rel)
        if args.skip_existing and out_path.exists():
            continue
        tasks.append((input_dir, ns, f, out_path, inter_dir))

    if not tasks:
        print("[INFO] nothing to run (all outputs exist and --skip-existing enabled).")
        return 0

    print(f"[INFO] input-dirs: {len(input_dirs)}")
    for d in input_dirs:
        print(f"[INFO]   - {d} -> namespace={input_ns_map[d]}")
    print(f"[INFO] total input files: {len(rs_files_with_root)}")
    print(f"[INFO] queued tasks: {len(tasks)}")
    print(f"[INFO] workers: {args.workers}")
    print(f"[INFO] executor: {args.executor}")
    print(f"[INFO] output-dir: {output_dir}")
    print(f"[INFO] intermediate-root: {intermediate_root}")

    ok_count = 0
    fail_count = 0
    total_task_elapsed = 0.0

    executor_cls = (
        concurrent.futures.ThreadPoolExecutor
        if args.executor == "thread"
        else concurrent.futures.ProcessPoolExecutor
    )

    futures: dict[concurrent.futures.Future, str] = {}

    with executor_cls(max_workers=args.workers) as ex:
        for input_dir, ns, in_file, out_file, inter_dir in tasks:
            rel = in_file.relative_to(input_dir)
            log_path = run_log_dir / f"{sanitize_name(ns / rel)}.log"
            fut = ex.submit(
                run_one,
                main_py,
                config,
                in_file,
                out_file,
                inter_dir,
                log_path,
                args.repair,
                args.temp,
                args.extra_arg,
            )
            futures[fut] = str(in_file)

        for fut in progress_iter(
            concurrent.futures.as_completed(futures.keys()),
            total=len(futures.keys()),
            desc="Refinement progress (rs files)",
        ):
            fallback_file = futures[fut]
            try:
                in_file, code, one_elapsed = fut.result()
                total_task_elapsed += one_elapsed
                if code == 0:
                    ok_count += 1
                    print(f"[OK] {in_file} ({one_elapsed:.1f}s)")
                else:
                    fail_count += 1
                    print(f"[FAIL] {in_file} (exit={code}, {one_elapsed:.1f}s)")
            except Exception as e:
                fail_count += 1
                print(f"[FAIL] {fallback_file} (runner exception: {e})")

    elapsed = time.time() - start_time
    elapsed_hms = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(f"[DONE] success={ok_count}, failed={fail_count}, total={len(tasks)}")
    print(f"[DONE] elapsed_seconds={elapsed:.2f}, elapsed_hms={elapsed_hms}")
    print(f"[DONE] sum_task_seconds={total_task_elapsed:.2f}")
    print(f"[DONE] per-file logs: {run_log_dir}")
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

