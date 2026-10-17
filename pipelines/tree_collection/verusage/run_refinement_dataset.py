"""Batch parallel runner for verusage repair mode (dataset construction).. See docs/pipelines.md for portable commands."""

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


def sanitize_name(path: Path) -> str:
    return str(path.with_suffix("")).replace(os.sep, "__").replace(":", "_")


def safe_rel_to_parent(path: Path, parent: Path) -> Path:
    try:
        return path.relative_to(parent)
    except ValueError:
        return Path(path.name)


def progress_iter(iterable, total=None, desc=""):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


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
        "agent",
        "--config",
        str(config),
        "--input",
        str(input_file),
        "--output",
        str(output_file),
        "--outdir",
        str(intermediate_dir),
        "--repair",
        str(repair_steps),
        "--temp",
        str(temp),
    ]
    cmd.extend(extra_args)

    one_start = time.time()
    try:
        with log_file.open("w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
        return str(input_file), proc.returncode, (time.time() - one_start)
    except Exception as e:
        with log_file.open("a", encoding="utf-8") as lf:
            lf.write(f"\n[RUNNER_ERROR] {type(e).__name__}: {e}\n")
        return str(input_file), 99, (time.time() - one_start)


def main() -> int:
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Batch parallel runner for verusage dataset refinement"
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to verusage config JSON",
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
    parser.add_argument("--repair", type=int, default=5, help="Repair steps")
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
        help=(
            "Extra arg to pass through to main.py (repeatable), "
            "e.g. --extra-arg=--ablation-mode"
        ),
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    verusage_dir_candidates = [
        script_dir,
        script_dir / "verus-proof-synthesis" / "verusage",
        script_dir.parent / "verus-proof-synthesis" / "verusage",
    ]
    verusage_dir = None
    for cand in verusage_dir_candidates:
        if (cand / "main.py").is_file():
            verusage_dir = cand
            break
    if verusage_dir is None:
        print(
            "[ERROR] verusage directory not found; checked: "
            + ", ".join(str(p) for p in verusage_dir_candidates)
        )
        return 1

    main_py = verusage_dir / "main.py"
    config = (
        Path(args.config)
        if Path(args.config).is_absolute()
        else (verusage_dir / args.config)
    ).resolve()

    input_dir_args = args.input_dir or ["repair_candidate/zero-shot"]
    input_dirs = [
        (Path(p) if Path(p).is_absolute() else (verusage_dir / p)).resolve()
        for p in input_dir_args
    ]
    output_dir = (
        Path(args.output_dir)
        if Path(args.output_dir).is_absolute()
        else (verusage_dir / args.output_dir)
    ).resolve()
    intermediate_root = (
        Path(args.intermediate_root)
        if Path(args.intermediate_root).is_absolute()
        else (verusage_dir / args.intermediate_root)
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
    verified_count = 0
    not_verified_count = 0
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
            desc="Verusage refinement progress (rs files)",
        ):
            fallback_file = futures[fut]
            try:
                in_file, code, one_elapsed = fut.result()
                total_task_elapsed += one_elapsed
                if code == 0:
                    ok_count += 1
                    verified_count += 1
                    print(f"[OK] {in_file} (verified, {one_elapsed:.1f}s)")
                elif code == 233:
                    # 233 means run completed but did not verify.
                    ok_count += 1
                    not_verified_count += 1
                    print(f"[OK] {in_file} (not_verified, exit=233, {one_elapsed:.1f}s)")
                else:
                    fail_count += 1
                    print(f"[FAIL] {in_file} (exit={code}, {one_elapsed:.1f}s)")
            except Exception as e:
                fail_count += 1
                print(f"[FAIL] {fallback_file} (runner exception: {e})")

    elapsed = time.time() - start_time
    elapsed_hms = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(
        f"[DONE] completed={ok_count}, verified={verified_count}, "
        f"not_verified={not_verified_count}, failed={fail_count}, total={len(tasks)}"
    )
    print(f"[DONE] elapsed_seconds={elapsed:.2f}, elapsed_hms={elapsed_hms}")
    print(f"[DONE] sum_task_seconds={total_task_elapsed:.2f}")
    print(f"[DONE] per-file logs: {run_log_dir}")
    print(
        "[DONE] dataset files are inside each task intermediate "
        "subdir under intermediate-root."
    )
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
