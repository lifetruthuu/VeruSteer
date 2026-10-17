#!/usr/bin/env python3
"""
Batch runner for AutoVerus repair tasks with efficiency logging.
"""

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


FIELDNAMES = [
    "file",
    "exit_code",
    "status",
    "time_seconds",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "llm_calls",
    "repair_rounds",
    "task_log_file",
]


def parse_metrics_from_text(content):
    input_tokens = [int(t) for t in re.findall(r"Input tokens:\s*(\d+)", content)]
    output_tokens = [int(t) for t in re.findall(r"Output tokens:\s*(\d+)", content)]
    explicit_llm_calls = [int(t) for t in re.findall(r"LLM calls:\s*(\d+)", content)]
    step_rounds = [int(t) for t in re.findall(r"\bStep\s+(\d+)\b", content)]
    fixed_rounds = [
        int(t) for t in re.findall(r"All errors are fixed within\s+(\d+)\s+steps", content)
    ]
    return {
        "input_tokens": sum(input_tokens),
        "output_tokens": sum(output_tokens),
        "llm_calls": sum(explicit_llm_calls) or len(input_tokens),
        "repair_rounds": max(step_rounds + fixed_rounds, default=0),
    }


def write_results(output_root, benchmark_name, config_path, all_rs_files, results):
    csv_file = output_root / "results.csv"
    summary_file = output_root / "summary.txt"
    results = sorted(results, key=lambda r: r["file"])

    verified_count = sum(1 for r in results if r["status"] == "VERIFIED")
    failed_count = sum(1 for r in results if r["status"] == "FAILED")
    timeout_count = sum(1 for r in results if r["status"] == "TIMEOUT")
    exception_count = sum(1 for r in results if r["status"] == "EXCEPTION")
    error_count = sum(1 for r in results if r["status"] == "ERROR")
    total_input_tokens = sum(r["input_tokens"] for r in results)
    total_output_tokens = sum(r["output_tokens"] for r in results)
    total_tokens = sum(r["total_tokens"] for r in results)
    total_llm_calls = sum(r["llm_calls"] for r in results)
    total_repair_rounds = sum(r["repair_rounds"] for r in results)

    with open(csv_file, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

    with open(summary_file, "w") as sf:
        sf.write("AutoVerus Batch Repair Run\n")
        sf.write(f"Benchmark: {benchmark_name}\n")
        sf.write(f"Timestamp: {time.strftime('%Y%m%d-%H%M%S')}\n")
        sf.write(f"Config: {config_path}\n")
        sf.write(f"Total files: {len(all_rs_files)}\n")
        sf.write(f"Processed: {len(results)}\n")
        sf.write("=" * 80 + "\n\n")
        for idx, task_result in enumerate(results, 1):
            sf.write(f"[{idx:3d}] {task_result['file']}\n")
            sf.write(f"Status: {task_result['status']}\n")
            sf.write(f"Exit code: {task_result['exit_code']}\n")
            sf.write(f"Time: {task_result['time_seconds']:.2f} seconds\n")
            sf.write(f"Input tokens: {task_result['input_tokens']}\n")
            sf.write(f"Output tokens: {task_result['output_tokens']}\n")
            sf.write(f"Total tokens: {task_result['total_tokens']}\n")
            sf.write(f"LLM calls: {task_result['llm_calls']}\n")
            sf.write(f"Repair rounds: {task_result['repair_rounds']}\n")
            sf.write(f"Task log: {task_result['task_log_file']}\n")
            sf.write("-" * 80 + "\n\n")

        sf.write("\n" + "=" * 80 + "\n")
        sf.write("FINAL SUMMARY\n")
        sf.write("=" * 80 + "\n")
        sf.write(f"Total files:   {len(all_rs_files)}\n")
        sf.write(f"Processed:     {len(results)}\n")
        if results:
            sf.write(f"Verified:      {verified_count} ({100*verified_count/len(results):.1f}%)\n")
            sf.write(f"Failed:        {failed_count} ({100*failed_count/len(results):.1f}%)\n")
            sf.write(f"Timeout:       {timeout_count} ({100*timeout_count/len(results):.1f}%)\n")
            sf.write(f"Exception:     {exception_count} ({100*exception_count/len(results):.1f}%)\n")
            sf.write(f"Error:         {error_count} ({100*error_count/len(results):.1f}%)\n")
        sf.write("\nToken Usage:\n")
        sf.write(f"Input tokens:  {total_input_tokens:,}\n")
        sf.write(f"Output tokens: {total_output_tokens:,}\n")
        sf.write(f"Total tokens:  {total_tokens:,}\n")
        sf.write(f"LLM calls:     {total_llm_calls:,}\n")
        sf.write(f"Repair rounds: {total_repair_rounds:,}\n")

    results_json = {
        "benchmark": benchmark_name,
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "config": str(config_path),
        "total": len(all_rs_files),
        "processed": len(results),
        "verified": verified_count,
        "failed": failed_count,
        "timeout": timeout_count,
        "exception": exception_count,
        "error": error_count,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_llm_calls": total_llm_calls,
        "total_repair_rounds": total_repair_rounds,
        "tasks": results,
    }
    (output_root / "results.json").write_text(json.dumps(results_json, indent=2), encoding="utf-8")

    return results_json


def run_one_task(rs_file, config_path, output_root, task_log_dir, repair, timeout, autoverus_dir):
    output_file = output_root / f"{rs_file.stem}_output.rs"
    task_log_file = task_log_dir / f"{rs_file.stem}.log"
    cmd = [
        sys.executable,
        "main.py",
        "--config",
        str(config_path),
        "--mode",
        "repair",
        "--input",
        str(rs_file),
        "--output",
        str(output_file),
        "--outdir",
        str(output_root),
        "--repair",
        str(repair),
    ]

    start_time = time.time()
    try:
        with subprocess.Popen(
            cmd,
            cwd=str(autoverus_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        ) as proc:
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                raise

        elapsed = time.time() - start_time
        combined_log = "STDOUT:\n" + stdout + "\nSTDERR:\n" + stderr
        task_log_file.write_text(combined_log, encoding="utf-8")
        metrics = parse_metrics_from_text(combined_log)

        if exit_code == 0:
            status = "VERIFIED"
        elif exit_code == 233:
            status = "FAILED"
        else:
            status = "EXCEPTION"

        return {
            "file": rs_file.name,
            "exit_code": exit_code,
            "status": status,
            "time_seconds": round(elapsed, 2),
            "input_tokens": metrics["input_tokens"],
            "output_tokens": metrics["output_tokens"],
            "total_tokens": metrics["input_tokens"] + metrics["output_tokens"],
            "llm_calls": metrics["llm_calls"],
            "repair_rounds": metrics["repair_rounds"],
            "task_log_file": str(task_log_file),
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        task_log_file.write_text("TIMEOUT\n", encoding="utf-8")
        return {
            "file": rs_file.name,
            "exit_code": -1,
            "status": "TIMEOUT",
            "time_seconds": round(elapsed, 2),
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "llm_calls": 0,
            "repair_rounds": 0,
            "task_log_file": str(task_log_file),
        }
    except Exception as exc:
        elapsed = time.time() - start_time
        task_log_file.write_text(f"ERROR: {exc}\n", encoding="utf-8")
        return {
            "file": rs_file.name,
            "exit_code": -2,
            "status": "ERROR",
            "time_seconds": round(elapsed, 2),
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "llm_calls": 0,
            "repair_rounds": 0,
            "task_log_file": str(task_log_file),
        }


def run_batch(
    input_dir,
    config_file,
    output_dir=None,
    repair=5,
    max_tasks=None,
    timeout=1200,
    num_workers=1,
):
    input_dir_path = Path(input_dir).resolve()
    config_path = Path(config_file).resolve()
    autoverus_dir = Path(__file__).resolve().parent

    benchmark_name = input_dir_path.parent.name + "_" + input_dir_path.name
    output_root = Path(output_dir) if output_dir else Path(f"batch-results-{benchmark_name}")
    output_root.mkdir(parents=True, exist_ok=True)
    task_log_dir = output_root / "task_logs"
    task_log_dir.mkdir(parents=True, exist_ok=True)

    all_rs_files = sorted(input_dir_path.glob("*.rs"))
    rs_files = all_rs_files[:max_tasks] if max_tasks is not None else all_rs_files

    if num_workers > 1:
        print(f"Found {len(rs_files)} files to process (total {len(all_rs_files)})")
        print(f"Output directory: {output_root}")
        print(f"AutoVerus num_workers: {num_workers}")
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    run_one_task,
                    rs_file,
                    config_path,
                    output_root,
                    task_log_dir,
                    repair,
                    timeout,
                    autoverus_dir,
                ): rs_file
                for rs_file in rs_files
            }
            for fut in as_completed(futures):
                task_result = fut.result()
                results.append(task_result)
                print(
                    "  {file}: {status} (time={time_seconds:.2f}s, tokens={total_tokens}, "
                    "llm_calls={llm_calls}, rounds={repair_rounds})".format(**task_result)
                )
        results_json = write_results(output_root, benchmark_name, config_path, all_rs_files, results)
        print(f"\nAutoVerus batch complete. Results saved to: {output_root}")
        print(f"Verified: {results_json['verified']}/{len(results)}")
        print(f"Total tokens: {results_json['total_tokens']:,}; LLM calls: {results_json['total_llm_calls']:,}")
        return

    csv_file = output_root / "results.csv"
    summary_file = output_root / "summary.txt"
    results = []

    with open(csv_file, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=FIELDNAMES)
        writer.writeheader()

    with open(summary_file, "w") as sf:
        sf.write("AutoVerus Batch Repair Run\n")
        sf.write(f"Benchmark: {benchmark_name}\n")
        sf.write(f"Timestamp: {time.strftime('%Y%m%d-%H%M%S')}\n")
        sf.write(f"Config: {config_path}\n")
        sf.write(f"Total files: {len(all_rs_files)}\n")
        sf.write(f"Scheduled files: {len(rs_files)}\n")
        sf.write("=" * 80 + "\n\n")

        for idx, rs_file in enumerate(rs_files, 1):
            output_file = output_root / f"{rs_file.stem}_output.rs"
            task_log_file = task_log_dir / f"{rs_file.stem}.log"
            cmd = [
                sys.executable,
                "main.py",
                "--config",
                str(config_path),
                "--mode",
                "repair",
                "--input",
                str(rs_file),
                "--output",
                str(output_file),
                "--outdir",
                str(output_root),
                "--repair",
                str(repair),
            ]

            print(f"\n[{idx}/{len(rs_files)}] AutoVerus processing: {rs_file.name}")
            sf.write(f"[{idx:3d}] {rs_file.name}\n")
            sf.write(f"Command: {' '.join(cmd)}\n")
            sf.flush()

            start_time = time.time()
            try:
                with subprocess.Popen(
                    cmd,
                    cwd=str(autoverus_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                ) as proc:
                    try:
                        stdout, stderr = proc.communicate(timeout=timeout)
                        exit_code = proc.returncode
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        raise

                elapsed = time.time() - start_time
                combined_log = "STDOUT:\n" + stdout + "\nSTDERR:\n" + stderr
                task_log_file.write_text(combined_log, encoding="utf-8")
                metrics = parse_metrics_from_text(combined_log)

                if exit_code == 0:
                    status = "VERIFIED"
                elif exit_code == 233:
                    status = "FAILED"
                else:
                    status = "EXCEPTION"

                task_result = {
                    "file": rs_file.name,
                    "exit_code": exit_code,
                    "status": status,
                    "time_seconds": round(elapsed, 2),
                    "input_tokens": metrics["input_tokens"],
                    "output_tokens": metrics["output_tokens"],
                    "total_tokens": metrics["input_tokens"] + metrics["output_tokens"],
                    "llm_calls": metrics["llm_calls"],
                    "repair_rounds": metrics["repair_rounds"],
                    "task_log_file": str(task_log_file),
                }
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start_time
                task_result = {
                    "file": rs_file.name,
                    "exit_code": -1,
                    "status": "TIMEOUT",
                    "time_seconds": round(elapsed, 2),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "llm_calls": 0,
                    "repair_rounds": 0,
                    "task_log_file": str(task_log_file),
                }
                task_log_file.write_text("TIMEOUT\n", encoding="utf-8")
            except Exception as exc:
                elapsed = time.time() - start_time
                task_result = {
                    "file": rs_file.name,
                    "exit_code": -2,
                    "status": "ERROR",
                    "time_seconds": round(elapsed, 2),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "llm_calls": 0,
                    "repair_rounds": 0,
                    "task_log_file": str(task_log_file),
                }
                task_log_file.write_text(f"ERROR: {exc}\n", encoding="utf-8")

            results.append(task_result)
            with open(csv_file, "a", newline="") as cf:
                writer = csv.DictWriter(cf, fieldnames=FIELDNAMES)
                writer.writerow(task_result)

            sf.write(f"Status: {task_result['status']}\n")
            sf.write(f"Exit code: {task_result['exit_code']}\n")
            sf.write(f"Time: {task_result['time_seconds']:.2f} seconds\n")
            sf.write(f"Input tokens: {task_result['input_tokens']}\n")
            sf.write(f"Output tokens: {task_result['output_tokens']}\n")
            sf.write(f"Total tokens: {task_result['total_tokens']}\n")
            sf.write(f"LLM calls: {task_result['llm_calls']}\n")
            sf.write(f"Repair rounds: {task_result['repair_rounds']}\n")
            sf.write(f"Task log: {task_result['task_log_file']}\n")
            sf.write("-" * 80 + "\n\n")
            sf.flush()

            print(
                "  Status: {status} (exit={exit_code}, time={time_seconds:.2f}s, "
                "tokens={total_tokens}, llm_calls={llm_calls}, rounds={repair_rounds})".format(
                    **task_result
                )
            )

        verified_count = sum(1 for r in results if r["status"] == "VERIFIED")
        failed_count = sum(1 for r in results if r["status"] == "FAILED")
        timeout_count = sum(1 for r in results if r["status"] == "TIMEOUT")
        exception_count = sum(1 for r in results if r["status"] == "EXCEPTION")
        error_count = sum(1 for r in results if r["status"] == "ERROR")
        total_input_tokens = sum(r["input_tokens"] for r in results)
        total_output_tokens = sum(r["output_tokens"] for r in results)
        total_tokens = sum(r["total_tokens"] for r in results)
        total_llm_calls = sum(r["llm_calls"] for r in results)
        total_repair_rounds = sum(r["repair_rounds"] for r in results)

        sf.write("\n" + "=" * 80 + "\n")
        sf.write("FINAL SUMMARY\n")
        sf.write("=" * 80 + "\n")
        sf.write(f"Total files:   {len(all_rs_files)}\n")
        sf.write(f"Processed:     {len(results)}\n")
        if results:
            sf.write(f"Verified:      {verified_count} ({100*verified_count/len(results):.1f}%)\n")
            sf.write(f"Failed:        {failed_count} ({100*failed_count/len(results):.1f}%)\n")
            sf.write(f"Timeout:       {timeout_count} ({100*timeout_count/len(results):.1f}%)\n")
            sf.write(f"Exception:     {exception_count} ({100*exception_count/len(results):.1f}%)\n")
            sf.write(f"Error:         {error_count} ({100*error_count/len(results):.1f}%)\n")
        sf.write("\nToken Usage:\n")
        sf.write(f"Input tokens:  {total_input_tokens:,}\n")
        sf.write(f"Output tokens: {total_output_tokens:,}\n")
        sf.write(f"Total tokens:  {total_tokens:,}\n")
        sf.write(f"LLM calls:     {total_llm_calls:,}\n")
        sf.write(f"Repair rounds: {total_repair_rounds:,}\n")

    results_json = {
        "benchmark": benchmark_name,
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "config": str(config_path),
        "total": len(all_rs_files),
        "processed": len(results),
        "verified": verified_count,
        "failed": failed_count,
        "timeout": timeout_count,
        "exception": exception_count,
        "error": error_count,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_llm_calls": total_llm_calls,
        "total_repair_rounds": total_repair_rounds,
        "tasks": results,
    }
    (output_root / "results.json").write_text(json.dumps(results_json, indent=2), encoding="utf-8")

    print(f"\nAutoVerus batch complete. Results saved to: {output_root}")
    print(f"Verified: {verified_count}/{len(results)}")
    print(f"Total tokens: {total_tokens:,}; LLM calls: {total_llm_calls:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch runner for AutoVerus repair tasks")
    parser.add_argument("input_dir", help="Directory containing .rs files to process")
    parser.add_argument("config_file", help="Path to config JSON file")
    parser.add_argument("--output-dir", default=None, help="Output directory for batch results")
    parser.add_argument("--repair", type=int, default=5, help="Number of repair steps")
    parser.add_argument("--max-tasks", type=int, default=None, help="Only process the first N tasks")
    parser.add_argument("--timeout", type=int, default=1200, help="Per-task timeout in seconds")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of repair tasks to run concurrently")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    if not os.path.exists(args.config_file):
        print(f"Error: Config file not found: {args.config_file}")
        sys.exit(1)

    run_batch(
        args.input_dir,
        args.config_file,
        args.output_dir,
        args.repair,
        args.max_tasks,
        args.timeout,
        args.num_workers,
    )
