#!/usr/bin/env python3
"""
Batch runner for Verus repair tasks.
Runs repair on all files in a directory and logs results.
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


def parse_tokens_from_log(log_file):
    """Parse input and output tokens from the log file"""
    total_input = 0
    total_output = 0
    llm_calls = 0
    repair_rounds = 0

    if not log_file or not Path(log_file).exists():
        return total_input, total_output, llm_calls, repair_rounds

    try:
        with open(log_file) as f:
            content = f.read()
            # Find all token counts (format: "Input tokens: 1234")
            input_tokens = re.findall(r"Input tokens:\s*(\d+)", content)
            output_tokens = re.findall(r"Output tokens:\s*(\d+)", content)
            explicit_llm_calls = re.findall(r"LLM calls:\s*(\d+)", content)
            repair_attempts = re.findall(r"Repair attempt\s+(\d+)\s*/", content)

            total_input = sum(int(t) for t in input_tokens)
            total_output = sum(int(t) for t in output_tokens)
            llm_calls = sum(int(t) for t in explicit_llm_calls) or len(input_tokens)
            repair_rounds = max([int(t) for t in repair_attempts], default=0)
    except Exception as e:
        print(f"Warning: Failed to parse tokens from {log_file}: {e}")

    return total_input, total_output, llm_calls, repair_rounds


def find_task_log(output_dir, rs_file):
    """Find the newest verus-repair.log for a task."""
    candidates = list(Path(output_dir).glob(f"o-{rs_file.stem}-*"))
    if not candidates:
        return None
    newest_subdir = max(candidates, key=lambda p: p.stat().st_mtime)
    log_file = newest_subdir / "verus-repair.log"
    return log_file if log_file.exists() else None


def write_results(output_dir, benchmark_name, config_file, all_rs_files, results):
    results = sorted(results, key=lambda r: r["file"])
    csv_file = output_dir / "results.csv"
    summary_file = output_dir / "summary.txt"

    verified_count = sum(1 for r in results if r["status"] == "VERIFIED")
    failed_count = sum(1 for r in results if r["status"] == "FAILED")
    timeout_count = sum(1 for r in results if r["status"] == "TIMEOUT")
    error_count = sum(1 for r in results if r["status"] == "ERROR")
    exception_count = sum(1 for r in results if r["status"] == "EXCEPTION")
    total_input_tokens = sum(r.get("input_tokens", 0) for r in results)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in results)
    total_tokens = total_input_tokens + total_output_tokens
    total_llm_calls = sum(r.get("llm_calls", 0) for r in results)
    total_repair_rounds = sum(r.get("repair_rounds", 0) for r in results)

    with open(csv_file, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

    with open(summary_file, "w") as sf:
        sf.write("Batch Repair Run\n")
        sf.write(f"Benchmark: {benchmark_name}\n")
        sf.write(f"Timestamp: {time.strftime('%Y%m%d-%H%M%S')}\n")
        sf.write(f"Config: {config_file}\n")
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
        sf.write(f"Total files:  {len(all_rs_files)}\n")
        sf.write(f"Processed:    {len(results)}\n")
        if results:
            sf.write(f"Verified:     {verified_count} ({100*verified_count/len(results):.1f}%)\n")
            sf.write(f"Failed:       {failed_count} ({100*failed_count/len(results):.1f}%)\n")
            sf.write(f"Timeout:      {timeout_count} ({100*timeout_count/len(results):.1f}%)\n")
            sf.write(f"Exception:    {exception_count} ({100*exception_count/len(results):.1f}%)\n")
            sf.write(f"Error:        {error_count} ({100*error_count/len(results):.1f}%)\n")
        sf.write("\nToken Usage:\n")
        sf.write(f"Input tokens:  {total_input_tokens:,}\n")
        sf.write(f"Output tokens: {total_output_tokens:,}\n")
        sf.write(f"Total tokens:  {total_tokens:,}\n")
        sf.write(f"LLM calls:     {total_llm_calls:,}\n")
        sf.write(f"Repair rounds: {total_repair_rounds:,}\n")

    results_json = {
        "benchmark": benchmark_name,
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "config": config_file,
        "total": len(all_rs_files),
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
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    return results_json


def run_one_task(rs_file, config_file, output_dir, verus_args, repair, timeout):
    output_file = output_dir / f"{rs_file.stem}_output.rs"
    cmd = [
        sys.executable,
        "main.py",
        "--config",
        config_file,
        "--mode",
        "agent",
        "--input",
        str(rs_file),
        "--output",
        str(output_file),
        "--outdir",
        str(output_dir),
        "--repair",
        str(repair),
    ]
    if verus_args:
        cmd.extend(["--verus-args", verus_args])

    start_time = time.time()
    try:
        with subprocess.Popen(
            cmd,
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
        if exit_code == 0:
            status = "VERIFIED"
        elif exit_code == 233:
            status = "FAILED"
        else:
            status = "EXCEPTION"
            with open(output_dir / f"{rs_file.stem}_error.log", "w") as ef:
                ef.write("STDOUT:\n")
                ef.write(stdout + "\n")
                ef.write("STDERR:\n")
                ef.write(stderr + "\n")
        log_file = find_task_log(output_dir, rs_file)
        input_tokens, output_tokens, llm_calls, repair_rounds = parse_tokens_from_log(log_file)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        status = "TIMEOUT"
        exit_code = -1
        log_file = find_task_log(output_dir, rs_file)
        input_tokens, output_tokens, llm_calls, repair_rounds = parse_tokens_from_log(log_file)
    except Exception:
        elapsed = time.time() - start_time
        status = "ERROR"
        exit_code = -2
        log_file = find_task_log(output_dir, rs_file)
        input_tokens, output_tokens, llm_calls, repair_rounds = parse_tokens_from_log(log_file)

    return {
        "file": rs_file.name,
        "exit_code": exit_code,
        "status": status,
        "time_seconds": round(elapsed, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "llm_calls": llm_calls,
        "repair_rounds": repair_rounds,
        "task_log_file": str(log_file) if log_file else "",
    }


def run_batch(
    input_dir,
    config_file,
    output_dir=None,
    verus_args="",
    continue_from=None,
    max_tasks=None,
    repair=5,
    timeout=1200,
    num_workers=1,
):
    """
    Run repair on all .rs files in the input directory.

    Args:
        input_dir: Path to directory containing .rs files
        config_file: Path to config JSON file
        output_dir: Output directory for batch results (optional)
        verus_args: Additional arguments to pass to Verus
        continue_from: Directory to continue execution from (optional)
    """
    input_dir_path = Path(input_dir)

    # Results tracking
    results = []
    processed_files = set()

    if continue_from:
        output_dir = Path(continue_from)
        if not output_dir.exists():
            print(f"Error: Continue directory not found: {continue_from}")
            return
        print(f"Continuing from: {output_dir}")

        # Load existing results from CSV (preferred as it's incremental)
        csv_file = output_dir / "results.csv"
        if csv_file.exists():
            try:
                with open(csv_file) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Convert types back from string
                        try:
                            task = {
                                "file": row["file"],
                                "exit_code": int(row["exit_code"]),
                                "status": row["status"],
                                "time_seconds": float(row["time_seconds"]),
                                "input_tokens": int(row["input_tokens"]),
                                "output_tokens": int(row["output_tokens"]),
                                "total_tokens": int(row["total_tokens"]),
                                "llm_calls": int(row.get("llm_calls", 0) or 0),
                                "repair_rounds": int(row.get("repair_rounds", 0) or 0),
                                "task_log_file": row.get("task_log_file", ""),
                            }
                            results.append(task)
                            processed_files.add(task["file"])
                        except (ValueError, KeyError) as e:
                            print(f"Warning: Skipping malformed row in CSV: {row} ({e})")

                print(f"Loaded {len(results)} previous results from CSV")
            except Exception as e:
                print(f"Warning: Failed to load existing results from CSV: {e}")

        # Fallback to JSON if no results loaded from CSV
        if not results:
            json_file = output_dir / "results.json"
            if json_file.exists():
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        if "tasks" in data:
                            results = data["tasks"]
                            processed_files = {task["file"] for task in results}
                    print(f"Loaded {len(results)} previous results from JSON")
                except Exception as e:
                    print(f"Warning: Failed to load existing results from JSON: {e}")
    else:
        # Create output directory
        benchmark_name = input_dir_path.parent.name + "_" + input_dir_path.name
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = Path(f"batch-results-{benchmark_name}-{timestamp}")

        output_dir.mkdir(parents=True, exist_ok=True)

    # Get all .rs files
    all_rs_files = sorted(input_dir_path.glob("*.rs"))

    # Filter out processed files
    rs_files = [f for f in all_rs_files if f.name not in processed_files]
    if max_tasks is not None:
        rs_files = rs_files[:max_tasks]

    if not rs_files:
        print(f"No new .rs files to process in {input_dir}")
        if not results:
            return
    else:
        print(f"Found {len(rs_files)} new files to process (total {len(all_rs_files)})")

    print(f"Output directory: {output_dir}")

    summary_file = output_dir / "summary.txt"
    csv_file = output_dir / "results.csv"

    if num_workers > 1:
        print(f"VeruSAGE num_workers: {num_workers}")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    run_one_task,
                    rs_file,
                    config_file,
                    output_dir,
                    verus_args,
                    repair,
                    timeout,
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
        results_json = write_results(
            output_dir,
            input_dir_path.parent.name + "_" + input_dir_path.name,
            config_file,
            all_rs_files,
            results,
        )
        print(f"\nBatch run complete! Results saved to: {output_dir}")
        print(f"Total tokens: {results_json['total_tokens']:,}; LLM calls: {results_json['total_llm_calls']:,}")
        return

    # Create CSV file with headers if it doesn't exist or we are starting new
    write_header = not continue_from or not csv_file.exists()
    if write_header:
        with open(csv_file, "w", newline="") as cf:
            fieldnames = [
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
            csv_writer = csv.DictWriter(cf, fieldnames=fieldnames)
            csv_writer.writeheader()

    # Open summary file in append mode if continuing, else write
    mode = "a" if continue_from else "w"
    with open(summary_file, mode) as sf:
        if not continue_from:
            sf.write("Batch Repair Run\n")
            sf.write(f"Benchmark: {input_dir_path.parent.name}_{input_dir_path.name}\n")
            sf.write(f"Timestamp: {time.strftime('%Y%m%d-%H%M%S')}\n")
            sf.write(f"Config: {config_file}\n")
            sf.write(f"Total files: {len(all_rs_files)}\n")
            sf.write("=" * 80 + "\n\n")
        else:
            sf.write(f"\nResuming run at {time.strftime('%Y%m%d-%H%M%S')}\n")
            sf.write("=" * 80 + "\n\n")

        start_idx = len(processed_files) + 1
        for idx, rs_file in enumerate(rs_files, start_idx):
            file_name = rs_file.name
            print(f"\n[{idx}/{len(all_rs_files)}] Processing: {file_name}")

            # Output file path
            output_file = output_dir / f"{rs_file.stem}_output.rs"

            # Build command
            cmd = [
                sys.executable,
                "main.py",
                "--config",
                config_file,
                "--mode",
                "agent",
                "--input",
                str(rs_file),
                "--output",
                str(output_file),
                "--outdir",
                str(output_dir),
                "--repair",
                str(repair),
            ]

            if verus_args:
                print(f"Verus args: {verus_args}")
                cmd.extend(["--verus-args", verus_args])

            # Create quoted string for logging
            cmd_log = [f"'{arg}'" if " " in arg else arg for arg in cmd]
            cmd_text = " ".join(cmd_log)
            sf.write(f"[{idx:3d}] {file_name}\n")
            sf.write(f"Command: {cmd_text}\n")
            sf.flush()

            # Run main.py
            start_time = time.time()
            try:
                # Use Popen to allow killing the process group on timeout
                with subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                ) as proc:
                    try:
                        stdout, stderr = proc.communicate(timeout=timeout)
                        exit_code = proc.returncode
                        result = subprocess.CompletedProcess(cmd, exit_code, stdout, stderr)
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        raise

                elapsed = time.time() - start_time
                if exit_code == 0:
                    status = "VERIFIED"
                elif exit_code == 233:
                    status = "FAILED"
                else:
                    status = "EXCEPTION"
                    with open(output_dir / f"{rs_file.stem}_error.log", "w") as ef:
                        ef.write("STDOUT:\n")
                        ef.write(result.stdout + "\n")
                        ef.write("STDERR:\n")
                        ef.write(result.stderr + "\n")

                # Find and parse log file for token counts
                # Log file is in output_dir/o-{filename}-{timestamp}/verus-repair.log
                log_file = find_task_log(output_dir, rs_file)

                input_tokens, output_tokens, llm_calls, repair_rounds = parse_tokens_from_log(
                    log_file
                )

                # Record result
                task_result = {
                    "file": file_name,
                    "exit_code": exit_code,
                    "status": status,
                    "time_seconds": round(elapsed, 2),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "llm_calls": llm_calls,
                    "repair_rounds": repair_rounds,
                    "task_log_file": str(log_file) if log_file else "",
                }
                results.append(task_result)

                # Write to summary
                sf.write(f"Status: {status}\n")
                sf.write(f"Exit code: {exit_code}\n")
                sf.write(f"Time: {elapsed:.2f} seconds\n")
                sf.write(f"Input tokens: {input_tokens}\n")
                sf.write(f"Output tokens: {output_tokens}\n")
                sf.write(f"Total tokens: {input_tokens + output_tokens}\n")
                sf.write(f"LLM calls: {llm_calls}\n")
                sf.write(f"Repair rounds: {repair_rounds}\n")
                sf.write(f"Task log: {log_file if log_file else ''}\n")
                sf.write("-" * 80 + "\n\n")
                sf.flush()

                print(
                    f"  Status: {status} (exit={exit_code}, time={elapsed:.2f}s, tokens={input_tokens + output_tokens}, llm_calls={llm_calls}, rounds={repair_rounds})"
                )

                # Append to CSV
                with open(csv_file, "a", newline="") as cf:
                    fieldnames = [
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
                    csv_writer = csv.DictWriter(cf, fieldnames=fieldnames)
                    csv_writer.writerow(task_result)

            except subprocess.TimeoutExpired:
                elapsed = time.time() - start_time
                status = "TIMEOUT"
                log_file = find_task_log(output_dir, rs_file)
                input_tokens, output_tokens, llm_calls, repair_rounds = parse_tokens_from_log(
                    log_file
                )

                task_result = {
                    "file": file_name,
                    "exit_code": -1,
                    "status": status,
                    "time_seconds": round(elapsed, 2),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "llm_calls": llm_calls,
                    "repair_rounds": repair_rounds,
                    "task_log_file": str(log_file) if log_file else "",
                }
                results.append(task_result)

                sf.write(f"Status: {status}\n")
                sf.write(f"Timeout after {elapsed:.2f} seconds\n")
                sf.write(f"Input tokens: {input_tokens}\n")
                sf.write(f"Output tokens: {output_tokens}\n")
                sf.write(f"Total tokens: {input_tokens + output_tokens}\n")
                sf.write(f"LLM calls: {llm_calls}\n")
                sf.write(f"Repair rounds: {repair_rounds}\n")
                sf.write(f"Task log: {log_file if log_file else ''}\n")
                sf.write("-" * 80 + "\n\n")
                sf.flush()

                print(f"  Status: {status} (timeout)")

                # Append to CSV
                with open(csv_file, "a", newline="") as cf:
                    fieldnames = [
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
                    csv_writer = csv.DictWriter(cf, fieldnames=fieldnames)
                    csv_writer.writerow(task_result)

            except Exception as e:
                elapsed = time.time() - start_time
                status = "ERROR"

                task_result = {
                    "file": file_name,
                    "exit_code": -2,
                    "status": status,
                    "time_seconds": round(elapsed, 2),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "llm_calls": 0,
                    "repair_rounds": 0,
                    "task_log_file": "",
                }
                results.append(task_result)

                sf.write(f"Status: {status}\n")
                sf.write(f"Error: {str(e)}\n")
                sf.write("-" * 80 + "\n\n")
                sf.flush()

                print(f"  Status: {status} - {str(e)}")

                # Append to CSV
                with open(csv_file, "a", newline="") as cf:
                    fieldnames = [
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
                    csv_writer = csv.DictWriter(cf, fieldnames=fieldnames)
                    csv_writer.writerow(task_result)

        # Write final summary
        verified_count = sum(1 for r in results if r["status"] == "VERIFIED")
        failed_count = sum(1 for r in results if r["status"] == "FAILED")
        timeout_count = sum(1 for r in results if r["status"] == "TIMEOUT")
        error_count = sum(1 for r in results if r["status"] == "ERROR")
        exception_count = sum(1 for r in results if r["status"] == "EXCEPTION")
        total_input_tokens = sum(r.get("input_tokens", 0) for r in results)
        total_output_tokens = sum(r.get("output_tokens", 0) for r in results)
        total_tokens = total_input_tokens + total_output_tokens
        total_llm_calls = sum(r.get("llm_calls", 0) for r in results)
        total_repair_rounds = sum(r.get("repair_rounds", 0) for r in results)

        total_processed = len(results)

        sf.write("\n" + "=" * 80 + "\n")
        sf.write("FINAL SUMMARY\n")
        sf.write("=" * 80 + "\n")
        sf.write(f"Total files:  {len(all_rs_files)}\n")
        sf.write(f"Processed:    {total_processed}\n")
        if total_processed > 0:
            sf.write(
                f"Verified:     {verified_count} ({100*verified_count/total_processed:.1f}%)\n"
            )
            sf.write(f"Failed:       {failed_count} ({100*failed_count/total_processed:.1f}%)\n")
            sf.write(f"Timeout:      {timeout_count} ({100*timeout_count/total_processed:.1f}%)\n")
            sf.write(
                f"Exception:    {exception_count} ({100*exception_count/total_processed:.1f}%)\n"
            )
            sf.write(f"Error:        {error_count} ({100*error_count/total_processed:.1f}%)\n")
        sf.write("\nToken Usage:\n")
        sf.write(f"Input tokens:  {total_input_tokens:,}\n")
        sf.write(f"Output tokens: {total_output_tokens:,}\n")
        sf.write(f"Total tokens:  {total_tokens:,}\n")
        sf.write(f"LLM calls:     {total_llm_calls:,}\n")
        sf.write(f"Repair rounds: {total_repair_rounds:,}\n")

    # Save JSON results
    # If continuing, we overwrite the JSON with the updated list (which includes old + new)
    with open(output_dir / "results.json", "w") as f:
        json.dump(
            {
                "benchmark": input_dir_path.parent.name + "_" + input_dir_path.name,
                "timestamp": time.strftime("%Y%m%d-%H%M%S"),
                "config": config_file,
                "total": len(all_rs_files),
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
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 80}")
    print("Batch run complete!")
    print(f"Results saved to: {output_dir}")
    if total_processed > 0:
        print(
            f"Verified: {verified_count}/{total_processed} ({100*verified_count/total_processed:.1f}%)"
        )
    print(
        f"Total tokens: {total_tokens:,} (input: {total_input_tokens:,}, output: {total_output_tokens:,})"
    )
    print(f"LLM calls: {total_llm_calls:,}; repair rounds: {total_repair_rounds:,}")
    print("\nOutput files:")
    print(f"  - CSV:     {csv_file}")
    print(f"  - JSON:    {output_dir / 'results.json'}")
    print(f"  - Summary: {summary_file}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch runner for Verus repair tasks")
    parser.add_argument("input_dir", help="Directory containing .rs files to process")
    parser.add_argument("config_file", help="Path to config JSON file")
    parser.add_argument("--output-dir", default=None, help="Output directory for batch results")
    parser.add_argument(
        "--verus-args", type=str, default="", help="Additional arguments to pass to Verus"
    )
    parser.add_argument(
        "--continue-from", type=str, default=None, help="Directory to continue execution from"
    )
    parser.add_argument("--max-tasks", type=int, default=None, help="Only process the first N tasks")
    parser.add_argument("--repair", type=int, default=5, help="Number of repair steps")
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
        args.verus_args,
        args.continue_from,
        args.max_tasks,
        args.repair,
        args.timeout,
        args.num_workers,
    )
