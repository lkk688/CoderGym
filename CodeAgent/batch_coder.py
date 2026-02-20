#!/usr/bin/env python3
"""
batch_coder.py — Run mini_claude_code agent on all tasks in ml_tasks.json sequentially.

Generates code for each ML task, tracks pass/fail status, and saves results
to a JSON status file.

Usage:
    cd /Developer/AIserver
    python3 CodeAgent/batch_coder.py                           # run all tasks
    python3 CodeAgent/batch_coder.py --start-from 3            # skip first 3 tasks
    python3 CodeAgent/batch_coder.py --task-id linreg_lvl1_raw_tensors  # run one task
    python3 CodeAgent/batch_coder.py --status-file results.json  # custom output file
"""

import json
import sys
import os
import time
import subprocess
import shutil
from pathlib import Path
from datetime import datetime


# ---------------------
# Configuration
# ---------------------
TASKS_JSON = Path("CodeAgent/ml_tasks.json")
OUTPUT_DIR = Path("output")
DEFAULT_STATUS_FILE = Path("output/batch_status.json")

# Inherit from env or use defaults
BASE_URL = os.environ.get("VLLM_BASE_URL", "https://w0wqtv67-8000.usw3.devtunnels.ms/v1")
API_KEY = os.environ.get("VLLM_API_KEY", "myhpcvllmqwen123")
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-Coder-Next-FP8")


def load_tasks(tasks_json: Path) -> dict:
    """Load the full tasks configuration."""
    data = json.loads(tasks_json.read_text())
    return data


def build_goal_and_notes(task: dict, protocol: dict) -> tuple:
    """Build goal and notes strings from a task definition."""
    reqs = task.get("requirements", {})
    req_str = "\n".join(f"- {k.title()}: {v}" for k, v in reqs.items())

    eval_rules = ""
    if "evaluation_rules" in protocol:
        eval_rules = "\n".join(f"- {r}" for r in protocol["evaluation_rules"])

    goal = (
        f"Implement ML Task: {task['algorithm']}\n\n"
        f"Description: {task['description']}\n\n"
        f"Write a SINGLE self-contained Python file (task.py) with these functions:\n"
        f"get_task_metadata, set_seed, get_device, make_dataloaders, build_model, "
        f"train, evaluate, predict, save_artifacts.\n\n"
        f"CRITICAL: The if __name__ == '__main__' block must:\n"
        f"1. Train the model\n"
        f"2. Evaluate on BOTH train and validation splits\n"
        f"3. Print standard metrics (MSE, R2, accuracy, etc. as appropriate)\n"
        f"4. Assert quality thresholds so script exits non-zero on failure\n"
        f"5. Print a clear PASS/FAIL summary\n\n"
        f"Do NOT create separate test files or README. The script IS the test."
    )

    notes = (
        f"Requirements:\n{req_str}\n\n"
        f"Evaluation Rules:\n{eval_rules}\n\n"
        f"IMPORTANT: Only create task.py. No test_task.py, no README.md.\n"
        f"Protocol: {protocol.get('prompt_instructions', '')}"
    )

    return goal, notes


def run_single_task(task: dict, protocol: dict, output_dir: Path, verbose: bool = False) -> dict:
    """
    Run the mini_claude_code agent for a single task.
    Returns a status dict with success/failure, timing, and details.
    """
    task_id = task["id"]
    task_dir = output_dir / "tasks" / task_id
    task_file = task_dir / "task.py"

    # Clean previous output for this task
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    goal, notes = build_goal_and_notes(task, protocol)

    # Build the command
    cmd = [
        sys.executable, "-m", "CodeAgent.mini_claude_codev4",
        "--goal", goal,
        "--notes", notes,
        "--allowlist", str(task_file),
        "--yes",
        "--base-url", BASE_URL,
        "--api-key", API_KEY,
        "--model", MODEL,
        "--artifacts-dir", str(task_dir),  # Save artifacts directly to task folder
    ]

    start_time = time.time()
    result = {
        "task_id": task_id,
        "algorithm": task["algorithm"],
        "series": task["series"],
        "level": task["level"],
        "status": "unknown",
        "start_time": datetime.now().isoformat(),
        "duration_sec": 0,
        "task_file_exists": False,
        "verification_passed": False,
        "error": None,
        "output_snippet": "",
        "log_path": None,  # New field
    }

    try:
        print(f"\n{'='*70}")
        print(f"  Running: {task_id} — {task['algorithm']}")
        print(f"  Level: {task['level']} | Series: {task['series']}")
        print(f"{'='*70}\n")

        # Run the agent as a subprocess
        env = os.environ.copy()
        if verbose:
            env["FORCE_COLOR"] = "1"

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge stderr into stdout
            text=True,
            cwd=str(Path.cwd()),
            env=env,
            bufsize=1, # Line buffered
        )

        captured_lines = []
        try:
            # Stream output
            for line in proc.stdout:
                captured_lines.append(line)
                if verbose:
                    sys.stdout.write(line)
                    sys.stdout.flush()
            
            proc.wait(timeout=800)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise

        full_output = "".join(captured_lines)
        elapsed = time.time() - start_time
        result["duration_sec"] = round(elapsed, 1)
        result["output_snippet"] = full_output[-1000:]
        
        # Capture Log Path from stdout metadata
        # Look for: [METADATA] LOG_PATH: /path/to/logs
        import re
        m_log = re.search(r'\[METADATA\] LOG_PATH:\s*(.+)', full_output)
        if m_log:
            result["log_path"] = m_log.group(1).strip()
            print(f"  Logs: {result['log_path']}")

        # Check if task.py was created
        result["task_file_exists"] = task_file.exists()

        if proc.returncode == 0 and task_file.exists():
            # Try to verify the generated file
            verify_result = subprocess.run(
                [sys.executable, str(task_file)],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(Path.cwd()),
            )
            result["verification_passed"] = verify_result.returncode == 0
            if verify_result.returncode == 0:
                result["status"] = "success"
                result["output_snippet"] = (verify_result.stdout or "")[-500:]
                
                # Cleanup: Delete everything in task_dir except task.py
                try:
                    cleaned_count = 0
                    for item in task_dir.iterdir():
                        if item.name != "task.py":
                            if item.is_dir():
                                shutil.rmtree(item)
                            else:
                                item.unlink()
                            cleaned_count += 1
                    if verbose and cleaned_count > 0:
                        print(f"  Cleanup: Removed {cleaned_count} artifacts, kept only task.py")
                except Exception as e:
                    print(f"  [WARNING] Cleanup Failed: {e}")

            else:
                result["status"] = "verify_failed"
                result["error"] = (verify_result.stderr or verify_result.stdout or "")[-500:]
        elif task_file.exists():
            result["status"] = "agent_failed_file_exists"
            result["error"] = full_output[-500:]
        else:
            result["status"] = "agent_failed_no_file"
            result["error"] = full_output[-500:]

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Task exceeded 10 minute timeout"
        result["duration_sec"] = round(time.time() - start_time, 1)
    except Exception as e:
        result["status"] = "exception"
        result["error"] = str(e)
        result["duration_sec"] = round(time.time() - start_time, 1)

    status_icon = "✓" if result["status"] == "success" else "✗"
    print(f"\n  {status_icon} {task_id}: {result['status']} ({result['duration_sec']}s)")

    return result


def save_status(results: list, status_file: Path):
    """Save batch results to JSON."""
    summary = {
        "total": len(results),
        "success": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] != "success"),
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
    }

    output = {
        "summary": summary,
        "tasks": results,
    }

    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nStatus saved to: {status_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch run ML tasks through mini_claude_code")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Skip first N tasks (0-indexed)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Maximum number of tasks to run")
    parser.add_argument("--task-id", type=str, default=None,
                        help="Run only this specific task ID")
    parser.add_argument("--redo-failed", action="store_true",
                        help="Retry all tasks that failed in the previous run")
    parser.add_argument("--status-file", type=str, default=str(DEFAULT_STATUS_FILE),
                        help="Path to save status JSON")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Base output directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output from code agent")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    status_file = Path(args.status_file)

    # Load tasks
    data = load_tasks(TASKS_JSON)
    tasks = data["tasks"]
    protocols = data.get("interface_protocols", {})

    # Filter tasks
    if args.task_id:
        tasks = [t for t in tasks if t["id"] == args.task_id]
        if not tasks:
            print(f"Error: task '{args.task_id}' not found in {TASKS_JSON}")
            sys.exit(1)
    else:
        tasks = tasks[args.start_from:]
        if args.max_tasks:
            tasks = tasks[:args.max_tasks]

    print(f"\n{'#'*70}")
    print(f"  Batch Coder — {len(tasks)} task(s) to process")
    print(f"  Model: {MODEL}")
    print(f"  Output: {output_dir}")
    print(f"  Status: {status_file}")
    print(f"{'#'*70}")

    # Load existing results if resuming or redoing
    results = []
    if status_file.exists():
        try:
            existing = json.loads(status_file.read_text())
            results = existing.get("tasks", [])
        except Exception:
            pass

    # Filter tasks based on args
    # Pre-check for missing files:
    # If a task is marked success but task.py is missing, mark it failed so redo picks it up
    for r in results:
        t_id = r["task_id"]
        t_dir = output_dir / "tasks" / t_id
        t_file = t_dir / "task.py"
        if not t_file.exists() and r["status"] == "success":
            print(f"  [WARNING] Task {t_id} marked success but task.py missing. Marking for redo.")
            r["status"] = "missing_file"
        elif not t_file.exists() and r["status"] not in ("success", "missing_file"):
             # Ensure failures with empty dirs are also caught if needed
             pass

    if args.task_id:
        tasks = [t for t in tasks if t["id"] == args.task_id]
        if not tasks:
            print(f"Error: task '{args.task_id}' not found in {TASKS_JSON}")
            sys.exit(1)
    elif args.redo_failed:
        # Find failed task IDs from existing results
        failed_ids = {r["task_id"] for r in results if r["status"] != "success"}
        
        # Also check for ORPHANED tasks (folder exists but missing from results, likely crashed)
        existing_ids = {r["task_id"] for r in results}
        for t in tasks:
            t_id = t["id"]
            if t_id not in existing_ids:
                t_dir = output_dir / "tasks" / t_id
                # If folder exists, assume we tried to run passing, or crashed.
                # If empty (no task.py), treat as failed.
                t_file = t_dir / "task.py"
                if t_dir.exists() and not t_file.exists():
                     print(f"  [WARNING] Task {t_id} folder exists but no result/task.py. Marking as orphaned failure.")
                     failed_ids.add(t_id)

        if not failed_ids:
            print("No failed tasks found in status file (or orphans). Nothing to redo.")
            sys.exit(0)
        
        # Filter tasks to only those that failed
        tasks = [t for t in tasks if t["id"] in failed_ids]
        print(f"  Redo Mode: Retrying {len(tasks)} failed tasks...")
        
        # Remove failed tasks from 'results' so we don't duplicate entries
        # ONLY remove those present in results. Orphans are already missing.
        results = [r for r in results if r["task_id"] not in failed_ids]
    else:
        # Normal run (resume mode)
        # Skip tasks that are already in results (success OR failure)
        # unless user explicitly asked to redo them? Default behavior is resume.
        if not args.start_from and status_file.exists():
            existing_ids = {r["task_id"] for r in results}
            tasks = [t for t in tasks if t["id"] not in existing_ids]

        tasks = tasks[args.start_from:]
        if args.max_tasks:
            tasks = tasks[:args.max_tasks]

    # Run each task
    for i, task in enumerate(tasks):
        task_id = task["id"]
        proto_id = task.get("interface_protocol", "pytorch_task_v1")
        protocol = protocols.get(proto_id, {})

        print(f"\n[{i+1}/{len(tasks)}] Starting {task_id}...")

        result = run_single_task(task, protocol, output_dir, verbose=args.verbose)
        results.append(result)

        # Save after each task (in case of crash)
        save_status(results, status_file)

    # Final summary
    success = sum(1 for r in results if r["status"] == "success")
    total = len(results)

    print(f"\n{'='*70}")
    print(f"  BATCH COMPLETE: {success}/{total} tasks succeeded")
    print(f"{'='*70}")

    # Print per-task table
    print(f"\n  {'Task ID':<40} {'Status':<20} {'Time':>8}")
    print(f"  {'-'*40} {'-'*20} {'-':->8}")
    for r in results:
        icon = "✓" if r["status"] == "success" else "✗"
        print(f"  {icon} {r['task_id']:<38} {r['status']:<20} {r['duration_sec']:>6.1f}s")

    save_status(results, status_file)


if __name__ == "__main__":
    main()

"""
python3 CodeAgent/batch_coder.py --task-id linreg_lvl3_regularization_optim --status-file output/batch_status.json

python3 CodeAgent/batch_coder.py --task-id linreg_lvl4_sklearn_production --status-file output/batch_status.json

"""