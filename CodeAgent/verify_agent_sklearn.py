"""
verify_agent_sklearn.py – Comprehensive test suite for mini_claude_code agent.

Tests the agent's ability to generate working code across multiple domains:
1. Sklearn regression (simple, single-file)
2. ML Task: Linear Regression with metric-based evaluation (self-contained)
3. Sorting algorithm with built-in tests (pure algorithm)
4. Data processing script (pandas/csv)

Usage:
    cd /Developer/AIserver
    python3 tests/verify_agent_sklearn.py            # run all tests
    python3 tests/verify_agent_sklearn.py --test 1    # run only test 1
    python3 tests/verify_agent_sklearn.py --test 2    # run only test 2

    python3 CodeAgent/batch_coder.py --task-id linreg_lvl3_regularization_optim --status-file output/batch_status.json
"""

import sys
import os
import json
import shutil
import argparse
from pathlib import Path
from unittest.mock import patch

# Add project root to path
sys.path.append(os.getcwd())

from CodeAgent import mini_claude_code

OUTPUT_DIR = Path("output")
TASKS_JSON = Path("CodeAgent/ml_tasks.json")


def setup_output():
    """Clean output directory for fresh test run."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ml_task(task_index: int = 0):
    """Load ML task definition from ml_tasks.json."""
    if not TASKS_JSON.exists():
        print(f"Error: {TASKS_JSON} not found.")
        return None
    data = json.loads(TASKS_JSON.read_text())
    return data["tasks"][task_index]


def run_agent_test(name: str, goal: str, notes: str, allowlist: list):
    """Run the agent with given parameters and return success status."""
    print(f"\n{'='*60}")
    print(f"Running Test: {name}")
    print(f"Goal: {goal[:120]}...")
    print(f"Allowlist: {allowlist}")
    print(f"{'='*60}\n")
    
    args = [
        "mini_claude_code.py",
        "--goal", goal,
        "--notes", notes,
        "--allowlist", ",".join(str(p) for p in allowlist),
        "--yes"
    ]
    
    # Mock interactive prompts (shouldn't be needed with --yes, but as safety)
    def side_effect_prompt(prompt_text, **kwargs):
        if "Verification Command" in prompt_text:
            target = allowlist[0]
            if str(target).endswith(".py"):
                return f"python3 {target}" 
            return ""
        if "Skill tag" in prompt_text: return "auto_test"
        if "Skill text" in prompt_text: return "tested"
        if "Verdict" in prompt_text: return "success"
        return kwargs.get("default", "")

    def side_effect_confirm(prompt_text, **kwargs):
        return True

    with patch.object(sys, 'argv', args):
        with patch('rich.prompt.Prompt.ask', side_effect=side_effect_prompt):
            with patch('rich.prompt.Confirm.ask', side_effect=side_effect_confirm):
                try:
                    mini_claude_code.main()
                except SystemExit:
                    pass
                except Exception as e:
                    print(f"Agent crashed: {e}")
                    import traceback
                    traceback.print_exc()


# ======================================================================
# TEST CASE 1: Simple sklearn regression
# ======================================================================
def test_sklearn_regression():
    """Simple single-file sklearn script. Tests basic code generation."""
    sklearn_file = OUTPUT_DIR / "generated_sklearn.py"
    run_agent_test(
        name="Sklearn Regression",
        goal=(
            "Write a python script that loads the sklearn california housing dataset, "
            "splits it into train/test (80/20), trains a LinearRegression model, "
            "and prints the MSE and R2 score. The script must run successfully when "
            "executed with `python3`. Include proper imports and a main() function."
        ),
        notes="Use sklearn.datasets.fetch_california_housing. Handle imports. "
              "Print results clearly with labels.",
        allowlist=[sklearn_file]
    )


# ======================================================================
# TEST CASE 2: ML Task - Linear Regression (self-contained with metrics)
# ======================================================================
def test_ml_linreg():
    """
    Multi-function ML task with metric-based evaluation.
    Tests: complex code gen, train/val split, R2/MSE computation, assertions.
    """
    task = load_ml_task(0)
    if not task:
        print("SKIP: Could not load ML task definition.")
        return
    
    task_id = task["id"]
    base_path = OUTPUT_DIR / f"tasks/{task_id}"
    task_file = base_path / "task.py"
    
    # Build requirements string
    reqs = task["requirements"]
    req_str = "\n".join(f"- {k.title()}: {v}" for k, v in reqs.items())
    
    # Load protocol rules
    protocol = None
    data = json.loads(TASKS_JSON.read_text())
    proto_id = task.get("interface_protocol", "")
    if proto_id and proto_id in data.get("interface_protocols", {}):
        protocol = data["interface_protocols"][proto_id]
    
    eval_rules = ""
    if protocol and "evaluation_rules" in protocol:
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
        f"3. Print standard metrics: MSE, R2 score, learned parameters vs true values\n"
        f"4. Assert quality thresholds (R2 > 0.9 on validation, parameter error < 1.0)\n"
        f"5. Exit 0 on success, non-zero on failure\n\n"
        f"Do NOT create separate test files or README. The script IS the test."
    )
    
    notes = (
        f"Requirements:\n{req_str}\n\n"
        f"Evaluation Rules:\n{eval_rules}\n\n"
        f"Output file: {task_file}\n\n"
        f"IMPORTANT: Only create task.py. No test_task.py, no README.md."
    )
    
    run_agent_test(
        name=f"ML Task: {task['id']}",
        goal=goal,
        notes=notes,
        allowlist=[task_file]
    )


# ======================================================================
# TEST CASE 3: Sorting algorithm with self-test
# ======================================================================
def test_sorting_algorithm():
    """
    Pure algorithm task: implement sorting with built-in verification.
    Tests: algorithmic code generation, test logic, assertions.
    """
    sort_file = OUTPUT_DIR / "sorting_algorithms.py"
    run_agent_test(
        name="Sorting Algorithm",
        goal=(
            "Write a Python file implementing three sorting algorithms: "
            "bubble_sort, merge_sort, and quicksort. Each takes a list and "
            "returns a sorted list (do NOT sort in place).\n\n"
            "Include a main() block that tests all three algorithms with:\n"
            "1. Random lists of various sizes (0, 1, 10, 1000 elements)\n"
            "2. Already sorted and reverse-sorted lists\n"
            "3. Lists with duplicates\n"
            "4. Assert each result equals Python's built-in sorted()\n"
            "5. Print timing for the 1000-element case\n"
            "6. Print 'ALL TESTS PASSED' at the end if everything succeeds"
        ),
        notes="Pure Python, no external libraries except random and time. "
              "Each function must have a proper docstring.",
        allowlist=[sort_file]
    )


# ======================================================================
# TEST CASE 4: Data processing script
# ======================================================================
def test_data_processing():
    """
    Data processing task: generate synthetic CSV, process it, report stats.
    Tests: file I/O, data manipulation, statistics computation.
    """
    data_file = OUTPUT_DIR / "data_processor.py"
    run_agent_test(
        name="Data Processing",
        goal=(
            "Write a Python script that:\n"
            "1. Generates a synthetic CSV file 'output/sample_data.csv' with columns: "
            "id, name, age, salary, department (at least 50 rows of realistic fake data)\n"
            "2. Reads the CSV back and computes:\n"
            "   - Mean, median, std of salary\n"
            "   - Average salary per department\n"
            "   - Age distribution (min, max, mean)\n"
            "   - Count of employees per department\n"
            "3. Saves a summary report to 'output/data_summary.txt'\n"
            "4. Prints all statistics to stdout\n"
            "5. Asserts the CSV has >= 50 rows and the expected columns exist\n\n"
            "The script must run successfully with `python3` and produce both output files."
        ),
        notes="Use only Python standard library (csv module). Do NOT use pandas. "
              "Generate realistic-looking fake data with random module.",
        allowlist=[data_file]
    )


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Verify mini_claude_code agent")
    parser.add_argument("--test", type=int, help="Run only test N (1-4)", default=None)
    args = parser.parse_args()
    
    setup_output()
    
    tests = {
        1: ("Sklearn Regression", test_sklearn_regression),
        2: ("ML Task: LinReg", test_ml_linreg),
        3: ("Sorting Algorithm", test_sorting_algorithm),
        4: ("Data Processing", test_data_processing),
    }
    
    if args.test:
        if args.test in tests:
            name, func = tests[args.test]
            print(f"\n>>> Running single test: {name}")
            func()
        else:
            print(f"Error: --test must be 1-{len(tests)}")
            sys.exit(1)
    else:
        print(f"\n>>> Running all {len(tests)} tests")
        for num, (name, func) in tests.items():
            print(f"\n{'#'*60}")
            print(f"# Test {num}: {name}")
            print(f"{'#'*60}")
            func()
    
    print(f"\n{'='*60}")
    print("All requested tests completed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
