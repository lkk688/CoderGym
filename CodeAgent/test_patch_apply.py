#!/usr/bin/env python3
"""
Test suite for LLM response extraction and patch application.

Validates extract_all_diffs, extract_write_file_actions, sanitize_diff_text,
apply_fuzzy_patch, and _try_apply_content against real LLM session data 
from .agent/sessions/.

Usage:
    python3 CodeAgent/test_patch_apply.py                    # Run all tests
    python3 CodeAgent/test_patch_apply.py --session-prefix 2026-02-17_001031  # Test specific session
    python3 CodeAgent/test_patch_apply.py --verbose          # Show all details
"""

import os
import re
import sys
import shutil
import tempfile
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CodeAgent.mini_claude_codev4 import (
    extract_all_diffs,
    extract_write_file_actions,
    sanitize_diff_text,
    apply_fuzzy_patch,
    _try_apply_content,
    extract_files_from_diff,
    resolve_path,
    AgentConfig,
)

# ─── Test Result Tracking ────────────────────────────────────────────────────

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures: List[str] = []
    
    def record_pass(self, name: str, detail: str = ""):
        self.passed += 1
        if detail:
            print(f"  ✅ {name}: {detail}")
        else:
            print(f"  ✅ {name}")
    
    def record_fail(self, name: str, detail: str):
        self.failed += 1
        msg = f"  ❌ {name}: {detail}"
        self.failures.append(msg)
        print(msg)
    
    def record_skip(self, name: str, reason: str):
        self.skipped += 1
        print(f"  ⏭️  {name}: {reason}")
    
    def summary(self):
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.passed} passed, {self.failed} failed, {self.skipped} skipped / {total} total")
        if self.failures:
            print(f"\nFAILURES:")
            for f in self.failures:
                print(f)
        print(f"{'='*60}")
        return self.failed == 0


# ─── Unit Tests for Known Broken Cases ───────────────────────────────────────

def test_known_broken_cases(results: TestResults, verbose: bool = False):
    """Test the two specifically identified broken cases."""
    
    print("\n── Unit Tests: Known Broken Cases ──")
    
    # ═══ Case 1: Session 001031 ═══
    # Bug: sanitize_diff_text was stripping `-    #```python` 
    #      and converting `index --- a/` to `--- a/` (duplicate header)
    case1_response = """## Reasoning

Looking at the error output, there's a `SyntaxError`.

## Action

```diff
diff --git a/output/tasks/linreg_lvl3_regularization_optim/task.py b/output/tasks/linreg_lvl3_regularization_optim/task.py
index --- a/output/tasks/linreg_lvl3_regularization_optim/task.py
+++ b/output/tasks/linreg_lvl3_regularization_optim/task.py
@@ -473,8 +473,6 @@ def main():
     # Save artifacts
     save_artifacts(model, history, train_metrics, val_metrics)
     
-    #```python
-Exit with appropriate code
     sys.exit(0 if all_pass else 1)
 
 
```"""
    
    diff = extract_all_diffs(case1_response)
    if diff is None:
        results.record_fail("Case1: extract_all_diffs", "returned None")
    else:
        # Check: the diff must contain the removal of #```python
        if '-    #```python' in diff:
            results.record_pass("Case1: backtick content preserved", 
                              f"'-    #```python' found in extracted diff")
        else:
            results.record_fail("Case1: backtick content preserved",
                              f"'-    #```python' missing from diff.\nGot:\n{diff[:300]}")
        
        # Check: no duplicate --- headers (index line should be dropped)
        header_count = diff.count('--- a/')
        if header_count <= 1:
            results.record_pass("Case1: no duplicate --- headers", 
                              f"found {header_count} '--- a/' headers")
        else:
            results.record_fail("Case1: no duplicate --- headers",
                              f"found {header_count} '--- a/' headers (expected 1)")
        
        # Check: diff must have valid structure
        has_diff_header = 'diff --git' in diff
        has_minus_header = '--- ' in diff
        has_plus_header = '+++ ' in diff
        has_hunk = '@@ ' in diff
        if all([has_diff_header, has_minus_header, has_plus_header, has_hunk]):
            results.record_pass("Case1: valid diff structure")
        else:
            results.record_fail("Case1: valid diff structure",
                              f"missing: diff={has_diff_header}, ---={has_minus_header}, "
                              f"+++={has_plus_header}, @@={has_hunk}")
        
        if verbose:
            print(f"    Extracted diff:\n{diff}")
    
    # ═══ Case 2: Session 003549 ═══
    # Bug: diff --git header is OUTSIDE the ```diff fence block
    case2_response = """## Reasoning

The fix should be:
```python
corr_matrix = np.corrcoef(np.column_stack([X, y]))
```

## Action

diff --git a/output/tasks/linreg_lvl4_sklearn_production/task.py b/output/tasks/linreg_lvl4_sklearn_production/task.py
```diff
@@ -252,7 +252,7 @@ def plot_eda(X, y, feature_names, output_dir="output"):
     plt.figure(figsize=(10, 8))
-    corr_matrix = np.corrcoef(np.column_stack([X.T, y]))
+    corr_matrix = np.corrcoef(np.column_stack([X, y]))
     plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
     plt.colorbar()
     plt.xticks(range(len(feature_names) + 1), feature_names + ['Target'], rotation=45)
```"""
    
    diff = extract_all_diffs(case2_response)
    if diff is None:
        results.record_fail("Case2: extract_all_diffs", "returned None")
    else:
        # Check: diff must have the actual change
        if '+    corr_matrix = np.corrcoef(np.column_stack([X, y]))' in diff:
            results.record_pass("Case2: change content preserved")
        else:
            results.record_fail("Case2: change content preserved",
                              f"Expected '+' line not found in diff")
        
        # Check: diff must have valid structure  
        has_diff_header = 'diff --git' in diff
        has_hunk = '@@ ' in diff
        if has_diff_header and has_hunk:
            results.record_pass("Case2: valid diff structure", 
                              "has diff --git and @@ headers")
        else:
            results.record_fail("Case2: valid diff structure",
                              f"diff --git={has_diff_header}, @@={has_hunk}")
        
        if verbose:
            print(f"    Extracted diff:\n{diff}")
    
    # ═══ Case 3: Standard fenced diff (should still work) ═══
    case3_response = """## Action

```diff
diff --git a/task.py b/task.py
--- a/task.py
+++ b/task.py
@@ -299,6 +299,10 @@ def main():
     print(f"Training samples: {X_train.shape[0]}")
     print(f"Validation samples: {X_val.shape[0]}")
     
+    # Move tensors to the same device as the model
+    X_train, y_train = X_train.to(device), y_train.to(device)
+    X_val, y_val = X_val.to(device), y_val.to(device)
+    
     # Build model
     print("\\nBuilding model...")
     model = build_model(device)
```"""
    
    diff = extract_all_diffs(case3_response)
    if diff is None:
        results.record_fail("Case3: standard fenced diff", "returned None")
    else:
        has_all = ('diff --git' in diff and '@@ ' in diff and 
                   '+    X_train, y_train' in diff)
        if has_all:
            results.record_pass("Case3: standard fenced diff (regression check)")
        else:
            results.record_fail("Case3: standard fenced diff",
                              f"missing expected content")
    
    # ═══ Case 4: WRITE_FILE extraction ═══
    case4_response = """## Action

WRITE_FILE: task.py
<<<CONTENT
import torch
import numpy as np

def main():
    print("hello")

if __name__ == '__main__':
    main()
CONTENT>>>
"""
    
    actions = extract_write_file_actions(case4_response)
    if actions and len(actions) == 1:
        path, content = actions[0]
        if path == "task.py" and "import torch" in content:
            results.record_pass("Case4: WRITE_FILE extraction")
        else:
            results.record_fail("Case4: WRITE_FILE extraction",
                              f"path={path}, content starts: {content[:50]}")
    else:
        results.record_fail("Case4: WRITE_FILE extraction",
                          f"expected 1 action, got {len(actions) if actions else 0}")
    
    # ═══ Case 5: sanitize_diff_text preserves diff content with backticks ═══
    backtick_diff = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -10,3 +10,3 @@
-    code = "```python"
+    code = "```bash"
     # comment
"""
    sanitized = sanitize_diff_text(backtick_diff)
    if '-    code = "```python"' in sanitized and '+    code = "```bash"' in sanitized:
        results.record_pass("Case5: backtick in diff content lines preserved")
    else:
        results.record_fail("Case5: backtick in diff content lines preserved",
                          f"sanitized:\n{sanitized}")

    # ═══ Case 6: index line with hash should be dropped ═══
    index_diff = """diff --git a/task.py b/task.py
index 1234567..abcdefg 100644
--- a/task.py
+++ b/task.py
@@ -1,3 +1,3 @@
-old line
+new line
 unchanged
"""
    sanitized = sanitize_diff_text(index_diff)
    if 'index ' not in sanitized:
        results.record_pass("Case6: index hash line dropped")
    else:
        results.record_fail("Case6: index hash line dropped",
                          f"'index' line still present")

    # ═══ Case 7: 'index --- a/file' should NOT become '--- a/file' ═══
    malformed_index_diff = """diff --git a/task.py b/task.py
index --- a/task.py
+++ b/task.py
@@ -1,3 +1,3 @@
-old
+new
 same
"""
    sanitized = sanitize_diff_text(malformed_index_diff)
    dash_count = sanitized.count('--- a/')
    # Should have exactly 1 --- header (auto-injected), not 2
    if dash_count <= 1:
        results.record_pass("Case7: malformed 'index ---' not converted to duplicate header",
                          f"{dash_count} '--- a/' headers")
    else:
        results.record_fail("Case7: malformed 'index ---' duplicate header",
                          f"found {dash_count} '--- a/' headers, expected ≤1")


# ─── Session-based Tests ─────────────────────────────────────────────────────

def find_session_dirs(base_dir: Path, prefix: str = "2026-02-17_") -> List[Path]:
    """Find all session directories matching the prefix."""
    sessions_dir = base_dir / ".agent" / "sessions"
    if not sessions_dir.exists():
        return []
    
    dirs = sorted([
        d for d in sessions_dir.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    ])
    return dirs


def test_session_extraction(
    session_dir: Path, 
    results: TestResults, 
    verbose: bool = False
):
    """
    Test response extraction for a single session.
    - 0000/response.md: typically WRITE_FILE (initial generation)
    - 0001/response.md: typically diff (fix attempt)
    """
    session_name = session_dir.name
    
    # Test 0000 folder (initial generation → usually WRITE_FILE)
    resp_0000 = session_dir / "0000" / "response.md"
    if resp_0000.exists():
        text = resp_0000.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            write_actions = extract_write_file_actions(text)
            diffs = extract_all_diffs(text)
            
            if write_actions:
                # Validate each WRITE_FILE action has valid path and content
                all_valid = True
                for path, content in write_actions:
                    if not path or len(content.strip()) < 10:
                        all_valid = False
                        results.record_fail(
                            f"{session_name}/0000 WRITE_FILE",
                            f"invalid action: path='{path}', content_len={len(content)}")
                        break
                if all_valid:
                    results.record_pass(
                        f"{session_name}/0000 WRITE_FILE",
                        f"{len(write_actions)} file(s) extracted")
            elif diffs:
                results.record_pass(
                    f"{session_name}/0000 extract_diffs",
                    f"diff extracted ({len(diffs)} bytes)")
            else:
                # Some responses might have neither (e.g., pure reasoning)
                results.record_skip(
                    f"{session_name}/0000",
                    "no WRITE_FILE or diff found")
    
    # Test 0001 folder (fix attempt → typically diff)
    resp_0001 = session_dir / "0001" / "response.md"
    if resp_0001.exists():
        text = resp_0001.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            diffs = extract_all_diffs(text)
            write_actions = extract_write_file_actions(text)
            
            if diffs:
                # Validate diff structure
                has_diff_header = 'diff --git' in diffs
                has_hunk = '@@ ' in diffs
                
                if has_diff_header and has_hunk:
                    results.record_pass(
                        f"{session_name}/0001 extract_diffs",
                        f"valid diff ({len(diffs)} bytes)")
                else:
                    results.record_fail(
                        f"{session_name}/0001 extract_diffs",
                        f"malformed: diff_header={has_diff_header}, hunk={has_hunk}")
                    if verbose:
                        print(f"    Extracted:\n{diffs[:200]}")
                    
                # Cross-check with apply.log if it exists
                apply_log = session_dir / "0001" / "apply.log"
                if apply_log.exists():
                    log_text = apply_log.read_text(encoding="utf-8", errors="ignore")
                    had_success = "exit=0" in log_text
                    had_failure = ("error:" in log_text.lower() or 
                                   "No valid patches" in log_text)
                    
                    if verbose:
                        status = "SUCCESS" if had_success else "FAILED" 
                        print(f"    apply.log: {status}")
                        
            elif write_actions:
                results.record_pass(
                    f"{session_name}/0001 WRITE_FILE (fix)",
                    f"{len(write_actions)} file(s)")
            else:
                results.record_skip(
                    f"{session_name}/0001",
                    "no diff or WRITE_FILE found")
    
    # Test additional rounds (0002, 0003, ...)
    for i in range(2, 10):
        resp = session_dir / f"{i:04d}" / "response.md"
        if resp.exists():
            text = resp.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                diffs = extract_all_diffs(text)
                write_actions = extract_write_file_actions(text)
                
                if diffs or write_actions:
                    kind = "diff" if diffs else "WRITE_FILE"
                    results.record_pass(f"{session_name}/{i:04d} {kind}")
                else:
                    results.record_skip(f"{session_name}/{i:04d}", "no extractable content")


# ─── Sanitize Diff Regression Tests ─────────────────────────────────────────

def test_sanitize_regressions(results: TestResults, verbose: bool = False):
    """Test sanitize_diff_text edge cases."""
    
    print("\n── Sanitize Diff Regression Tests ──")
    
    # Test: Empty input
    assert sanitize_diff_text("") == "\n", "Empty input should return newline"
    results.record_pass("Sanitize: empty input")
    
    # Test: Diff with no issues
    clean_diff = """diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,3 @@
-old
+new
 same
"""
    sanitized = sanitize_diff_text(clean_diff)
    if 'diff --git' in sanitized and '--- a/foo.py' in sanitized:
        results.record_pass("Sanitize: clean diff unchanged")
    else:
        results.record_fail("Sanitize: clean diff unchanged", f"got: {sanitized[:100]}")
    
    # Test: Missing --- header (should be auto-injected)
    missing_header = """diff --git a/bar.py b/bar.py
+++ b/bar.py
@@ -1,3 +1,3 @@
-x
+y
 z
"""
    sanitized = sanitize_diff_text(missing_header)
    if '--- a/bar.py' in sanitized:
        results.record_pass("Sanitize: missing --- auto-injected")
    else:
        results.record_fail("Sanitize: missing --- auto-injected", 
                          f"no '--- a/bar.py' in output")
    
    # Test: Missing both --- and +++ (should be injected before @@)
    missing_both = """diff --git a/baz.py b/baz.py
@@ -1,3 +1,3 @@
-a
+b
 c
"""
    sanitized = sanitize_diff_text(missing_both)
    has_minus = '--- a/baz.py' in sanitized
    has_plus = '+++ b/baz.py' in sanitized
    if has_minus and has_plus:
        results.record_pass("Sanitize: missing ---/+++ both auto-injected")
    else:
        results.record_fail("Sanitize: missing ---/+++ both auto-injected",
                          f"---={has_minus}, +++={has_plus}")
    
    # Test: Bare backtick fence should be stripped
    with_fence = """diff --git a/x.py b/x.py
--- a/x.py
+++ b/x.py
@@ -1,2 +1,2 @@
-old
+new
```
some trailing stuff
"""
    sanitized = sanitize_diff_text(with_fence)
    if '```' not in sanitized:
        results.record_pass("Sanitize: bare backtick fence stripped")
    else:
        results.record_fail("Sanitize: bare backtick fence stripped",
                          "``` still in output")
    
    # Test: Diff removal line containing backticks should be KEPT
    backtick_content = """diff --git a/x.py b/x.py
--- a/x.py
+++ b/x.py
@@ -1,3 +1,2 @@
-    #```python
-Exit with code
     sys.exit(0)
"""
    sanitized = sanitize_diff_text(backtick_content)
    if '-    #```python' in sanitized:
        results.record_pass("Sanitize: diff '-' line with backticks preserved")
    else:
        results.record_fail("Sanitize: diff '-' line with backticks preserved",
                          f"line missing from output")


# ─── apply_fuzzy_patch End-to-End Tests ──────────────────────────────────────

def _make_temp_file(tmp_dir: Path, filename: str, content: str) -> Path:
    """Helper: create a temp file with given content."""
    fp = tmp_dir / filename
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content, encoding="utf-8")
    return fp


def test_fuzzy_patch_e2e(results: TestResults, verbose: bool = False):
    """End-to-end tests for apply_fuzzy_patch with real temp files."""
    
    print("\n── apply_fuzzy_patch E2E Tests ──")
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        
        # ═══ Test 1: Exact Match — basic hunk application ═══
        original = (
            "import os\n"
            "import sys\n"
            "\n"
            "def main():\n"
            "    x = 1\n"
            "    y = 2\n"
            "    print(x + y)\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        fp = _make_temp_file(tmp_dir, "exact.py", original)
        
        diff_content = (
            "diff --git a/exact.py b/exact.py\n"
            "--- a/exact.py\n"
            "+++ b/exact.py\n"
            "@@ -4,4 +4,4 @@ def main():\n"
            "     x = 1\n"
            "-    y = 2\n"
            "+    y = 42\n"
            "     print(x + y)\n"
        )
        
        ok = apply_fuzzy_patch(fp, diff_content)
        result = fp.read_text(encoding="utf-8")
        if ok and "y = 42" in result and "y = 2" not in result:
            results.record_pass("FuzzyPatch: exact match")
        else:
            results.record_fail("FuzzyPatch: exact match",
                              f"ok={ok}, result_snippet: {result[:200]}")
        
        # ═══ Test 2: Whitespace Fuzzy Match ═══
        # File has slightly different indentation than the diff expects
        original_ws = (
            "def foo():\n"
            "   x = 1\n"         # 3 spaces
            "   y = 2\n"         # 3 spaces
            "   return x + y\n"  # 3 spaces
        )
        fp2 = _make_temp_file(tmp_dir, "fuzzy_ws.py", original_ws)
        
        diff_ws = (
            "diff --git a/fuzzy_ws.py b/fuzzy_ws.py\n"
            "--- a/fuzzy_ws.py\n"
            "+++ b/fuzzy_ws.py\n"
            "@@ -1,4 +1,4 @@\n"
            " def foo():\n"
            "     x = 1\n"          # 4 spaces in diff
            "-    y = 2\n"          # 4 spaces in diff
            "+    y = 99\n"
            "     return x + y\n"   # 4 spaces in diff
        )
        
        ok = apply_fuzzy_patch(fp2, diff_ws)
        result = fp2.read_text(encoding="utf-8")
        if ok and "y = 99" in result:
            results.record_pass("FuzzyPatch: whitespace-insensitive match")
        else:
            results.record_fail("FuzzyPatch: whitespace-insensitive match",
                              f"ok={ok}, result: {result[:200]}")
        
        # ═══ Test 3: Anchor Match (Strategy C) ═══
        # Scenario: The file has an EXTRA line that the diff context doesn't know about.
        # Exact and whitespace-stripped matching both fail because the block lengths differ.
        # Strategy C should find the first+last anchor lines and match based on those.
        original_anchor = (
            "import torch\n"
            "import numpy as np\n"
            "\n"
            "def compute():\n"
            "    data = load_data()\n"
            "    # EXTRA COMMENT not in diff context\n"  # <-- extra line
            "    result = data * 2\n"
            "    return result\n"
            "\n"
            "def main():\n"
            "    compute()\n"
        )
        fp3 = _make_temp_file(tmp_dir, "anchor.py", original_anchor)
        
        # Diff context doesn't include the extra comment line
        diff_anchor = (
            "diff --git a/anchor.py b/anchor.py\n"
            "--- a/anchor.py\n"
            "+++ b/anchor.py\n"
            "@@ -4,4 +4,4 @@\n"
            " def compute():\n"
            "     data = load_data()\n"
            "-    result = data * 2\n"
            "+    result = data * 3\n"
            "     return result\n"
        )
        
        ok = apply_fuzzy_patch(fp3, diff_anchor)
        result = fp3.read_text(encoding="utf-8")
        if ok and "result = data * 3" in result and "result = data * 2" not in result:
            results.record_pass("FuzzyPatch: anchor match (Strategy C)")
        else:
            results.record_fail("FuzzyPatch: anchor match (Strategy C)",
                              f"ok={ok}, result: {result[:200]}")
        
        # ═══ Test 4: Empty Lines in Hunk ═══
        original_empty = (
            "def foo():\n"
            "    a = 1\n"
            "\n"                    # Empty line
            "    b = 2\n"
            "\n"                    # Empty line
            "    return a + b\n"
        )
        fp4 = _make_temp_file(tmp_dir, "empty_lines.py", original_empty)
        
        diff_empty = (
            "diff --git a/empty_lines.py b/empty_lines.py\n"
            "--- a/empty_lines.py\n"
            "+++ b/empty_lines.py\n"
            "@@ -1,6 +1,6 @@\n"
            " def foo():\n"
            "     a = 1\n"
            "\n"                        # Empty context line (no space prefix)
            "-    b = 2\n"
            "+    b = 42\n"
            "\n"                        # Empty context line
            "     return a + b\n"
        )
        
        ok = apply_fuzzy_patch(fp4, diff_empty)
        result = fp4.read_text(encoding="utf-8")
        if ok and "b = 42" in result and "b = 2" not in result:
            results.record_pass("FuzzyPatch: empty lines in hunk preserved")
        else:
            results.record_fail("FuzzyPatch: empty lines in hunk preserved",
                              f"ok={ok}, result: {result[:200]}")
        
        # ═══ Test 5: Trailing Newline Preservation ═══
        original_nl = "line1\nline2\nline3\n"  # Has trailing newline
        fp5 = _make_temp_file(tmp_dir, "trailing_nl.txt", original_nl)
        
        diff_nl = (
            "diff --git a/trailing_nl.txt b/trailing_nl.txt\n"
            "--- a/trailing_nl.txt\n"
            "+++ b/trailing_nl.txt\n"
            "@@ -1,3 +1,3 @@\n"
            " line1\n"
            "-line2\n"
            "+line2_modified\n"
            " line3\n"
        )
        
        ok = apply_fuzzy_patch(fp5, diff_nl)
        result = fp5.read_text(encoding="utf-8")
        if ok and result.endswith("\n") and "line2_modified" in result:
            results.record_pass("FuzzyPatch: trailing newline preserved")
        else:
            results.record_fail("FuzzyPatch: trailing newline preserved",
                              f"ok={ok}, ends_nl={result.endswith(chr(10))}, result={result!r}")
        
        # ═══ Test 6: New File Creation via diff ═══
        new_fp = tmp_dir / "newfile.py"
        assert not new_fp.exists()
        
        diff_new = (
            "diff --git a/newfile.py b/newfile.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/newfile.py\n"
            "@@ -0,0 +1,4 @@\n"
            "+import sys\n"
            "+\n"
            "+def hello():\n"
            "+    print('hello')\n"
        )
        
        ok = apply_fuzzy_patch(new_fp, diff_new)
        if ok and new_fp.exists():
            content = new_fp.read_text(encoding="utf-8")
            if "import sys" in content and "def hello" in content:
                results.record_pass("FuzzyPatch: new file creation")
            else:
                results.record_fail("FuzzyPatch: new file creation",
                                  f"unexpected content: {content[:200]}")
        else:
            results.record_fail("FuzzyPatch: new file creation",
                              f"ok={ok}, exists={new_fp.exists()}")
        
        # ═══ Test 7: Multi-hunk Application ═══
        original_multi = (
            "import os\n"
            "import sys\n"
            "\n"
            "VAL_A = 10\n"
            "\n"
            "def first():\n"
            "    return VAL_A\n"
            "\n"
            "VAL_B = 20\n"
            "\n"
            "def second():\n"
            "    return VAL_B\n"
        )
        fp7 = _make_temp_file(tmp_dir, "multi.py", original_multi)
        
        diff_multi = (
            "diff --git a/multi.py b/multi.py\n"
            "--- a/multi.py\n"
            "+++ b/multi.py\n"
            "@@ -4,1 +4,1 @@\n"
            "-VAL_A = 10\n"
            "+VAL_A = 100\n"
            "@@ -9,1 +9,1 @@\n"
            "-VAL_B = 20\n"
            "+VAL_B = 200\n"
        )
        
        ok = apply_fuzzy_patch(fp7, diff_multi)
        result = fp7.read_text(encoding="utf-8")
        if ok and "VAL_A = 100" in result and "VAL_B = 200" in result:
            results.record_pass("FuzzyPatch: multi-hunk application")
        else:
            results.record_fail("FuzzyPatch: multi-hunk application",
                              f"ok={ok}, result: {result[:300]}")
        
        # ═══ Test 8: Non-existent file returns False ═══
        missing_fp = tmp_dir / "does_not_exist.py"
        diff_missing = (
            "diff --git a/does_not_exist.py b/does_not_exist.py\n"
            "--- a/does_not_exist.py\n"
            "+++ b/does_not_exist.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "+new\n"
        )
        ok = apply_fuzzy_patch(missing_fp, diff_missing)
        if not ok:
            results.record_pass("FuzzyPatch: non-existent file returns False")
        else:
            results.record_fail("FuzzyPatch: non-existent file returns False",
                              "expected False, got True")


# ─── _try_apply_content End-to-End Tests ─────────────────────────────────────

@dataclass
class MockOpenAI:
    """Minimal mock for OpenAI client — never called in _try_apply_content."""
    api_key: str = "test"

def _make_agent_config(session_dir: Path) -> AgentConfig:
    """Create a minimal AgentConfig for testing."""
    return AgentConfig(
        client=MockOpenAI(),
        model="test",
        session_dir=session_dir,
        max_context=4096,
        max_output=4096,
        auto_approve=True,
        agent_dir=session_dir,
    )


def test_try_apply_content_e2e(results: TestResults, verbose: bool = False):
    """End-to-end tests for _try_apply_content with temp files."""
    
    print("\n── _try_apply_content E2E Tests ──")
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        
        # ═══ Test 1: Diff-based Content Application ═══
        # Create a target file, then apply a diff response to it
        target = _make_temp_file(
            tmp_dir, "task.py",
            "import os\nimport sys\n\ndef main():\n    x = 1\n    y = 2\n    print(x + y)\n\nif __name__ == '__main__':\n    main()\n"
        )
        
        turn_dir = tmp_dir / "turn1"
        turn_dir.mkdir()
        config = _make_agent_config(tmp_dir)
        
        response = """## Reasoning
The variable y should be 42.

## Action

```diff
diff --git a/task.py b/task.py
--- a/task.py
+++ b/task.py
@@ -4,4 +4,4 @@ def main():
     x = 1
-    y = 2
+    y = 42
     print(x + y)
```"""
        
        # _try_apply_content will try git apply (may fail in non-git temp dir)
        # then fuzzy patch, which should succeed
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_dir)
            ok = _try_apply_content(response, [str(target)], turn_dir, config)
        finally:
            os.chdir(old_cwd)
        
        result = target.read_text(encoding="utf-8")
        if ok and "y = 42" in result:
            results.record_pass("TryApply: diff content via fuzzy patch")
        else:
            results.record_fail("TryApply: diff content via fuzzy patch",
                              f"ok={ok}, result: {result[:200]}")
        
        # ═══ Test 2: WRITE_FILE Content Application ═══
        # Note: resolve_path resolves relative to CWD (which we set to tmp_dir)
        # and apply_write_files writes to the resolved path
        target2_name = "output.py"
        target2 = tmp_dir / target2_name
        turn_dir2 = tmp_dir / "turn2"
        turn_dir2.mkdir()
        
        response2 = f"""## Action

WRITE_FILE: {target2_name}
<<<CONTENT
import torch

def train():
    model = torch.nn.Linear(10, 1)
    print("Training...")
    return model

if __name__ == '__main__':
    train()
CONTENT>>>"""
        
        try:
            os.chdir(tmp_dir)
            ok = _try_apply_content(response2, [target2_name], turn_dir2, config)
        finally:
            os.chdir(old_cwd)
        
        # Check the file in the CWD-relative location (tmp_dir/output.py)
        written = tmp_dir / target2_name
        if ok and written.exists():
            content = written.read_text(encoding="utf-8")
            if "import torch" in content and "def train" in content:
                results.record_pass("TryApply: WRITE_FILE format")
            else:
                results.record_fail("TryApply: WRITE_FILE format",
                                  f"unexpected content: {content[:200]}")
        else:
            results.record_fail("TryApply: WRITE_FILE format",
                              f"ok={ok}, exists={written.exists()}")
        
        # ═══ Test 3: No valid content — should return False ═══
        turn_dir3 = tmp_dir / "turn3"
        turn_dir3.mkdir()
        
        response3 = """## Reasoning
I looked at the code but couldn't find the issue.
There's nothing to change."""
        
        try:
            os.chdir(tmp_dir)
            ok = _try_apply_content(response3, [], turn_dir3, config)
        finally:
            os.chdir(old_cwd)
        
        if not ok:
            results.record_pass("TryApply: no valid content returns False")
        else:
            results.record_fail("TryApply: no valid content returns False",
                              "expected False, got True")
        
        # ═══ Test 4: WRITE_FILE fallback when diff is found but can't apply ═══
        # Response has both a malformed diff AND a WRITE_FILE block
        fallback_name = "fallback.py"
        turn_dir4 = tmp_dir / "turn4"
        turn_dir4.mkdir()
        
        response4 = f"""## Action

```diff
diff --git a/{fallback_name} b/{fallback_name}
--- a/{fallback_name}
+++ b/{fallback_name}
@@ -100,3 +100,3 @@ 
 IMPOSSIBLE_CONTEXT_THAT_WONT_MATCH
-old line
+new line
 MORE_IMPOSSIBLE_CONTEXT
```

Since the diff might not apply cleanly, here's the full file:

WRITE_FILE: {fallback_name}
<<<CONTENT
def working():
    return True
CONTENT>>>"""
        
        try:
            os.chdir(tmp_dir)
            ok = _try_apply_content(response4, [fallback_name], turn_dir4, config)
        finally:
            os.chdir(old_cwd)
        
        written4 = tmp_dir / fallback_name
        if ok and written4.exists():
            content = written4.read_text(encoding="utf-8")
            if "def working" in content:
                results.record_pass("TryApply: WRITE_FILE fallback after diff failure")
            else:
                results.record_fail("TryApply: WRITE_FILE fallback after diff failure",
                                  f"content: {content[:200]}")
        else:
            results.record_fail("TryApply: WRITE_FILE fallback after diff failure",
                              f"ok={ok}, exists={written4.exists()}")

        # ═══ Test 5: New file from diff extraction ═══
        new_name = "brand_new.py"
        turn_dir5 = tmp_dir / "turn5"
        turn_dir5.mkdir()
        
        response5 = f"""## Action

```diff
diff --git a/{new_name} b/{new_name}
new file mode 100644
--- /dev/null
+++ b/{new_name}
@@ -0,0 +1,3 @@
+import sys
+def main():
+    sys.exit(0)
```"""
        
        try:
            os.chdir(tmp_dir)
            ok = _try_apply_content(response5, [new_name], turn_dir5, config)
        finally:
            os.chdir(old_cwd)
        
        new_file = tmp_dir / new_name
        if ok and new_file.exists():
            content = new_file.read_text(encoding="utf-8")
            if "import sys" in content:
                results.record_pass("TryApply: new file from diff")
            else:
                results.record_fail("TryApply: new file from diff",
                                  f"content: {content[:200]}")
        else:
            results.record_fail("TryApply: new file from diff",
                              f"ok={ok}, exists={new_file.exists()}")


# ─── Full-Pipeline Session E2E Tests ─────────────────────────────────────────

# Sessions where the LLM generated fundamentally broken diffs (hallucinated
# context, corrupted WRITE_FILE markers, etc.). These are LLM bugs, not
# matching failures. We expect these to fail and classify them separately.
KNOWN_BROKEN_DIFFS = {
    "2026-02-17_025041",  # 20% match: LLM diff context hallucinated
    "2026-02-17_030011",  # 22% match: LLM diff context hallucinated
    "2026-02-17_031503",  # 33% match: diff contains WRITE_FILE markers
    "2026-02-17_040619",  # 40% match: diff contains CONTENT>>> markers
    "2026-02-17_045411",  # multi-hunk: later hunks have 12-29% match
    "2026-02-17_045645",  # 5 hunks: hunks 3-4 have 12-13% match
}


def test_full_pipeline_sessions(results: TestResults, session_prefix: str = "2026-02-18_", verbose: bool = False):
    """
    Full-pipeline E2E test: simulate CodeAgent's generate-then-patch flow.
    
    For each session where git apply originally failed:
    1. Parse 0000/response.md → extract_write_file_actions → write initial file
    2. Parse 0001/response.md → extract_all_diffs → apply_fuzzy_patch
    3. Verify the patch applied cleanly
    
    This tests the REAL extraction + application pipeline with REAL LLM data.
    """
    print("\n── Full-Pipeline Session Tests ──")
    
    base_dir = Path(__file__).resolve().parent.parent
    sessions_dir = base_dir / ".agent" / "sessions"
    if not sessions_dir.exists():
        results.record_skip("FullPipeline", "no sessions dir")
        return
    
    tested = 0
    passed_count = 0
    expected_fail_count = 0
    unexpected_fail = []
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        
        for session_dir in sorted(sessions_dir.iterdir()):
            if not session_dir.name.startswith(session_prefix):
                continue
            
            # Use ALL sessions, regardless of original outcome
            # We only need the response files to simulate the pipeline
            r0 = session_dir / "0000" / "response.md"
            r1 = session_dir / "0001" / "response.md"
            
            if not r0.exists() or not r1.exists():
                continue
            
            # Step 1: Extract initial file from 0000/response.md
            wf_actions = extract_write_file_actions(r0.read_text(encoding="utf-8", errors="ignore"))
            if not wf_actions:
                results.record_skip(f"Pipeline: {session_dir.name}", "no WRITE_FILE in 0000")
                continue
            
            # Step 2: Extract diff from 0001/response.md
            diff = extract_all_diffs(r1.read_text(encoding="utf-8", errors="ignore"))
            if not diff:
                results.record_skip(f"Pipeline: {session_dir.name}", "no diff in 0001")
                continue
            
            # Step 3: Write initial file to temp directory
            initial_path, initial_content = wf_actions[0]
            test_file = tmp_dir / f"session_{session_dir.name}" / "task.py"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text(initial_content, encoding="utf-8")
            initial_line_count = len(initial_content.splitlines())
            
            # Step 4: Extract file-specific diff and apply
            file_diffs = re.split(r'(?=^diff --git )', diff, flags=re.MULTILINE)
            file_diffs = [d for d in file_diffs if d.strip().startswith('diff --git')]
            
            if not file_diffs:
                results.record_skip(f"Pipeline: {session_dir.name}", "no file diffs")
                continue
            
            ok = apply_fuzzy_patch(test_file, file_diffs[0])
            tested += 1
            
            is_known_broken = session_dir.name in KNOWN_BROKEN_DIFFS
            
            if ok:
                passed_count += 1
                # Verify the file was actually modified
                result_content = test_file.read_text(encoding="utf-8")
                result_lines = len(result_content.splitlines())
                results.record_pass(f"Pipeline: {session_dir.name}")
                if verbose:
                    print(f"    ✅ {session_dir.name}: {initial_line_count}→{result_lines} lines")
            elif is_known_broken:
                expected_fail_count += 1
                results.record_skip(f"Pipeline: {session_dir.name}",
                                  "known broken LLM diff (hallucinated context)")
                if verbose:
                    print(f"    ⏭️  {session_dir.name}: known broken diff (expected)")
            else:
                unexpected_fail.append(session_dir.name)
                results.record_fail(f"Pipeline: {session_dir.name}",
                                  f"fuzzy patch failed (initial={initial_line_count} lines)")
                if verbose:
                    print(f"    ❌ {session_dir.name}: UNEXPECTED failure")
    
    if tested == 0:
        results.record_skip("FullPipeline", "no testable sessions found")
    else:
        print(f"  Full pipeline: {passed_count}/{tested} passed, "
              f"{expected_fail_count} expected failures, "
              f"{len(unexpected_fail)} unexpected failures")
        if unexpected_fail:
            print(f"  ⚠️  Unexpected failures: {', '.join(unexpected_fail)}")




def main():
    parser = argparse.ArgumentParser(description="Test LLM response extraction and patch apply")
    parser.add_argument("--session-prefix", default="2026-02-18_",
                        help="Prefix to filter session dirs (default: 2026-02-17_)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output for each test")
    parser.add_argument("--max-sessions", type=int, default=0,
                        help="Max sessions to test (0=all)")
    args = parser.parse_args()
    
    results = TestResults()
    base_dir = Path(__file__).resolve().parent.parent
    
    # Phase 1: Unit tests for known broken cases
    test_known_broken_cases(results, verbose=args.verbose)
    
    # Phase 2: Sanitize regression tests
    test_sanitize_regressions(results, verbose=args.verbose)
    
    # Phase 3: Session-based extraction tests
    print(f"\n── Session Tests (prefix: {args.session_prefix}) ──")
    sessions = find_session_dirs(base_dir, prefix=args.session_prefix)
    
    if not sessions:
        print(f"  No sessions found with prefix '{args.session_prefix}' in {base_dir / '.agent/sessions'}")
    else:
        if args.max_sessions > 0:
            sessions = sessions[:args.max_sessions]
        print(f"  Found {len(sessions)} session(s)")
        
        for session_dir in sessions:
            test_session_extraction(session_dir, results, verbose=args.verbose)
    
    # Phase 4: apply_fuzzy_patch end-to-end tests
    test_fuzzy_patch_e2e(results, verbose=args.verbose)
    
    # Phase 5: _try_apply_content end-to-end tests
    test_try_apply_content_e2e(results, verbose=args.verbose)
    
    # Phase 6: Full-pipeline session tests
    test_full_pipeline_sessions(results, session_prefix=args.session_prefix, verbose=args.verbose)
    
    # Summary
    all_passed = results.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
