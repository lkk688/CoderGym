#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mini_claude_code.py (v2 – Robust)
A minimal, non-interactive "Claude Code"-like coding agent with:
- Multi-diff extraction & sanitization
- Write-file fallback when diffs fail
- Robust continuation stitching
- Fault-tolerant JSON planner
- SkillDB injection
- Prompt ledger & session logging

Requirements:
  pip install openai rich tiktoken

Env (overridden by CLI args):
  VLLM_BASE_URL (default https://w0wqtv67-8000.usw3.devtunnels.ms/v1)
  VLLM_API_KEY  (default myhpcvllmqwen)
  VLLM_MODEL    (default Qwen/Qwen3-Coder-Next-FP8)
"""

import os
import re
import json
import time
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from openai import OpenAI
try:
    import tiktoken
except ImportError:
    tiktoken = None

console = Console()


# ---------------------------
# Config Defaults
# ---------------------------

AGENT_DIR = Path(".agent")
SESSIONS_DIR = AGENT_DIR / "sessions"
SKILL_DIR = AGENT_DIR / "skilldb"
SKILL_SUCCESS = SKILL_DIR / "successes.jsonl"
SKILL_FAIL = SKILL_DIR / "failures.jsonl"
RUNS_LOG = AGENT_DIR / "runs.jsonl"

DEFAULT_SYSTEM = """\
You are an advanced AI coding agent.  Your ONLY job is to produce file changes.

## Output Format (STRICT — follow exactly)

You MUST output in ONE of these two formats per response.  Never mix them.

### Format A: Unified Diff (PREFERRED)

1. Start with a brief `## Reasoning` section (plain text, keep short).
2. Then output `## Action` followed by a SINGLE fenced code block of type `diff`.
3. Inside that block, include ALL file changes as concatenated unified diffs.
4. Each file diff starts with `diff --git a/<path> b/<path>`.
5. For NEW files use `--- /dev/null` and `+++ b/<path>`.
6. Do NOT put any prose, markdown, or commentary between diffs inside the block.
7. Make sure hunk line-counts are correct (@@ -X,Y +A,B @@).

Example:
```diff
diff --git a/foo.py b/foo.py
new file mode 100644
--- /dev/null
+++ b/foo.py
@@ -0,0 +1,3 @@
+import os
+
+print("hello")
diff --git a/bar.py b/bar.py
--- a/bar.py
+++ b/bar.py
@@ -1,3 +1,3 @@
 import os
-print("old")
+print("new")
 # end
```

### Format B: WRITE_FILE (fallback for new files / very long content)

Use when generating many new files or when diff would be impractical.

```
WRITE_FILE: path/to/file.py
<<<CONTENT
#!/usr/bin/env python3
... file content here ...
CONTENT>>>

WRITE_FILE: path/to/other.py
<<<CONTENT
... file content here ...
CONTENT>>>
```

## Rules
- NEVER embed ``` fences inside a diff block.
- NEVER mix Format A and Format B in the same response.
- If output will be very long, prefer Format B (WRITE_FILE) so content isn't truncated.
- Always include `Verification: <command>` on its own line if you know how to verify.
"""

# Skill injection limits (keep short to save tokens)
SKILL_INJECT_TOPK = 6
SKILL_INJECT_MAX_LINES = 40  # total lines injected into prompt

@dataclass
class AgentConfig:
    client: OpenAI
    model: str
    session_dir: Path
    max_context: int
    max_output: int
    auto_approve: bool
    agent_dir: Path
    model_max_context: int = 0  # 0 = auto-detected from model, fallback to max_context

# ---------------------------
# Utilities
# ---------------------------

def now_stamp() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if tiktoken:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    # Rough fallback: 1 token ~= 4 chars
    return len(text) // 4

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    target_chars = int(max_tokens * 3.5)
    return text[:target_chars] + "\n...[TRUNCATED]..."


def query_model_context_length(client: OpenAI, model_name: str) -> int:
    """
    Query the vLLM /v1/models endpoint to discover the model's max context length.
    Returns 0 if the query fails (caller should use a fallback).
    """
    try:
        models = client.models.list()
        for m in models.data:
            if m.id == model_name:
                # vLLM exposes max_model_len in the model info
                ctx = getattr(m, 'max_model_len', 0)
                if ctx and ctx > 0:
                    console.print(f"[green]Auto-detected model context length: {ctx}[/green]")
                    return int(ctx)
        console.print(f"[yellow]Model '{model_name}' not found in /v1/models. Using fallback.[/yellow]")
    except Exception as e:
        console.print(f"[yellow]Could not query model context length: {e}. Using fallback.[/yellow]")
    return 0


def compute_safe_max_tokens(input_tokens: int, model_max_context: int, desired_max_output: int,
                            safety_margin: int = 200, min_output: int = 1024) -> int:
    """
    Compute the largest safe max_tokens value that won't exceed the model's context limit.
    
    Args:
        input_tokens: Estimated token count of the input (system + user messages)
        model_max_context: Model's maximum context window
        desired_max_output: The user's requested max output tokens
        safety_margin: Extra buffer for tokenizer estimation errors
        min_output: Minimum output tokens; below this, signal an error condition
    
    Returns:
        Clamped max_tokens value, or min_output if budget is very tight.
    """
    available = model_max_context - input_tokens - safety_margin
    if available < min_output:
        console.print(f"[red]Context budget very tight: {available} tokens available "
                      f"(input={input_tokens}, limit={model_max_context}). "
                      f"Clamping to min={min_output}.[/red]")
        return min_output
    safe = min(desired_max_output, available)
    return safe


def ensure_dirs(base_dir: Path):
    (base_dir / "sessions").mkdir(parents=True, exist_ok=True)
    (base_dir / "skilldb").mkdir(parents=True, exist_ok=True)
    for p in [base_dir / "skilldb/successes.jsonl", base_dir / "skilldb/failures.jsonl", base_dir / "runs.jsonl"]:
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("", encoding="utf-8")

def run_shell(cmd: str, cwd: Optional[str] = None, cap: int = 20000) -> Tuple[int, str]:
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True, cwd=cwd)
    out = (p.stdout or "") + (p.stderr or "")
    if len(out) > cap:
        out = out[-cap:]
    return p.returncode, out

def is_git_repo() -> bool:
    code, _ = run_shell("git rev-parse --is-inside-work-tree")
    return code == 0

def git_status() -> str:
    code, out = run_shell("git status -sb")
    return out if code == 0 else ""

def git_diff() -> str:
    code, out = run_shell("git diff")
    return out if code == 0 else ""

def read_file(path: str, max_chars: int = 16000) -> str:
    p = Path(path)
    if not p.exists():
        return f"[MISSING FILE] {path}"
    data = p.read_text(encoding="utf-8", errors="ignore")
    if len(data) > max_chars:
        return data[:max_chars] + "\n\n[TRUNCATED]\n"
    return data

def top_level_tree(max_items: int = 200) -> str:
    items = []
    try:
        for p in Path(".").iterdir():
            if p.name.startswith(".agent") or p.name.startswith(".git"):
                continue
            items.append(p.name + ("/" if p.is_dir() else ""))
    except Exception:
        pass
    items = sorted(items)[:max_items]
    return "\n".join(items)

def write_jsonl(path: Path, obj: Dict[str, Any]):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------------------
# Diff Extraction & Sanitization (IMPROVED)
# ---------------------------

def sanitize_diff_text(diff_text: str) -> str:
    """
    Clean up a diff block extracted from LLM output.
    Removes common LLM artifacts that corrupt patches:
    - Stray ``` fence markers embedded in diff body
    - HTML tags (<details>, etc.)
    - Trailing prose after last hunk
    """
    lines = diff_text.split("\n")
    cleaned = []
    for line in lines:
        # Skip stray fence markers
        if re.match(r'^```', line.strip()):
            continue
        # Skip HTML tags
        if re.match(r'^</?(?:details|summary|br|hr)', line.strip(), re.IGNORECASE):
            continue
        cleaned.append(line)
    
    result = "\n".join(cleaned)
    # Ensure trailing newline
    if not result.endswith("\n"):
        result += "\n"
    return result


def extract_all_diffs(text: str) -> Optional[str]:
    """
    Extract unified diffs from model output.
    
    IMPORTANT: When the model outputs multiple diff blocks (reasoning drafts + final),
    we use only the LAST one, which is typically the final/correct version.
    
    Handles:
      - Multiple diffs inside a single fenced ```diff block
      - Multiple separate fenced blocks each containing a diff
      - Raw diffs starting with 'diff --git' (unfenced)
    Returns the last (most likely correct) diff text or None.
    """
    t = text.strip()
    
    # Strategy 1: Look for fenced ```diff blocks and extract content
    fenced_diffs = []
    # Match all ```diff ... ``` blocks (or just ``` blocks containing diffs)
    fence_pattern = re.compile(r'```(?:diff)?\s*\n(.*?)```', re.DOTALL)
    for m in fence_pattern.finditer(t):
        block = m.group(1).strip()
        if 'diff --git' in block:
            fenced_diffs.append(block)
    
    if fenced_diffs:
        # Use the LAST diff block — model often puts reasoning diffs first,
        # then the final correct diff last ("Final Answer", "## Action", etc.)
        last_diff = fenced_diffs[-1]
        return sanitize_diff_text(last_diff)
    
    # Strategy 2: Find all raw 'diff --git' blocks (unfenced)
    # Split text at each 'diff --git' boundary and collect
    parts = re.split(r'(?=^diff --git )', t, flags=re.MULTILINE)
    raw_diffs = []
    for part in parts:
        part = part.strip()
        if part.startswith('diff --git'):
            raw_diffs.append(part)
    
    if raw_diffs:
        # For raw diffs, use the last complete diff block
        return sanitize_diff_text(raw_diffs[-1])
    
    return None


def extract_write_file_actions(text: str) -> List[Tuple[str, str]]:
    """
    Extract WRITE_FILE actions from model output.
    Supports multiple formats the model might use:
    
    Format 1 (preferred): WRITE_FILE: path
                          <<<CONTENT
                          ... content ...
                          CONTENT>>>
    
    Format 2 (tool_call XML): <tool_call><function=write_file>
                              <parameter=file_path>path</parameter>
                              <parameter=content>...</parameter>
                              </function></tool_call>
    
    Format 3 (implicit): ## File: path/to/file.py
                         ```python
                         ... content ...
                         ```
    
    Returns list of (filepath, content) tuples.
    """
    results = []
    
    # Format 1: WRITE_FILE: path\n<<<CONTENT\n...\nCONTENT>>>
    wf_pattern = re.compile(
        r'WRITE_FILE:\s*(.+?)\s*\n<<<CONTENT\n(.*?)CONTENT>>>',
        re.DOTALL
    )
    for m in wf_pattern.finditer(text):
        filepath = m.group(1).strip()
        content = m.group(2)
        if filepath and content:
            results.append((filepath, content))
    
    if results:
        return results
    
    # Format 2: <tool_call> XML style
    tc_pattern = re.compile(
        r'<(?:tool_call|function)[^>]*>.*?'
        r'<parameter[= ]+(?:file_path|path)[^>]*>\s*(.+?)\s*</parameter>'
        r'.*?<parameter[= ]+content[^>]*>\s*(.*?)\s*</parameter>',
        re.DOTALL
    )
    for m in tc_pattern.finditer(text):
        filepath = m.group(1).strip()
        content = m.group(2)
        if filepath and content:
            results.append((filepath, content))
    
    if results:
        return results
    
    # Format 3: Implicit file path + code block
    # Look for patterns like:
    #   # File: path/to/file.py  (or ## File: or ### or just path followed by ```)
    implicit_pattern = re.compile(
        r'(?:^#+\s*(?:File:\s*)?|^(?:Create|Writing|Output)\s+(?:file\s+)?)'
        r'[`]*([^\n`]+?\.\w{1,5})[`]*\s*\n'
        r'```\w*\n(.*?)```',
        re.MULTILINE | re.DOTALL
    )
    for m in implicit_pattern.finditer(text):
        filepath = m.group(1).strip().strip('`').strip()
        content = m.group(2)
        # Validate it looks like a real path
        if filepath and '/' in filepath and content and len(content) > 10:
            results.append((filepath, content))
    
    return results


# ---------------------------
# SkillDB
# ---------------------------

@dataclass
class Skill:
    tag: str
    kind: str  # "success" or "failure"
    text: str  # short guidance
    pattern: str  # keyword/pattern
    evidence: str  # session/turn summary
    created_at: str

def load_skills(skill_dir: Path) -> List[Skill]:
    skills: List[Skill] = []
    for kind, filename in [("success", "successes.jsonl"), ("failure", "failures.jsonl")]:
        path = skill_dir / filename
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    skills.append(Skill(
                        tag=obj.get("tag", ""),
                        kind=kind,
                        text=obj.get("text", ""),
                        pattern=obj.get("pattern", ""),
                        evidence=obj.get("evidence", ""),
                        created_at=obj.get("created_at", ""),
                    ))
                except Exception:
                    continue
        except Exception:
            pass
    return skills

def score_skill(skill: Skill, query: str) -> int:
    """
    Simple lexical scoring:
    - +2 if pattern token appears in query
    - +1 for each word from skill.text appearing in query (cap)
    """
    q = query.lower()
    s = 0
    patt = (skill.pattern or "").lower().strip()
    if patt and patt in q:
        s += 2
    words = re.findall(r"[a-zA-Z0-9_]{3,}", (skill.text or "").lower())
    hits = 0
    for w in set(words):
        if w in q:
            hits += 1
    s += min(hits, 3)
    return s

def select_relevant_skills(goal_and_notes: str, skill_dir: Path, topk: int = SKILL_INJECT_TOPK) -> List[Skill]:
    skills = load_skills(skill_dir)
    scored = [(score_skill(sk, goal_and_notes), sk) for sk in skills]
    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [sk for sc, sk in scored if sc > 0][:topk]
    return picked

def format_skill_injection(skills: List[Skill]) -> str:
    if not skills:
        return ""
    lines = ["## History-based guardrails (SkillDB)"]
    for sk in skills:
        prefix = "✅" if sk.kind == "success" else "⛔"
        text = sk.text.strip().replace("\n", " ")
        evidence = sk.evidence.strip().replace("\n", " ")
        lines.append(f"- {prefix} [{sk.tag}] {text} (evidence: {evidence})")
    if len(lines) > SKILL_INJECT_MAX_LINES:
        lines = lines[:SKILL_INJECT_MAX_LINES] + ["- (truncated)"]
    return "\n".join(lines).strip() + "\n"


# ---------------------------
# Prompt Ledger
# ---------------------------

def build_prompt_md(
    goal: str,
    allowlist: List[str],
    context_files: List[str],
    extra_notes: str,
    inject_skills: str,
    max_context: int,
    max_output: int
) -> str:
    allow_txt = "\n".join(f"- {p}" for p in allowlist) if allowlist else "- (none)"

    # Detect if ALL files are new (don't exist yet) — skip repo context to save tokens
    all_new_files = all(not Path(f).exists() for f in allowlist) if allowlist else False

    # Estimate if this is a multi-file task or new-file task — suggest WRITE_FILE format
    n_files = len(allowlist)
    format_hint = ""
    if n_files > 1 or all_new_files:
        format_hint = (
            "\n> **IMPORTANT**: Use **Format B (WRITE_FILE)** to create all files in a single response. "
            "This avoids diff truncation issues and is more reliable for new files.\n"
        )

    # Base prompt structure
    base_md = f"""# Turn Prompt

## Goal
{goal}

## ALLOWLIST (files you must create/modify)
{allow_txt}
{format_hint}
{inject_skills if inject_skills else ""}

## Notes / Constraints
{extra_notes.strip() if extra_notes.strip() else "(none)"}

## Output Contract
1. Return changes using EITHER unified diff (Format A) OR WRITE_FILE (Format B) as described in the system prompt.
2. ALL files in the ALLOWLIST must be addressed.
3. (Optional) Include a line: "Verification: <command>" before the changes.
"""
    
    # Calculate remaining budget for context
    safety_margin = 1000
    usable_context = max_context - max_output - safety_margin
    used_tokens = estimate_tokens(base_md) + estimate_tokens(DEFAULT_SYSTEM)
    remaining = usable_context - used_tokens
    
    if remaining < 1000:
        console.print("[red]Warning: Context budget extremely low![/red]")
    
    # When budget is tight, add a hint
    if remaining < 2000:
        base_md += "\n> **OUTPUT HINT**: Context budget is tight. Use WRITE_FILE format and keep code concise.\n"
        
    context_md = ""
    
    # Priority 1: Git Status/Diff — SKIP for new file creation (saves ~10K tokens)
    if not all_new_files and is_git_repo():
        st = git_status()
        df = git_diff()
        git_section = ""
        if st:
            # Filter git status to only show relevant files (in allowlist)
            relevant_lines = []
            for line in st.split('\n'):
                line_stripped = line.strip()
                # Keep the branch line and lines mentioning allowlisted files
                if line_stripped.startswith('##') or any(str(a) in line for a in allowlist):
                    relevant_lines.append(line)
            if relevant_lines:
                git_section += f"### git status (relevant)\n```\n{'\n'.join(relevant_lines)}\n```\n"
        if df.strip():
            df_toks = estimate_tokens(df)
            if df_toks > 2000:
                df = truncate_to_tokens(df, 2000)
            git_section += f"### git diff\n```diff\n{df}\n```\n"
            
        if git_section and estimate_tokens(git_section) < remaining:
            context_md += "## Repo snapshot\n" + git_section
            remaining -= estimate_tokens(git_section)
    elif all_new_files:
        console.print("[dim]Skipping git status/diff (all files are new).[/dim]")
    
    # Priority 2: File Content (only existing files that are being modified)
    files_md = ""
    for f in context_files:
        content = read_file(f)
        if not content or content.startswith("[MISSING FILE]"):
            continue  # Skip missing files to save tokens
        if estimate_tokens(content) > 6000: 
             content = truncate_to_tokens(content, 6000)
             
        f_block = f"## File: {f}\n```python\n{content}\n```\n"
        if estimate_tokens(f_block) < remaining:
            files_md += f_block
            remaining -= estimate_tokens(f_block)
        else:
            files_md += f"## File: {f}\n[Content Omitted - Context Limit]\n"
            
    context_md += files_md
    
    # Priority 3: Tree — SKIP for new file tasks (not useful)
    if not all_new_files:
        tree = top_level_tree()
        if estimate_tokens(tree) < remaining:
            context_md += "### Top-level tree\n" + tree + "\n"

    if context_md.strip():
        return base_md + "\n## Context\n" + context_md
    return base_md


def build_bugfix_prompt(file_path: str, error_output: str, original_goal: str = "") -> str:
    """
    Build a focused prompt for fixing a specific error in generated code.
    Much shorter than build_prompt_md — just the file content + error.
    Forces WRITE_FILE format only (no diffs) for reliable application.
    """
    content = read_file(file_path)
    if not content:
        content = "[FILE NOT FOUND]"
    
    return f"""# Bug Fix Required

## Original Goal
{original_goal if original_goal else "(see previous context)"}

## Current File: {file_path}
```python
{content}
```

## Error Output
```
{error_output[-3000:]}
```

## STRICT Instructions
1. Identify and fix ONLY the specific error shown above.
2. Output the COMPLETE corrected file using WRITE_FILE format.
3. Do NOT use diffs. Do NOT include reasoning diff examples.
4. Do NOT change the overall structure — only fix the error.
5. Output EXACTLY one WRITE_FILE block, nothing else after it.

WRITE_FILE: {file_path}
<<<CONTENT
... your complete corrected file here ...
CONTENT>>>
"""


# ---------------------------
# Core Loop
# ---------------------------

def apply_patch_guarded(diff_text: str, turn_dir: Path, auto_approve: bool = False) -> bool:
    """
    Apply patches robustly with multiple fallback strategies.
    1. Sanitize the diff text.
    2. Create parent directories for new files.
    3. Try git apply --check with --recount.
    4. If combined patch fails, try each diff block separately.
    5. Apply on success.
    """
    patch_path = turn_dir / "patch.diff"
    
    # Pre-sanitize
    diff_text = sanitize_diff_text(diff_text)
    patch_path.write_text(diff_text, encoding="utf-8")

    # Pre-create directories for new files mentioned in the diff
    for m in re.finditer(r'^\+\+\+ b/(.+)$', diff_text, re.MULTILINE):
        fpath = Path(m.group(1))
        fpath.parent.mkdir(parents=True, exist_ok=True)

    apply_log_parts = []

    def try_apply(patch_file: Path, label: str) -> bool:
        """Try applying a patch file with multiple strategies. Returns True on success."""
        strategies = [
            f"git apply --check --recount {patch_file.as_posix()}",
            f"git apply --check {patch_file.as_posix()}",
        ]
        for cmd_check in strategies:
            check_code, check_out = run_shell(cmd_check)
            apply_log_parts.append(f"[{cmd_check}] exit={check_code}\n{check_out}\n")
            
            if check_code == 0:
                # Check passed — apply
                cmd_apply = cmd_check.replace("--check ", "")
                app_code, app_out = run_shell(cmd_apply)
                apply_log_parts.append(f"[{cmd_apply}] exit={app_code}\n{app_out}\n")
                
                if app_code == 0:
                    console.print(f"[green]Patch applied ({label}).[/green]")
                    return True
                else:
                    console.print(f"[yellow]Apply failed after check passed ({label}): {app_out[:200]}[/yellow]")
        return False

    # Strategy 1: Try the full combined patch
    success = try_apply(patch_path, "combined")
    
    if not success:
        # Strategy 2: Split into individual file diffs and try each
        individual_diffs = re.split(r'(?=^diff --git )', diff_text, flags=re.MULTILINE)
        individual_diffs = [d for d in individual_diffs if d.strip().startswith('diff --git')]
        
        if len(individual_diffs) > 1:
            console.print(f"[yellow]Combined patch failed. Trying {len(individual_diffs)} individual patches...[/yellow]")
            all_ok = True
            for idx, single_diff in enumerate(individual_diffs):
                single_path = turn_dir / f"patch_part{idx}.diff"
                single_path.write_text(sanitize_diff_text(single_diff), encoding="utf-8")
                if not try_apply(single_path, f"part {idx+1}/{len(individual_diffs)}"):
                    all_ok = False
                    # Extract filename for error message
                    fname_m = re.search(r'diff --git a/(\S+)', single_diff)
                    fname = fname_m.group(1) if fname_m else f"part {idx+1}"
                    console.print(f"[red]Individual patch for {fname} also failed.[/red]")
            success = all_ok
    
    # Write full log
    (turn_dir / "apply.log").write_text("\n".join(apply_log_parts), encoding="utf-8")
    
    if not success:
        console.print(Panel(
            apply_log_parts[-1][:500] if apply_log_parts else "(no output)",
            title="Patch check failed", style="red"
        ))
    
    return success


def extract_files_from_diff(diff_text: str) -> List[Tuple[str, str]]:
    """
    Extract file contents from diff '+' lines.
    When git apply fails, we can still reconstruct new files from
    the diff by collecting all '+' lines (ignoring the leading '+').
    
    For NEW files (--- /dev/null): extracts the full file content.
    For EDIT diffs: extracts the intended final content by combining
    context lines and '+' lines. This may fail for partial diffs,
    so we check for indentation consistency.
    
    Returns list of (filepath, content) tuples.
    """
    results = []
    
    # Split into individual file diffs
    file_diffs = re.split(r'(?=^diff --git )', diff_text, flags=re.MULTILINE)
    file_diffs = [d for d in file_diffs if d.strip().startswith('diff --git')]
    
    for single_diff in file_diffs:
        # Extract target filename from diff header
        fname_match = re.search(r'diff --git a/\S+ b/(\S+)', single_diff)
        if not fname_match:
            continue
        filepath = fname_match.group(1)
        
        # Determine if this is a new file
        is_new_file = ('new file mode' in single_diff or 
                       '--- /dev/null' in single_diff)
        
        # Collect all '+' lines (content lines), skipping the diff headers
        lines = single_diff.split('\n')
        content_lines = []
        has_minus_lines = False
        in_hunk = False
        
        for line in lines:
            # Skip diff metadata lines
            if line.startswith('diff --git') or line.startswith('---') or line.startswith('+++'):
                continue
            if line.startswith('@@'):
                in_hunk = True
                continue
            if line.startswith('\\ No newline'):
                continue
            
            if in_hunk:
                if line.startswith('+'):
                    content_lines.append(line[1:])  # Remove leading '+'
                elif line.startswith('-'):
                    has_minus_lines = True
                    pass  # Skip removed lines
                elif line.startswith(' '):
                    content_lines.append(line[1:])  # Context line, remove leading space
                elif line == '':
                    content_lines.append('')  # Empty line
        
        if not content_lines:
            continue
            
        # For edit diffs (has minus lines, not new file), validate the content
        # Edit diffs may produce fragments with wrong base indentation
        if has_minus_lines and not is_new_file:
            # Check if the first non-empty line has suspicious indentation
            first_code = next((l for l in content_lines if l.strip()), '')
            if first_code and first_code != first_code.lstrip():
                # First line is indented — this is likely a fragment, not full file
                # Only accept if the content looks like a complete Python file
                full_text = '\n'.join(content_lines)
                if not (full_text.lstrip().startswith(('import ', 'from ', '#', '"""', "'''", '#!/'))): 
                    console.print(f"[yellow]Skipping edit-diff fragment for {filepath} "
                                  f"(starts with indented code, likely incomplete)[/yellow]")
                    continue
        
        # Join with newlines, ensure trailing newline
        content = '\n'.join(content_lines)
        if not content.endswith('\n'):
            content += '\n'
        results.append((filepath, content))
        console.print(f"[cyan]Extracted {filepath} from diff ({len(content)} bytes)"
                      f"{' [new file]' if is_new_file else ' [edit]'}[/cyan]")
    
    return results


def apply_write_files(
    actions: List[Tuple[str, str]], 
    allowlist: List[str], 
    turn_dir: Path
) -> bool:
    """
    Write files directly from WRITE_FILE actions.
    Validates paths against the allowlist.
    Returns True if at least one file was written.
    """
    written = 0
    log_parts = []
    
    # Normalize allowlist for comparison — convert PosixPath to str, extract basenames too
    norm_allowlist = set()
    for p in allowlist:
        s = str(p)
        norm_allowlist.add(s)                           # full path as string
        norm_allowlist.add(str(Path(s)))                 # normalized
        norm_allowlist.add(os.path.basename(s))          # just filename
        # Also add without leading dirs that might differ
        # e.g. "output/foo.py" from "./output/foo.py" or "/abs/output/foo.py"
        parts = Path(s).parts
        for i in range(len(parts)):
            norm_allowlist.add(str(Path(*parts[i:])))
    
    for filepath, content in actions:
        # Normalize the filepath
        clean_path = filepath.strip().lstrip('/')
        
        # Check if file is in allowlist (flexible matching)
        allowed = False
        for ap in norm_allowlist:
            ap_str = str(ap)
            if (clean_path == ap_str or 
                clean_path.endswith(ap_str) or 
                ap_str.endswith(clean_path) or
                os.path.basename(clean_path) == ap_str):
                allowed = True
                break
        # Also allow if no strict allowlist or auto mode
        if not norm_allowlist or not allowlist:
            allowed = True
            
        if not allowed:
            log_parts.append(f"SKIPPED (not in allowlist): {filepath} (allowlist: {[str(a) for a in allowlist]})")
            console.print(f"[yellow]Skipping {filepath} — not in allowlist ({[str(a) for a in allowlist]})[/yellow]")
            continue
        
        try:
            target = Path(clean_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            log_parts.append(f"WROTE: {filepath} ({len(content)} bytes)")
            console.print(f"[green]Wrote file: {filepath}[/green]")
            written += 1
            
            # Also git add if in a repo
            if is_git_repo():
                run_shell(f"git add {target.as_posix()}")
        except Exception as e:
            log_parts.append(f"FAILED: {filepath} — {e}")
            console.print(f"[red]Failed to write {filepath}: {e}[/red]")
    
    (turn_dir / "write_files.log").write_text("\n".join(log_parts), encoding="utf-8")
    return written > 0


# ---------------------------
# LLM Interaction
# ---------------------------

def complete_with_continuation(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
    model_max_context: int = 16384,
) -> str:
    """
    Calls the LLM. If finish_reason is 'length', appends the partial response
    to messages and asks it to continue, stitching the results.
    Improved: diff-aware continuation prompting.
    
    Adaptively caps max_tokens based on input size to avoid context overflow.
    """
    full_content = ""
    current_messages = list(messages)
    
    max_loops = 5  # Increased from 3 for complex multi-file tasks
    
    for i in range(max_loops):
        console.print(f"[dim]Generation loop {i+1}/{max_loops}...[/dim]")
        
        # Adaptive max_tokens: estimate input and cap output accordingly
        input_text = "\n".join(m.get("content", "") for m in current_messages)
        input_est = estimate_tokens(input_text)
        safe_tokens = compute_safe_max_tokens(
            input_tokens=input_est,
            model_max_context=model_max_context,
            desired_max_output=max_output_tokens
        )
        
        if safe_tokens < max_output_tokens:
            console.print(f"[yellow]Adaptive max_tokens: {safe_tokens} "
                          f"(input≈{input_est}, limit={model_max_context}, "
                          f"requested={max_output_tokens})[/yellow]")
        
        # Retry with backoff on API errors (including context overflow)
        resp = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=current_messages,
                    temperature=temperature,
                    max_tokens=safe_tokens
                )
                break
            except Exception as e:
                err_str = str(e)
                if 'max_tokens' in err_str or 'context length' in err_str or 'maximum context' in err_str:
                    # Context overflow — reduce tokens further
                    safe_tokens = max(1024, safe_tokens // 2)
                    console.print(f"[red]Context overflow (attempt {attempt+1}). "
                                  f"Retrying with max_tokens={safe_tokens}...[/red]")
                    time.sleep(1)
                    continue
                console.print(f"[red]LLM Call failed: {e}[/red]")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                break
        
        if resp is None:
            console.print(f"[red]All LLM retry attempts failed.[/red]")
            break
            
        choice = resp.choices[0]
        console.print(f"[dim]Finish Reason: {choice.finish_reason}[/dim]")
        content = choice.message.content or ""
        full_content += content
        
        if choice.finish_reason == "length":
            console.print("[yellow]Output truncated (limit reached). Continuing...[/yellow]")
            
            # Detect what kind of output we're in the middle of
            # and craft appropriate continuation prompt
            if content.rstrip().endswith('```'):
                # Cleanly ended a code block — can continue normally
                cont_prompt = (
                    "Continue. If there are more files to create, "
                    "continue with the next WRITE_FILE block or diff. "
                    "Do not repeat already-generated content."
                )
            elif 'WRITE_FILE:' in content or '<<<CONTENT' in content:
                cont_prompt = (
                    "You were truncated mid-output. Continue EXACTLY where you left off. "
                    "You were in the middle of a WRITE_FILE block. "
                    "Continue the file content, then close with CONTENT>>> "
                    "and continue with remaining files."
                )
            elif 'diff --git' in content:
                cont_prompt = (
                    "You were truncated mid-output. Continue EXACTLY where you left off. "
                    "You were in the middle of a unified diff. "
                    "Continue the diff hunks. Do NOT repeat diff headers already generated. "
                    "Do NOT restart the response."
                )
            else:
                cont_prompt = (
                    "You were truncated. Continue exactly where you left off. "
                    "Do not repeat previous content."
                )
            
            current_messages.append({"role": "assistant", "content": content})
            current_messages.append({"role": "user", "content": cont_prompt})
        else:
            break
            
    return full_content


# ---------------------------
# Task Planning
# ---------------------------

def extract_json_robust(text: str) -> Optional[dict]:
    """
    Robustly extract JSON from LLM output.
    Tries multiple strategies including truncation repair.
    """
    text = text.strip()
    
    # Strip <think>...</think> tags (Qwen thinking mode)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # Strategy 2: Extract from ```json block
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    
    # Strategy 3: Find first {...} in text using brace-matching
    start = text.find('{')
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except Exception:
                        break
    
    # Strategy 4: Truncation repair — model hit max_tokens and JSON was cut off
    # Common pattern: {"complex": true, "steps": ["step1", "step2"  (missing ]})
    if start is not None and start >= 0:
        candidate = text[start:]
        # Try appending common missing closers
        for suffix in [']}', ']', '}', '"]}', '"]}']:
            try:
                return json.loads(candidate + suffix)
            except Exception:
                pass
        # Try finding last complete string and closing from there
        # Find the last complete quoted string
        last_quote = candidate.rfind('"')
        if last_quote > 0:
            # Try closing after last complete string
            trimmed = candidate[:last_quote+1]
            for suffix in [']}', ']}', ']}\n']:
                try:
                    return json.loads(trimmed + suffix)
                except Exception:
                    pass
    
    # Strategy 5: Try to fix common JSON issues (unquoted keys)
    m = re.search(r'\{[^{}]+\}', text)
    if m:
        candidate = m.group(0)
        fixed = re.sub(r'(\w+)\s*:', r'"\1":', candidate)
        try:
            return json.loads(fixed)
        except Exception:
            pass
    
    return None


def plan_tasks(config: AgentConfig, goal: str, notes: str) -> List[str]:
    """
    Ask the model to analyze complexity. 
    Returns a list of sub-tasks (strings). 
    If simple, returns a list with just the original goal.
    """
    system_prompt = """You are a task planner. Analyze the request.
- If simple (1 file, 1 change): output {"complex": false}
- If complex: output {"complex": true, "steps": ["step1", "step2", ...]}

Rules:
- Output ONLY valid JSON, no markdown, no explanations.
- Keep each step description under 10 words.
- Maximum 5 steps.
"""
    
    user_prompt = f"Goal: {goal}\nNotes: {notes}\n\nJSON:"
    
    console.print("[cyan]Analyzing task complexity...[/cyan]")
    try:
        # Adaptive max_tokens for planner
        planner_input = system_prompt + user_prompt
        planner_input_est = estimate_tokens(planner_input)
        ctx_limit = config.model_max_context or config.max_context
        planner_max_tokens = compute_safe_max_tokens(
            input_tokens=planner_input_est,
            model_max_context=ctx_limit,
            desired_max_output=2048,
            min_output=512
        )
        
        resp = config.client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=planner_max_tokens
        )
        content = resp.choices[0].message.content or "{}"
        
        # Debug: log raw planning response
        planning_log = config.session_dir / "planning_response.md"
        planning_log.write_text(f"# Raw Planning Response\n\n```\n{content}\n```\n", encoding="utf-8")
        console.print(f"[dim]Planning response logged to {planning_log}[/dim]")
        
        data = extract_json_robust(content)
        if data is None:
            console.print(f"[yellow]Could not parse planning JSON. Raw response:[/yellow]")
            console.print(f"[dim]{content[:300]}[/dim]")
            return [goal]
        
        if data.get("complex"):
            steps = data.get("steps", [])
            if steps:
                console.print(Panel(
                    "\n".join([f"{i+1}. {s}" for i,s in enumerate(steps)]), 
                    title="Complex Task Plan", style="magenta"
                ))
                if config.auto_approve:
                    return steps
                if Confirm.ask("Proceed with this plan?"):
                    return steps
                else:
                    return [goal]
            else:
                console.print("[yellow]Complex plan had no steps. Proceeding as single task.[/yellow]")
                return [goal]
        else:
            console.print("[green]Task assessed as Simple.[/green]")
            return [goal]
            
    except Exception as e:
        console.print(f"[red]Planning failed ({e}). Proceeding as single task.[/red]")
        return [goal]


# ---------------------------
# Sub-task Execution
# ---------------------------

def _try_apply_content(content: str, allowlist: List[str], turn_dir: Path, 
                       config: AgentConfig) -> bool:
    """
    Try all methods to apply model output as file changes.
    Order: git apply diff → WRITE_FILE → diff extraction.
    
    Key insight: when model outputs BOTH diffs and WRITE_FILE,
    WRITE_FILE is more reliable (full file, no patch issues).
    So we try WRITE_FILE before diff extraction.
    """
    # --- TRY FORMAT A: Unified Diff ---
    diff = extract_all_diffs(content)
    changes_applied = False
    
    if diff:
        (turn_dir / "patch.diff").write_text(diff, encoding="utf-8")
        if is_git_repo():
            changes_applied = apply_patch_guarded(diff, turn_dir, auto_approve=config.auto_approve)
        else:
            console.print("[red]Not a git repo, skipping diff apply.[/red]")
    
    # --- TRY FORMAT B: WRITE_FILE (try BEFORE diff extraction — more reliable) ---
    if not changes_applied:
        write_actions = extract_write_file_actions(content)
        if write_actions:
            console.print(f"[cyan]Found {len(write_actions)} WRITE_FILE action(s). Applying...[/cyan]")
            changes_applied = apply_write_files(write_actions, allowlist, turn_dir)
            if changes_applied:
                console.print("[green]Applied via WRITE_FILE.[/green]")
    
    # --- TRY FORMAT A.5: Extract files from diff (last resort) ---
    if not changes_applied and diff:
        console.print("[yellow]Diff + WRITE_FILE failed. Extracting from diff lines...[/yellow]")
        diff_files = extract_files_from_diff(diff)
        if diff_files:
            changes_applied = apply_write_files(diff_files, allowlist, turn_dir)
            if changes_applied:
                console.print("[green]Wrote files extracted from diff.[/green]")
    
    if not changes_applied and not diff:
        write_actions = extract_write_file_actions(content)
        if not write_actions:
            console.print("[red]No valid diff or WRITE_FILE actions found.[/red]")
    
    return changes_applied


def _determine_verify_cmd(allowlist: List[str], auto_verify_cmd: str, 
                          config: AgentConfig) -> str:
    """
    Determine the verification command to run.
    """
    cmd_to_run = ""
    if auto_verify_cmd:
        if config.auto_approve:
            cmd_to_run = auto_verify_cmd
        elif Confirm.ask(f"Run parsed verification: [bold]{auto_verify_cmd}[/bold]?"):
            cmd_to_run = auto_verify_cmd
    
    # Auto-detect: if no verification command, infer from allowlist
    if not cmd_to_run and config.auto_approve:
        py_files = [str(f) for f in allowlist if str(f).endswith('.py')]
        if py_files:
            cmd_to_run = f"python3 {py_files[0]}"
            console.print(f"[cyan]Auto-detected verification: {cmd_to_run}[/cyan]")
    
    if not cmd_to_run and not config.auto_approve:
        if Confirm.ask("Run verification command?"):
            cmd_to_run = Prompt.ask("Command", default="")
    
    return cmd_to_run


def run_subtask_loop(
    config: AgentConfig,
    subtask: str,
    subtask_idx: int,
    allowlist: List[str],
    context_files: List[str],
    global_notes: str,
) -> bool:
    """
    Execute a single sub-task with a Generate → Test → Fix flow.
    
    Phase 1 (Generate): One model call to produce code. Try all apply methods.
                        If apply fails, one retry with WRITE_FILE hint.
    Phase 2 (Test+Fix): Run verification immediately. If it fails, use a focused
                        bug-fix prompt (just code + error) for up to 2 fix iterations.
    """
    
    turn_base = subtask_idx * 10
    skill_dir = config.agent_dir / "skilldb"
    turn_counter = 0  # tracks turn directories
    
    console.rule(f"Executing Sub-task {subtask_idx+1}: {subtask}")
    
    # ========================================
    # PHASE 1: Generate code
    # ========================================
    console.print("[bold cyan]Phase 1: Generating code...[/bold cyan]")
    
    changes_applied = False
    for gen_attempt in range(2):  # Max 2 generation attempts (first try + WRITE_FILE retry)
        turn_id = turn_base + turn_counter
        turn_dir = config.session_dir / f"{turn_id:04d}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        turn_counter += 1
        
        # Build appropriate notes
        current_notes = global_notes
        if gen_attempt > 0:
            current_notes += (
                "\n\nIMPORTANT: Previous attempt's diff could not be applied. "
                "You MUST use WRITE_FILE format (Format B) this time. "
                "Output the COMPLETE file content for each file."
                f"\nFiles needed: {', '.join(str(p) for p in allowlist)}"
            )
        
        # Inject skills
        inject = format_skill_injection(select_relevant_skills(subtask + "\n" + current_notes, skill_dir))
        
        # Build prompt
        prompt_md = build_prompt_md(subtask, allowlist, context_files, current_notes, inject, config.max_context, config.max_output)
        (turn_dir / "prompt.md").write_text(prompt_md, encoding="utf-8")
        
        # Call Model
        console.print(f"[cyan]Generating (attempt {gen_attempt+1})...[/cyan]")
        content = complete_with_continuation(
            client=config.client,
            model=config.model,
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": prompt_md}
            ],
            temperature=0.2,
            max_output_tokens=config.max_output,
            model_max_context=config.model_max_context or config.max_context,
        )
        (turn_dir / "response.md").write_text(content, encoding="utf-8")
        
        # Display truncated output
        display_content = content[:500] + "..." if len(content) > 500 else content
        console.print(Panel(display_content, title="Model Output (Generate)"))
        
        # Try all apply methods
        changes_applied = _try_apply_content(content, allowlist, turn_dir, config)
        
        if changes_applied:
            console.print("[green]Phase 1: Code applied successfully.[/green]")
            break
        else:
            console.print(f"[yellow]Generation attempt {gen_attempt+1}: apply failed.[/yellow]")
            if gen_attempt == 0:
                console.print("[yellow]Retrying with WRITE_FILE format hint...[/yellow]")
    
    if not changes_applied:
        console.print("[red]Phase 1 failed: Could not apply code after 2 attempts.[/red]")
        return False
    
    # ========================================
    # PHASE 2: Test and Fix
    # ========================================
    console.print("[bold cyan]Phase 2: Testing and fixing...[/bold cyan]")
    
    # Extract verification command from the last model response
    auto_verify_cmd = None
    v_match = re.search(r"^Verification:\s*(.+)$", content, re.MULTILINE)
    if v_match:
        auto_verify_cmd = v_match.group(1).strip()
    
    verify_cmd = _determine_verify_cmd(allowlist, auto_verify_cmd, config)
    
    if not verify_cmd:
        if config.auto_approve:
            console.print("[yellow]No verification command. Assuming success.[/yellow]")
            return True
        elif Confirm.ask("No verification command. Mark as DONE?"):
            return True
        else:
            return False
    
    max_fix_attempts = 2
    for fix_attempt in range(max_fix_attempts + 1):  # 0 = initial test, 1-2 = fix attempts
        # Run verification
        turn_id = turn_base + turn_counter
        # For the initial test, reuse the last turn_dir; for fix attempts, create new ones
        if fix_attempt > 0:
            turn_dir = config.session_dir / f"{turn_id:04d}"
            turn_dir.mkdir(parents=True, exist_ok=True)
            turn_counter += 1
        
        console.print(f"[blue]Running verification: {verify_cmd}[/blue]")
        code, out = run_shell(verify_cmd, cap=20000)
        (turn_dir / "verify_stdout.txt").write_text(out, encoding='utf-8')
        
        if code == 0:
            console.print(f"[green]Verification PASSED! "
                          f"({'initial' if fix_attempt == 0 else f'after fix {fix_attempt}'})[/green]")
            return True
        
        console.print(f"[red]Verification FAILED (exit={code}).[/red]")
        
        # If we've used all fix attempts, bail out
        if fix_attempt >= max_fix_attempts:
            console.print("[red]Max fix attempts reached. Sub-task failed.[/red]")
            break
        
        # Interactive mode: ask before fixing
        if not config.auto_approve and not Confirm.ask("Verification failed. Attempt bug fix?"):
            break
        
        # --- Bug Fix Turn ---
        console.print(f"[yellow]Bug fix attempt {fix_attempt + 1}/{max_fix_attempts}...[/yellow]")
        
        fix_turn_id = turn_base + turn_counter
        fix_turn_dir = config.session_dir / f"{fix_turn_id:04d}"
        fix_turn_dir.mkdir(parents=True, exist_ok=True)
        turn_counter += 1
        
        # Build focused bug-fix prompt (much shorter than full prompt)
        py_files = [str(f) for f in allowlist if str(f).endswith('.py')]
        target_file = py_files[0] if py_files else str(allowlist[0])
        
        bugfix_prompt = build_bugfix_prompt(
            file_path=target_file,
            error_output=out,
            original_goal=subtask
        )
        (fix_turn_dir / "prompt.md").write_text(bugfix_prompt, encoding="utf-8")
        
        # Call model with bug-fix prompt
        console.print("[cyan]Generating bug fix...[/cyan]")
        fix_content = complete_with_continuation(
            client=config.client,
            model=config.model,
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": bugfix_prompt}
            ],
            temperature=0.2,
            max_output_tokens=config.max_output,
            model_max_context=config.model_max_context or config.max_context,
        )
        (fix_turn_dir / "response.md").write_text(fix_content, encoding="utf-8")
        
        display_content = fix_content[:500] + "..." if len(fix_content) > 500 else fix_content
        console.print(Panel(display_content, title=f"Bug Fix Output ({fix_attempt + 1})"))
        
        # Apply the fix
        fix_applied = _try_apply_content(fix_content, allowlist, fix_turn_dir, config)
        
        if not fix_applied:
            console.print("[red]Bug fix could not be applied.[/red]")
            continue
        
        console.print("[green]Bug fix applied. Re-testing...[/green]")
        # The verification will run on the next iteration of the for loop
        # Update turn_dir for verification output
        turn_dir = fix_turn_dir
    
    return False

# ---------------------------
# Main Orchestrator
# ---------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", help="Task goal/description")
    parser.add_argument("--allowlist", help="Comma-separated list of files to allow editing")
    parser.add_argument("--context", help="Comma-separated list of read-only context files")
    parser.add_argument("--notes", help="Extra notes/constraints", default="")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-approve patches and verification")
    
    # Configurable Model/Env
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "https://w0wqtv67-8000.usw3.devtunnels.ms/v1"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "myhpcvllmqwen"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-Coder-Next-FP8"))
    
    # Configurable Agent config
    parser.add_argument("--agent-dir", default=".agent", help="Directory for agent artifacts")
    parser.add_argument("--max-context", type=int, default=16000, help="Max context length")
    parser.add_argument("--max-output", type=int, default=4096, help="Max output tokens")
    
    args = parser.parse_args()

    agent_dir = Path(args.agent_dir)
    ensure_dirs(agent_dir)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # Auto-detect model context length from vLLM server
    detected_ctx = query_model_context_length(client, args.model)
    effective_ctx = detected_ctx if detected_ctx > 0 else args.max_context
    console.print(f"[dim]Effective context limit: {effective_ctx} tokens[/dim]")

    session_id = now_stamp()
    session_dir = agent_dir / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    config = AgentConfig(
        client=client,
        model=args.model,
        session_dir=session_dir,
        max_context=args.max_context,
        max_output=args.max_output,
        auto_approve=args.yes,
        agent_dir=agent_dir,
        model_max_context=effective_ctx,
    )

    console.print(Panel(
        f"Session: {session_id}\nbase_url={args.base_url}\nmodel={args.model}\nlogs: {session_dir}",
        title="mini-claude-code (Refactored)",
        style="cyan"
    ))

    # Goal
    goal = args.goal
    if not goal:
        goal = Prompt.ask("Goal").strip()

    # Gather allowlist and context files
    allowlist: List[str] = []
    if args.allowlist:
        allowlist = [x.strip() for x in args.allowlist.split(",") if x.strip()]
    elif not args.yes:
        console.print("\n[bold]ALLOWLIST[/bold] (only these files may be modified)")
        while True:
            p = Prompt.ask("Add allowlisted file path (empty to stop)", default="").strip()
            if not p:
                break
            allowlist.append(p)

    context_files = list(dict.fromkeys(allowlist))  # start with allowlist
    if args.context:
        extra = [x.strip() for x in args.context.split(",") if x.strip()]
        for e in extra:
            if e not in context_files:
                context_files.append(e)
    elif not args.yes:
        console.print("\n[bold]Extra context files[/bold] (read-only context)")
        while True:
            p = Prompt.ask("Add context file path (empty to stop)", default="").strip()
            if not p:
                break
            if p not in context_files:
                context_files.append(p)

    if args.goal:
        console.print(f"\n[bold]Goal:[/bold] {goal}")
        
    if args.notes:
        extra_notes = args.notes
        console.print(f"[bold]Notes:[/bold] {extra_notes}")
    elif not args.yes:
        extra_notes = Prompt.ask("Constraints / notes (optional)", default="").strip()
    else:
        extra_notes = ""
    
    # 1. Plan
    subtasks = plan_tasks(config, goal, extra_notes)
    
    # 2. Execute
    success_count = 0
    for i, subtask in enumerate(subtasks):
        ok = run_subtask_loop(
            config=config,
            subtask=subtask,
            subtask_idx=i,
            allowlist=allowlist,
            context_files=context_files,
            global_notes=extra_notes,
        )
        if ok:
            success_count += 1
        else:
            console.print(f"[red]Sub-task {i+1} failed. Stopping sequence.[/red]")
            break
            
    console.print(Panel(f"Task Complete. Success: {success_count}/{len(subtasks)}", subtitle=str(session_dir)))



if __name__ == "__main__":
    main()