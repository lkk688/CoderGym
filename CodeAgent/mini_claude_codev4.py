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

python CodeAgent/qwen_coder_evalv4_1.py   --model_source remote_vllm   --remote_vllm_url "https://w0wqtv67-8000.usw3.devtunnels.ms/v1"   --models "Qwen/Qwen3-Coder-Next-FP8"   --run_all --out_dir ./eval_results_remote

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
from dataclasses import dataclass, asdict
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
SKILL_TEACHER = SKILL_DIR / "teacher.jsonl"

# DEFAULT_SYSTEM is now centralized in PromptRegistry.SYSTEM (see below)

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

def _handle_missing_modules(error_output: str) -> Optional[str]:
    """
    Detects ModuleNotFoundError and installs the missing package.
    Returns the installation log if an installation occurred, else None.
    """
    # Regex for "ModuleNotFoundError: No module named 'xyz'"
    # Also handle "ImportError: No module named xyz" (older python)
    match = re.search(r"ModuleNotFoundError: No module named '(.+?)'", error_output)
    if not match:
        match = re.search(r"ImportError: No module named '(.+?)'", error_output)
    
    if not match:
        return None
        
    module_name = match.group(1)
    
    # Map common imports to package names
    # This is a heuristic mapping.
    package_map = {
        "sklearn": "scikit-learn",
        "PIL": "Pillow",
        "cv2": "opencv-python",
        "yaml": "PyYAML",
        "bs4": "beautifulsoup4",
        "dotenv": "python-dotenv",
        "dateutil": "python-dateutil"
    }
    
    package_name = package_map.get(module_name, module_name)
    
    console.print(f"[yellow]Detected missing module: '{module_name}'. Attempting auto-install of '{package_name}'...[/yellow]")
    
    # Use --no-input to prevent hanging on prompts
    cmd = f"pip install --no-input {package_name}"
    code, out = run_shell(cmd)
    
    log = f"\n[Auto-Install: {cmd}]\nExit Code: {code}\nOutput:\n{out}\n"
    
    if code == 0:
        console.print(f"[green]Successfully installed '{package_name}'.[/green]")
    else:
        console.print(f"[red]Failed to install '{package_name}'.[/red]")
        
    return log


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

def read_file(path: str, max_chars: int = 64000) -> str:
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
    - Stray ``` fence markers (but NOT inside diff hunk content)
    - ALL 'index' lines (never converts them to --- headers)
    - Missing ---/+++ headers (injects them from diff --git header)
    - Enforces strict --- before +++ order
    """
    lines = diff_text.split("\n")
    cleaned = []
    
    # State tracking for header repair
    current_file_a = None
    current_file_b = None
    seen_header_a = False
    seen_header_b = False
    in_hunk = False  # Track if we're inside a @@ hunk
    
    for line in lines:
        stripped = line.strip()
        
        # 1. Skip stray fence markers — BUT ONLY if they are NOT diff content.
        #    Lines like '-    #```python' are valid diff removals and must be kept.
        #    A fence is "stray" only if it's a bare fence line (not prefixed by +/-/space).
        if re.match(r'^```', stripped):
            # Check: is this a diff content line? (starts with +, -, or space)
            if not (line.startswith('+') or line.startswith('-') or line.startswith(' ')):
                continue  # Bare fence — skip it
            # Otherwise it's diff content like '-    #```python' — keep it
            
        # 2. Skip HTML tags (only bare ones, not diff content)
        if re.match(r'^</?(?:details|summary|br|hr)', stripped, re.IGNORECASE):
            if not (line.startswith('+') or line.startswith('-') or line.startswith(' ')):
                continue
        
        # 3. Handle 'diff --git' header to reset state
        if line.startswith('diff --git'):
            m = re.match(r'^diff --git a/(\S+) b/(\S+)', line)
            if m:
                current_file_a = m.group(1)
                current_file_b = m.group(2)
            else:
                current_file_a = None
                current_file_b = None
            seen_header_a = False
            seen_header_b = False
            in_hunk = False
            cleaned.append(line)
            continue
            
        # 4. Handle ALL 'index' lines — always skip.
        #    This includes 'index abc123..def456' and malformed 'index --- a/foo'.
        #    Never convert to --- headers (causes duplicate headers).
        if line.startswith('index '):
            continue
            
        # 5. Handle --- (Original file)
        if line.startswith('--- '):
            seen_header_a = True
            in_hunk = False
            cleaned.append(line)
            continue
            
        # 6. Handle +++ (New file)
        if line.startswith('+++ '):
            # CRITICAL: If we see +++ but missed ---, inject --- NOW to preserve order
            if not seen_header_a and current_file_a:
                cleaned.append(f"--- a/{current_file_a}")
                seen_header_a = True
                
            seen_header_b = True
            in_hunk = False
            cleaned.append(line)
            continue
            
        # 7. Handle Hunk Header @@
        # If we hit @@ and missed headers, inject them (respecting order)
        if line.startswith('@@ ') and current_file_a and current_file_b:
            if not seen_header_a:
                cleaned.append(f"--- a/{current_file_a}")
                seen_header_a = True
            if not seen_header_b:
                cleaned.append(f"+++ b/{current_file_b}")
                seen_header_b = True
            in_hunk = True
                
        cleaned.append(line)
    
    result = "\n".join(cleaned)
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
      - Mixed format: 'diff --git' header OUTSIDE a fenced block with hunks inside
    Returns the last (most likely correct) diff text or None.
    """
    t = text.strip()
    
    # --- Pre-processing: Merge split diffs ---
    # LLMs sometimes put 'diff --git ...' on a line BEFORE ```diff,
    # with the actual hunks inside the fenced block. Merge them.
    # Pattern: diff --git a/X b/X\n```diff\n@@ ...\n```
    t = re.sub(
        r'^(diff --git [^\n]+)\n\s*```(?:diff|python|python3)?\s*\n',
        r'\1\n',
        t,
        flags=re.MULTILINE
    )
    
    # Strategy 1: Look for fenced ```diff blocks and extract content
    fenced_diffs = []
    # Match all ```diff ... ``` blocks (or just ``` blocks containing diffs)
    # CRITICAL: Use line-anchored closing fence (^```\s*$) to avoid matching
    # backticks inside diff content lines like '-    #```python'
    fence_pattern = re.compile(r'```(?:diff)?\s*\n(.*?)^```\s*$', re.DOTALL | re.MULTILINE)
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
            # Clean trailing prose — stop at blank line followed by non-diff text
            # Keep everything that looks like diff content (lines starting with
            # diff, ---, +++, @@, +, -, space, or 'new file mode', backslash)
            diff_lines = []
            for line in part.split('\n'):
                # Lines that are valid diff content
                if (line.startswith('diff --git') or
                    line.startswith('---') or
                    line.startswith('+++') or
                    line.startswith('@@') or
                    line.startswith('+') or
                    line.startswith('-') or
                    line.startswith(' ') or
                    line.startswith('\\') or
                    line.startswith('index ') or
                    line.startswith('new file') or
                    line.startswith('old mode') or
                    line.startswith('new mode') or
                    line.startswith('deleted file') or
                    line.startswith('similarity') or
                    line.startswith('rename') or
                    line == ''):
                    diff_lines.append(line)
                else:
                    # Non-diff line (prose, markdown, etc.) — stop
                    break
            if diff_lines:
                raw_diffs.append('\n'.join(diff_lines))
    
    if raw_diffs:
        # For raw diffs, use the last complete diff block
        return sanitize_diff_text(raw_diffs[-1])
    
    return None


def extract_write_file_actions(text: str) -> List[Tuple[str, str]]:
    """
    Extract WRITE_FILE actions with high-robustness regex.
    Handles:
    - Merged headers (e.g. 'code...WRITE_FILE: path')
    - Malformed closers (CONTENT>>, CONTENT]>>)
    - Truncated output (EOF)
    - Prose injection (stops at '## Reasoning')
    - Diff artifacts (ignores '-WRITE_FILE' or '-<<<CONTENT')
    """
    results = []
    
    # Regex Breakdown:
    # 1. (?:^|\n)(?!\-).*?WRITE_FILE:
    #    - Matches start of line OR new line.
    #    - (?!\-) Negative lookahead: Ensure line does NOT start with '-' (diff removal).
    #    - .*? Consumes garbage prefix (e.g. 'model = ...').
    
    # 2. \s*(\S+)
    #    - Capture filepath (stops at whitespace).
    
    # 3. .*?\n
    #    - Consume rest of the header line.
    
    # 4. \s*<<<CONTENT\n
    #    - Match start tag. 
    #    - \s* matches spaces but NOT hyphens (diff safety).
    
    # 5. (.*?)
    #    - Capture content non-greedily.
    
    # 6. Terminator Group:
    #    - CONTENT>{2,3}        -> Normal closer (>>> or >>)
    #    - (?=\n.*?WRITE_FILE:) -> Lookahead: Next file starts
    #    - (?=\ndiff --git)     -> Lookahead: Diff starts
    #    - (?=\n\#\#\s)         -> Lookahead: Markdown header (e.g. ## Reasoning)
    #    - (?=\n```)            -> Lookahead: Code block fence
    #    - $                    -> EOF (Truncation)
    
    pattern = re.compile(
        r'(?:^|\n)(?!\-).*?WRITE_FILE:\s*(\S+).*?\n'  # Header (safe from diffs)
        r'\s*<<<CONTENT\n'                            # Start Tag
        r'(.*?)'                                      # Content Capture
        r'(?:CONTENT>{2,3}|(?=\n.*?WRITE_FILE:)|(?=\ndiff --git)|(?=\n\#\#\s)|(?=\n```)|$)', # Robust Terminator
        re.DOTALL
    )
    
    for m in pattern.finditer(text):
        filepath = m.group(1).strip()
        content = m.group(2)
        
        # Post-processing: Strip "CONTENT>>>" manually if regex captured it due to whitespace
        if "CONTENT>>>" in content:
            content = content.replace("CONTENT>>>", "")
        
        # Post-processing checks
        
        # 1. Diff Artifact check (double safety)
        # If the path looks like a diff path (a/foo.py, b/foo.py), ignore it
        if filepath.startswith("a/") or filepath.startswith("b/") or filepath == "/dev/null":
            continue
            
        # 2. Content validation
        # If content is extremely short (< 15 chars), it's likely a parsing artifact or hallucination
        if len(content.strip()) < 15:
            continue
            
        results.append((filepath, content))
        
    return results



# ---------------------------
# SkillDB
# ---------------------------

@dataclass
class Skill:
    category: str      # e.g., "PyTorch", "Syntax", "Logic", "API"
    pattern: str       # Trigger keywords (e.g., "conv2d", "plot")
    insight: str       # The lesson (e.g. "Do not use .cuda() on inputs...")
    evidence: str      # Short snippet or original output
    count: int = 1     # Dedup counter
    created_at: str = ""

def load_skills(skill_dir: Path) -> List[Skill]:
    skills: List[Skill] = []
    # Load separate success/failure logs or a unified DB
    # For v2, we can just load all jsonl files in skilldb
    if not skill_dir.exists():
        return []
        
    for path in skill_dir.glob("*.jsonl"):
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    # Migration: Handle legacy format
                    if "insight" not in obj:
                        # Legacy skill
                        tag = obj.get("tag", "general")
                        kind = obj.get("kind", "unknown")
                        text = obj.get("text", "")
                        evidence = obj.get("evidence", "")
                        # Construct a basic migration
                        skills.append(Skill(
                            category="Legacy",
                            pattern=obj.get("pattern", "general"),
                            insight=f"Legacy {kind}: {text[:100]}...",
                            evidence=evidence[:200],  # Truncate legacy evidence
                            created_at=obj.get("created_at", "")
                        ))
                    else:
                        # New format
                        skills.append(Skill(
                            category=obj.get("category", "Uncategorized"),
                            pattern=obj.get("pattern", ""),
                            insight=obj.get("insight", ""),
                            evidence=obj.get("evidence", ""),
                            count=obj.get("count", 1),
                            created_at=obj.get("created_at", "")
                        ))
                except Exception:
                    continue
        except Exception:
            pass
    return skills

def score_skill(skill: Skill, query: str) -> int:
    """
    Scoring Strategy v2:
    - High match on Pattern
    - Medium match on Insight words
    """
    q = query.lower()
    s = 0
    
    # 1. Pattern Match (Strong signal)
    patt = (skill.pattern or "").lower().strip()
    if patt and patt in q:
        s += 5
        
    # 2. Insight Match (Contextual signal)
    # Tokenize insight
    words = re.findall(r"[a-zA-Z0-9_]{3,}", (skill.insight or "").lower())
    hits = 0
    for w in set(words):
        if w in q:
            hits += 1
    s += min(hits, 5)  # Cap at 5 points
    
    return s

def select_relevant_skills(goal_and_notes: str, skill_dir: Path, topk: int = SKILL_INJECT_TOPK) -> List[Skill]:
    skills = load_skills(skill_dir)
    scored = [(score_skill(sk, goal_and_notes), sk) for sk in skills]
    # Sort by score desc
    scored.sort(key=lambda x: x[0], reverse=True)
    # Filter only relevant ones (score > 2)
    picked = [sk for sc, sk in scored if sc >= 2][:topk]
    return picked

def format_skill_injection(skills: List[Skill]) -> str:
    if not skills:
        return ""
    
    # Group by category
    by_cat = {}
    for sk in skills:
        by_cat.setdefault(sk.category, []).append(sk)
    
    lines = ["## Teacher Guidelines (From Experience)"]
    for cat, sk_list in by_cat.items():
        if cat == "Legacy": continue # Skip legacy unless very relevant?
        lines.append(f"### {cat}")
        for sk in sk_list:
            # Format: "- [Pattern] Insight"
            lines.append(f"- [{sk.pattern}] {sk.insight}")
            
    if len(lines) == 1: # Only header
        return ""
        
    return "\n".join(lines).strip() + "\n"


# ---------------------------
# Prompt Logic (centralized in PromptRegistry below)
# ---------------------------
# All prompt construction functions have been merged into the PromptRegistry class.
# See PromptRegistry.format_task(), format_bugfix(), format_fix_diff(), format_fix_rewrite().


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

def apply_fuzzy_patch(file_path: Path, diff_content: str, log_buffer: list = None) -> bool:
    """
    Applies a Unified Diff with 'fuzzy' matching logic.
    1. Ignores line numbers (@@ -12,4 +12,5 @@).
    2. Matches context by stripping whitespace (ignoring indentation changes).
    3. Falls back to anchor-based matching (first+last lines of search block).
    4. Handles 'New File' creation via diff.
    5. Preserves trailing newline state from original file.
    """
    def log(msg: str):
        if log_buffer is not None:
            log_buffer.append(msg)

    # 1. Handle New File Creation
    if "new file mode" in diff_content or "--- /dev/null" in diff_content:
         new_content = []
         for line in diff_content.splitlines():
             if line.startswith('+') and not line.startswith('+++'):
                 new_content.append(line[1:]) # Remove '+'
         
         # Sanity check: verify it's not just an empty file or metadata
         if len(new_content) > 0:
             file_path.parent.mkdir(parents=True, exist_ok=True)
             file_path.write_text("\n".join(new_content) + "\n", encoding="utf-8")
             msg = f"[green]Created new file from diff: {file_path}[/green]"
             console.print(msg)
             log(msg)
             return True
         log(f"New file creation failed: content empty for {file_path}")
         return False

    if not file_path.exists():
        msg = f"[red]Target file {file_path} not found for diff.[/red]"
        console.print(msg)
        log(msg)
        return False


    original_text = file_path.read_text(encoding="utf-8")
    had_trailing_newline = original_text.endswith("\n")
    original_lines = original_text.splitlines()
    # Work on a copy
    modified_lines = list(original_lines)
    
    # 2. Parse Hunks
    # Regex to split by @@ ... @@ header, consuming the rest of the line (e.g. function context)
    hunks = re.split(r'^@@\s.*?\s@@.*$', diff_content, flags=re.MULTILINE)
    # The first part is the header (diff --git ...), skip it
    hunks = hunks[1:]
    
    if not hunks:
        msg = "[yellow]No hunks found in diff.[/yellow]"
        console.print(msg)
        log(msg)
        return False
    
    applied_hunks = 0
    for hunk in hunks:
        # IMPORTANT: Don't filter out empty lines — they are valid context!
        # An empty line in a hunk (no leading +/-/space) is a context line.
        hunk_lines = hunk.splitlines()
        # Skip the very first element if it's empty (artifact from split)
        if hunk_lines and hunk_lines[0] == '':
            hunk_lines = hunk_lines[1:]
        if not hunk_lines:
            continue
            
        # 3. Identify the "Search Block" (Context + Removed lines)
        search_block = []
        replace_block = []
        
        for line in hunk_lines:
            if line.startswith(' '): # Context — remove leading space
                search_block.append(line[1:])
                replace_block.append(line[1:])
            elif line.startswith('-'): # Remove
                search_block.append(line[1:])
            elif line.startswith('+'): # Add
                replace_block.append(line[1:])
            elif line.startswith('\\'): # '\ No newline at end of file'
                pass
            elif line == '': # Empty context line (common in diffs)
                search_block.append('')
                replace_block.append('')
            # Any other line (shouldn't happen) — treat as context
            else:
                search_block.append(line)
                replace_block.append(line)
        
        if not search_block:
            # Pure addition hunk — no context. Insert at beginning if first hunk.
            if replace_block:
                for i, rl in enumerate(replace_block):
                    modified_lines.insert(i, rl)
                msg = f"[green]Applied pure-addition hunk ({len(replace_block)} lines)[/green]"
                console.print(msg)
                log(msg)
                applied_hunks += 1
            continue

        # Strategy 0: Already Applied?
        # If the *result* (replace_block) is already in the file, we can skip this hunk.
        # This handles cases where the file was partially updated or the model is repetitive.
        if replace_block:
            replace_stripped = [l.strip() for l in replace_block]
            n_replace = len(replace_block)
            found_already = -1
            
            # Fuzzy match (strip whitespace) for replace block
            for i in range(len(modified_lines) - n_replace + 1):
                file_subset = modified_lines[i : i+n_replace]
                if [l.strip() for l in file_subset] == replace_stripped:
                    found_already = i
                    break
            
            if found_already != -1:
                msg = f"[green]Hunk already applied (found replacement at line {found_already+1})[/green]"
                console.print(msg)
                log(msg)
                applied_hunks += 1
                continue

        # 4. Find where this block exists in the file
        match_index = -1
        n_search = len(search_block)
        
        # Strategy A: Exact Match
        for i in range(len(modified_lines) - n_search + 1):
            if modified_lines[i : i+n_search] == search_block:
                match_index = i
                log(f"Strategy A (Exact) match at line {i+1}")
                break
        
        # Strategy B: Fuzzy Match (strip whitespace)
        if match_index == -1:
            search_stripped = [l.strip() for l in search_block]
            for i in range(len(modified_lines) - n_search + 1):
                file_subset = modified_lines[i : i+n_search]
                file_stripped = [l.strip() for l in file_subset]
                if file_stripped == search_stripped:
                    match_index = i
                    msg = f"[green]Fuzzy-matched hunk at line {match_index+1} (whitespace-insensitive)[/green]"
                    console.print(msg)
                    log(msg)
                    break
        
        # Strategy C: Anchor Match (first + last non-empty lines)
        # When context has drifted or the file has extra/missing lines,
        # match using boundary lines as anchors. Search for the last
        # anchor independently within a window after the first anchor.
        n_delete = n_search  # Default: delete same number of lines as search block
        if match_index == -1 and n_search >= 2:
            # Find first and last non-empty search lines
            anchors = [(idx, l) for idx, l in enumerate(search_block) if l.strip()]
            if len(anchors) >= 2:
                first_idx, first_line = anchors[0]
                last_idx, last_line = anchors[-1]
                expected_span = last_idx - first_idx
                first_stripped = first_line.strip()
                last_stripped = last_line.strip()
                
                # Search for the first anchor
                for i in range(len(modified_lines)):
                    if modified_lines[i].strip() != first_stripped:
                        continue
                    
                    # Search for the last anchor within 2x expected span
                    max_end = min(len(modified_lines), i + expected_span * 2 + 2)
                    found_last = -1
                    # Allow search to start immediately after first anchor (handle missing/hallucinated context lines)
                    for j in range(i + 1, max_end):
                        if j < len(modified_lines) and modified_lines[j].strip() == last_stripped:
                            found_last = j
                            break
                    
                    if found_last != -1:
                        # Found both anchors!
                        match_index = i - first_idx
                        # Calculate how many lines to actually delete from the file
                        # This bridges the gap between where the first anchor matched and where the last matched.
                        # The 'last' anchor in the file is at index 'found_last'.
                        # The 'last' anchor in the search block is at index 'last_idx'.
                        # The match starts at `i - first_idx`.
                        # So the theoretical end of the match block is `i - first_idx + n_search`.
                        # But in the file, the end is at `found_last + (len(search_block) - last_idx - 1)`.
                        # Actually simpler: we are deleting from `match_index` to `found_last + <lines after last anchor>`.
                        
                        lines_after_last_anchor = len(search_block) - last_idx - 1
                        actual_end = found_last + lines_after_last_anchor + 1
                        n_delete = actual_end - match_index
                        
                        msg = f"[cyan]Anchor-matched hunk at line {match_index+1} (anchors at {match_index+1}..{actual_end}, deleting {n_delete} lines)[/cyan]"
                        console.print(msg)
                        log(msg)
                        break

        # Strategy D: Sliding Window Fuzzy Match
        if match_index == -1 and n_search >= 4:
            # Try to match a significant subset of lines (e.g. 60%) in sequence
            search_stripped = [l.strip() for l in search_block]
            best_ratio = 0
            best_pos = -1
            best_wsize = 0
            
            # Use a slightly smaller window to allow for missing lines
            min_window = max(3, int(n_search * 0.5))
            
            for wsize in range(n_search, min_window - 1, -1):
                # If we found a good match already, stop shrinking
                if best_ratio > 0.8: break
                
                window_stripped = search_stripped[:wsize]
                for i in range(len(modified_lines) - wsize + 1):
                    file_subset = [l.strip() for l in modified_lines[i : i+wsize]]
                    # Calculate similarity ratio
                    matches = sum(1 for a, b in zip(file_subset, window_stripped) if a == b)
                    ratio = matches / wsize
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_pos = i
                        best_wsize = wsize
            
            if best_ratio >= 0.5 and best_pos >= 0:
                match_index = best_pos
                n_delete = best_wsize
                msg = f"[cyan]Partial-matched hunk at line {best_pos+1} ({best_ratio:.0%} match, window={best_wsize})[/cyan]"
                console.print(msg)
                log(msg)

        if match_index != -1:
            # Apply replacement
            # Remove n_delete lines starting from match_index
            # Insert replace_block
            
            # Handles index out of bounds if match is near end?
            # List slicing is forgiving.
            del modified_lines[match_index : match_index + n_delete]
            
            for i, line in enumerate(replace_block):
                modified_lines.insert(match_index + i, line)
            
            msg = f"[green]Applied hunk at line {match_index+1}[/green]"
            console.print(msg)
            log(msg)
            applied_hunks += 1
        else:
            msg = "[red]Failed to find matching context for hunk:[/red]"
            console.print(msg)
            log(msg)
            
            # Print context for debugging
            ctx_head = search_block[:5]
            ctx_box = Panel("\n".join(ctx_head) + ("\n..." if len(search_block) > 5 else ""), title="Expected Context (First 5 lines)")
            console.print(ctx_box)
            # Log expected context
            log("Expected Context snippet:")
            for l in ctx_head:
                log(f"| {l}")
    
    if applied_hunks == len(hunks):
        # Success! Write back
        new_text = "\n".join(modified_lines)
        if had_trailing_newline and not new_text.endswith("\n"):
            new_text += "\n"
        elif not had_trailing_newline and new_text.endswith("\n"):
            new_text = new_text[:-1]
            
        file_path.write_text(new_text, encoding="utf-8")
        return True
    
    return False
def extract_files_from_diff(diff_text: str) -> List[Tuple[str, str]]:
    """
    Extract file contents from diff '+' lines — ONLY FOR NEW FILES.
    
    SAFETY: This function ONLY extracts from diffs where `--- /dev/null`
    (i.e., entirely new files). For EDIT diffs (partial patches), scraping
    '+' lines would produce a tiny fragment and OVERWRITE the existing
    file, destroying it. This was the root cause of the 'task.py destroyed'
    bug in session 2026-02-16_215657.
    
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
        
        # CRITICAL SAFETY: Only extract from NEW FILE diffs
        is_new_file = ('new file mode' in single_diff or 
                       '--- /dev/null' in single_diff)
        
        if not is_new_file:
            console.print(f"[yellow]Skipping diff extraction for '{filepath}' "
                          f"(edit diff — would destroy existing file)[/yellow]")
            continue
        
        # Collect all '+' lines (for new files, every line is a '+' line)
        lines = single_diff.split('\n')
        content_lines = []
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
                elif line.startswith(' '):
                    content_lines.append(line[1:])  # Context line
                elif line == '':
                    content_lines.append('')
        
        if not content_lines:
            continue
        
        # Join with newlines, ensure trailing newline
        content = '\n'.join(content_lines)
        if not content.endswith('\n'):
            content += '\n'
        results.append((filepath, content))
        console.print(f"[cyan]Extracted NEW file '{filepath}' from diff ({len(content)} bytes)[/cyan]")
    
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
    
    Robustness Features:
    - Strips conversational filler from continuations ("Here is the rest...").
    - Prevents hallucinated headers/markdown injection inside code blocks.
    - Adaptively caps max_tokens to prevent context overflow.
    """
    full_content = ""
    current_messages = list(messages)
    
    max_loops = 5  # Max continuation loops
    
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
                          f"(input≈{input_est}, limit={model_max_context})[/yellow]")
        
        # Retry with backoff on API errors
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
                if 'max_tokens' in err_str or 'context length' in err_str:
                    safe_tokens = max(1024, safe_tokens // 2)
                    console.print(f"[red]Context overflow. Retrying with max_tokens={safe_tokens}...[/red]")
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
            
        
        # Custom Environment Handling: If client returns a string, use it directly
        if isinstance(resp, str):
            full_content += resp
            break

        choice = resp.choices[0]
        console.print(f"[dim]Finish Reason: {choice.finish_reason}[/dim]")
        content = choice.message.content or ""
        
        # --- Robust Stitching Logic ---
        # If this is a continuation (loop > 0), filter out conversational prefixes.
        if i > 0:
            original_len = len(content)
            
            # Check if we were inside a code block in the previous chunk
            # (Odd number of triple-backticks implies we are inside a block)
            prev_chunk_fences = full_content.count("```")
            is_inside_code = (prev_chunk_fences % 2 == 1)
            
            # Check if we are inside a WRITE_FILE block (<<<CONTENT without CONTENT>>>)
            open_tags = len(re.findall(r'<<<CONTENT', full_content))
            close_tags = len(re.findall(r'CONTENT>{2,3}', full_content))
            is_inside_write_file = (open_tags > close_tags)
            
            if is_inside_code or is_inside_write_file:
                # 1. Strip re-opened code fences (e.g. "```python")
                # Models often restart the block when continued
                content = re.sub(r'^\s*```\w*\n', '', content)
                
                # 2. Strip "Here is the rest..." prose if it precedes code
                # If the content starts with prose lines that end in a colon or look like chat
                # (Heuristic: remove lines until we hit what looks like code)
                # Be careful not to remove actual code comments.
                if not content.strip().startswith(('#', 'def ', 'class ', 'print', 'import ')):
                     # Remove first line if it looks like conversation
                     content = re.sub(r'^(Here is the rest.*?|Sure.*?|Continuing.*?)\n', '', content, flags=re.IGNORECASE)

            # 3. Strip hallucinated headers immediately (e.g. "## Reasoning")
            # If we are inside code, a markdown header is almost always a hallucination
            if is_inside_code and content.lstrip().startswith("## "):
                # Stop processing here? Or strip the header? 
                # Usually implies model switched context. We treat it as end of code.
                console.print("[red]Detected hallucinated header in code block. Truncating.[/red]")
                content = content.split("## ")[0]

            if len(content) < original_len:
                console.print(f"[dim]Stitched continuation (stripped {original_len - len(content)} chars)[/dim]")

        full_content += content
        
        if choice.finish_reason == "length":
            console.print("[yellow]Output truncated (limit reached). Continuing...[/yellow]")
            
            # Append partial content to history
            current_messages.append({"role": "assistant", "content": content})
            
            # Strict Continuation Prompt
            cont_prompt = (
                "You were cut off. "
                "IMMEDIATELY continue the code/text exactly where you left off. "
                "DO NOT repeat the last line. "
                "DO NOT output conversational text (e.g. 'Here is the rest'). "
                "DO NOT output markdown headers or code fences. "
                "Just output the missing characters."
            )
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


def plan_tasks(config: AgentConfig, goal: str, notes: str, allowlist: List[str]) -> List[str]:
    """
    Analyze complexity. 
    Optimized: Skips LLM call if task is constrained to 1 file or allowlist is empty (assuming new file).
    """
    
    # --- Optimization 1: Explicit Single File Constraint ---
    # If the user provided --allowlist task.py, we know we can't edit anything else.
    # Plan = [goal]. No LLM needed.
    if allowlist and len(allowlist) == 1:
        console.print(f"[green]Single file target ({allowlist[0]}) detected. Skipping planner.[/green]")
        return [goal]

    # --- Optimization 2: Implicit Single File Goal ---
    # If allowlist is empty (meaning "create whatever you need"), but the goal 
    # explicitly mentions creating a specific file, assume single task.
    # Regex looks for "Create task.py", "Write script.py", etc.
    if not allowlist:
        # Check for explicit file creation intent in goal
        # m = re.search(r"(?:create|write|implement)\s+(\S+\.py)", goal, re.IGNORECASE)
        
        # NEW: allows words in between (e.g. "Write a new test.py")
        # We use dotall to match across newlines and \b to ensure clean filename start
        m = re.search(r"(?:create|write|implement).*?\b([a-zA-Z0-9_]+\.py)", goal, re.IGNORECASE | re.DOTALL)
        if m:
            filename = m.group(1)
            console.print(f"[green]Goal targets single file ({filename}). Skipping planner.[/green]")
            # Side effect: We can hint to the main loop to verify this file later
            return [goal]

    system_prompt = """You are a technical lead. Plan the execution steps.

**CRITICAL GUIDELINES**:
1. **Prefer Single Step**: Modern LLMs can write 500+ lines at once. Do NOT split a task just because it has multiple functions.
2. **One File = One Step**: Never split the creation of a single file into multiple steps.
3. **Split Only for Isolation**: Only split if the task touches completely different parts of the system (e.g., "Step 1: Update SQL Schema", "Step 2: Update React Frontend").

Output JSON: {"steps": ["step1", ...]}
"""
    
    files_context = f"Target Files: {', '.join(str(p) for p in allowlist)}" if allowlist else "Target Files: (Open)"
    user_prompt = f"Goal: {goal}\nNotes: {notes}\n{files_context}\n\nJSON:"
    
    console.print("[cyan]Analyzing task complexity...[/cyan]")
    try:
        # Calculate adaptive tokens
        planner_input = system_prompt + user_prompt
        planner_input_est = estimate_tokens(planner_input)
        ctx_limit = config.model_max_context or config.max_context
        planner_max_tokens = compute_safe_max_tokens(
            input_tokens=planner_input_est,
            model_max_context=ctx_limit,
            desired_max_output=1024,
            min_output=256
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
        
        # Log planning
        (config.session_dir / "planning_response.md").write_text(content, encoding="utf-8")
        
        data = extract_json_robust(content)
        if not data or "steps" not in data:
            return [goal]
        
        steps = data["steps"]
        
        # --- Heuristic 3: Collapse micro-plans ---
        # If the model outputs many small steps for a small file list, collapse them.
        if len(steps) > 3 and (allowlist and len(allowlist) <= 2):
            console.print("[yellow]Plan too fragmented for small file count. Collapsing to single task.[/yellow]")
            return [goal]

        if len(steps) > 1:
            console.print(Panel(
                "\n".join([f"{i+1}. {s}" for i,s in enumerate(steps)]), 
                title="Task Plan", style="magenta"
            ))
            if config.auto_approve:
                # Still check: if steps look like "Step 1: Imports", collapse them
                return steps
            
            if Confirm.ask("Execute as separate sub-tasks? (No = run as one big task)"):
                return steps
            
        return [goal]

    except Exception as e:
        console.print(f"[red]Planning failed ({e}). Defaulting to single task.[/red]")
        return [goal]


# ---------------------------
# Sub-task Execution
# ---------------------------
def resolve_path(raw_path: str, allowlist: List[str], root_dir: Path = Path(".")) -> Optional[Path]:
    """
    Robustly resolves an LLM-generated path to a valid local file path.
    Prioritizes:
    1. Exact match in allowlist.
    2. Basename match in allowlist (e.g. '/abs/path/task.py' -> 'task.py').
    3. Relative path from root_dir.
    """
    # Clean up formatting artifacts
    clean = raw_path.strip().strip("'").strip('"')
    
    # 1. Safety Check: Absolute paths are suspicious. Strip root.
    # Logic: If model says /Developer/AIserver/task.py, we only care about src/main.py relative to us.
    if clean.startswith("/"):
        clean = clean.lstrip("/")
    
    # 2. Check Allowlist (Highest Priority)
    # This fixes the exact case you saw: 'Developer/AIserver/task.py' vs 'task.py'
    target_name = Path(clean).name
    for allowed in allowlist:
        allowed_p = Path(allowed)
        # If basenames match (e.g. task.py == task.py), map it!
        if allowed_p.name == target_name:
            # Optional: Check if the full suffix matches to be safer
            # e.g. 'server/task.py' matches 'task.py' -> maybe unsafe?
            # For a mini-agent, basename matching is usually the desired behavior.
            return allowed_p

    # 3. Direct resolution relative to CWD
    candidate = root_dir / clean
    if candidate.exists() or candidate.parent.exists():
        return candidate

    return None

def _try_apply_content(content: str, allowlist: List[str], turn_dir: Path, 
                       config: AgentConfig) -> bool:
    """
    Try all methods to apply model output as file changes.
    Order: 
    1. git apply (Strict Diff)
    2. apply_fuzzy_patch (Loose Diff - handles line/whitespace errors)
    3. WRITE_FILE (Full rewrite) — tried even if diff was found
    4. Diff Extraction (Last resort reconstruction for new files)
    """
    
    # --- Extract Diff once ---
    diff = extract_all_diffs(content)
    changes_applied = False
    apply_method = None
    
    # --- TRY FORMAT A: Unified Diff Strategies ---
    if diff:
        (turn_dir / "patch.diff").write_text(diff, encoding="utf-8")
        
        # Strategy 1: Strict Git Apply
        if is_git_repo():
            changes_applied = apply_patch_guarded(diff, turn_dir, auto_approve=config.auto_approve)
            if changes_applied:
                apply_method = "git_apply"
        else:
            console.print("[red]Not a git repo, skipping strict diff apply.[/red]")
        
        # Strategy 2: Fuzzy Patch
        if not changes_applied:
            console.print("[yellow]Strict apply failed. Attempting fuzzy patch...[/yellow]")
            file_diffs = re.split(r'(?=^diff --git )', diff, flags=re.MULTILINE)
            fuzzy_successes = 0
            fuzzy_total = 0
            
            fuzzy_logs = ["\n--- Fuzzy Patch Attempt ---"]
            
            for fd in file_diffs:
                if not fd.strip().startswith("diff --git"): continue
                fuzzy_total += 1
                
                # Extract raw path from header
                match = re.search(r'diff --git a/\S+ b/(\S+)', fd)
                if match:
                    raw_path = match.group(1)
                    fuzzy_logs.append(f"Processing diff for: {raw_path}")
                    
                    # Resolve Path
                    target_path = resolve_path(raw_path, allowlist)
                    
                    if target_path:
                        if target_path != Path(raw_path):
                            msg = f"[dim]Redirecting '{raw_path}' -> '{target_path}'[/dim]"
                            console.print(msg)
                            fuzzy_logs.append(msg)
                        
                        if apply_fuzzy_patch(target_path, fd, log_buffer=fuzzy_logs):
                            fuzzy_successes += 1
                            fuzzy_logs.append(">> Success")
                        else:
                            fuzzy_logs.append(">> Failed")
                    else:
                        msg = f"[red]Skipping diff for unresolved path: {raw_path}[/red]"
                        console.print(msg)
                        fuzzy_logs.append(msg)
            
            # Append logs to apply.log
            try:
                with open(turn_dir / "apply.log", "a", encoding="utf-8") as f:
                    f.write("\n".join(fuzzy_logs) + "\n")
            except Exception as e:
                console.print(f"Failed to append to apply.log: {e}")

            # Mark success if at least one file was patched
            if fuzzy_successes > 0:
                changes_applied = True
                apply_method = "fuzzy_patch"
                console.print(f"[green]Fuzzy patch applied ({fuzzy_successes}/{fuzzy_total} files).[/green]")


    # --- TRY FORMAT B: WRITE_FILE ---
    # Try WRITE_FILE regardless of whether a diff was found — some responses
    # contain both a diff AND a WRITE_FILE block. If the diff failed, the
    # WRITE_FILE may still work.
    if not changes_applied:
        write_actions = extract_write_file_actions(content)
        if write_actions:
            valid_actions = []
            for path, text in write_actions:
                # Resolve Path
                target_path = resolve_path(path, allowlist)
                if target_path:
                    valid_actions.append((str(target_path), text))
                else:
                    console.print(f"[red]Skipping WRITE_FILE for unresolved path: {path}[/red]")
            
            if valid_actions:
                changes_applied = apply_write_files(valid_actions, allowlist, turn_dir)
                if changes_applied:
                    apply_method = "write_file"
    
    # --- TRY FORMAT C: Extract NEW files from diff (Last resort) ---
    # SAFETY: extract_files_from_diff ONLY extracts new files (--- /dev/null).
    # For edit diffs, it safely skips to avoid overwriting existing files
    # with tiny fragments (the session 2026-02-16_215657 bug).
    if not changes_applied and diff:
        console.print("[yellow]All patch methods failed. Checking for extractable new files in diff...[/yellow]")
        diff_files = extract_files_from_diff(diff)
        if diff_files:
            changes_applied = apply_write_files(diff_files, allowlist, turn_dir)
            if changes_applied:
                apply_method = "diff_extraction"
                console.print("[green]Wrote new files extracted from diff.[/green]")
        else:
            console.print("[red]No new files to extract. Edit diffs cannot be safely applied as rewrites.[/red]")
    
    # --- Log result ---
    if apply_method:
        console.print(f"[green]Changes applied via: {apply_method}[/green]")
    elif not changes_applied:
        # Check if we missed a WRITE_FILE due to bad formatting
        if "WRITE_FILE:" in content and "CONTENT" in content:
             console.print("[red]Potential malformed WRITE_FILE block detected but extraction failed.[/red]")
        
        if not diff and not extract_write_file_actions(content):
            console.print("[red]No valid diff or WRITE_FILE actions found in response.[/red]")
            
            # --- TRY FORMAT E: Fenced Block Fallback (Session 213156 fix) ---
            # If model wraps code in markdown fences but forgets WRITE_FILE
            if len(allowlist) == 1 and not changes_applied:
                # Look for ```python ... ``` or just ``` ... ```
                # We want EXACTLY ONE code block to be safe
                code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', content, re.DOTALL)
                
                if len(code_blocks) == 1:
                    target_file = Path(allowlist[0])
                    block_content = code_blocks[0].strip()
                    
                    # Heuristic: does it look like Python code?
                    if "def " in block_content or "import " in block_content:
                        console.print(f"[yellow]Fallback E: Extracting single fenced block for {target_file}[/yellow]")
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        target_file.write_text(block_content + "\n", encoding="utf-8")
                        apply_method = "fenced_fallback"
                        changes_applied = True
            
            # --- TRY FORMAT D: Raw Code Fallback (Session 153128 fix) ---
            # If the model outputs *just* the code without formatting, and we expect 1 file.
            if len(allowlist) == 1 and not changes_applied:
                target_file = Path(allowlist[0])
                # Heuristic: does it look like Python code?
                if "def " in content or "import " in content:
                    console.print(f"[yellow]Fallback D: Treating entire response as content for {target_file}[/yellow]")
                    
                    # Sanitize: Remove markdown fences if they wrap the whole content
                    clean_content = content.strip()
                    if clean_content.startswith("```python"):
                        clean_content = clean_content[len("```python"):].strip()
                    elif clean_content.startswith("```"):
                        clean_content = clean_content[3:].strip()
                    
                    if clean_content.endswith("```"):
                        clean_content = clean_content[:-3].strip()
                        
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    target_file.write_text(clean_content + "\n", encoding="utf-8")
                    apply_method = "raw_fallback"
                    changes_applied = True
    
    return changes_applied


def _determine_verify_cmd(
    allowlist: List[str], 
    modified_files: List[str], 
    auto_verify_cmd: Optional[str], 
    config: AgentConfig
) -> str:
    """
    Determine the verification command.
    Priority:
    1. Model's explicit 'Verification:' line.
    2. Python file found in 'modified_files' (the file just generated).
    3. Python file found in 'allowlist'.
    """
    # 1. Start with Model Suggestion
    candidate = auto_verify_cmd
    
    # 2. If no suggestion, look for a runnable Python file in modified files
    if not candidate:
        py_files = [str(f) for f in modified_files if str(f).endswith('.py')]
        if py_files:
            candidate = f"python3 {py_files[0]}"
            
    # 3. If still nothing, check allowlist
    if not candidate:
        py_files = [str(f) for f in allowlist if str(f).endswith('.py')]
        if py_files:
            candidate = f"python3 {py_files[0]}"
    
    # Interactive Mode
    if not config.auto_approve:
        if Confirm.ask("Run verification?", default=True):
            # Pre-fill the prompt with our best guess
            # User can just hit Enter to accept "python3 task.py"
            return Prompt.ask("Command", default=candidate or "").strip()
        return ""

    # Auto Mode
    return candidate or ""

def run_linter(files: List[str]) -> Optional[str]:
    """
    Run fast static analysis (Ruff) on Python files.
    Catches syntax errors and undefined names before execution.
    Requires: pip install ruff
    """
    py_files = [str(f) for f in files if str(f).endswith('.py')]
    if not py_files:
        return None
    
    # E9: Syntax, F821: Undefined name, F823: Local var referenced before assign
    cmd = f"ruff check --select=E9,F821,F823 --output-format=text {' '.join(py_files)}"
    code, out = run_shell(cmd)
    
    if code != 0:
        return f"STATIC ANALYSIS FAILED (Ruff):\n{out}\n(Fix these syntax/name errors first!)"
    return None

def extract_skill_insight(
    client: OpenAI, 
    model: str, 
    goal: str, 
    success: bool, 
    evidence: str
) -> Skill:
    """
    Uses the LLM to distill the execution result into a concise Skill.
    """
    outcome = "SUCCESS" if success else "FAILURE"
    prompt = (
        f"Analyze this CodeAgent execution ({outcome}).\n"
        f"Goal: {goal}\n"
        f"Evidence/Output:\n{evidence[:2000]}\n\n"
        f"Extract a SINGLE, concise 'Skill' or 'Insight' to help future agents avoid this failure or repeat this success.\n"
        f"Return ONLY a JSON object with these keys:\n"
        f"- category: One of [PyTorch, NumPy, Syntax, Logic, API, General]\n"
        f"- pattern: A short trigger keyword/phrase (e.g. 'conv2d', 'plot', 'json.load')\n"
        f"- insight: A concise rule (max 15 words). E.g. 'Use .detach().cpu() before plotting tensors.'\n"
    )
    
    try:
        # Use valid messages format for complete_with_continuation
        messages = [
            {"role": "system", "content": "You are an expert developer extracting coding insights."},
            {"role": "user", "content": prompt}
        ]
        
        # Use the robust completion helper
        content = complete_with_continuation(
            client, model, messages, 
            max_output_tokens=200,
            model_max_context=4000
        )
        
        # Strip markdown fences if present
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```json\s*|```$", "", content).strip()
            
        # JSON extraction heuristic (find first { and last })
        json_start = content.find('{')
        json_end = content.rfind('}')
        if json_start != -1 and json_end != -1:
             content = content[json_start:json_end+1]

        data = json.loads(content)
        return Skill(
            category=data.get("category", "General"),
            pattern=data.get("pattern", "general"),
            insight=data.get("insight", "Always check outputs."),
            evidence=evidence[:500],
            created_at=now_stamp()
        )
    except Exception as e:
        console.print(f"[yellow]Failed to extract insight: {e}[/yellow]")
        console.print(f"[dim]Raw content: {content[:200]}...[/dim]")
        # Fallback
        return Skill(
            category="General",
            pattern="general",
            insight=f"Review output for {outcome} details.",
            evidence=evidence[:500],
            created_at=now_stamp()
        )

def save_skill(config: AgentConfig, goal: str, notes: str, success: bool, evidence: str):
    """Save the session outcome to the SkillDB (new structured format)."""
    # Only save if there's meaningful evidence
    if not evidence.strip():
        return

    # Use a unified skills file for v2
    skill_file = config.agent_dir / "skilldb" / "skills.jsonl"
    
    # 1. Extract Insight
    console.print("[cyan]Extracting experience insight...[/cyan]")
    skill = extract_skill_insight(config.client, config.model, goal, success, evidence)
    
    # 2. Load existing to deduplicate
    current_skills = []
    if skill_file.exists():
        for line in skill_file.read_text(errors="ignore").splitlines():
            try: current_skills.append(json.loads(line))
            except: pass
            
    # 3. Check for duplicates (same insight + category)
    found = False
    for existing in current_skills:
        if (existing.get("category") == skill.category and 
            existing.get("insight") == skill.insight):
            existing["count"] = existing.get("count", 1) + 1
            existing["evidence"] = skill.evidence # Update with latest evidence
            existing["created_at"] = now_stamp()
            found = True
            console.print(f"[green]Updated existing skill: [{skill.category}] {skill.insight}[/green]")
            break
            
    if not found:
        current_skills.append(asdict(skill))
        console.print(f"[green]Saved new skill: [{skill.category}] {skill.insight}[/green]")
        
    # 4. Write back
    with open(skill_file, "w", encoding="utf-8") as f:
        for s in current_skills:
            f.write(json.dumps(s) + "\n")

class PromptRegistry:
    """
    Centralized manager for all LLM prompts.
    Optimized to reduce token waste by removing redundant git context.
    """

    SYSTEM = (
        "You are an advanced AI coding agent. Your ONLY job is to produce file changes.\n"
        "\n"
        "## Output Format (STRICT)\n"
        "You MUST output in ONE of these two formats per response. Never mix them.\n"
        "\n"
        "### Format A: Unified Diff (For small edits)\n"
        "1. Start with a brief `## Reasoning` section.\n"
        "2. Then output `## Action` followed by a SINGLE fenced diff code block.\n"
        "3. Each file diff starts with `diff --git a/<path> b/<path>`.\n"
        "4. For NEW files use `--- /dev/null` and `+++ b/<path>`.\n"
        "5. Make sure hunk line-counts are correct (@@ -X,Y +A,B @@).\n"
        "6. Do NOT put prose between diffs inside the block.\n"
        "\n"
        "### Format B: WRITE_FILE (For new files or full rewrites)\n"
        "Use when creating new files or when diffs are too complex.\n"
        "\n"
        "WRITE_FILE: path/to/file.py\n"
        "<<<CONTENT\n"
        "... file content here ...\n"
        "CONTENT>>>\n"
        "\n"
        "## Rules\n"
        "- NEVER embed triple-backtick fences inside a diff block.\n"
        "- NEVER mix Format A and Format B in the same response.\n"
        "- If output will be very long, prefer Format B (WRITE_FILE) to avoid truncation.\n"
        "- Always include `Verification: <command>` on its own line if you know how to verify.\n"
        "\n"
        "## Teacher Guidelines (CRITICAL)\n"
        "If provided, you MUST follow the language-specific guidelines in the User Prompt.\n"
    )

    @staticmethod
    def format_task(
        goal: str,
        allowlist: List[str],
        context_files: List[str],
        notes: str,
        skills: str,
        max_context: int,
        max_output: int = 4096,
    ) -> str:
        """
        Builds the main Turn Prompt.
        Optimized: Removed 'Repo Snapshot' (git status/diff) to save tokens.
        
        Prioritizes context usage:
          1. Essential Instructions & Goal (Base)
          2. File Contents (Critical Context)
          3. Directory Tree (Navigation Context - if space permits)
        """
        allow_txt = "\n".join(f"- {p}" for p in allowlist) if allowlist else "- (none)"

        # Detect if ALL files are new
        all_new_files = all(not Path(f).exists() for f in allowlist) if allowlist else False

        # Suggest WRITE_FILE for new or multi-file tasks
        format_hint = ""
        if (allowlist and len(allowlist) > 1) or all_new_files:
            format_hint = (
                "\n> **IMPORTANT**: Use **Format B (WRITE_FILE)** to create all files. "
                "This avoids diff truncation issues and is more reliable for new files.\n"
            )
        
        # Get current relative context
        cwd = Path.cwd().name
        
        # Explicit Workspace Instruction
        workspace_block = (
            f"## Workspace Context\n"
            f"You are working in the directory: `./` (inside `{cwd}/`)\n"
            f"Use ONLY relative paths (e.g. `task.py` or `src/utils.py`).\n"
            f"DO NOT use absolute paths (e.g. `/home/user/...`).\n"
        )

        base_md = (
            f"# Turn Prompt\n\n"
            f"## Goal\n{goal}\n\n"
            f"{workspace_block}\n"  # <--- Added here
            f"## Target Files (Allowlist)\n{allow_txt}\n"
            f"{format_hint}\n"
            f"{skills if skills else ''}\n"
            f"## Constraints / Teacher Guidelines\n"
            f"{notes.strip() if notes.strip() else '(none)'}\n\n"
            f"## Output Contract\n"
            f"1. Return changes using EITHER Format A (Diff) OR Format B (WRITE_FILE).\n"
            f"2. ALL files in the Target Files list must be addressed.\n"
            f"3. (Optional) Include: \"Verification: <command>\" before the changes.\n"
        )

        # --- Token Budgeting ---
        safety_margin = 1000
        usable_context = max_context - max_output - safety_margin
        used_tokens = estimate_tokens(base_md) + estimate_tokens(PromptRegistry.SYSTEM)
        remaining = usable_context - used_tokens

        if remaining < 500:
            console.print("[red]Critical Warning: Goal + Constraints exceed context limit![/red]")
            base_md += "\n> **CRITICAL**: Input too long. Context truncated.\n"
            return base_md

        if remaining < 2000:
            base_md += "\n> **OUTPUT HINT**: Context budget is tight. Use WRITE_FILE format and keep code concise.\n"

        context_sections = []

        # --- Priority 1: File Contents (The most important context) ---
        # Ensure allowlist files come first
        priority_files = list(dict.fromkeys(list(allowlist) + list(context_files)))
        files_md = ""
        
        for f in priority_files:
            content = read_file(str(f))
            if not content or content.startswith("[MISSING FILE]"):
                continue
            
            # Smart truncation: prioritize seeing start/end of large files if needed
            # But for now, simple truncation
            if estimate_tokens(content) > 8000:
                content = truncate_to_tokens(content, 8000)
                
            file_block = f"## File: {f}\n```python\n{content}\n```\n"
            block_cost = estimate_tokens(file_block)
            
            if block_cost < remaining:
                files_md += file_block
                remaining -= block_cost
            else:
                files_md += f"## File: {f}\n[Content Omitted - Context Limit Reached]\n"
        
        if files_md:
            context_sections.append(files_md)

        # --- Priority 2: Directory Tree (Navigation context) ---
        # Only include if we have a healthy buffer (e.g. >500 tokens)
        if not all_new_files and remaining > 500:
            tree = top_level_tree()
            if estimate_tokens(tree) < remaining:
                context_sections.append(f"### File Tree\n{tree}\n")

        if context_sections:
            base_md += "\n## Context\n" + "\n".join(context_sections)

        return base_md

    @staticmethod
    def format_bugfix(file_path: str, error_output: str, original_goal: str = "") -> str:
        """
        Focused bug-fix prompt. Forces WRITE_FILE output.
        """
        content = read_file(str(file_path))
        if not content:
            content = "[FILE NOT FOUND]"

        return (
            f"# Bug Fix Required\n\n"
            f"## Original Goal\n{original_goal if original_goal else '(see previous context)'}\n\n"
            f"## Current File: {file_path}\n```python\n{content}\n```\n\n"
            f"## Error Output\n```\n{error_output[-3000:]}\n```\n\n"
            f"## STRICT Instructions\n"
            f"1. Analyze the Traceback to find the failing function.\n"
            f"2. Fix the specific error shown.\n"
            f"3. **CRITICAL: Scan the rest of that function for similar issues.**\n"
            f"   (e.g., if you change a variable from Tensor to Numpy, ensure ALL subsequent usages handle Numpy).\n"
            f"4. Output the COMPLETE corrected file using WRITE_FILE format.\n"
            f"5. Do NOT use diffs.\n"
            f"6. Output EXACTLY one WRITE_FILE block, nothing else after it.\n\n"
            f"WRITE_FILE: {file_path}\n"
            f"<<<CONTENT\n"
            f"... your complete corrected file here ...\n"
            f"CONTENT>>>\n"
        )

    @staticmethod
    def format_fix_diff(file_path: str, code_content: str, error_log: str, teacher_guidelines: str = "") -> str:
        """
        Prompt for Strategy 1: Quick Fix via Diff.
        """
        return (
            f"# Bug Fix Required (Diff Strategy)\n\n"
            f"The previous code for `{file_path}` failed verification.\n\n"
            f"## Error Output\n```\n{error_log[-3000:]}\n```\n\n"
            f"## Instructions\n"
            f"1. **Analyze**: Look at the error and the code below.\n"
            f"2. **Scope**: Fix ONLY the specific error.\n"
            f"3. **Consistency**: Check the *entire function* for related issues.\n"
            f"4. **Output**: Use **Format A (Unified Diff)**.\n"
            f"{teacher_guidelines}\n\n"
            f"## Current Code: {file_path}\n```python\n{code_content}\n```\n"
        )

    @staticmethod
    def format_fix_rewrite(file_path: str, current_code: str, error_history: str, teacher_guidelines: str = "") -> str:
        """
        Prompt for Strategy 2: Full Rewrite.
        Ensures the model sees the broken code so it can recover logic.
        """
        return (
            f"# Rewrite Required (Fresh Start)\n\n"
            f"Diff-based fixes have failed. We need a clean rewrite of `{file_path}`.\n\n"
            f"## Context: Current File Content (Broken)\n"
            f"```python\n{current_code}\n```\n\n"
            f"## Failure History\n```\n{error_history[-4000:]}\n```\n\n"
            f"## Instructions\n"
            f"1. **Recover**: Use the logic from the 'Current File' above, but fix the errors.\n"
            f"2. **Format**: Output the **COMPLETE** file using **Format B (WRITE_FILE)**.\n"
            f"3. **Constraint**: Do NOT use diffs. Do NOT use placeholders.\n"
            f"4. **Completeness**: You must output every single line of code.\n"
            f"{teacher_guidelines}\n\n"
            f"WRITE_FILE: {file_path}\n"
            f"<<<CONTENT\n"
            f"... complete fixed code ...\n"
            f"CONTENT>>>\n"
        )

def run_subtask_loop(
    config: AgentConfig,
    subtask: str,
    subtask_idx: int,
    allowlist: List[str],
    context_files: List[str],
    global_notes: str,
) -> bool:
    """
    Modular execution loop: Generate -> Verify -> Fix(Diff) -> Fix(Rewrite) -> Exit
    """
    skill_dir = config.agent_dir / "skilldb"
    turn_base = subtask_idx * 10
    console.rule(f"Executing Sub-task {subtask_idx+1}: {subtask}")

    def get_turn_dir(offset: int) -> Path:
        d = config.session_dir / f"{turn_base + offset:04d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # =========================================================================
    # PHASE 1: GENERATION
    # =========================================================================
    console.print("[bold cyan]Phase 1: Generating Code[/bold cyan]")
    turn_dir = get_turn_dir(0)
    
    # 1. Prepare Prompt
    # Enhanced Selection: Include global_notes (tech stack) in the query
    skill_query = f"{subtask}\n{global_notes}"
    inject = format_skill_injection(select_relevant_skills(skill_query, skill_dir))
    
    combined_guidelines = f"{global_notes}\n\n{inject}".strip()
    
    prompt_md = PromptRegistry.format_task(
        subtask, allowlist, context_files, global_notes, inject, 
        config.max_context, config.max_output
    )
    (turn_dir / "prompt.md").write_text(prompt_md, encoding="utf-8")

    # 2. Call Model
    console.print("[cyan]Generating solution...[/cyan]")
    content = complete_with_continuation(
        config.client, config.model,
        [{"role": "system", "content": PromptRegistry.SYSTEM}, 
         {"role": "user", "content": prompt_md}],
        max_output_tokens=config.max_output,
        model_max_context=config.model_max_context
    )
    (turn_dir / "response.md").write_text(content, encoding="utf-8")

    # 3. Detect Modified Files (Critical for Verification)
    # We parse the output to see what files are being touched
    modified_files = []
    
    # Scan for WRITE_FILE targets
    w_actions = extract_write_file_actions(content)
    for p, _ in w_actions: 
        modified_files.append(p)
    
    # Scan for Diff targets
    diff_text = extract_all_diffs(content)
    if diff_text:
        # Regex to find '+++ b/filename'
        diff_paths = re.findall(r'^\+\+\+ b/(.+)$', diff_text, re.MULTILINE)
        modified_files.extend(diff_paths)
    
    # Deduplicate
    modified_files = list(set(modified_files))

    # 4. Apply Code
    if not _try_apply_content(content, allowlist, turn_dir, config):
        # Retry logic for malformed WRITE_FILE could go here
        if "WRITE_FILE:" in content and "CONTENT" in content and not w_actions:
             console.print("[yellow]Detected malformed WRITE_FILE. Retrying...[/yellow]")
             # (Optional: Insert retry logic here)
        
        console.print("[red]Failed to apply generated code. Stopping.[/red]")
        return False
    console.print("[green]Code generated and applied.[/green]")

    # =========================================================================
    # PHASE 2: VERIFICATION & FIX
    # =========================================================================
    console.print("[bold cyan]Phase 2: Verification[/bold cyan]")
    
    # Check for explicit verification command in output
    auto_verify_cmd = None
    v_match = re.search(r"^Verification:\s*(.+)$", content, re.MULTILINE)
    if v_match:
        auto_verify_cmd = v_match.group(1).strip()
    
    # Determine the actual command to run
    # We pass 'modified_files' so we can default to 'python3 task.py' 
    # even if allowlist is empty.
    verify_cmd = _determine_verify_cmd(allowlist, modified_files, auto_verify_cmd, config)
    
    if not verify_cmd:
        console.print("[yellow]No verification command selected. Assuming success.[/yellow]")
        return True

    # --- Verification Loop ---
    error_history = []
    
    # Increase from 3 to 4 attempts (0=Initial, 1=Diff, 2=Rewrite, 3=Final Rewrite)
    MAX_RETRIES = 4
    
    for fix_stage in range(MAX_RETRIES): 
        
        console.print(f"[blue]Running verification (Stage {fix_stage})...[/blue]")
        code, out = run_shell(verify_cmd, cap=20000)
        
        # --- Auto-Install Missing Modules ---
        if code != 0:
            install_log = _handle_missing_modules(out)
            if install_log:
                out += install_log
                # Retry verification immediately
                console.print("[blue]Retrying verification after installation...[/blue]")
                code, out_retry = run_shell(verify_cmd, cap=20000)
                out += f"\n[Post-Install Verification]\n{out_retry}\n"
        
        (turn_dir / "verify_stdout.txt").write_text(out, encoding='utf-8')
        
        if code == 0:
            console.print(f"[green]Verification PASSED at Stage {fix_stage}![/green]")
            save_skill(config, subtask, global_notes, True, out)
            return True
        
        console.print(f"[red]Verification Failed (exit={code})[/red]")
        error_history.append(f"Stage {fix_stage} Output:\n{out}\n{'-'*20}")
        
        if fix_stage == MAX_RETRIES - 1:
            console.print("[bold red]All fix attempts failed. Exiting subtask.[/bold red]")
            save_skill(config, subtask, global_notes, False, out)
            return False

        # --- PREPARE FIX ---
        turn_dir = get_turn_dir(fix_stage + 1)
        
        # Pick the most relevant file to fix (heuristic: first python file modified)
        target_file = next((f for f in modified_files if str(f).endswith('.py')), None)
        if not target_file and allowlist:
             target_file = next((f for f in allowlist if str(f).endswith('.py')), allowlist[0])
        
        if not target_file:
            console.print("[red]Cannot identify a target file to fix. Aborting.[/red]")
            return False

        current_code = read_file(str(target_file))

        if fix_stage == 0:
            # STRATEGY 1: DIFF FIX
            console.print("[yellow]Attempting Fix 1: Targeted Diff...[/yellow]")
            fix_prompt = PromptRegistry.format_fix_diff(
                target_file, current_code, out,
                teacher_guidelines=combined_guidelines
            )
        else:
            # STRATEGY 2: FULL REWRITE
            console.print("[yellow]Attempting Fix 2: Full Rewrite (Accumulated Errors)...[/yellow]")
            full_history = "\n".join(error_history)
            # UPDATE: Pass current_code here
            fix_prompt = PromptRegistry.format_fix_rewrite(
                target_file, current_code, full_history,
                teacher_guidelines=combined_guidelines
            )
            #fix_prompt = PromptRegistry.format_fix_rewrite(target_file, full_history)

        (turn_dir / "prompt.md").write_text(fix_prompt, encoding="utf-8")

        # Generate Fix
        console.print("[cyan]Generating fix...[/cyan]")
        fix_content = complete_with_continuation(
            config.client, config.model,
            [{"role": "system", "content": PromptRegistry.SYSTEM}, 
             {"role": "user", "content": fix_prompt}],
            max_output_tokens=config.max_output,
            model_max_context=config.model_max_context
        )
        (turn_dir / "response.md").write_text(fix_content, encoding="utf-8")

        # Apply Fix
        if not _try_apply_content(fix_content, allowlist, turn_dir, config):
            console.print("[red]Failed to apply fix. Moving to next strategy...[/red]")
            # Loop continues to next stage (Rewrite) automatically
    
    return False


def detect_tech_stack(goal: str, allowlist: List[str]) -> str:
    """
    Heuristics to detect the tech stack (PyTorch, NumPy, etc.) 
    and return strict 'Teacher Guidelines' to prevent common runtime errors.
    Loads guidelines from SKILL_TEACHER (teacher.jsonl).
    """
    if not SKILL_TEACHER.exists():
        return ""

    goal_lower = goal.lower()
    combined_text = goal_lower + " ".join(str(x).lower() for x in allowlist)
    
    guidelines = []
    
    try:
        # Load teacher guidelines from JSONL
        # Format: {"category": "...", "triggers": [...], "header": "...", "guidelines": [...]}
        with open(SKILL_TEACHER, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    entry = json.loads(line)
                    triggers = entry.get("triggers", [])
                    
                    # Check if any trigger matches the context
                    if any(t.lower() in combined_text for t in triggers):
                        header = entry.get("header")
                        if header:
                            guidelines.append(header)
                        
                        rules = entry.get("guidelines", [])
                        guidelines.extend(rules)
                        guidelines.append("") # Spacer
                        
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        console.print(f"[yellow]Failed to load teacher guidelines: {e}[/yellow]")
        return ""

    if guidelines:
        return "\n".join(guidelines).strip()
    return ""
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
    
    parser.add_argument("--migrate-skills", action="store_true", help="Migrate legacy skill DB to new format")
    parser.add_argument("--artifacts-dir", help="Directory where the agent should save task artifacts (plots, models)")
    
    args = parser.parse_args()

    agent_dir = Path(args.agent_dir)
    ensure_dirs(agent_dir)

    # Initialize Client
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    
    # Migration Mode
    if args.migrate_skills:
        skill_dir = agent_dir / "skilldb"
        console.print("[bold yellow]Starting Skill DB Migration...[/bold yellow]")
        
        # Load legacy skills
        legacy_skills = []
        for kind, filename in [("success", "successes.jsonl"), ("failure", "failures.jsonl")]:
            path = skill_dir / filename
            if path.exists():
                for line in path.read_text(errors="ignore").splitlines():
                    if line.strip(): legacy_skills.append((kind == "success", json.loads(line)))
        
        console.print(f"Found {len(legacy_skills)} legacy records.")
        
        # Process each
        new_db = skill_dir / "skills.jsonl"
        for i, (success, obj) in enumerate(legacy_skills):
            console.print(f"[{i+1}/{len(legacy_skills)}] Extracting insight...")
            goal = obj.get("text", "").split("\n")[0].replace("Goal: ", "")
            evidence = obj.get("evidence", "")
            
            # Use the extraction logic
            skill = extract_skill_insight(client, args.model, goal, success, evidence)
            
            # Save (append to new DB)
            with open(new_db, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(skill)) + "\n")
                
        console.print("[green]Migration Complete![/green]")
        return

    # 1. Auto-detect model context
    detected_ctx = query_model_context_length(client, args.model)
    effective_ctx = detected_ctx if detected_ctx > 0 else args.max_context
    console.print(f"[dim]Effective context limit: {effective_ctx} tokens[/dim]")

    # 2. Setup Session
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
        title="mini-claude-code (Teacher-Enhanced)",
        style="cyan"
    ))

    # 3. Gather Inputs (Goal & Allowlist)
    goal = args.goal
    if not goal:
        goal = Prompt.ask("Goal").strip()

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
    
    # Default to allowlist empty -> "create whatever" (Handled in planner)

    # 4. Context Files
    context_files = list(dict.fromkeys(allowlist)) 
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

    # 5. User Notes + TEACHER INJECTION
    extra_notes = args.notes if args.notes else ""
    if not args.yes and not args.notes:
        extra_notes = Prompt.ask("Constraints / notes (optional)", default="").strip()

    # --- INJECT TEACHER GUIDELINES ---
    console.print("[dim]Scanning task for technical risks...[/dim]")
    teacher_guidelines = detect_tech_stack(goal, allowlist)
    if teacher_guidelines:
        console.print(Panel(teacher_guidelines, title="Teacher Guidelines Injected", style="yellow"))
        # Append to extra_notes so it persists through Planning AND Execution
        extra_notes = f"{extra_notes}\n\n{teacher_guidelines}"

    # --- ARTIFACTS DIR INJECTION ---
    if args.artifacts_dir:
        abs_artifacts = Path(args.artifacts_dir).resolve()
        abs_artifacts.mkdir(parents=True, exist_ok=True)
        artifact_instr = (
            f"\n\n**ARTIFACT MANAGMENT RULE**:\n"
            f"You MUST save ALL generated assets (plots, models, logs, images) to this directory:\n"
            f"`{abs_artifacts}`\n"
            f"Example: `plt.savefig('{abs_artifacts}/plot.png')`\n"
            f"DO NOT save to `./` or `output/` unless explicitly asked."
        )
        extra_notes += artifact_instr
        console.print(f"[cyan]Artifacts directory set: {abs_artifacts}[/cyan]")

    # Print Machine-Readable Log Path for Batch Coder
    print(f"[METADATA] LOG_PATH: {session_dir.resolve()}")

    # 6. Plan (Optimized: Skips LLM for single file tasks)
    # The 'extra_notes' now contains the Teacher Guidelines, so the planner sees them too!
    subtasks = plan_tasks(config, goal, extra_notes, allowlist)
    
    # 7. Execute
    success_count = 0
    for i, subtask in enumerate(subtasks):
        # We pass the same 'extra_notes' (with guidelines) to the subtask loop
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

"""

python CodeAgent//mini_claude_codev4.py --goal "Implement Univariate Linear Regression using ONLY PyTorch tensors. Do NOT use torch.nn, torch.optim, or autograd. Write everything in a single task.py file with a complete main() that trains, evaluates, and validates."

python CodeAgent/mini_claude_codev4.py --goal "Implement ML Task: SVM (Score Calibration + ROC/PR). Description: Calibrate decision scores; produce ROC/PR curves and AUC. Write a SINGLE self-contained Python file (task.py) with these functions: get_task_metadata, set_seed, get_device, make_dataloaders, build_model, train, evaluate, predict, save_artifacts."

python CodeAgent/mini_claude_codev4.py --api-key "myhpcvllmqwen123" --goal "Implement Multivariate Linear Regression using torch.autograd. Visualize training. Description: Calibrate decision scores; produce ROC/PR curves and AUC. Write a SINGLE self-contained Python file (task.py) with these functions: get_task_metadata, set_seed, get_device, make_dataloaders, build_model, train, evaluate, predict, save_artifacts."

"""