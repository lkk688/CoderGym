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
        
        # Post-processing checks
        
        # 1. Diff Artifact check (double safety)
        # If the path looks like a diff path (a/foo.py, b/foo.py), ignore it
        if filepath.startswith("a/") or filepath.startswith("b/") or filepath == "/dev/null":
            continue
            
        # 2. Content validation
        # If content is extremely short (< 5 chars), it's likely a parsing artifact or hallucination
        if len(content.strip()) < 5:
            continue
            
        results.append((filepath, content))
        
    return results
# def extract_write_file_actions(text: str) -> List[Tuple[str, str]]:
#     """
#     Extract WRITE_FILE actions with robust regex to handle LLM artifacts:
#     1. Merged lines (e.g., 'code...WRITE_FILE: path')
#     2. Missing closers (truncated output)
#     3. Typos in closers (CONTENT>> instead of CONTENT>>>)
#     4. Safe ignores (doesn't match diff removals like '-<<<CONTENT')
#     """
#     results = []
    
#     # Regex Breakdown:
#     # 1. (?:^|\n).*?WRITE_FILE: -> Start match anywhere on a line (consumes garbage prefix)
#     # 2. \s*(\S+)               -> Capture the filepath (stops at whitespace)
#     # 3. .*?\n                  -> Consume the rest of the header line
#     # 4. \s*<<<CONTENT\n        -> Match start tag. STRICTLY requires newline before and after.
#     #                              (Note: \s* matches spaces/tabs but NOT '-' or '+'. 
#     #                               This safely ignores diff contexts like '-<<<CONTENT')
#     # 5. (.*?)                  -> Capture content (DOTALL)
#     # 6. Terminator             -> Stops at CONTENT>>, new WRITE_FILE, diff header, or EOF.
    
#     pattern = re.compile(
#         r'(?:^|\n).*?WRITE_FILE:\s*(\S+).*?\n'   # Header (fuzzy start)
#         r'\s*<<<CONTENT\n'                       # Start Tag (strict structure)
#         r'(.*?)'                                 # Content Capture
#         r'(?:CONTENT>{2,3}|(?=\n.*?WRITE_FILE:)|(?=\ndiff --git)|$)', # Robust Terminator
#         re.DOTALL
#     )
    
#     for m in pattern.finditer(text):
#         filepath = m.group(1).strip()
#         content = m.group(2)
        
#         # Sanity Checks
#         # 1. Skip if filepath looks like a diff artifact
#         if filepath.startswith("a/") or filepath.startswith("b/") or filepath == "/dev/null":
#             continue
            
#         # 2. Skip if content is extremely short (likely parsing noise)
#         if len(content.strip()) < 1:
#             continue
            
#         results.append((filepath, content))
        
#     return results

# def extract_write_file_actions(text: str) -> List[Tuple[str, str]]:
#     """
#     Extract WRITE_FILE actions with robust regex to handle LLM typos
#     (e.g. missing > in closing tag, spacing issues).
#     """
#     results = []
    
#     # Strategy 1: Strict Match (Preferred)
#     # WRITE_FILE: path\n<<<CONTENT\n...\nCONTENT>>>
#     wf_pattern = re.compile(
#         r'WRITE_FILE:\s*(.+?)\s*\n<<<CONTENT\n(.*?)CONTENT>{2,3}', # Allow >> or >>>
#         re.DOTALL
#     )
#     for m in wf_pattern.finditer(text):
#         filepath = m.group(1).strip()
#         content = m.group(2)
#         if filepath and content:
#             results.append((filepath, content))
    
#     if results:
#         return results
        
#     # Strategy 2: Fallback for "forgot to close" or "weird spacing"
#     # Looks for <<<CONTENT and just takes everything until the next likely block or EOF
#     # This captures cases where the model stops generating or messes up the closer completely.
#     fallback_pattern = re.compile(
#         r'WRITE_FILE:\s*(.+?)\s*\n<<<CONTENT\n(.*?)(?:(?=\nWRITE_FILE:)|(?=\n#)|$)',
#         re.DOTALL
#     )
    
#     for m in fallback_pattern.finditer(text):
#         filepath = m.group(1).strip()
#         content = m.group(2)
        
#         # Clean up the end of content if it captured the broken closer
#         content = re.sub(r'CONTENT>+$', '', content).strip()
        
#         # Validate: Don't accept if it's too short (likely a hallucination)
#         if filepath and len(content) > 10:
#              # Sanity check: ensure it doesn't look like a diff
#             if "diff --git" not in content[:50]:
#                 results.append((filepath, content + "\n"))
    
#     return results

# def extract_write_file_actions(text: str) -> List[Tuple[str, str]]:
#     """
#     Extract WRITE_FILE actions from model output.
#     Supports multiple formats the model might use:
    
#     Format 1 (preferred): WRITE_FILE: path
#                           <<<CONTENT
#                           ... content ...
#                           CONTENT>>>
    
#     Format 2 (tool_call XML): <tool_call><function=write_file>
#                               <parameter=file_path>path</parameter>
#                               <parameter=content>...</parameter>
#                               </function></tool_call>
    
#     Format 3 (implicit): ## File: path/to/file.py
#                          ```python
#                          ... content ...
#                          ```
    
#     Returns list of (filepath, content) tuples.
#     """
#     results = []
    
#     # Format 1: WRITE_FILE: path\n<<<CONTENT\n...\nCONTENT>>>
#     wf_pattern = re.compile(
#         r'WRITE_FILE:\s*(.+?)\s*\n<<<CONTENT\n(.*?)CONTENT>>>',
#         re.DOTALL
#     )
#     for m in wf_pattern.finditer(text):
#         filepath = m.group(1).strip()
#         content = m.group(2)
#         if filepath and content:
#             results.append((filepath, content))
    
#     if results:
#         return results
    
#     # Format 2: <tool_call> XML style
#     tc_pattern = re.compile(
#         r'<(?:tool_call|function)[^>]*>.*?'
#         r'<parameter[= ]+(?:file_path|path)[^>]*>\s*(.+?)\s*</parameter>'
#         r'.*?<parameter[= ]+content[^>]*>\s*(.*?)\s*</parameter>',
#         re.DOTALL
#     )
#     for m in tc_pattern.finditer(text):
#         filepath = m.group(1).strip()
#         content = m.group(2)
#         if filepath and content:
#             results.append((filepath, content))
    
#     if results:
#         return results
    
#     # Format 3: Implicit file path + code block
#     # Look for patterns like:
#     #   # File: path/to/file.py  (or ## File: or ### or just path followed by ```)
#     implicit_pattern = re.compile(
#         r'(?:^#+\s*(?:File:\s*)?|^(?:Create|Writing|Output)\s+(?:file\s+)?)'
#         r'[`]*([^\n`]+?\.\w{1,5})[`]*\s*\n'
#         r'```\w*\n(.*?)```',
#         re.MULTILINE | re.DOTALL
#     )
#     for m in implicit_pattern.finditer(text):
#         filepath = m.group(1).strip().strip('`').strip()
#         content = m.group(2)
#         # Validate it looks like a real path
#         if filepath and '/' in filepath and content and len(content) > 10:
#             results.append((filepath, content))
    
#     return results


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

def apply_fuzzy_patch(file_path: Path, diff_content: str) -> bool:
    """
    Applies a Unified Diff with 'fuzzy' matching logic.
    1. Ignores line numbers (@@ -12,4 +12,5 @@).
    2. Matches context by stripping whitespace (ignoring indentation changes).
    3. Handles 'New File' creation via diff.
    """
    # 1. Handle New File Creation
    if "new file mode" in diff_content or "--- /dev/null" in diff_content:
         new_content = []
         for line in diff_content.splitlines():
             if line.startswith('+') and not line.startswith('+++'):
                 new_content.append(line[1:]) # Remove '+'
         
         # Sanity check: verify it's not just an empty file or metadata
         if len(new_content) > 0:
             file_path.parent.mkdir(parents=True, exist_ok=True)
             file_path.write_text("\n".join(new_content), encoding="utf-8")
             console.print(f"[green]Created new file from diff: {file_path}[/green]")
             return True
         return False

    if not file_path.exists():
        console.print(f"[red]Target file {file_path} not found for diff.[/red]")
        return False

    original_lines = file_path.read_text(encoding="utf-8").splitlines()
    # Work on a copy
    modified_lines = list(original_lines)
    
    # 2. Parse Hunks
    # Regex to split by @@ ... @@ header
    hunks = re.split(r'^@@\s.*?\s@@', diff_content, flags=re.MULTILINE)
    # The first part is the header (diff --git ...), skip it
    hunks = hunks[1:]
    
    if not hunks:
        console.print("[yellow]No hunks found in diff.[/yellow]")
        return False
        
    for hunk in hunks:
        hunk_lines = [l for l in hunk.splitlines() if l]
        if not hunk_lines:
            continue
            
        # 3. Identify the "Search Block" (Context + Removed lines)
        search_block = []
        replace_block = []
        
        for line in hunk_lines:
            if line.startswith(' '): # Context
                search_block.append(line[1:])
                replace_block.append(line[1:])
            elif line.startswith('-'): # Remove
                search_block.append(line[1:])
            elif line.startswith('+'): # Add
                replace_block.append(line[1:])
            # Ignore '\ No newline at end of file'
        
        if not search_block:
            continue

        # 4. Find where this block exists in the file
        # We try strict match first, then whitespace-insensitive match
        match_index = -1
        n_search = len(search_block)
        
        # Strategy A: Exact Match
        for i in range(len(modified_lines) - n_search + 1):
            if modified_lines[i : i+n_search] == search_block:
                match_index = i
                break
        
        # Strategy B: Fuzzy Match (strip whitespace)
        if match_index == -1:
            search_stripped = [l.strip() for l in search_block]
            for i in range(len(modified_lines) - n_search + 1):
                file_subset = modified_lines[i : i+n_search]
                file_stripped = [l.strip() for l in file_subset]
                if file_stripped == search_stripped:
                    match_index = i
                    # CAUTION: We matched loosely. We must be careful not to delete 
                    # wrong indentation, but typically replacing the whole block is safer.
                    break
        
        if match_index != -1:
            # 5. Apply the patch
            # Remove old lines
            del modified_lines[match_index : match_index + n_search]
            # Insert new lines
            for i, line in enumerate(replace_block):
                modified_lines.insert(match_index + i, line)
            console.print(f"[green]Applied hunk at line {match_index+1}[/green]")
        else:
            console.print(f"[red]Failed to find matching context for hunk:[/red]")
            console.print(Panel("\n".join(search_block[:5]) + "\n...", title="Expected Context (First 5 lines)"))
            return False # Fail the whole patch if one hunk fails (atomic apply)
            
    # Save result
    file_path.write_text("\n".join(modified_lines), encoding="utf-8")
    return True

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
            
            if is_inside_code:
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
# def complete_with_continuation(
#     client: OpenAI,
#     model: str,
#     messages: List[Dict[str, str]],
#     temperature: float = 0.2,
#     max_output_tokens: int = 4096,
#     model_max_context: int = 16384,
# ) -> str:
#     """
#     Calls the LLM. If finish_reason is 'length', appends the partial response
#     to messages and asks it to continue, stitching the results.
#     Improved: diff-aware continuation prompting.
    
#     Adaptively caps max_tokens based on input size to avoid context overflow.
#     """
#     full_content = ""
#     current_messages = list(messages)
    
#     max_loops = 5  # Increased from 3 for complex multi-file tasks
    
#     for i in range(max_loops):
#         console.print(f"[dim]Generation loop {i+1}/{max_loops}...[/dim]")
        
#         # Adaptive max_tokens: estimate input and cap output accordingly
#         input_text = "\n".join(m.get("content", "") for m in current_messages)
#         input_est = estimate_tokens(input_text)
#         safe_tokens = compute_safe_max_tokens(
#             input_tokens=input_est,
#             model_max_context=model_max_context,
#             desired_max_output=max_output_tokens
#         )
        
#         if safe_tokens < max_output_tokens:
#             console.print(f"[yellow]Adaptive max_tokens: {safe_tokens} "
#                           f"(input≈{input_est}, limit={model_max_context}, "
#                           f"requested={max_output_tokens})[/yellow]")
        
#         # Retry with backoff on API errors (including context overflow)
#         resp = None
#         for attempt in range(3):
#             try:
#                 resp = client.chat.completions.create(
#                     model=model,
#                     messages=current_messages,
#                     temperature=temperature,
#                     max_tokens=safe_tokens
#                 )
#                 break
#             except Exception as e:
#                 err_str = str(e)
#                 if 'max_tokens' in err_str or 'context length' in err_str or 'maximum context' in err_str:
#                     # Context overflow — reduce tokens further
#                     safe_tokens = max(1024, safe_tokens // 2)
#                     console.print(f"[red]Context overflow (attempt {attempt+1}). "
#                                   f"Retrying with max_tokens={safe_tokens}...[/red]")
#                     time.sleep(1)
#                     continue
#                 console.print(f"[red]LLM Call failed: {e}[/red]")
#                 if attempt < 2:
#                     time.sleep(2 ** attempt)
#                     continue
#                 break
        
#         if resp is None:
#             console.print(f"[red]All LLM retry attempts failed.[/red]")
#             break
            
#         choice = resp.choices[0]
#         console.print(f"[dim]Finish Reason: {choice.finish_reason}[/dim]")
#         content = choice.message.content or ""
#         full_content += content
        
#         if choice.finish_reason == "length":
#             console.print("[yellow]Output truncated (limit reached). Continuing...[/yellow]")
            
#             # Detect what kind of output we're in the middle of
#             # and craft appropriate continuation prompt
#             if content.rstrip().endswith('```'):
#                 # Cleanly ended a code block — can continue normally
#                 cont_prompt = (
#                     "Continue. If there are more files to create, "
#                     "continue with the next WRITE_FILE block or diff. "
#                     "Do not repeat already-generated content."
#                 )
#             elif 'WRITE_FILE:' in content or '<<<CONTENT' in content:
#                 cont_prompt = (
#                     "You were truncated mid-output. Continue EXACTLY where you left off. "
#                     "You were in the middle of a WRITE_FILE block. "
#                     "Continue the file content, then close with CONTENT>>> "
#                     "and continue with remaining files."
#                 )
#             elif 'diff --git' in content:
#                 cont_prompt = (
#                     "You were truncated mid-output. Continue EXACTLY where you left off. "
#                     "You were in the middle of a unified diff. "
#                     "Continue the diff hunks. Do NOT repeat diff headers already generated. "
#                     "Do NOT restart the response."
#                 )
#             else:
#                 cont_prompt = (
#                     "You were truncated. Continue exactly where you left off. "
#                     "Do not repeat previous content."
#                 )
            
#             current_messages.append({"role": "assistant", "content": content})
#             current_messages.append({"role": "user", "content": cont_prompt})
#         else:
#             break
            
#     return full_content


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
        m = re.search(r"(?:create|write|implement)\s+(\S+\.py)", goal, re.IGNORECASE)
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
    # Logic: If model says /Developer/src/main.py, we only care about src/main.py relative to us.
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
    3. WRITE_FILE (Full rewrite)
    4. Diff Extraction (Last resort reconstruction)
    """
    
    # --- Extract Diff once ---
    diff = extract_all_diffs(content)
    changes_applied = False
    
    # --- TRY FORMAT A: Unified Diff Strategies ---
    if diff:
        (turn_dir / "patch.diff").write_text(diff, encoding="utf-8")
        
        # Strategy 1: Strict Git Apply
        if is_git_repo():
            changes_applied = apply_patch_guarded(diff, turn_dir, auto_approve=config.auto_approve)
        else:
            console.print("[red]Not a git repo, skipping strict diff apply.[/red]")
        
        # Strategy 2: Fuzzy Patch
        if not changes_applied:
            console.print("[yellow]Strict apply failed. Attempting fuzzy patch...[/yellow]")
            file_diffs = re.split(r'(?=^diff --git )', diff, flags=re.MULTILINE)
            fuzzy_successes = 0
            fuzzy_total = 0
            
            for fd in file_diffs:
                if not fd.strip().startswith("diff --git"): continue
                fuzzy_total += 1
                
                # Extract raw path from header
                match = re.search(r'diff --git a/\S+ b/(\S+)', fd)
                if match:
                    raw_path = match.group(1)
                    
                    # Resolve Path
                    target_path = resolve_path(raw_path, allowlist)
                    
                    if target_path:
                        if target_path != Path(raw_path):
                            console.print(f"[dim]Redirecting '{raw_path}' -> '{target_path}'[/dim]")
                        
                        if apply_fuzzy_patch(target_path, fd):
                            fuzzy_successes += 1
                    else:
                        console.print(f"[red]Skipping diff for unresolved path: {raw_path}[/red]")
            
            # Mark success if at least one file was patched
            if fuzzy_successes > 0:
                changes_applied = True
                console.print(f"[green]Fuzzy patch applied ({fuzzy_successes}/{fuzzy_total} files).[/green]")

    # --- TRY FORMAT B: WRITE_FILE (Try if diffs failed or didn't exist) ---
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
                console.print("[green]Wrote new files extracted from diff.[/green]")
        else:
            console.print("[red]No new files to extract. Edit diffs cannot be safely applied as rewrites.[/red]")
    
    # --- Final Failure Check ---
    if not changes_applied:
        # Check if we missed a WRITE_FILE due to bad formatting (optional check)
        if "WRITE_FILE:" in content and "CONTENT" in content:
             console.print("[red]Potential malformed WRITE_FILE block detected but extraction failed.[/red]")
        
        if not diff and not extract_write_file_actions(content):
            console.print("[red]No valid diff or WRITE_FILE actions found in response.[/red]")
    
    return changes_applied
# def _try_apply_content(content: str, allowlist: List[str], turn_dir: Path, 
#                        config: AgentConfig) -> bool:
#     """
#     Try all methods to apply model output as file changes.
#     Order: git apply diff → WRITE_FILE → diff extraction.
    
#     Key insight: when model outputs BOTH diffs and WRITE_FILE,
#     WRITE_FILE is more reliable (full file, no patch issues).
#     So we try WRITE_FILE before diff extraction.
#     """
#     # --- TRY FORMAT A: Unified Diff ---
#     diff = extract_all_diffs(content)
#     changes_applied = False
    
#     if diff:
#         (turn_dir / "patch.diff").write_text(diff, encoding="utf-8")
#         if is_git_repo():
#             changes_applied = apply_patch_guarded(diff, turn_dir, auto_approve=config.auto_approve)
#         else:
#             console.print("[red]Not a git repo, skipping diff apply.[/red]")
    
#     # --- TRY FORMAT B: WRITE_FILE (try BEFORE diff extraction — more reliable) ---
#     if not changes_applied:
#         write_actions = extract_write_file_actions(content)
#         if write_actions:
#             console.print(f"[cyan]Found {len(write_actions)} WRITE_FILE action(s). Applying...[/cyan]")
#             changes_applied = apply_write_files(write_actions, allowlist, turn_dir)
#             if changes_applied:
#                 console.print("[green]Applied via WRITE_FILE.[/green]")
    
#     # --- TRY FORMAT A.5: Extract files from diff (last resort) ---
#     if not changes_applied and diff:
#         console.print("[yellow]Diff + WRITE_FILE failed. Extracting from diff lines...[/yellow]")
#         diff_files = extract_files_from_diff(diff)
#         if diff_files:
#             changes_applied = apply_write_files(diff_files, allowlist, turn_dir)
#             if changes_applied:
#                 console.print("[green]Wrote files extracted from diff.[/green]")
    
#     if not changes_applied and not diff:
#         write_actions = extract_write_file_actions(content)
#         if not write_actions:
#             console.print("[red]No valid diff or WRITE_FILE actions found.[/red]")
    
#     return changes_applied


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
# def _determine_verify_cmd(allowlist: List[str], auto_verify_cmd: str, 
#                           config: AgentConfig) -> str:
#     """
#     Determine the verification command to run.
#     """
#     cmd_to_run = ""
#     if auto_verify_cmd:
#         if config.auto_approve:
#             cmd_to_run = auto_verify_cmd
#         elif Confirm.ask(f"Run parsed verification: [bold]{auto_verify_cmd}[/bold]?"):
#             cmd_to_run = auto_verify_cmd
    
#     # Auto-detect: if no verification command, infer from allowlist
#     if not cmd_to_run and config.auto_approve:
#         py_files = [str(f) for f in allowlist if str(f).endswith('.py')]
#         if py_files:
#             cmd_to_run = f"python3 {py_files[0]}"
#             console.print(f"[cyan]Auto-detected verification: {cmd_to_run}[/cyan]")
    
#     if not cmd_to_run and not config.auto_approve:
#         if Confirm.ask("Run verification command?"):
#             cmd_to_run = Prompt.ask("Command", default="")
    
#     return cmd_to_run

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

def save_skill(config: AgentConfig, goal: str, notes: str, success: bool, evidence: str):
    """Save the session outcome to the SkillDB."""
    kind = "success" if success else "failure"
    filename = "successes.jsonl" if success else "failures.jsonl"
    
    skill = {
        "tag": "execution",
        "kind": kind,
        "text": f"Goal: {goal[:100]}",
        "pattern": goal.split()[0].lower() if goal else "general",
        "evidence": evidence[-1000:] if evidence else "No output",
        "created_at": now_stamp()
    }
    write_jsonl(config.agent_dir / "skilldb" / filename, skill)

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
    def format_fix_diff(file_path: str, code_content: str, error_log: str) -> str:
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
            f"4. **Output**: Use **Format A (Unified Diff)**.\n\n"
            f"## Current Code: {file_path}\n```python\n{code_content}\n```\n"
        )

    @staticmethod
    def format_fix_rewrite(file_path: str, current_code: str, error_history: str) -> str:
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
            f"4. **Completeness**: You must output every single line of code.\n\n"
            f"WRITE_FILE: {file_path}\n"
            f"<<<CONTENT\n"
            f"... complete fixed code ...\n"
            f"CONTENT>>>\n"
        )
# class PromptRegistry:
#     """
#     Centralized manager for all LLM prompts.
#     Handles token budgeting, context prioritization, and format enforcement.
#     """

#     # Use a raw string + concatenation to avoid f-string/backtick issues
#     SYSTEM = (
#         "You are an advanced AI coding agent. Your ONLY job is to produce file changes.\n"
#         "\n"
#         "## Output Format (STRICT)\n"
#         "You MUST output in ONE of these two formats per response. Never mix them.\n"
#         "\n"
#         "### Format A: Unified Diff (For small edits)\n"
#         "1. Start with a brief `## Reasoning` section (plain text, keep short).\n"
#         "2. Then output `## Action` followed by a SINGLE fenced diff code block.\n"
#         "3. Each file diff starts with `diff --git a/<path> b/<path>`.\n"
#         "4. For NEW files use `--- /dev/null` and `+++ b/<path>`.\n"
#         "5. Make sure hunk line-counts are correct (@@ -X,Y +A,B @@).\n"
#         "6. Do NOT put prose between diffs inside the block.\n"
#         "\n"
#         "### Format B: WRITE_FILE (For new files or full rewrites)\n"
#         "Use when creating new files or when diffs are too complex.\n"
#         "\n"
#         "WRITE_FILE: path/to/file.py\n"
#         "<<<CONTENT\n"
#         "... file content here ...\n"
#         "CONTENT>>>\n"
#         "\n"
#         "## Rules\n"
#         "- NEVER embed triple-backtick fences inside a diff block.\n"
#         "- NEVER mix Format A and Format B in the same response.\n"
#         "- If output will be very long, prefer Format B (WRITE_FILE) to avoid truncation.\n"
#         "- Always include `Verification: <command>` on its own line if you know how to verify.\n"
#         "\n"
#         "## Teacher Guidelines (CRITICAL)\n"
#         "If provided, you MUST follow the language-specific guidelines in the User Prompt.\n"
#     )

#     @staticmethod
#     def format_task(
#         goal: str,
#         allowlist: List[str],
#         context_files: List[str],
#         notes: str,
#         skills: str,
#         max_context: int,
#         max_output: int = 4096,
#     ) -> str:
#         """
#         Builds the main Turn Prompt with dynamic context management.
#         Prioritizes context usage:
#           1. Essential Instructions & Goal (Base)
#           2. Git Status/Diff (for existing repos)
#           3. File Contents (read-only context + allowlist)
#           4. Directory Tree (if space permits)
#         """
#         allow_txt = "\n".join(f"- {p}" for p in allowlist) if allowlist else "- (none)"

#         # Detect if ALL files are new (don't exist yet)
#         all_new_files = all(not Path(f).exists() for f in allowlist) if allowlist else False

#         # Suggest WRITE_FILE for new or multi-file tasks
#         format_hint = ""
#         if len(allowlist) > 1 or all_new_files:
#             format_hint = (
#                 "\n> **IMPORTANT**: Use **Format B (WRITE_FILE)** to create all files. "
#                 "This avoids diff truncation issues and is more reliable for new files.\n"
#             )

#         base_md = (
#             f"# Turn Prompt\n\n"
#             f"## Goal\n{goal}\n\n"
#             f"## Target Files (Allowlist)\n{allow_txt}\n"
#             f"{format_hint}\n"
#             f"{skills if skills else ''}\n"
#             f"## Constraints / Teacher Guidelines\n"
#             f"{notes.strip() if notes.strip() else '(none)'}\n\n"
#             f"## Output Contract\n"
#             f"1. Return changes using EITHER Format A (Diff) OR Format B (WRITE_FILE).\n"
#             f"2. ALL files in the Target Files list must be addressed.\n"
#             f"3. (Optional) Include: \"Verification: <command>\" before the changes.\n"
#         )

#         # --- Token Budgeting ---
#         safety_margin = 1000
#         usable_context = max_context - max_output - safety_margin
#         used_tokens = estimate_tokens(base_md) + estimate_tokens(PromptRegistry.SYSTEM)
#         remaining = usable_context - used_tokens

#         if remaining < 500:
#             console.print("[red]Critical Warning: Goal + Constraints exceed context limit![/red]")
#             base_md += "\n> **CRITICAL**: Input too long. Context truncated.\n"
#             return base_md

#         if remaining < 2000:
#             base_md += "\n> **OUTPUT HINT**: Context budget is tight. Use WRITE_FILE format and keep code concise.\n"

#         context_sections = []

#         # --- Priority 1: Git Status/Diff (skip for new files) ---
#         if not all_new_files and is_git_repo():
#             st = git_status()
#             df = git_diff()
#             git_block = ""
#             if st:
#                 rel_lines = [line for line in st.split('\n')
#                              if line.strip().startswith('##') or any(str(a) in line for a in allowlist)]
#                 if rel_lines:
#                     status_text = '\n'.join(rel_lines)
#                     git_block += f"### git status (relevant)\n```\n{status_text}\n```\n"
#             if df.strip():
#                 if estimate_tokens(df) > 2000:
#                     df = truncate_to_tokens(df, 2000)
#                 git_block += f"### git diff\n```diff\n{df}\n```\n"
#             git_cost = estimate_tokens(git_block)
#             if git_block and git_cost < remaining:
#                 context_sections.append("## Repo Snapshot\n" + git_block)
#                 remaining -= git_cost
#         elif all_new_files:
#             console.print("[dim]Skipping git context (new files mode)[/dim]")

#         # --- Priority 2: File Contents ---
#         # Ensure allowlist files come first
#         priority_files = list(dict.fromkeys(list(allowlist) + list(context_files)))
#         files_md = ""
#         for f in priority_files:
#             content = read_file(str(f))
#             if not content or content.startswith("[MISSING FILE]"):
#                 continue
#             if estimate_tokens(content) > 6000:
#                 content = truncate_to_tokens(content, 6000)
#             file_block = f"## File: {f}\n```python\n{content}\n```\n"
#             block_cost = estimate_tokens(file_block)
#             if block_cost < remaining:
#                 files_md += file_block
#                 remaining -= block_cost
#             else:
#                 files_md += f"## File: {f}\n[Content Omitted - Context Limit Reached]\n"
#         if files_md:
#             context_sections.append(files_md)

#         # --- Priority 3: Directory Tree (least important) ---
#         if not all_new_files and remaining > 500:
#             tree = top_level_tree()
#             if estimate_tokens(tree) < remaining:
#                 context_sections.append(f"### File Tree\n{tree}\n")

#         if context_sections:
#             base_md += "\n## Context\n" + "\n".join(context_sections)

#         return base_md

#     @staticmethod
#     def format_bugfix(file_path: str, error_output: str, original_goal: str = "") -> str:
#         """
#         Focused bug-fix prompt. Much shorter than format_task.
#         Forces WRITE_FILE output to avoid broken diffs in fix responses.
#         """
#         content = read_file(str(file_path))
#         if not content:
#             content = "[FILE NOT FOUND]"

#         return (
#             f"# Bug Fix Required\n\n"
#             f"## Original Goal\n{original_goal if original_goal else '(see previous context)'}\n\n"
#             f"## Current File: {file_path}\n```python\n{content}\n```\n\n"
#             f"## Error Output\n```\n{error_output[-3000:]}\n```\n\n"
#             f"## STRICT Instructions\n"
#             f"1. Analyze the Traceback to find the failing function.\n"
#             f"2. Fix the specific error shown.\n"
#             f"3. **CRITICAL: Scan the rest of that function for similar issues.**\n"
#             f"   (e.g., if you change a variable from Tensor to Numpy, ensure ALL subsequent usages handle Numpy).\n"
#             f"4. Output the COMPLETE corrected file using WRITE_FILE format.\n"
#             f"5. Do NOT use diffs. Do NOT include reasoning diff examples.\n"
#             f"6. Output EXACTLY one WRITE_FILE block, nothing else after it.\n\n"
#             f"WRITE_FILE: {file_path}\n"
#             f"<<<CONTENT\n"
#             f"... your complete corrected file here ...\n"
#             f"CONTENT>>>\n"
#         )

#     @staticmethod
#     def format_fix_diff(file_path: str, code_content: str, error_log: str) -> str:
#         """
#         Prompt for Strategy 1: Quick Fix via Diff.
#         Focuses on specific error location.
#         """
#         return (
#             f"# Bug Fix Required (Diff Strategy)\n\n"
#             f"The previous code for `{file_path}` failed verification.\n\n"
#             f"## Error Output\n```\n{error_log[-3000:]}\n```\n\n"
#             f"## Instructions\n"
#             f"1. **Analyze**: Look at the error and the code below.\n"
#             f"2. **Scope**: Fix ONLY the specific error.\n"
#             f"3. **Consistency**: Check the *entire function* for related issues "
#             f"(e.g. if changing variable type, update all usages).\n"
#             f"4. **Output**: Use **Format A (Unified Diff)**.\n\n"
#             f"## Current Code: {file_path}\n```python\n{code_content}\n```\n"
#         )

#     @staticmethod
#     def format_fix_rewrite(file_path: str, error_history: str) -> str:
#         """
#         Prompt for Strategy 2: Full Rewrite.
#         Used when diffs fail repeatedly. Force clean slate.
#         """
#         return (
#             f"# Rewrite Required (Fresh Start)\n\n"
#             f"Diff-based fixes have failed multiple times. "
#             f"We need a clean rewrite of `{file_path}`.\n\n"
#             f"## Failure History\n```\n{error_history[-4000:]}\n```\n\n"
#             f"## Instructions\n"
#             f"1. **Format**: Output the **COMPLETE** file using **Format B (WRITE_FILE)**.\n"
#             f"2. **Constraint**: Do NOT use diffs.\n"
#             f"3. **Requirement**: Ensure ALL errors listed above are resolved.\n"
#             f"4. **Completeness**: Do not use placeholders. Output every line.\n\n"
#             f"WRITE_FILE: {file_path}\n"
#             f"<<<CONTENT\n"
#             f"... your complete corrected file here ...\n"
#             f"CONTENT>>>\n"
#         )


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
    inject = format_skill_injection(select_relevant_skills(subtask, skill_dir))
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
    
    for fix_stage in range(3): # 0=Initial, 1=Diff Fix, 2=Rewrite Fix
        
        console.print(f"[blue]Running verification (Stage {fix_stage})...[/blue]")
        code, out = run_shell(verify_cmd, cap=20000)
        (turn_dir / "verify_stdout.txt").write_text(out, encoding='utf-8')
        
        if code == 0:
            console.print(f"[green]Verification PASSED at Stage {fix_stage}![/green]")
            save_skill(config, subtask, global_notes, True, out)
            return True
        
        console.print(f"[red]Verification Failed (exit={code})[/red]")
        error_history.append(f"Stage {fix_stage} Output:\n{out}\n{'-'*20}")
        
        if fix_stage == 2:
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
            fix_prompt = PromptRegistry.format_fix_diff(target_file, current_code, out)
        else:
            # STRATEGY 2: FULL REWRITE
            console.print("[yellow]Attempting Fix 2: Full Rewrite (Accumulated Errors)...[/yellow]")
            full_history = "\n".join(error_history)
            # UPDATE: Pass current_code here
            fix_prompt = PromptRegistry.format_fix_rewrite(target_file, current_code, full_history)
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

# def run_subtask_loop(
#     config: AgentConfig,
#     subtask: str,
#     subtask_idx: int,
#     allowlist: List[str],
#     context_files: List[str],
#     global_notes: str,
# ) -> bool:
#     """
#     Modular execution loop: Generate -> Verify -> Fix(Diff) -> Fix(Rewrite) -> Exit
    
#     Flow:
#       1. Generate code (single attempt with continuation for long output)
#       2. Verify (run test/check command)
#       3. If fail: Fix 1 — targeted diff fix
#       4. If fail: Fix 2 — full file rewrite
#       5. If fail: Stop
#     """
#     skill_dir = config.agent_dir / "skilldb"
#     turn_base = subtask_idx * 10
#     console.rule(f"Executing Sub-task {subtask_idx+1}: {subtask}")

#     # Helper to get turn directory
#     def get_turn_dir(offset: int) -> Path:
#         d = config.session_dir / f"{turn_base + offset:04d}"
#         d.mkdir(parents=True, exist_ok=True)
#         return d

#     # =========================================================================
#     # PHASE 1: GENERATION
#     # =========================================================================
#     console.print("[bold cyan]Phase 1: Generating Code[/bold cyan]")
#     turn_dir = get_turn_dir(0)
    
#     # 1. Prepare Prompt (format_task handles file context + token budgeting)
#     inject = format_skill_injection(select_relevant_skills(subtask, skill_dir))
#     prompt_md = PromptRegistry.format_task(
#         goal=subtask,
#         allowlist=allowlist,
#         context_files=context_files,
#         notes=global_notes,
#         skills=inject,
#         max_context=config.model_max_context or config.max_context,
#         max_output=config.max_output,
#     )

#     (turn_dir / "prompt.md").write_text(prompt_md, encoding="utf-8")

#     # 2. Call Model (Internal loop handles truncation/continuation)
#     console.print("[cyan]Generating solution...[/cyan]")
#     content = complete_with_continuation(
#         config.client, config.model,
#         [{"role": "system", "content": PromptRegistry.SYSTEM}, 
#          {"role": "user", "content": prompt_md}],
#         max_output_tokens=config.max_output,
#         model_max_context=config.model_max_context
#     )
#     (turn_dir / "response.md").write_text(content, encoding="utf-8")

#     # 3. Apply
#     if not _try_apply_content(content, allowlist, turn_dir, config):
#         console.print("[red]Failed to apply generated code. Stopping.[/red]")
#         return False
#     console.print("[green]Code generated and applied.[/green]")

#     # =========================================================================
#     # PHASE 2: VERIFICATION & FIX
#     # =========================================================================
#     console.print("[bold cyan]Phase 2: Verification[/bold cyan]")
    
#     # Determine Verify Command
#     # If no explicit command found, default to running the python file (main block)
#     verify_cmd = None
#     v_match = re.search(r"^Verification:\s*(.+)$", content, re.MULTILINE)
#     if v_match:
#         verify_cmd = v_match.group(1).strip()
    
#     verify_cmd = _determine_verify_cmd(allowlist, verify_cmd, config)
    
#     # Fallback: Just run the file if it's python
#     if not verify_cmd:
#         py_files = [f for f in allowlist if str(f).endswith('.py')]
#         if py_files:
#             verify_cmd = f"python3 {py_files[0]}"
#             console.print(f"[dim]No verify cmd found. Defaulting to: {verify_cmd}[/dim]")

#     if not verify_cmd:
#         console.print("[yellow]No verification possible. Assuming success.[/yellow]")
#         return True

#     # --- Verification Loop ---
#     # Attempt 0: Initial Verification
#     # Attempt 1: Fix using Diff
#     # Attempt 2: Fix using Full Rewrite
    
#     error_history = []
    
#     for fix_stage in range(3): # 0=Initial, 1=Diff Fix, 2=Rewrite Fix
        
#         # Run Verification
#         console.print(f"[blue]Running verification (Stage {fix_stage})...[/blue]")
#         code, out = run_shell(verify_cmd, cap=20000)
        
#         if code == 0:
#             console.print(f"[green]Verification PASSED at Stage {fix_stage}![/green]")
#             return True
        
#         console.print(f"[red]Verification Failed (exit={code})[/red]")
#         error_history.append(f"Stage {fix_stage} Output:\n{out}\n{'-'*20}")
        
#         # If we just failed the final rewrite stage, give up.
#         if fix_stage == 2:
#             console.print("[bold red]All fix attempts (Diff & Rewrite) failed. Exiting subtask.[/bold red]")
#             return False

#         # --- PREPARE FIX ---
#         turn_dir = get_turn_dir(fix_stage + 1)
#         target_file = next((f for f in allowlist if str(f).endswith('.py')), allowlist[0])
#         current_code = read_file(str(target_file))

#         if fix_stage == 0:
#             # STRATEGY 1: DIFF FIX
#             console.print("[yellow]Attempting Fix 1: Targeted Diff...[/yellow]")
#             fix_prompt = PromptRegistry.format_fix_diff(target_file, current_code, out)
#         else:
#             # STRATEGY 2: FULL REWRITE
#             console.print("[yellow]Attempting Fix 2: Full Rewrite (Accumulated Errors)...[/yellow]")
#             full_history = "\n".join(error_history)
#             fix_prompt = PromptRegistry.format_fix_rewrite(target_file, full_history)

#         (turn_dir / "prompt.md").write_text(fix_prompt, encoding="utf-8")

#         # Generate Fix
#         fix_content = complete_with_continuation(
#             config.client, config.model,
#             [{"role": "system", "content": PromptRegistry.SYSTEM}, 
#              {"role": "user", "content": fix_prompt}],
#             max_output_tokens=config.max_output,
#             model_max_context=config.model_max_context
#         )
#         (turn_dir / "response.md").write_text(fix_content, encoding="utf-8")

#         # Apply Fix
#         if not _try_apply_content(fix_content, allowlist, turn_dir, config):
#             console.print("[red]Failed to apply fix. Moving to next strategy...[/red]")
#             # If diff failed to apply, the loop continues to 'Rewrite' stage naturally
#             # because verify will fail (code didn't change)
    
#     return False


def detect_tech_stack(goal: str, allowlist: List[str]) -> str:
    """
    Heuristics to detect the tech stack (PyTorch, NumPy, etc.) 
    and return strict 'Teacher Guidelines' to prevent common runtime errors.
    """
    goal_lower = goal.lower()
    combined_text = goal_lower + " ".join(str(x).lower() for x in allowlist)
    
    guidelines = []

    # --- PYTORCH TEACHER ---
    if any(k in combined_text for k in ["torch", "pytorch", "neural", "train", "model", "nn.linear"]):
        guidelines.append("**PYTORCH CRITICAL RULES (Strict Compliance Required):**")
        guidelines.append("1. **Device Safety**: Always move input tensors to `model.device` or `device` before forward pass.")
        guidelines.append("2. **Type Safety**: NEVER pass Numpy arrays to `model()`. Always wrap in `torch.FloatTensor(X).to(device)`.")
        guidelines.append("3. **Shape Safety**: Check shapes before MatMul! `nn.Linear` expects (N, Din), not (N). Use `.unsqueeze(1)` for targets if needed.")
        guidelines.append("4. **Numpy Conversion**: To convert a Tensor to Numpy, YOU MUST chain: `.detach().cpu().numpy()`.")
        guidelines.append("5. **Eval Mode**: Always call `model.eval()` and use `with torch.no_grad():` during evaluation.")
    
    # --- NUMPY TEACHER ---
    elif any(k in combined_text for k in ["numpy", "matrix", "array", "pandas"]):
        guidelines.append("**NUMPY/DATA RULES:**")
        guidelines.append("1. Check for `NaN` or `Inf` in data generation.")
        guidelines.append("2. Ensure matrix dimensions match for dot products (`@`).")

    # --- GENERAL PYTHON ---
    if any(str(f).endswith(".py") for f in allowlist):
        guidelines.append("3. **Reproducibility**: Set `torch.manual_seed(42)` and `np.random.seed(42)` at start.")

    if guidelines:
        return "\n".join(guidelines)
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
    
    args = parser.parse_args()

    agent_dir = Path(args.agent_dir)
    ensure_dirs(agent_dir)

    # Initialize Client
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

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

python CodeAgent//mini_claude_codev2.py --goal "Implement Univariate Linear Regression using ONLY PyTorch tensors. Do NOT use torch.nn, torch.optim, or autograd. Write everything in a single task.py file with a complete main() that trains, evaluates, and validates."

python CodeAgent/mini_claude_codev2.py --goal "Implement ML Task: SVM (Score Calibration + ROC/PR). Description: Calibrate decision scores; produce ROC/PR curves and AUC. Write a SINGLE self-contained Python file (task.py) with these functions: get_task_metadata, set_seed, get_device, make_dataloaders, build_model, train, evaluate, predict, save_artifacts."

python CodeAgent/mini_claude_codev2.py --goal "Implement Multivariate Linear Regression using torch.autograd. Visualize training. Description: Calibrate decision scores; produce ROC/PR curves and AUC. Write a SINGLE self-contained Python file (task.py) with these functions: get_task_metadata, set_seed, get_device, make_dataloaders, build_model, train, evaluate, predict, save_artifacts."

"""