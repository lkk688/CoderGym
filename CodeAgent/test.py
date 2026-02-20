Here is the complete, robust `PromptRegistry` class.

This version encapsulates all prompt logic in one place. It retains your smart **Token Budgeting** system (prioritizing git diffs → file content → directory tree) and integrates the new "Teacher Guidelines" and "Fix Strategy" logic we discussed.

### `PromptRegistry` Class Implementation

```python
class PromptRegistry:
    """
    Centralized manager for all LLM prompts.
    Handles token budgeting, context prioritization, and format enforcement.
    """
    
    SYSTEM = """\
You are an advanced AI coding agent. Your ONLY job is to produce file changes.

## Output Format (STRICT)
You MUST output in ONE of these two formats per response. Never mix them.

### Format A: Unified Diff (For small edits)
1. Start with `## Reasoning`.
2. Then `## Action` with a SINGLE `diff` block.
3. Use `diff --git a/path b/path` headers.
```diff
diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -10,2 +10,2 @@
-print("old")
+print("new")

```

### Format B: WRITE_FILE (For new files or full rewrites)

Use this when creating new files or when diffs are too complex.
WRITE_FILE: path/to/file.py
<<<CONTENT
... content ...
CONTENT>>>

## Teacher Guidelines (CRITICAL)

If provided, you MUST follow the language-specific guidelines in the User Prompt.
"""

```
@staticmethod
def format_task(
    goal: str,
    allowlist: List[str],
    context_files: List[str],
    notes: str,
    skills: str,
    max_context: int,
    max_output: int
) -> str:
    """
    Builds the main Turn Prompt with dynamic context management.
    
    Prioritizes context usage:
    1. Essential Instructions & Goal (Base)
    2. Git Status/Diff (for existing repos)
    3. File Contents (read-only context + allowlist)
    4. Directory Tree (if space permits)
    """
    
    # --- 1. Base Prompt Construction ---
    allow_txt = "\n".join(f"- {p}" for p in allowlist) if allowlist else "- (none)"
    
    # Heuristic: If all files are new, we skip git context to save tokens
    all_new_files = all(not Path(f).exists() for f in allowlist) if allowlist else False
    
    # Heuristic: Suggest WRITE_FILE if multiple files or new files
    format_hint = ""
    if (allowlist and len(allowlist) > 1) or all_new_files:
        format_hint = (
            "\n> **Hint**: Use **Format B (WRITE_FILE)**. "
            "It is more reliable for creating multiple or completely new files."
        )

    base_md = f"""# Turn Prompt

```

## Goal

{goal}

## Target Files (Allowlist)

{allow_txt}
{format_hint}
{skills if skills else ""}

## Constraints / Teacher Guidelines

{notes.strip() if notes.strip() else "(none)"}

## Output Contract

1. Return changes using EITHER Format A (Diff) OR Format B (WRITE_FILE).
2. ALL files in the Target Files list must be addressed if relevant.
3. (Optional) Include: "Verification: <command>" before the changes.
"""
```
 # --- 2. Token Budgeting ---
 # Reserve space for the model's output and a safety buffer
 safety_margin = 1000  # Buffer for tokenizer mismatch/system prompt
 usable_context = max_context - max_output - safety_margin

 # Initial usage (Base Prompt + System Prompt estimate)
 used_tokens = estimate_tokens(base_md) + estimate_tokens(PromptRegistry.SYSTEM)
 remaining = usable_context - used_tokens

 if remaining < 500:
     console.print("[red]Critical Warning: Goal + Constraints exceed context limit![/red]")
     base_md += "\n> **CRITICAL**: Input too long. Context truncated.\n"
     return base_md

 context_sections = []

 # --- 3. Context Priority 1: Git Status/Diff ---
 # (Skip for fresh files to save massive tokens)
 if not all_new_files and is_git_repo():
     st = git_status()
     df = git_diff()

     git_block = ""
     if st:
         # Filter status to only relevant files
         rel_lines = [line for line in st.split('\n') 
                      if line.startswith('##') or any(str(a) in line for a in allowlist)]
         if rel_lines:
             git_block += f"### git status (relevant)\n```\n{'\n'.join(rel_lines)}\n```\n"

     if df.strip():
         # Cap diff size to 2k tokens to prevent context flooding
         df_toks = estimate_tokens(df)
         if df_toks > 2000:
             df = truncate_to_tokens(df, 2000)
         git_block += f"### git diff\n```diff\n{df}\n```\n"

     git_cost = estimate_tokens(git_block)
     if git_block and git_cost < remaining:
         context_sections.append("## Repo Snapshot\n" + git_block)
         remaining -= git_cost
     elif all_new_files:
         console.print("[dim]Skipping git context (new files mode)[/dim]")

 # --- 4. Context Priority 2: File Contents ---
 files_md = ""
 # Ensure allowlist files are always first in context
 priority_files = list(dict.fromkeys(allowlist + context_files))

 for f in priority_files:
     fpath = str(f)
     content = read_file(fpath)

     if not content or content.startswith("[MISSING FILE]"):
         continue

     # Smart truncation for massive files
     if estimate_tokens(content) > 6000:
         content = truncate_to_tokens(content, 6000)

     file_block = f"## File: {fpath}\n```python\n{content}\n```\n"
     block_cost = estimate_tokens(file_block)

     if block_cost < remaining:
         files_md += file_block
         remaining -= block_cost
     else:
         files_md += f"## File: {fpath}\n[Content Omitted - Context Limit Reached]\n"

 if files_md:
     context_sections.append(files_md)

 # --- 5. Context Priority 3: Directory Tree ---
 # Least important, only if space permits
 if not all_new_files and remaining > 500:
     tree = top_level_tree()
     if estimate_tokens(tree) < remaining:
         context_sections.append(f"### File Tree\n{tree}\n")

 # Combine
 if context_sections:
     base_md += "\n## Context\n" + "\n".join(context_sections)

 return base_md

```


@staticmethod
def format_fix_diff(file_path: str, code_content: str, error_log: str) -> str:
"""
Prompt for Strategy 1: Quick Fix via Diff.
Focuses on specific error location.
"""
return f"""# Bug Fix Required (Diff Strategy)
The verification command failed.

## Error Output

```
{error_log[-3000:]}

```

## Instructions

1. **Analyze**: Look at the error and the code below.
2. **Scope**: Fix ONLY the specific error.
3. **Consistency**: Check the *entire function* for related issues (e.g. if changing variable type, update all usages).
4. **Output**: Use **Format A (Unified Diff)**.

## Current Code: {file_path}

```python
{code_content}

```

"""

```
@staticmethod
def format_fix_rewrite(file_path: str, error_history: str) -> str:
    """
    Prompt for Strategy 2: Full Rewrite.
    Used when diffs fail repeatedly. Force clean slate.
    """
    return f"""# Rewrite Required (Fresh Start)

```

Diff-based fixes have failed multiple times. We need a clean rewrite of `{file_path}`.

## Failure History

{error_history[-4000:]}

## Instructions

1. **Format**: output the **COMPLETE** file using **Format B (WRITE_FILE)**.
2. **Constraint**: Do NOT use diffs.
3. **Requirement**: Ensure ALL errors listed above are resolved in this new version.
4. **Completeness**: Do not use placeholders like `# ... rest of code`. Output every line.
"""

```

### Integration Check

1.  **Delete** the old standalone `build_prompt_md` function.
2.  **Delete** the old `DEFAULT_SYSTEM` constant string.
3.  Ensure `estimate_tokens`, `read_file`, `git_status`, `git_diff`, `truncate_to_tokens` are imported or available in your utils section.
4.  In `run_subtask_loop`, use `PromptRegistry.format_task(...)` instead of `build_prompt_md(...)`.

```