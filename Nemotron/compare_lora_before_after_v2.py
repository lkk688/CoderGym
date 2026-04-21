#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_lora_before_after.py

Full comparison script for:
1) Local BASE model (Unsloth / local HF model)
2) Local LoRA adapter model (Unsloth adapter path)
3) Multiple remote OpenAI-compatible models
4) Multiple NVIDIA API models

Features:
- Load examples from train.csv or JSONL
- Random / first / task-balanced sampling
- Unified prompt building
- Unified answer extraction
- Task-aware normalization and scoring
- JSONL output with all model results per sample
- Summary JSON with per-model and per-task accuracy

Typical usage:

# Local BASE + local LoRA + several NVIDIA API models
export NVIDIA_API_KEY="nvapi-xxxx"

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
python compare_lora_before_after.py \
  --base_model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --lora_path ./nemotron_sft_0419 \
  --train_csv ./train.csv \
  --num_samples 24 \
  --sample_mode task_balanced \
  --seed 3407 \
  --max_seq_length 4096 \
  --max_new_tokens 640 \
  --temperature 0.0 \
  --prompt_style minimal \
  --remote_models "nvidia/llama-3.3-nemotron-super-49b-v1,deepseek-ai/deepseek-v3.1,qwen/qwen3-coder-480b-a35b-instruct" \
  --nvidia_api \
  --output_jsonl compare_all.jsonl \
  --summary_json compare_all_summary.json

# Remote-only compare against a local vLLM endpoint
python compare_lora_before_after.py \
  --train_csv ./train.csv \
  --num_samples 24 \
  --sample_mode task_balanced \
  --remote_models "qwen-27b,qwen-72b" \
  --api_base http://127.0.0.1:8000/v1 \
  --api_key EMPTY \
  --output_jsonl compare_remote.jsonl \
  --summary_json compare_remote_summary.json
"""

import os
import gc
import re
import csv
import json
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# IMPORTANT: import unsloth before transformers-related internals
import unsloth  # noqa: F401
import torch
import httpx
from openai import OpenAI
from unsloth import FastLanguageModel


# -----------------------------------------------------------------------------
# Environment / proxy handling
# -----------------------------------------------------------------------------

os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

# ANSI color codes for terminal streaming output
COLOR_GREY = "\033[90m"
COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[92m"
COLOR_CYAN = "\033[96m"

# Per-model NVIDIA API configs based on official sample code
NVIDIA_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "deepseek-ai/deepseek-v3.2": {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 8192,
        "extra_body": {"chat_template_kwargs": {"thinking": True}},
        "has_thinking": True,
        "inline_thinking": False,
    },
    "minimaxai/minimax-m2.7": {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 8192,
        "extra_body": None,
        "has_thinking": False,
        "inline_thinking": True,  # uses <think>...</think> tags in content
    },
    "z-ai/glm4.7": {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 16384,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": False}},
        "has_thinking": True,
        "inline_thinking": False,
    },
    "google/gemma-4-31b-it":{
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16384,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        "has_thinking": True,
        "inline_thinking": False,
    },
    "qwen/qwen3.5-122b-a10b":{
        "temperature": 0.60,
        "top_p": 0.95,
        "max_tokens": 16384,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        "has_thinking": True,
        "inline_thinking": False,
    }
}


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class Sample:
    sample_id: str
    prompt: str
    answer: str
    task_type: str
    raw: Dict[str, Any]


# -----------------------------------------------------------------------------
# Task inference and loading
# -----------------------------------------------------------------------------

def infer_task_type(obj: Dict[str, Any], prompt: str) -> str:
    task = (obj.get("task_type") or obj.get("task") or "").strip().lower()
    if task:
        return task

    p = prompt.lower()
    if "wonderland numeral system" in p or "roman numeral" in p:
        return "roman"
    if "secret unit conversion" in p:
        return "unit_conversion"
    if "gravitational constant has been secretly changed" in p:
        return "gravity"
    if "secret encryption rules are used on text" in p:
        return "cipher_text"
    if "bit manipulation rule transforms 8-bit binary numbers" in p:
        return "bit_manipulation"
    if "secret set of transformation rules is applied to equations" in p:
        return "symbol_transform"
    return ""


def load_samples_from_train_csv(path: str) -> List[Sample]:
    samples: List[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = (row.get("prompt") or "").strip()
            answer = str(row.get("answer") or "").strip()
            sample_id = str(row.get("id") or f"row_{len(samples)}").strip()
            if not prompt or answer == "":
                continue
            task_type = infer_task_type(row, prompt)
            samples.append(
                Sample(
                    sample_id=sample_id,
                    prompt=prompt,
                    answer=answer,
                    task_type=task_type,
                    raw=row,
                )
            )
    return samples


def load_samples_from_jsonl(path: str) -> List[Sample]:
    samples: List[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = str(obj.get("prompt") or obj.get("question") or obj.get("user") or "").strip()
            answer = str(obj.get("answer") or obj.get("output") or obj.get("target") or "").strip()
            sample_id = str(obj.get("id") or f"jsonl_{idx}").strip()
            if not prompt or answer == "":
                continue
            task_type = infer_task_type(obj, prompt)
            samples.append(
                Sample(
                    sample_id=sample_id,
                    prompt=prompt,
                    answer=answer,
                    task_type=task_type,
                    raw=obj,
                )
            )
    return samples


def load_candidate_examples(args: argparse.Namespace) -> List[Sample]:
    out: List[Sample] = []

    if args.train_csv:
        print("[*] Loading examples from train.csv...")
        out.extend(load_samples_from_train_csv(args.train_csv))

    if args.input_jsonl:
        print("[*] Loading examples from JSONL...")
        out.extend(load_samples_from_jsonl(args.input_jsonl))

    # Deduplicate by sample_id, keep first
    seen = set()
    deduped: List[Sample] = []
    for s in out:
        if s.sample_id in seen:
            continue
        seen.add(s.sample_id)
        deduped.append(s)

    return deduped


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------

def sample_examples(
    samples: List[Sample],
    num_samples: int,
    sample_mode: str,
    seed: int,
) -> List[Sample]:
    if num_samples <= 0 or num_samples >= len(samples):
        return list(samples)

    rng = random.Random(seed)

    if sample_mode == "first":
        return samples[:num_samples]

    if sample_mode == "random":
        return rng.sample(samples, num_samples)

    if sample_mode == "task_balanced":
        buckets: Dict[str, List[Sample]] = {}
        for s in samples:
            buckets.setdefault(s.task_type or "unknown", []).append(s)

        for k in buckets:
            rng.shuffle(buckets[k])

        task_names = sorted(buckets.keys())
        selected: List[Sample] = []
        while len(selected) < num_samples:
            progressed = False
            for task in task_names:
                if buckets[task]:
                    selected.append(buckets[task].pop())
                    progressed = True
                    if len(selected) >= num_samples:
                        break
            if not progressed:
                break
        return selected

    raise ValueError(f"Unsupported sample_mode={sample_mode}")


# -----------------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------------

def build_system_prompt(task_type: str, prompt_style: str) -> str:
    generic_minimal = (
        "You solve pattern induction tasks. "
        "Return only the final answer, enclosed in \\boxed{...}. "
        "Do not include any explanation."
    )

    if prompt_style == "minimal":
        if task_type == "roman":
            return (
                "You solve Roman numeral conversion tasks. "
                "Return only the final Roman numeral enclosed in \\boxed{...}. "
                "No explanation."
            )
        if task_type == "unit_conversion":
            return (
                "You solve unit conversion pattern tasks. "
                "Infer the conversion from the examples and return only the final numeric answer enclosed in \\boxed{...}. "
                "No explanation."
            )
        if task_type == "gravity":
            return (
                "You solve gravity formula pattern tasks. "
                "Infer the effective gravitational constant from the examples, compute the final distance, "
                "and return only the final numeric answer enclosed in \\boxed{...}. No explanation."
            )
        if task_type == "cipher_text":
            return (
                "You solve text decryption tasks. "
                "Return only the final decrypted plaintext enclosed in \\boxed{...}. No explanation."
            )
        if task_type == "bit_manipulation":
            return (
                "You solve 8-bit pattern induction tasks. "
                "Return only the final 8-bit binary answer enclosed in \\boxed{...}. No explanation."
            )
        if task_type == "symbol_transform":
            return (
                "You solve symbol transformation tasks. "
                "Return only the final transformed symbol string enclosed in \\boxed{...}. No explanation."
            )
        return generic_minimal

    if prompt_style == "strict_boxed":
        return (
            "Return exactly one line in this format: \\boxed{final_answer}\n"
            "Do not output anything else."
        )

    # fallback
    return generic_minimal


def build_messages(sample: Sample, prompt_style: str) -> List[Dict[str, str]]:
    system_prompt = build_system_prompt(sample.task_type, prompt_style)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample.prompt},
    ]


# -----------------------------------------------------------------------------
# Extraction / normalization / scoring
# -----------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_BIN8_RE = re.compile(r"\b[01]{8}\b")
_ROMAN_RE = re.compile(r"\b[IVXLCDM]+\b")
_LINE_ANSWER_RE = re.compile(
    r"(?:final answer|answer|result|therefore|thus)\s*[:：]\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_boxed(text: str) -> Optional[str]:
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def extract_last_numeric(text: str) -> Optional[str]:
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    return matches[-1]


def extract_last_bin8(text: str) -> Optional[str]:
    matches = _BIN8_RE.findall(text)
    if not matches:
        return None
    return matches[-1]


def extract_last_roman(text: str) -> Optional[str]:
    matches = _ROMAN_RE.findall(text)
    if not matches:
        return None
    return matches[-1]


def extract_last_answer_line(text: str) -> Optional[str]:
    matches = _LINE_ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None

    # Prefer the last short line over a very long reasoning block
    for ln in reversed(lines):
        if len(ln) <= 120:
            return ln
    return lines[-1]


def extract_prediction(text: str, task_type: str) -> Tuple[str, str]:
    """
    Returns:
        (prediction, extract_method)
    """
    text = text or ""

    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed, "boxed"

    if task_type == "bit_manipulation":
        x = extract_last_bin8(text)
        if x is not None:
            return x, "heuristic_bin8"

    if task_type == "roman":
        x = extract_last_roman(text)
        if x is not None:
            return x, "heuristic_roman"

    if task_type in {"gravity", "unit_conversion"}:
        x = extract_last_numeric(text)
        if x is not None:
            return x, "heuristic_numeric"

    if task_type == "symbol_transform":
        # Try answer line first
        x = extract_last_answer_line(text)
        if x:
            # Remove obvious wrappers
            x = x.strip().strip("`").strip()
            return x, "heuristic_symbol_line"

    if task_type == "cipher_text":
        x = extract_last_answer_line(text)
        if x:
            x = x.strip().strip("`").strip().strip('"').strip("'")
            return x, "heuristic_text_line"

    # Generic fallback
    x = extract_last_answer_line(text)
    if x is not None:
        return x.strip(), "heuristic"

    return "", "empty"


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def normalize_prediction(pred: str, task_type: str) -> str:
    pred = pred.strip()

    if task_type == "roman":
        return re.sub(r"\s+", "", pred.upper())

    if task_type == "bit_manipulation":
        m = _BIN8_RE.search(pred)
        return m.group(0) if m else pred.replace(" ", "")

    if task_type in {"gravity", "unit_conversion"}:
        # Keep numeric string normalization light
        pred = pred.strip()
        # Strip surrounding punctuation
        pred = pred.strip(".,;: ")
        try:
            val = float(pred)
            # Normalize 19.00 -> 19, but preserve stable decimal string if needed
            s = f"{val:.10f}".rstrip("0").rstrip(".")
            return s
        except Exception:
            return pred

    if task_type == "cipher_text":
        return normalize_whitespace(pred).lower().strip(" .,:;!?")

    if task_type == "symbol_transform":
        return pred.strip()

    return normalize_whitespace(pred)


def normalize_gold(gold: str, task_type: str) -> str:
    return normalize_prediction(gold, task_type)


def try_parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def official_like_match(
    pred_norm: str,
    gold_norm: str,
    task_type: str,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-6,
) -> Tuple[bool, str]:
    # Exact match first
    if pred_norm == gold_norm:
        return True, "exact"

    # Numeric tolerance match
    if task_type in {"gravity", "unit_conversion"}:
        pv = try_parse_float(pred_norm)
        gv = try_parse_float(gold_norm)
        if pv is not None and gv is not None:
            if math.isclose(pv, gv, rel_tol=rel_tol, abs_tol=abs_tol):
                return True, "numeric_tolerance"

    return False, "none"


# -----------------------------------------------------------------------------
# Remote API client
# -----------------------------------------------------------------------------

def resolve_remote_base_and_key(args: argparse.Namespace) -> Tuple[str, str]:
    if args.nvidia_api:
        api_base = "https://integrate.api.nvidia.com/v1"
        # Always read from env for NVIDIA; only use --api_key if explicitly provided (not default "EMPTY")
        api_key = os.getenv("NVIDIA_API_KEY", "").strip().strip("'").strip('"')
        if not api_key and args.api_key and args.api_key != "EMPTY":
            api_key = args.api_key
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY env var not set and no --api_key provided.")
        return api_base, api_key

    return args.api_base, args.api_key


def build_openai_client(api_base: str, api_key: str, timeout_s: float) -> OpenAI:
    return OpenAI(
        base_url=api_base,
        api_key=api_key,
        timeout=timeout_s,
        max_retries=3,
        http_client=httpx.Client(trust_env=False),
    )


def get_nvidia_model_config(model_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Return per-model API config, preferring NVIDIA_MODEL_CONFIGS for known models."""
    if args.nvidia_api and model_name in NVIDIA_MODEL_CONFIGS:
        return dict(NVIDIA_MODEL_CONFIGS[model_name])

    cfg: Dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": 0.95,
        "max_tokens": None,  # use per_task_max_new_tokens at call site
        "extra_body": None,
        "has_thinking": False,
        "inline_thinking": False,
    }
    if args.nvidia_api and args.nvidia_enable_thinking is not None:
        cfg["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": bool(args.nvidia_enable_thinking),
                "clear_thinking": False,
            }
        }
        cfg["has_thinking"] = bool(args.nvidia_enable_thinking)
    return cfg


def print_stream_metrics(usage: Any, ttft_seconds: Optional[float], total_seconds: float) -> None:
    print(f"\n{COLOR_CYAN}{'-'*50}{COLOR_RESET}")
    if not usage:
        print("[Warning] No usage data available.")
        return
    p_tokens = getattr(usage, "prompt_tokens", 0)
    c_tokens = getattr(usage, "completion_tokens", 0)
    t_tokens = getattr(usage, "total_tokens", 0)
    print(f"Tokens : Prompt={p_tokens} | Completion={c_tokens} | Total={t_tokens}")
    if ttft_seconds and total_seconds and ttft_seconds > 0:
        gen_seconds = max(total_seconds - ttft_seconds, 0.001)
        print(f"Time   : Prefill(TTFT)={ttft_seconds:.3f}s | Generate={gen_seconds:.3f}s")
        prefill_speed = p_tokens / ttft_seconds if p_tokens > 0 else 0
        gen_speed = c_tokens / gen_seconds if c_tokens > 0 else 0
        print(f"Speed  : Prefill={prefill_speed:.1f} t/s | Generate={gen_speed:.1f} t/s")
    elif total_seconds:
        print(f"Time   : Total={total_seconds:.3f}s")
        if t_tokens > 0:
            print(f"Speed  : Overall={t_tokens/total_seconds:.1f} t/s")
    print(f"{COLOR_CYAN}{'-'*50}{COLOR_RESET}")


def stream_single_request(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    model_cfg: Dict[str, Any],
    sample_id: str,
    model_label: str,
) -> Tuple[str, str, Optional[Any], Optional[float], float]:
    """
    Stream one request, printing thinking in grey and content normally.
    Returns (answer_text, thinking_text, usage, ttft_sec, elapsed_sec).
    answer_text excludes thinking for scoring; thinking_text holds the full reasoning.
    """
    start_time = time.perf_counter()
    first_token_time: Optional[float] = None
    final_usage = None
    full_content: List[str] = []
    thinking_parts: List[str] = []
    in_thinking_block = False

    temperature = model_cfg.get("temperature", 0.7)
    top_p = model_cfg.get("top_p", 0.95)
    extra_body = model_cfg.get("extra_body")
    inline_thinking = model_cfg.get("inline_thinking", False)

    create_kwargs: Dict[str, Any] = dict(
        model=model_name,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )
    if extra_body:
        create_kwargs["extra_body"] = extra_body

    print(f"\n{COLOR_GREEN}[{model_label}] sample={sample_id}{COLOR_RESET} ", end="", flush=True)

    try:
        stream = client.chat.completions.create(**create_kwargs)

        for chunk in stream:
            if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
                if getattr(chunk, "usage", None):
                    final_usage = chunk.usage
                continue

            delta = getattr(chunk.choices[0], "delta", None)
            if delta is None:
                continue

            reasoning = getattr(delta, "reasoning_content", None)
            content = getattr(delta, "content", None) or ""

            if first_token_time is None and (reasoning or content):
                first_token_time = time.perf_counter()

            # Dedicated reasoning_content field (DeepSeek / GLM style)
            if reasoning:
                if not in_thinking_block:
                    print(f"\n{COLOR_GREY}[Thinking...]\n", end="", flush=True)
                    in_thinking_block = True
                thinking_parts.append(reasoning)
                print(reasoning, end="", flush=True)
                continue

            # Close reasoning block when answer content starts
            if in_thinking_block and not inline_thinking and content:
                in_thinking_block = False
                print(f"{COLOR_RESET}\n[Answer]\n", end="", flush=True)

            # Inline <think> tags (MiniMax style)
            if inline_thinking and content:
                if "<think>" in content:
                    print(f"\n{COLOR_GREY}[Thinking...]\n", end="", flush=True)
                    content = content.replace("<think>", "")
                    in_thinking_block = True
                if "</think>" in content:
                    content = content.replace("</think>", "")
                    in_thinking_block = False
                    print(f"{COLOR_RESET}\n[Answer]\n", end="", flush=True)
                if in_thinking_block:
                    thinking_parts.append(content)
                    print(content, end="", flush=True)
                    continue  # exclude thinking from full_content for scoring

            if content:
                full_content.append(content)
                print(content, end="", flush=True)

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        err = f"[REMOTE_ERROR] {type(e).__name__}: {e}"
        print(f"\n{err}")
        return err, "", None, None, elapsed

    elapsed = time.perf_counter() - start_time
    print(COLOR_RESET, flush=True)

    ttft = (first_token_time - start_time) if first_token_time else None
    print_stream_metrics(final_usage, ttft, elapsed)

    return "".join(full_content), "".join(thinking_parts), final_usage, ttft, elapsed


def _build_result_dict(
    model_label: str,
    model_name: str,
    sample: "Sample",
    answer_text: str,
    thinking_text: str,
    usage: Any,
    ttft: Optional[float],
    elapsed: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Shared helper: score the answer and return a full result dict."""
    pred, method = extract_prediction(answer_text, sample.task_type)
    pred_norm = normalize_prediction(pred, sample.task_type)
    gold_norm = normalize_gold(sample.answer, sample.task_type)
    match, match_mode = official_like_match(
        pred_norm, gold_norm, sample.task_type,
        rel_tol=args.numeric_rel_tol, abs_tol=args.numeric_abs_tol,
    )

    p_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    c_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    gen_sec = max(elapsed - (ttft or 0), 0.001)
    prefill_tps = round(p_tokens / ttft, 1) if ttft and ttft > 0 and p_tokens > 0 else None
    gen_tps = round(c_tokens / gen_sec, 1) if c_tokens > 0 else None

    return {
        "model_label": model_label,
        "model_name": model_name,
        "sample_id": sample.sample_id,
        "task_type": sample.task_type,
        "thinking_raw": thinking_text,
        "prediction_raw": answer_text,
        "prediction_extracted": pred,
        "prediction_normalized": pred_norm,
        "gold_answer": sample.answer,
        "gold_answer_normalized": gold_norm,
        "extract_method": method,
        "correct": match,
        "match_mode": match_mode,
        "latency_sec": round(elapsed, 4),
        "ttft_sec": round(ttft, 4) if ttft else None,
        "prefill_speed_tps": prefill_tps,
        "gen_speed_tps": gen_tps,
        "prompt_tokens": p_tokens,
        "completion_tokens": c_tokens,
    }


def run_remote_single(
    client: OpenAI,
    model_name: str,
    model_label: str,
    sample: "Sample",
    model_cfg: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run one remote model on one sample. Returns a result dict."""
    messages = build_messages(sample, args.prompt_style)
    max_tokens = int(model_cfg.get("max_tokens") or per_task_max_new_tokens(sample.task_type, args))

    answer_text, thinking_text, usage, ttft, elapsed = stream_single_request(
        client=client,
        model_name=model_name,
        messages=messages,
        max_tokens=max_tokens,
        model_cfg=model_cfg,
        sample_id=sample.sample_id,
        model_label=model_label,
    )

    result = _build_result_dict(
        model_label, model_name, sample,
        answer_text, thinking_text, usage, ttft, elapsed, args,
    )

    status = f"{COLOR_GREEN}CORRECT{COLOR_RESET}" if result["correct"] else "WRONG"
    print(
        f"[{model_label}] id={sample.sample_id} | task={sample.task_type} | "
        f"pred={result['prediction_extracted']!r} | gold={sample.answer!r} | {status}"
    )
    return result


# -----------------------------------------------------------------------------
# Local model loading / inference
# -----------------------------------------------------------------------------

def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_dtype(dtype_name: str):
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return None  # auto


def load_local_model_and_tokenizer(
    model_path: str,
    args: argparse.Namespace,
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=args.max_seq_length,
        dtype=get_dtype(args.dtype),
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def per_task_max_new_tokens(task_type: str, args: argparse.Namespace) -> int:
    if args.task_max_new_tokens_json:
        try:
            cfg = json.loads(args.task_max_new_tokens_json)
            if task_type in cfg:
                return int(cfg[task_type])
        except Exception:
            pass
    return int(args.max_new_tokens)


def run_local_model_pass(
    model_label: str,
    model_path: str,
    samples: List[Sample],
    args: argparse.Namespace,
    skip_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Run a local model over samples, skipping any sample_id in skip_ids."""
    print(f"[*] Loading local model: {model_label} -> {model_path}")
    model, tokenizer = load_local_model_and_tokenizer(model_path, args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results: List[Dict[str, Any]] = []

    try:
        for i, sample in enumerate(samples, start=1):
            if skip_ids and sample.sample_id in skip_ids:
                print(f"[{model_label}] Skipping {sample.sample_id} (already done)")
                continue

            messages = build_messages(sample, args.prompt_style)
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)

            n_prompt_tokens = inputs.shape[1]
            start_time = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs,
                    max_new_tokens=int(per_task_max_new_tokens(sample.task_type, args)),
                    temperature=float(args.temperature),
                    do_sample=bool(args.temperature > 0.0),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            elapsed = time.perf_counter() - start_time

            gen_ids = outputs[0][n_prompt_tokens:]
            raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            n_gen_tokens = len(gen_ids)
            gen_tps = round(n_gen_tokens / elapsed, 1) if elapsed > 0 else None

            result = _build_result_dict(
                model_label, model_path, sample,
                raw_text, "",  # no separate thinking for local models
                None, None, elapsed, args,
            )
            # Override token counts with what we measured directly
            result["prompt_tokens"] = n_prompt_tokens
            result["completion_tokens"] = n_gen_tokens
            result["gen_speed_tps"] = gen_tps

            results.append(result)

            status = f"{COLOR_GREEN}CORRECT{COLOR_RESET}" if result["correct"] else "WRONG"
            print(
                f"[{model_label}] {i}/{len(samples)} | "
                f"id={sample.sample_id} | task={sample.task_type} | "
                f"pred={result['prediction_extracted']!r} | gold={sample.answer!r} | {status} | "
                f"{n_gen_tokens} tok @ {gen_tps} t/s"
            )

    finally:
        print(f"[*] Releasing local model: {model_label}")
        del model
        del tokenizer
        cleanup_cuda()

    return results


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

def make_summary(
    samples: List[Sample],
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    # rows are per-sample records with model_results
    summary: Dict[str, Any] = {
        "args": vars(args),
        "num_samples": len(samples),
        "models": {},
    }

    # Flatten per-model stats
    by_model: Dict[str, List[Tuple[str, bool]]] = {}
    for row in rows:
        task_type = row["task_type"]
        for model_key, res in row["results"].items():
            by_model.setdefault(model_key, []).append((task_type, bool(res["correct"])))

    for model_key, entries in by_model.items():
        total = len(entries)
        correct = sum(1 for _, ok in entries if ok)
        model_info = {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total if total else 0.0, 6),
            "by_task": {},
        }

        by_task: Dict[str, List[bool]] = {}
        for task, ok in entries:
            by_task.setdefault(task or "unknown", []).append(ok)

        for task, vals in sorted(by_task.items()):
            c = sum(1 for x in vals if x)
            t = len(vals)
            model_info["by_task"][task] = {
                "correct": c,
                "total": t,
                "accuracy": round(c / t if t else 0.0, 6),
            }

        summary["models"][model_key] = model_info

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("[*] SUMMARY")
    print("=" * 80)
    for model_key, info in summary["models"].items():
        print(f"{model_key}: {info['correct']}/{info['total']} = {info['accuracy']:.4f}")
        for task, tinfo in info["by_task"].items():
            print(f"  - {task}: {tinfo['correct']}/{tinfo['total']} = {tinfo['accuracy']:.4f}")
    print("=" * 80)


# -----------------------------------------------------------------------------
# Incremental I/O helpers
# -----------------------------------------------------------------------------

def load_existing_output(path: str) -> Dict[str, Dict[str, Any]]:
    """Load output JSONL into {sample_id: row}. Returns empty dict if file absent."""
    data: Dict[str, Dict[str, Any]] = {}
    if not path or not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                sid = row.get("id") or row.get("sample_id", "")
                if sid:
                    data[sid] = row
            except json.JSONDecodeError:
                pass
    return data


def flush_output_jsonl(path: str, data: Dict[str, Dict[str, Any]], samples: List["Sample"]) -> None:
    """Write all rows to output JSONL in sample order."""
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            if s.sample_id in data:
                f.write(json.dumps(data[s.sample_id], ensure_ascii=False) + "\n")
        # Also write any rows for samples not in our current list (from prior runs)
        present = {s.sample_id for s in samples}
        for sid, row in data.items():
            if sid not in present:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def upsert_result_into_output(
    output_data: Dict[str, Dict[str, Any]],
    sample: "Sample",
    result: Dict[str, Any],
) -> None:
    """Add a single model result into the in-memory output dict."""
    sid = sample.sample_id
    if sid not in output_data:
        output_data[sid] = {
            "id": sid,
            "task_type": sample.task_type,
            "prompt": sample.prompt,
            "gold_answer": sample.answer,
            "results": {},
        }
    output_data[sid]["results"][result["model_label"]] = result


def load_clean_answers(path: str) -> Dict[str, Any]:
    """Load the clean-answers JSON. Returns empty dict if absent."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def update_clean_answers_file(
    path: str,
    sample: "Sample",
    result: Dict[str, Any],
) -> None:
    """
    If result is correct, append it to the clean-answers JSON (keyed by sample_id).
    Each entry accumulates correct answers from multiple models across re-runs.
    """
    if not path or not result.get("correct"):
        return

    data = load_clean_answers(path)
    sid = sample.sample_id

    if sid not in data:
        data[sid] = {
            "task_type": sample.task_type,
            "gold_answer": sample.answer,
            "prompt": sample.prompt,
            "correct_models": [],
        }

    # Avoid duplicate entries for the same model
    model_label = result["model_label"]
    existing_labels = {e["model"] for e in data[sid]["correct_models"]}
    if model_label not in existing_labels:
        data[sid]["correct_models"].append({
            "model": model_label,
            "model_name": result.get("model_name", ""),
            "thinking": result.get("thinking_raw", ""),
            "answer": result.get("prediction_raw", ""),
            "answer_extracted": result.get("prediction_extracted", ""),
            "latency_sec": result.get("latency_sec"),
            "ttft_sec": result.get("ttft_sec"),
            "gen_speed_tps": result.get("gen_speed_tps"),
            "prompt_tokens": result.get("prompt_tokens"),
            "completion_tokens": result.get("completion_tokens"),
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare local BASE/LoRA and remote API models on the same samples.")

    # Data
    p.add_argument("--train_csv", type=str, default="", help="Path to local train.csv")
    p.add_argument("--input_jsonl", type=str, default="", help="Path to local JSONL with prompt/answer records")

    # Sampling / targeting
    p.add_argument("--num_samples", type=int, default=0,
                   help="Randomly select N unfinished samples (0 = all unfinished)")
    p.add_argument("--task_id", type=str, default="",
                   help="Evaluate a single sample by id with all models (overrides --num_samples)")
    p.add_argument("--sample_mode", type=str, default="random", choices=["first", "random", "task_balanced"])
    p.add_argument("--seed", type=int, default=3407)

    # Local models
    p.add_argument("--base_model", type=str, default="", help="Local HF / Unsloth base model path or model id")
    p.add_argument("--lora_path", type=str, default="", help="Local LoRA adapter path saved by Unsloth")

    # Remote models
    p.add_argument("--remote_models", type=str, default="",
                   help="Comma-separated remote model names for OpenAI-compatible or NVIDIA API compare")
    p.add_argument("--api_base", type=str, default="http://127.0.0.1:8000/v1")
    p.add_argument("--api_key", type=str, default="EMPTY")
    p.add_argument("--nvidia_api", action="store_true",
                   help="Use NVIDIA API: https://integrate.api.nvidia.com/v1")
    p.add_argument("--nvidia_enable_thinking", action="store_true", default=None,
                   help="For NVIDIA API only: enable thinking if the model supports it")
    p.add_argument("--no_nvidia_enable_thinking", dest="nvidia_enable_thinking", action="store_false")

    # Prompt / extraction
    p.add_argument("--prompt_style", type=str, default="minimal",
                   choices=["minimal", "strict_boxed"],
                   help="Minimal generally works best for fair compare")
    p.add_argument("--numeric_rel_tol", type=float, default=1e-3)
    p.add_argument("--numeric_abs_tol", type=float, default=1e-6)

    # Generation
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument(
        "--task_max_new_tokens_json",
        type=str,
        default='{"roman":64,"unit_conversion":96,"gravity":160,"cipher_text":320,"bit_manipulation":640,"symbol_transform":640}',
        help="Optional JSON mapping task_type -> max_new_tokens"
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout_s", type=float, default=180.0)

    # Local loading
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    p.add_argument("--load_in_8bit", action="store_true", default=False)
    p.add_argument("--dtype", type=str, default="bf16", choices=["auto", "bf16", "fp16"])
    p.add_argument("--attn_implementation", type=str, default="eager")

    # Output
    p.add_argument("--output_jsonl", type=str, required=True,
                   help="Path to JSONL file (created/appended incrementally per sample)")
    p.add_argument("--clean_answers_json", type=str, default="",
                   help="Path to JSON file that collects correct answers across runs (appended)")
    p.add_argument("--summary_json", type=str, default="")

    return p.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def _get_already_done(output_data: Dict[str, Dict[str, Any]], model_label: str) -> set:
    """Return set of sample_ids that already have a result for model_label."""
    return {sid for sid, row in output_data.items() if model_label in row.get("results", {})}


def main() -> None:
    args = parse_args()

    if torch.cuda.is_available():
        print(f"[*] Visible CUDA devices: {torch.cuda.device_count()}")
        print(f"[*] Using device: cuda:0 -> {torch.cuda.get_device_name(0)}")
    else:
        print("[*] CUDA not available, running on CPU.")

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("[*] Loading candidate examples...")
    all_samples = load_candidate_examples(args)
    if not all_samples:
        raise RuntimeError("No samples loaded. Provide --train_csv and/or --input_jsonl")

    # -------------------------------------------------------------------------
    # Load existing output (for resume / skip logic)
    # -------------------------------------------------------------------------
    output_data = load_existing_output(args.output_jsonl)
    print(f"[*] Loaded {len(output_data)} existing results from {args.output_jsonl}")

    # -------------------------------------------------------------------------
    # Build the working sample list
    # -------------------------------------------------------------------------
    remote_models = [x.strip() for x in args.remote_models.split(",") if x.strip()]
    all_model_labels = (
        (["BASE"] if args.base_model else [])
        + (["LORA"] if args.lora_path else [])
        + [f"REMOTE::{rm}" for rm in remote_models]
    )

    if args.task_id:
        # Single-task mode: find the sample by id
        matches = [s for s in all_samples if s.sample_id == args.task_id]
        if not matches:
            raise RuntimeError(f"--task_id '{args.task_id}' not found in dataset")
        samples = matches
        print(f"[*] Single-task mode: id={args.task_id}")
    else:
        # Unfinished-task mode: a sample is "unfinished" if at least one model hasn't run on it
        def is_unfinished(s: Sample) -> bool:
            row = output_data.get(s.sample_id, {})
            done = set(row.get("results", {}).keys())
            return bool(set(all_model_labels) - done)

        unfinished = [s for s in all_samples if is_unfinished(s)]
        print(f"[*] {len(unfinished)}/{len(all_samples)} samples have unfinished models")

        samples = sample_examples(
            unfinished,
            num_samples=args.num_samples,
            sample_mode=args.sample_mode,
            seed=args.seed,
        )

    print(f"[*] Working on {len(samples)} samples")
    for s in samples:
        print(f"    - id={s.sample_id} task={s.task_type}")

    # -------------------------------------------------------------------------
    # Build remote clients once per model (reuse across samples)
    # -------------------------------------------------------------------------
    remote_clients: Dict[str, Tuple[OpenAI, Dict[str, Any]]] = {}
    if remote_models:
        api_base, api_key = resolve_remote_base_and_key(args)
        client = build_openai_client(api_base, api_key, args.timeout_s)
        for rm in remote_models:
            model_cfg = get_nvidia_model_config(rm, args)
            remote_clients[rm] = (client, model_cfg)
            print(f"[*] Remote model ready: {rm} | temp={model_cfg['temperature']} "
                  f"thinking={model_cfg.get('has_thinking', False)}")

    # -------------------------------------------------------------------------
    # Local BASE model: load once, run all unfinished samples, unload
    # -------------------------------------------------------------------------
    if args.base_model:
        skip = _get_already_done(output_data, "BASE")
        base_results = run_local_model_pass(
            model_label="BASE",
            model_path=args.base_model,
            samples=samples,
            args=args,
            skip_ids=skip,
        )
        for r in base_results:
            sid = r["sample_id"]
            sample = next(s for s in samples if s.sample_id == sid)
            upsert_result_into_output(output_data, sample, r)
            update_clean_answers_file(args.clean_answers_json, sample, r)
        flush_output_jsonl(args.output_jsonl, output_data, samples)
        cleanup_cuda()

    # -------------------------------------------------------------------------
    # Local LoRA model: same
    # -------------------------------------------------------------------------
    if args.lora_path:
        skip = _get_already_done(output_data, "LORA")
        lora_results = run_local_model_pass(
            model_label="LORA",
            model_path=args.lora_path,
            samples=samples,
            args=args,
            skip_ids=skip,
        )
        for r in lora_results:
            sid = r["sample_id"]
            sample = next(s for s in samples if s.sample_id == sid)
            upsert_result_into_output(output_data, sample, r)
            update_clean_answers_file(args.clean_answers_json, sample, r)
        flush_output_jsonl(args.output_jsonl, output_data, samples)
        cleanup_cuda()

    # -------------------------------------------------------------------------
    # Remote models: iterate per-sample × per-model
    # After every sample, flush both output files so partial results are safe.
    # -------------------------------------------------------------------------
    if remote_clients:
        for si, sample in enumerate(samples, start=1):
            print(f"\n{'='*60}")
            print(f"[*] Sample {si}/{len(samples)}: id={sample.sample_id} task={sample.task_type}")
            print(f"{'='*60}")

            row_done = set(output_data.get(sample.sample_id, {}).get("results", {}).keys())

            for rm in remote_models:
                model_label = f"REMOTE::{rm}"
                if model_label in row_done:
                    print(f"[skip] {model_label} already done for {sample.sample_id}")
                    continue

                api_client, model_cfg = remote_clients[rm]
                result = run_remote_single(
                    client=api_client,
                    model_name=rm,
                    model_label=model_label,
                    sample=sample,
                    model_cfg=model_cfg,
                    args=args,
                )
                upsert_result_into_output(output_data, sample, result)
                update_clean_answers_file(args.clean_answers_json, sample, result)

            # Flush after every sample so a crash loses at most one sample's data
            flush_output_jsonl(args.output_jsonl, output_data, samples)
            if args.clean_answers_json:
                print(f"[*] Clean answers updated: {args.clean_answers_json}")

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    # Rebuild rows list from output_data for summary
    rows = [output_data[s.sample_id] for s in samples if s.sample_id in output_data]
    summary = make_summary(samples, rows, args)
    print_summary(summary)

    print(f"\n[*] Final output JSONL: {args.output_jsonl}")
    if args.clean_answers_json:
        clean = load_clean_answers(args.clean_answers_json)
        print(f"[*] Clean answers JSON: {args.clean_answers_json} ({len(clean)} tasks with correct answers)")

    if args.summary_json:
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[*] Wrote summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()

"""
# Example: NVIDIA-only, 30 random unfinished samples, saves correct answers separately
# Re-running is safe: already-done (sample, model) pairs are skipped automatically.
export NVIDIA_API_KEY="nvapi-xxxx"

python compare_lora_before_after_v2.py \
  --train_csv ./train.csv \
  --num_samples 30 \
  --remote_models "deepseek-ai/deepseek-v3.2,z-ai/glm4.7,minimaxai/minimax-m2.7" \
  --nvidia_api \
  --output_jsonl compare_nvidia_only.jsonl \
  --clean_answers_json correct_answers.json \
  --summary_json compare_nvidia_only_summary.json

# Example: evaluate a single task id with all models
python compare_lora_before_after_v2.py \
  --train_csv ./train.csv \
  --task_id e526fae1 \
  --remote_models "deepseek-ai/deepseek-v3.2,z-ai/glm4.7,minimaxai/minimax-m2.7" \
  --nvidia_api \
  --output_jsonl compare_nvidia_only.jsonl \
  --clean_answers_json correct_answers.json

# Example: local BASE + LoRA + NVIDIA API models
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
python compare_lora_before_after_v2.py \
  --base_model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --lora_path ./nemotron_sft_0419 \
  --train_csv ./train.csv \
  --num_samples 24 \
  --seed 3407 \
  --max_seq_length 4096 \
  --max_new_tokens 640 \
  --remote_models "deepseek-ai/deepseek-v3.2,z-ai/glm4.7" \
  --nvidia_api \
  --output_jsonl compare_all.jsonl \
  --clean_answers_json correct_answers.json \
  --summary_json compare_all_summary.json

# Example: local vLLM endpoint
python compare_lora_before_after_v2.py \
  --train_csv ./train.csv \
  --num_samples 30 \
  --remote_models "qwen-27b,qwen-72b" \
  --api_base http://127.0.0.1:8000/v1 \
  --api_key EMPTY \
  --output_jsonl compare_vllm.jsonl \
  --clean_answers_json correct_answers.json \
  --summary_json compare_vllm_summary.json

#[REMOTE::deepseek-ai/deepseek-v3.2] 8/30 | id=e526fae1 | task=cipher_text | pred='the hidden student dreams' | gold='the hidden student dreams' | CORRECT
#[REMOTE::deepseek-ai/deepseek-v3.2] 10/30 | id=3ef556a2 | task=roman | pred='LVII' | gold='LVII' | CORRECT
python compare_lora_before_after_v2.py \
  --train_csv ./train.csv \
  --task_id e526fae1 \
  --remote_models "google/gemma-4-31b-it, qwen/qwen3.5-122b-a10b, z-ai/glm4.7, minimaxai/minimax-m2.7, deepseek-ai/deepseek-v3.2" \
  --nvidia_api \
  --output_jsonl outputs/compare_nvidia_only.jsonl \
  --clean_answers_json outputs/correct_answers.json

python compare_lora_before_after_v2.py \
  --train_csv ./train.csv \
  --num_samples 10 \
  --remote_models "google/gemma-4-31b-it, qwen/qwen3.5-122b-a10b, z-ai/glm4.7, minimaxai/minimax-m2.7, deepseek-ai/deepseek-v3.2" \
  --nvidia_api \
  --output_jsonl outputs/compare_nvidia_only.jsonl \
  --clean_answers_json outputs/correct_answers.json \
  --summary_json outputs/compare_nvidia_only_summary.json

python compare_lora_before_after_v2.py \
  --train_csv ./train.csv \
  --num_samples 100 \
  --remote_models "google/gemma-4-31b-it, qwen/qwen3.5-122b-a10b, z-ai/glm4.7, minimaxai/minimax-m2.7, deepseek-ai/deepseek-v3.2" \
  --nvidia_api \
  --output_jsonl outputs/compare_nvidia_only.jsonl \
  --clean_answers_json outputs/correct_answers.json \
  --summary_json outputs/compare_nvidia_only_summary.json
"""