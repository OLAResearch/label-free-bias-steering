#!/usr/bin/env python3
"""TTRL bias-only training on MathVista with Qwen2.5-VL-7B-Instruct.

Same approach as train_ttrl_steer.py but for multimodal VQA:
  - Model : Qwen2.5-VL-7B-Instruct (same Qwen2MLP structure, added down_proj.bias)
  - Data  : AI4Math/MathVista testmini (1000 problems, images)
  - Reward: GRPO binary via majority-vote on normalized VQA answers
  - Grade : score_vqa (MCQ + free-form + numeric) — no SymPy needed
"""
import sys
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "refs", "steering-reasoning"))
sys.path.insert(0, os.path.join(_ROOT, "pydeps"))

import argparse
import difflib
import json
import random
import re
import string
from collections import Counter
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor

from steering_reasoning.train.rl.reward_processor import GRPORewardProcessor


# ── MathVista instruction ─────────────────────────────────────────────────────

COT_INSTRUCTION = (
    "Answer the question based on the image.\n"
    "If the question includes answer choices, choose the best option. "
    "For free-form questions, give a short answer.\n"
    "Use exactly this format:\n"
    "Reasoning: <step-by-step reasoning>\n"
    "Final answer: <short answer>\n\n"
    "Example (multi-choice):\n"
    "Question: Which shape has the largest area?\nChoices:\nA. triangle\nB. circle\n"
    "Reasoning: The circle fills more area.\nFinal answer: B\n\n"
    "Example (free-form):\n"
    "Question: What is the value of x?\n"
    "Reasoning: x + 3 = 15, so x = 12.\nFinal answer: 12"
)

BOXED_INSTRUCTION = (
    "Answer the question based on the image. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)


# ── MathVista dataset helpers ─────────────────────────────────────────────────

def get_image(sample: dict):
    img = sample.get("decoded_image") or sample.get("image")
    if img is None:
        raise ValueError(f"No image in sample (keys={list(sample.keys())})")
    return img.convert("RGB")


def get_question(sample: dict) -> str:
    question = str(sample.get("question", "")).strip()
    choices = sample.get("choices") or []
    if str(sample.get("question_type", "")).strip().lower() == "multi_choice" and choices:
        choice_block = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        return f"{question}\nChoices:\n{choice_block}"
    return question


def get_scoring_kwargs(sample: dict) -> dict:
    return {
        "choices":       sample.get("choices") or [],
        "question_type": str(sample.get("question_type", "")).strip().lower(),
        "answer_type":   str(sample.get("answer_type",  "")).strip().lower(),
        "precision":     sample.get("precision", None),
    }


# ── VQA metrics (inlined from vqa/core/metrics.py) ───────────────────────────

_UNIT_RE = re.compile(
    r"\s*(ml|cl|dl|liters?|litres?|gallons?|oz|ounces?|lbs?|pounds?|"
    r"ft|feet|foot|yards?|miles?|inches?|in\b|"
    r"cm|mm|km|meters?|metres?|m\b|kg|g\b|mg|"
    r"°[CF]?|degrees?[CF]?|radians?|"
    r"%|percent|"
    r"hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?|months?|years?)"
    r"\s*$",
    flags=re.IGNORECASE,
)
_CURRENCY_RE = re.compile(r"^[\$€£¥]")


def _strip_units(text: str) -> str:
    text = _CURRENCY_RE.sub("", text.strip())
    return _UNIT_RE.sub("", text).strip()


def normalize_answer(text: str) -> str:
    text = str(text).lower().strip()
    text = _strip_units(text)
    keep = set("./")
    text = "".join(c for c in text if c not in string.punctuation or c in keep)
    return " ".join(text.split())


def _try_numeric(text: str):
    try:
        return float(text.strip())
    except ValueError:
        pass
    m = re.match(r"^(-?\d+)\s*/\s*(-?\d+)$", text.strip())
    if m:
        n, d = int(m.group(1)), int(m.group(2))
        return n / d if d != 0 else None
    return None


def extract_final_answer(text: str) -> str:
    m = re.search(r"final answer[:\s]+(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".,;:!? ")
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return (lines[-1] if lines else text.strip()).rstrip(".,;:!? ")


def _last_boxed_only_string(string):
    """Brace-depth-aware \\boxed{} span finder (unlike a plain regex, this
    correctly handles nested braces from \\frac{a}{b}, \\sqrt{}, etc.).
    Ported from eval_mathvista.py's fix -- train-side extractor was never
    updated when the eval-side bug was found+fixed there (2026-08-20)."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i, num_open, right = idx, 0, None
    while i < len(string):
        if string[i] == "{":
            num_open += 1
        elif string[i] == "}":
            num_open -= 1
            if num_open == 0:
                right = i
                break
        i += 1
    return None if right is None else string[idx:right + 1]


def extract_final_answer_boxed(text: str) -> str:
    # Brace-depth aware, not a naive regex (a naive r"\\boxed\{([^}]*)\}"
    # truncates at the first inner '}', e.g. \\boxed{\\frac{5}{3}\\pi} would
    # extract as '\\frac{5' instead of the full answer). This corrupts both
    # the GRPO reward and the majority-vote pseudo-label for any boxed-format
    # training run whose gold or predicted answer contains nested braces.
    boxed = _last_boxed_only_string(text)
    if boxed is not None:
        left = "\\boxed{"
        if boxed.startswith(left) and boxed.endswith("}"):
            return boxed[len(left):-1].strip()
    return extract_final_answer(text)


def normalize_mathvista_prediction(extraction: str, choices: list = None,
                                   question_type: str = "", answer_type: str = "",
                                   precision=None) -> str:
    raw = (extraction or "").strip()
    qt  = (question_type or "").strip().lower()
    at  = (answer_type  or "").strip().lower()
    cvs = list(choices or [])

    if qt == "multi_choice" and cvs:
        labels = [chr(65 + i) for i in range(len(cvs))]
        # Official grader: only extract letter from explicit "(A)" format
        letter_match = re.findall(r"\(([a-zA-Z])\)", raw)
        if letter_match:
            letter = letter_match[0].upper()
            if letter in labels:
                return normalize_answer(cvs[labels.index(letter)])
        # Bare standalone letter (e.g. "A" or "A." as entire extracted answer)
        bare = re.match(r"^\s*([A-Z])\s*[.:]?\s*$", raw)
        if bare and bare.group(1) in labels:
            return normalize_answer(cvs[labels.index(bare.group(1))])
        # Exact match against normalized choice text
        rn = normalize_answer(raw)
        norm_cvs = [normalize_answer(c) for c in cvs]
        if rn in norm_cvs:
            return rn
        # Fallback: SequenceMatcher similarity (approximates official Levenshtein)
        sims = [difflib.SequenceMatcher(None, rn, c).ratio() for c in norm_cvs]
        return norm_cvs[sims.index(max(sims))]

    if at == "integer":
        try:
            return str(int(float(_strip_units(raw))))
        except Exception:
            return ""
    if at == "float":
        try:
            prec = int(precision) if precision is not None else 2
            return str(round(float(_strip_units(raw)), prec))
        except Exception:
            return ""
    return normalize_answer(raw)


def get_gt_forms(sample: dict) -> List[str]:
    choices = sample.get("choices") or []
    qt      = str(sample.get("question_type", "")).strip().lower()
    at      = str(sample.get("answer_type",   "")).strip().lower()
    prec    = sample.get("precision", None)
    raw     = str(sample.get("answer", "")).strip()
    gt_norm = normalize_mathvista_prediction(raw, choices=choices, question_type=qt,
                                             answer_type=at, precision=prec)
    forms = set()
    if gt_norm:
        forms.add(gt_norm)
    if qt == "multi_choice" and choices:
        m = re.match(r"^\(?([A-Z])\)?", raw)
        if m:
            forms.add(m.group(1).upper())
    return list(forms)


def score_vqa(pred: str, gt_forms: List[str], choices: list = None,
              question_type: str = "", answer_type: str = "", precision=None,
              extract_fn=None) -> bool:
    if extract_fn is None:
        extract_fn = extract_final_answer
    pred_norm = normalize_mathvista_prediction(
        extract_fn(pred), choices=choices,
        question_type=question_type, answer_type=answer_type, precision=precision)
    for g in gt_forms:
        gn = normalize_answer(g)
        if pred_norm == gn:
            return True
        pf, gf = _try_numeric(pred_norm), _try_numeric(gn)
        if pf is not None and gf is not None:
            tol = 10 ** (-(int(precision) if precision is not None else 2))
            if gf == 0:
                if abs(pf) <= tol:
                    return True
            elif abs(pf - gf) / abs(gf) < tol:
                return True
    return False


# ── Runtime patch: add down_proj.bias to HF Qwen2.5-VL ───────────────────────

def patch_hf_for_bias_vl():
    import torch.nn as nn
    # Qwen2.5-VL uses Qwen2_5_VLMLP for the vision encoder (bias=True, already has bias)
    # and Qwen2MLP for the text decoder (bias=False, needs bias added). Both must be
    # patched independently — the VL patch alone misses the text decoder.
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLMLP
        _orig_vl = Qwen2_5_VLMLP.__init__
        def _patched_vl(self, config, bias=False):
            _orig_vl(self, config, bias=bias)
            if self.down_proj.bias is None:
                old = self.down_proj
                self.down_proj = nn.Linear(old.in_features, old.out_features, bias=True)
        Qwen2_5_VLMLP.__init__ = _patched_vl
        print("HF: patched Qwen2_5_VLMLP.__init__ for native down_proj.bias")
    except (ImportError, AttributeError):
        pass
    # Patch Qwen2MLP from the qwen2_5_vl module — text decoder uses this class
    # (distinct from transformers.models.qwen2.modeling_qwen2.Qwen2MLP).
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2MLP as Qwen2_5_VLTextMLP
        _orig_text = Qwen2_5_VLTextMLP.__init__
        def _patched_text(self, config):
            _orig_text(self, config)
            if self.down_proj.bias is None:
                old = self.down_proj
                self.down_proj = nn.Linear(old.in_features, old.out_features, bias=True)
        Qwen2_5_VLTextMLP.__init__ = _patched_text
        print("HF: patched qwen2_5_vl.Qwen2MLP.__init__ for native down_proj.bias (text decoder)")
    except (ImportError, AttributeError):
        pass


# ── Runtime patch: add down_proj.bias to vLLM Qwen2.5-VL ─────────────────────

def patch_vllm_for_bias_vl():
    from vllm.model_executor.layers.linear import RowParallelLinear

    def _make_mlp_patch(orig):
        def _patched_init(self, hidden_size, intermediate_size, hidden_act,
                          quant_config=None, prefix=""):
            orig(self, hidden_size, intermediate_size, hidden_act,
                 quant_config=quant_config, prefix=prefix)
            self.down_proj = RowParallelLinear(
                intermediate_size, hidden_size, bias=True,
                quant_config=quant_config, prefix=f"{prefix}.down_proj")
        return _patched_init

    def _make_lw_patch(orig):
        def _patched_lw(self, weights):
            loaded = orig(self, weights)
            for name, p in self.named_parameters():
                if name.endswith("mlp.down_proj.bias"):
                    with torch.no_grad():
                        p.zero_()
                    loaded.add(name)
            return loaded
        return _patched_lw

    patched_vl = False
    try:
        import vllm.model_executor.models.qwen2_5_vl as qm_vl
        # Patch all MLP classes found — no break, both vision and text decoder need it.
        for cls_name in ("Qwen2_5_VLMLP", "Qwen2MLP"):
            if hasattr(qm_vl, cls_name):
                cls = getattr(qm_vl, cls_name)
                cls.__init__ = _make_mlp_patch(cls.__init__)
                patched_vl = True
        if hasattr(qm_vl, "Qwen2_5_VLForConditionalGeneration"):
            vl_cls = qm_vl.Qwen2_5_VLForConditionalGeneration
            vl_cls.load_weights = _make_lw_patch(vl_cls.load_weights)
    except (ImportError, AttributeError) as e:
        print(f"  vLLM VL patch warning: {e}")

    if not patched_vl:
        import vllm.model_executor.models.qwen2 as qm
        qm.Qwen2MLP.__init__ = _make_mlp_patch(qm.Qwen2MLP.__init__)
        qm.Qwen2ForCausalLM.load_weights = _make_lw_patch(qm.Qwen2ForCausalLM.load_weights)

    print("vLLM: patched Qwen2.5-VL for native down_proj.bias")


# ── Sync bias HF → vLLM (robust layer-path detection) ────────────────────────

_sync_diag_done = False  # print full diagnostics only once

def sync_bias_to_vllm_vl(llm, hf_model):
    global _sync_diag_done
    by_layer = {}
    for name, p in hf_model.named_parameters():
        if p.requires_grad and name.endswith("mlp.down_proj.bias"):
            parts = name.split(".")
            try:
                li = parts.index("layers")
                by_layer[int(parts[li + 1])] = p.detach().cpu().float()
            except (ValueError, IndexError):
                pass

    if not _sync_diag_done:
        print(f"  [sync diag] by_layer keys: {sorted(by_layer.keys())[:5]}... "
              f"total={len(by_layer)}", flush=True)

    _path_used = [None]

    def _find_layers(m):
        candidates = [
            ("x.language_model.model.layers",    lambda x: x.language_model.model.layers),
            ("x.model.language_model.model.layers", lambda x: x.model.language_model.model.layers),
            ("x.model.model.layers",             lambda x: x.model.model.layers),
            ("x.model.layers",                   lambda x: x.model.layers),
            ("x.language_model.layers",          lambda x: x.language_model.layers),
        ]
        for desc, getter in candidates:
            try:
                result = getter(m)
                _path_used[0] = desc
                return result
            except AttributeError:
                continue
        raise AttributeError("Cannot locate decoder layers in vLLM VL model")

    def _do_sync(vllm_model):
        layers = _find_layers(vllm_model)
        if not _sync_diag_done:
            print(f"  [sync diag] layer path: {_path_used[0]}, "
                  f"n_layers={len(layers)}", flush=True)
        for idx, bias_cpu in by_layer.items():
            dp = layers[idx].mlp.down_proj
            if dp.bias is None:
                if not _sync_diag_done:
                    print(f"  [sync diag] layer {idx}: dp.bias is None — patch didn't take!", flush=True)
                return
            with torch.no_grad():
                dp.bias.copy_(bias_cpu.to(dp.bias.device, dp.bias.dtype))
        if not _sync_diag_done:
            dp0 = layers[0].mlp.down_proj
            hf_norm = by_layer[0].norm().item() if 0 in by_layer else float("nan")
            vl_norm = dp0.bias.float().norm().item()
            print(f"  [sync diag] layer 0 bias norm — HF: {hf_norm:.4f}  vLLM: {vl_norm:.4f}", flush=True)

    llm.apply_model(_do_sync)
    if not _sync_diag_done:
        _sync_diag_done = True


# ── Majority vote on normalized VQA answers ───────────────────────────────────

def majority_vote_vqa(responses: List[str], sample: dict, extract_fn=None):
    if extract_fn is None:
        extract_fn = extract_final_answer
    kwargs = get_scoring_kwargs(sample)
    preds = [normalize_mathvista_prediction(extract_fn(r), **kwargs)
             for r in responses]
    preds = [p for p in preds if p]
    if not preds:
        return None, 0.0
    vote, count = Counter(preds).most_common(1)[0]
    return vote, count / len(responses)


# ── Log-prob for one rollout (VL forward pass, gradients enabled) ─────────────

def compute_log_prob_vl(model, processor, user_msg: dict, image,
                        prompt_len: int, response_text: str,
                        temperature: float = 1.0) -> torch.Tensor:
    """Return mean log-prob over response tokens (scalar, gradients flow)."""
    device = next(model.parameters()).device
    messages = [user_msg, {"role": "assistant", "content": response_text}]
    full_text = processor.apply_chat_template(messages, tokenize=False)
    enc = processor(text=[full_text], images=[image], return_tensors="pt")

    kwargs = {
        "input_ids":      enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
    }
    for key in ("pixel_values", "image_grid_thw"):
        if enc.get(key) is not None:
            kwargs[key] = enc[key].to(device)

    logits  = model(**kwargs).logits[0, :-1] / temperature  # [T-1, V]
    ids     = enc["input_ids"][0].to(device)
    tok_lp  = F.log_softmax(logits, dim=-1).gather(1, ids[1:].unsqueeze(1)).squeeze(1)

    T = tok_lp.shape[0]
    mask = torch.zeros(T, device=device)
    mask[prompt_len - 1:] = 1.0          # response starts at prompt_len
    return (tok_lp * mask).sum() / mask.sum().clamp(min=1)


def compute_log_probs_vl_batch(model, processor, user_msg: dict, image,
                                prompt_len: int, response_texts: list,
                                temperature: float = 1.0,
                                mini_batch: int = 8) -> list:
    """Batched log-prob: processes responses in mini-batches for GPU efficiency.

    Same image+prompt for all responses (one TTRL step). Returns a list of G
    scalar tensors with gradients — drop-in replacement for G calls to
    compute_log_prob_vl. Speedup: ~mini_batch× fewer forward passes.
    """
    device = next(model.parameters()).device
    all_lps = []

    for start in range(0, len(response_texts), mini_batch):
        batch = response_texts[start : start + mini_batch]
        B = len(batch)

        full_texts = [
            processor.apply_chat_template(
                [user_msg, {"role": "assistant", "content": r}], tokenize=False)
            for r in batch
        ]

        enc = processor(
            text=full_texts,
            images=[image] * B,
            return_tensors="pt",
            padding=True,          # right-pad all sequences to same length
        )

        input_ids = enc["input_ids"].to(device)        # [B, T]
        attn_mask = enc["attention_mask"].to(device)   # [B, T]

        kwargs = {"input_ids": input_ids, "attention_mask": attn_mask}
        for key in ("pixel_values", "image_grid_thw"):
            if enc.get(key) is not None:
                kwargs[key] = enc[key].to(device)

        logits = model(**kwargs).logits[:, :-1] / temperature  # [B, T-1, V]
        ids_next = input_ids[:, 1:]                             # [B, T-1]
        tok_lp = (F.log_softmax(logits, dim=-1)
                  .gather(2, ids_next.unsqueeze(2)).squeeze(2))  # [B, T-1]

        seq_lens = attn_mask.sum(1)  # actual (non-padded) length per sample
        for j in range(B):
            sl = int(seq_lens[j].item())
            # response tokens: [prompt_len-1, sl-2] in shifted (tok_lp) indexing
            resp = tok_lp[j, prompt_len - 1 : sl - 1]
            n = resp.shape[0]
            all_lps.append(resp.sum() / max(n, 1))

    return all_lps


# ── Evaluation (vLLM fast path or HF fallback) ───────────────────────────────

@torch.no_grad()
def evaluate_vl(model, processor, llm, samples, max_new_tokens,
                return_details=False, instruction=None, extract_fn=None, lora_request=None):
    if instruction is None:
        instruction = COT_INSTRUCTION
    if extract_fn is None:
        extract_fn = extract_final_answer
    device = next(model.parameters()).device

    if llm is not None:
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        requests = []
        for s in samples:
            image    = get_image(s)
            question = get_question(s)
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": f"{instruction}\nQuestion: {question}"},
            ]}]
            pt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            requests.append({"prompt": pt, "multi_modal_data": {"image": image}})
        vllm_outs = llm.generate(requests, sp, lora_request=lora_request)
        texts = [o.outputs[0].text for o in vllm_outs]
    else:
        texts = []
        for s in tqdm(samples, desc="eval", leave=False):
            image    = get_image(s)
            question = get_question(s)
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": f"{instruction}\nQuestion: {question}"},
            ]}]
            pt  = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = processor(text=[pt], images=[image], return_tensors="pt")
            kw  = {"input_ids": enc["input_ids"].to(device),
                   "attention_mask": enc["attention_mask"].to(device)}
            for k in ("pixel_values", "image_grid_thw"):
                if enc.get(k) is not None:
                    kw[k] = enc[k].to(device)
            out  = model.generate(**kw, do_sample=False, max_new_tokens=max_new_tokens,
                                  pad_token_id=processor.tokenizer.eos_token_id)
            resp = processor.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            texts.append(resp)

    correct = 0
    details = []
    for s, resp in zip(samples, texts):
        gt_forms = get_gt_forms(s)
        ok = score_vqa(resp, gt_forms, extract_fn=extract_fn, **get_scoring_kwargs(s))
        if ok:
            correct += 1
        if return_details:
            details.append({
                "id":        s.get("pid", len(details)),
                "question":  get_question(s),
                "answer":    str(s.get("answer", "")),
                "response":  resp,
                "predicted": extract_fn(resp),
                "correct":   ok,
            })

    acc = correct / len(samples)
    return (acc, details) if return_details else acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="TTRL bias-only steering on MathVista (VL)")
    ap.add_argument("--model_id",          default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--num_steps",         type=int,   default=200)
    ap.add_argument("--G",                 type=int,   default=32)
    ap.add_argument("--lr",                type=float, default=1e-3)
    ap.add_argument("--gen_temp",          type=float, default=0.7)
    ap.add_argument("--max_new_tokens",    type=int,   default=1024)
    ap.add_argument("--eval_every",        type=int,   default=20)
    ap.add_argument("--eval_start",        type=int,   default=20)
    ap.add_argument("--save_every",        type=int,   default=20)
    ap.add_argument("--eval_n",            type=int,   default=500)
    ap.add_argument("--log_prob_mini_batch", type=int, default=1,
                    help="Mini-batch size for batched log-prob forward passes (1=sequential, 8=fast)")
    ap.add_argument("--seed",              type=int,   default=42)
    ap.add_argument("--use_vllm",          action="store_true")
    ap.add_argument("--vllm_gpu_util",     type=float, default=0.30)
    ap.add_argument("--lr_scheduler",      default="cosine",
                    choices=["constant", "cosine", "delayed_cosine"])
    ap.add_argument("--lr_warmup_steps",   type=int, default=None,
                    help="For delayed_cosine: steps at full LR before decay starts. "
                         "Defaults to num_steps//2.")
    ap.add_argument("--prompt_format",     default="cot", choices=["cot", "boxed"],
                    help="cot: 'Final answer:' format; boxed: MM-UPT \\boxed{} format")
    ap.add_argument("--steering_at_layer", type=int,   default=None)
    ap.add_argument("--output_dir",        default=None)
    # max_pixels controls image resolution → visual token count.
    # Qwen2.5-VL uses 28×28 patches: max_pixels=448*448 → ~256 visual tokens.
    ap.add_argument("--max_pixels",        type=int,   default=262144)  # 512×512
    ap.add_argument("--mcq_only",          action="store_true",
                    help="Filter dataset to multi_choice problems only")
    ap.add_argument("--use_labels",        action="store_true",
                    help="use ground-truth labels instead of majority-vote pseudo-labels")
    ap.add_argument("--full_ft",           action="store_true",
                    help="full fine-tune all language_model params (no vLLM); bias-only otherwise")
    ap.add_argument("--lora_r",            type=int,   default=None,
                    help="LoRA rank; if set uses peft LoRA instead of bias-only")
    ap.add_argument("--lora_alpha",        type=int,   default=None,
                    help="LoRA alpha (defaults to 2*lora_r)")
    ap.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj",
                    help="comma-separated LoRA target modules (default: q_proj,v_proj)")
    ap.add_argument("--weight_decay",      type=float, default=0.0)
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir or
                   Path(_ROOT) / "outputs" /
                   f"ttrl_mathvista_{args.model_id.replace('/', '_')}_s{args.num_steps}_G{args.G}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # 1. Patch HF MLP
    patch_hf_for_bias_vl()

    # 2. Load model + processor
    print(f"\nLoading {args.model_id} …")
    from transformers import Qwen2_5_VLForConditionalGeneration
    n_visible = torch.cuda.device_count()
    _device_map = "balanced" if n_visible > 1 else {"": 0}
    print(f"GPUs visible: {n_visible}  →  device_map={_device_map!r}")
    processor = AutoProcessor.from_pretrained(args.model_id,
                                              max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=_device_map,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    # 3. Freeze all; zero-init and enable down_proj.bias
    trainable = []
    if args.lora_r:
        # peft LoRA — vLLM supported via native LoRA hot-swap (LoRARequest)
        from peft import LoraConfig, get_peft_model
        lora_alpha = args.lora_alpha if args.lora_alpha else args.lora_r * 2
        target_modules = args.lora_target_modules.replace(" ", "").split(",")
        lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
                              target_modules=target_modules, bias="none")
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        trainable = [p for p in model.parameters() if p.requires_grad]
        mode = f"LoRA r={args.lora_r} alpha={lora_alpha} {target_modules}"
    elif args.full_ft:
        if args.use_vllm:
            print("  [WARNING] --full_ft disables vLLM (weight sync not supported for all params)")
            args.use_vllm = False
        for name, p in model.named_parameters():
            if "language_model" in name:
                p.requires_grad_(True)
                trainable.append(p)
            else:
                p.requires_grad_(False)
        model.language_model.gradient_checkpointing_enable()
    else:
        for name, p in model.named_parameters():
            is_trainable = "language_model" in name and name.endswith("mlp.down_proj.bias")
            if args.steering_at_layer is not None and is_trainable:
                parts = name.split(".")
                try:
                    li = parts.index("layers")
                    is_trainable = int(parts[li + 1]) == args.steering_at_layer
                except (ValueError, IndexError):
                    is_trainable = False
            if is_trainable:
                torch.nn.init.zeros_(p)
                p.requires_grad_(True)
                trainable.append(p)
            else:
                p.requires_grad_(False)

    if not args.lora_r:
        n_trainable = sum(p.numel() for p in trainable)
        mode = "full LM fine-tune" if args.full_ft else (
            f"layer {args.steering_at_layer}" if args.steering_at_layer is not None
            else f"all {len(trainable)} layers [down_proj.bias]"
        )
        print(f"  Trainable: {n_trainable:,} params ({mode})")
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"  Trainable names (first 3): {trainable_names[:3]}")

    # 4. Optionally init vLLM
    llm = None
    lora_sync = None
    if args.use_vllm:
        from vllm import LLM, SamplingParams
        print(f"\nInitializing vLLM (gpu_util={args.vllm_gpu_util}) …")
        max_visual_tokens = args.max_pixels // (28 * 28)
        vllm_max_len = max_visual_tokens + args.max_new_tokens + 512
        print(f"  max_pixels={args.max_pixels} → ~{max_visual_tokens} visual tokens, "
              f"max_model_len={vllm_max_len}")
        if args.lora_r:
            import shutil as _shutil
            from vllm.lora.request import LoRARequest as _LoRARequest
            class _LoraSync:
                def __init__(self, staging_dir):
                    self._dir = Path(staging_dir)
                    self._dir.mkdir(parents=True, exist_ok=True)
                    self._id = 1
                def sync(self, m):
                    if self._dir.exists():
                        _shutil.rmtree(self._dir)
                    m.save_pretrained(str(self._dir))
                    req = _LoRARequest("adapter", self._id, str(self._dir))
                    self._id += 1
                    return req
            lora_sync = _LoraSync(out_dir / "_lora_sync_staging")
            llm = LLM(
                args.model_id, dtype="bfloat16", tensor_parallel_size=1,
                gpu_memory_utilization=args.vllm_gpu_util, enforce_eager=True,
                max_model_len=vllm_max_len, disable_log_stats=True,
                limit_mm_per_prompt={"image": 1}, mm_processor_kwargs={"max_pixels": args.max_pixels},
                enable_lora=True, max_loras=1, max_lora_rank=args.lora_r, max_cpu_loras=2,
            )
        else:
            patch_vllm_for_bias_vl()
            llm = LLM(
                args.model_id, dtype="bfloat16", tensor_parallel_size=1,
                gpu_memory_utilization=args.vllm_gpu_util, enforce_eager=True,
                max_model_len=vllm_max_len, disable_log_stats=True,
                limit_mm_per_prompt={"image": 1}, mm_processor_kwargs={"max_pixels": args.max_pixels},
            )
        print("  vLLM ready")

    # 5. Optimizer + scheduler + GRPO
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_steps, eta_min=0.0)
        print(f"  LR: cosine {args.lr} → 0 over {args.num_steps} steps")
    elif args.lr_scheduler == "delayed_cosine":
        import math as _math
        T_flat = args.lr_warmup_steps if args.lr_warmup_steps is not None else args.num_steps // 2
        T_decay = max(1, args.num_steps - T_flat)
        def _delayed_cosine_lambda(step):
            if step < T_flat:
                return 1.0
            progress = (step - T_flat) / T_decay
            return 0.5 * (1.0 + _math.cos(_math.pi * min(progress, 1.0)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _delayed_cosine_lambda)
        print(f"  LR: delayed_cosine {args.lr} — flat for {T_flat} steps, "
              f"then cosine decay to 0 over {T_decay} steps")
    else:
        scheduler = None
    grpo = GRPORewardProcessor(num_generations=args.G)

    # 6. Load MathVista
    print("\nLoading AI4Math/MathVista (testmini) …")
    dataset = list(load_dataset("AI4Math/MathVista", split="testmini"))
    if args.mcq_only:
        dataset = [s for s in dataset if
                   str(s.get("question_type", "")).strip().lower() == "multi_choice"]
        print(f"  MCQ-only filter: {len(dataset)} problems retained")
    eval_samples  = dataset[:args.eval_n]
    train_samples = list(dataset)
    random.shuffle(train_samples)
    device = next(model.parameters()).device
    print(f"  {len(dataset)} problems  (eval_n={args.eval_n})")

    # Prompt format
    if args.prompt_format == "boxed":
        instruction = BOXED_INSTRUCTION
        extract_fn  = extract_final_answer_boxed
        print(f"  Prompt format: boxed (\\boxed{{}})")
    else:
        instruction = COT_INSTRUCTION
        extract_fn  = extract_final_answer
        print(f"  Prompt format: cot (Final answer:)")

    # 7. Step-0 eval
    print("\nEVAL step 0 (baseline) …")
    _lora_req0 = None
    if llm is not None:
        if lora_sync is not None:
            _lora_req0 = lora_sync.sync(model)
        else:
            sync_bias_to_vllm_vl(llm, model)
    acc0, details0 = evaluate_vl(model, processor, llm, eval_samples,
                                 args.max_new_tokens, return_details=True,
                                 instruction=instruction, extract_fn=extract_fn,
                                 lora_request=_lora_req0)
    print(f"★  step 0  pass@1 = {acc0:.4f}  ({acc0*100:.2f}%)  [{args.eval_n} problems]")
    (out_dir / "eval_step0.json").write_text(
        json.dumps({"step": 0, "acc": acc0, "n": args.eval_n, "details": details0}, indent=2))

    if args.num_steps == 0:
        return

    # 8. Training loop
    print(f"\n{'═'*64}")
    print(f"TRAIN {args.num_steps} steps | G={args.G} | lr={args.lr} | "
          f"{'vLLM' if llm else 'HF'} generation")
    print(f"{'═'*64}")

    training_log = []

    for step in range(args.num_steps):
        sample  = train_samples[step % len(train_samples)]
        image   = get_image(sample)
        question = get_question(sample)

        # Build user message dict (reused for log-prob calls)
        user_msg = {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": f"{instruction}\nQuestion: {question}"},
        ]}

        # Prompt length including image tokens — computed once per step
        prompt_text = processor.apply_chat_template(
            [user_msg], tokenize=False, add_generation_prompt=True)
        prompt_enc  = processor(text=[prompt_text], images=[image], return_tensors="pt")
        prompt_len  = prompt_enc["input_ids"].shape[1]

        # (a) Generate G rollouts ─────────────────────────────────────────────
        print(f"  step {step+1:3d}: sampling {args.G} rollouts …", flush=True)

        if llm is not None:
            if lora_sync is not None:
                _lora_req = lora_sync.sync(model)
            else:
                sync_bias_to_vllm_vl(llm, model)
                _lora_req = None
            sp = SamplingParams(temperature=args.gen_temp,
                                max_tokens=args.max_new_tokens, n=args.G)
            vllm_out = llm.generate([{
                "prompt":           prompt_text,
                "multi_modal_data": {"image": image},
            }], sp, lora_request=_lora_req)
            texts = [o.text for o in vllm_out[0].outputs]
        else:
            enc_dev = {k: v.to(device) for k, v in prompt_enc.items()}
            texts   = []
            with torch.no_grad():
                for _ in range(args.G):
                    out  = model.generate(**enc_dev, do_sample=True,
                                          temperature=args.gen_temp,
                                          max_new_tokens=args.max_new_tokens,
                                          pad_token_id=processor.tokenizer.eos_token_id)
                    texts.append(processor.decode(out[0][prompt_len:],
                                                  skip_special_tokens=True))

        # (b) Pseudo-label: ground-truth or majority-vote ─────────────────────
        if args.use_labels:
            gt_forms = get_gt_forms(sample)
            pseudo_label = gt_forms[0] if gt_forms else None
            kwargs = get_scoring_kwargs(sample)
            majority_ratio = sum(
                1 for t in texts
                if normalize_mathvista_prediction(extract_fn(t), **kwargs) in gt_forms
            ) / len(texts)
        else:
            pseudo_label, majority_ratio = majority_vote_vqa(texts, sample, extract_fn=extract_fn)
        s1 = step + 1
        if not pseudo_label:
            print(f"  step {s1:3d}: skip — no valid answers extracted")
            training_log.append({"step": s1, "skip": "no_answer"})
            continue

        kwargs = get_scoring_kwargs(sample)
        correct_flags = [
            normalize_mathvista_prediction(extract_fn(t), **kwargs) == pseudo_label
            for t in texts
        ]
        raw_rewards = [1.0 if c else -1.0 for c in correct_flags]
        rewards     = torch.tensor(raw_rewards, dtype=torch.float32, device=device)
        n_match     = int((rewards > 0).sum().item())

        # (c) GRPO advantages ─────────────────────────────────────────────────
        if rewards.std() < 1e-6:
            print(f"  step {s1:3d}: skip — all {args.G} rewards equal "
                  f"(ratio={majority_ratio:.2f})")
            training_log.append({"step": s1, "skip": "all_same",
                                 "majority_ratio": majority_ratio})
            _do_checkpoint_and_eval(s1, model, processor, llm, eval_samples, args, out_dir,
                                    instruction=instruction, extract_fn=extract_fn, lora_sync=lora_sync)
            continue

        _no_mask   = torch.zeros(args.G, dtype=torch.bool, device=device)
        advantages, _ = grpo.baseline_rewards(rewards, invalid_mask=_no_mask)
        advantages = advantages.to(device)

        # (d) Log-probs + GRPO loss via (mini-)batched gradient accumulation ────
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        optimizer.zero_grad()
        accum_loss = 0.0
        mb = args.log_prob_mini_batch

        for mb_start in range(0, args.G, mb):
            mb_texts = texts[mb_start : mb_start + mb]
            mb_advs  = advantages[mb_start : mb_start + mb]
            if mb == 1:
                lps = [compute_log_prob_vl(model, processor, user_msg, image,
                                           prompt_len, mb_texts[0], args.gen_temp)]
            else:
                lps = compute_log_probs_vl_batch(model, processor, user_msg, image,
                                                 prompt_len, mb_texts, args.gen_temp, mb)
            mb_loss = -sum(a * lp for a, lp in zip(mb_advs, lps)) / args.G
            mb_loss.backward()
            accum_loss += mb_loss.item()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # (e) Optimizer step ──────────────────────────────────────────────────
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"]
        sv_rms = (sum(p.pow(2).mean().item() for p in trainable) / len(trainable)) ** 0.5
        print(f"  step {s1:3d} | loss={accum_loss:.4f} | "
              f"reward={rewards.mean():.3f} ({n_match}/{args.G}) | "
              f"ratio={majority_ratio:.2f} | ∥grad∥={float(grad_norm):.4f} | "
              f"sv_rms={sv_rms:.2e} | lr={cur_lr:.2e}")
        training_log.append({
            "step": s1, "loss": accum_loss, "reward_mean": rewards.mean().item(),
            "n_match": n_match, "majority_ratio": majority_ratio,
            "grad_norm": float(grad_norm), "sv_rms": sv_rms,
        })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _do_checkpoint_and_eval(s1, model, processor, llm, eval_samples, args, out_dir,
                                instruction=instruction, extract_fn=extract_fn, lora_sync=lora_sync)

    # 9. Find best checkpoint from per-step eval JSONs
    best_step, best_ckpt_acc = 0, acc0
    for jf in sorted(out_dir.glob("eval_step*.json")):
        try:
            d = json.loads(jf.read_text())
            if d.get("acc", 0) > best_ckpt_acc:
                best_ckpt_acc = d["acc"]
                best_step = d["step"]
        except Exception:
            pass
    print(f"\nBest checkpoint: step {best_step}  (eval acc={best_ckpt_acc:.4f})")

    # Load best checkpoint biases into model (and vLLM)
    best_ckpt_path = out_dir / f"steering_biases_step{best_step}.pt"
    if best_step > 0 and best_ckpt_path.exists():
        print(f"  Loading {best_ckpt_path.name} …")
        saved = torch.load(best_ckpt_path, map_location="cpu")
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in saved:
                    p.copy_(saved[name].to(p.device, p.dtype))
        _lora_req_final = None
        if llm is not None:
            if lora_sync is not None:
                _lora_req_final = lora_sync.sync(model)
            else:
                sync_bias_to_vllm_vl(llm, model)
    else:
        print("  No checkpoint found (step 0 was best or file missing) — using current weights")
        _lora_req_final = None
        if llm is not None:
            if lora_sync is not None:
                _lora_req_final = lora_sync.sync(model)
            else:
                sync_bias_to_vllm_vl(llm, model)

    # 10. Final evaluation (full dataset, best checkpoint)
    print(f"\n{'═'*64}\nFINAL EVAL ({len(dataset)} problems)  [best ckpt: step {best_step}]\n{'═'*64}")
    model.eval()
    acc_final, details_final = evaluate_vl(model, processor, llm, dataset,
                                           args.max_new_tokens, return_details=True,
                                           instruction=instruction, extract_fn=extract_fn,
                                           lora_request=_lora_req_final)
    print(f"★  final pass@1 = {acc_final:.4f}  (Δ vs step-0 = {acc_final - acc0:+.4f})")

    results = {
        "model_id": args.model_id, "num_steps": args.num_steps, "G": args.G, "lr": args.lr,
        "prompt_format": args.prompt_format,
        "acc_step0": acc0, "best_step": best_step, "best_ckpt_acc": best_ckpt_acc,
        "acc_final": acc_final, "delta": acc_final - acc0,
        "training_log": training_log,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    sv = {n: p.data.cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(sv, out_dir / "steering_biases.pt")
    print(f"\n  Results → {out_dir / 'results.json'}")
    print(f"  Biases  → {out_dir / 'steering_biases.pt'}  "
          f"({sum(v.numel() for v in sv.values()):,} params)")


def _do_checkpoint_and_eval(s1, model, processor, llm, eval_samples, args, out_dir,
                            instruction=None, extract_fn=None, lora_sync=None):
    if args.save_every > 0 and s1 % args.save_every == 0:
        ckpt = {n: p.data.cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(ckpt, out_dir / f"steering_biases_step{s1}.pt")
        torch.save(ckpt, out_dir / "steering_biases_latest.pt")
        if lora_sync is not None:
            model.save_pretrained(str(out_dir / f"lora_step{s1}"))
            model.save_pretrained(str(out_dir / "lora_latest"))

    do_eval = (args.eval_every > 0 and s1 % args.eval_every == 0
               and s1 >= args.eval_start)
    if not do_eval:
        return None

    model.eval()
    lora_req = None
    if llm is not None:
        if lora_sync is not None:
            lora_req = lora_sync.sync(model)
        else:
            sync_bias_to_vllm_vl(llm, model)
    acc, details_ = evaluate_vl(model, processor, llm, eval_samples,
                                args.max_new_tokens, return_details=True,
                                instruction=instruction, extract_fn=extract_fn,
                                lora_request=lora_req)
    print(f"★  step {s1:3d}  pass@1 = {acc:.4f}  ({acc*100:.2f}%)  "
          f"[{len(eval_samples)} problems]")
    if args.save_every == 0 or s1 % args.save_every != 0:
        ckpt = {n: p.data.cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(ckpt, out_dir / f"steering_biases_step{s1}.pt")
        torch.save(ckpt, out_dir / "steering_biases_latest.pt")
    (out_dir / f"eval_step{s1}.json").write_text(
        json.dumps({"step": s1, "acc": acc, "n": len(eval_samples),
                    "details": details_}, indent=2))
    return acc


if __name__ == "__main__":
    main()
