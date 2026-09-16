#!/usr/bin/env python3
"""TTRL training with bias-only steering vectors.

Imports directly from the official sub-repos (unmodified):
  GRPORewardProcessor  ← refs/steering-reasoning
  _majority_vote       ← refs/TTRL
  extract_answer, grade ← refs/TTRL

Adds a native down_proj.bias to the HF/vLLM MLP classes via runtime
monkey-patches (works on both ROCm and CUDA backends).
"""
import sys
import os
import time

# Ensure vLLM can serialize closures via pickle fallback
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── sys.path for both official sub-repos ─────────────────────────────────────
sys.path.insert(0, os.path.join(_ROOT, "refs", "steering-reasoning"))  # steering_reasoning.*
sys.path.insert(0, os.path.join(_ROOT, "refs", "TTRL", "verl"))        # verl.*
sys.path.insert(0, os.path.join(_ROOT, "pydeps"))                       # math_verify, sympy, …

# ── Stub the verl package hierarchy to bypass verl/__init__.py ───────────────
# verl/__init__.py imports ray + tensordict (heavy deps not in this container).
# We only need ttrl_utils and ttrl_math, which have no such deps.
# Namespace-package stubs let Python still locate sub-packages via __path__
# while skipping each __init__.py's problematic imports.
import types as _types
from pathlib import Path as _Path

def _verl_stub(name, rel):
    mod = _types.ModuleType(name)
    mod.__path__ = [str(_Path(_ROOT, "refs", "TTRL", "verl", *rel.split("/")))]
    mod.__package__ = name
    sys.modules[name] = mod
    return mod

_verl_stub("verl",                    "verl")
_verl_stub("verl.trainer",            "verl/trainer")
_verl_stub("verl.trainer.ppo",        "verl/trainer/ppo")
_verl_stub("verl.utils",              "verl/utils")
_verl_stub("verl.utils.reward_score", "verl/utils/reward_score")

# ── Import from official sub-repos (no modifications to those files) ──────────
from steering_reasoning.train.rl.reward_processor import GRPORewardProcessor  # noqa: E402
from verl.trainer.ppo.ttrl_utils import _majority_vote                          # noqa: E402
from verl.utils.reward_score.ttrl_math import extract_answer, grade, simplify_expression_string  # noqa: E402

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Reward check: mirrors _majority_vote's own normalization ──────────────────
# _majority_vote returns the SIMPLIFIED answer as pseudo_label.  Comparing raw
# extracted text to the simplified form via grade() fails on format mismatches.
# Instead simplify the extracted answer first (same pipeline), then compare.
def _reward_match(text, pseudo_label) -> float:
    """Return 1.0 if text's simplified answer matches pseudo_label, else 0.0."""
    raw = extract_answer(text)
    if raw is None:
        return 0.0
    try:
        simplified = simplify_expression_string(raw)
        if not simplified:
            return 0.0
        # String equality first (fast, exact match after same normalization)
        if simplified == pseudo_label:
            return 1.0
        # Fallback: sympy equality for different-but-equivalent forms
        return 1.0 if grade(simplified, pseudo_label) else 0.0
    except Exception:
        return 0.0


def shorter_better_reward(texts: list, is_correct: list, alpha: float = 0.5) -> list:
    """ShorterBetter reward: correct responses rewarded inversely to excess length.

    Correct responses get reward in [0, 1] based on how much longer they are
    than the shortest correct response in the group. Incorrect responses get -1.
    alpha controls how much the longest correct response is penalised:
      alpha=0.5 → longest correct gets reward 0.5 (never below 0)
      alpha=1.0 → longest correct gets reward 0.0

    Edge case: if no correct responses exist, all rewards are 0.0 (no gradient).
    """
    lengths = [len(t.split()) for t in texts]
    correct_lengths = [l for l, c in zip(lengths, is_correct) if c]

    if not correct_lengths:
        return [0.0] * len(texts)

    L_min = min(correct_lengths)
    L_max = max(lengths)
    denom = max(L_max - L_min, 1)

    rewards = []
    for length, correct in zip(lengths, is_correct):
        if not correct:
            rewards.append(-1.0)
        else:
            penalty = alpha * (length - L_min) / denom
            rewards.append(max(0.0, 1.0 - penalty))
    return rewards


def _try_grade(model_answer, reference_answer) -> bool:
    """grade() wrapper used for eval; handles None and other exceptions."""
    if model_answer is None:
        return False
    try:
        return bool(grade(model_answer, reference_answer))
    except Exception:
        return False


# ── Runtime patch: add down_proj.bias to HF Qwen2MLP ─────────────────────────
# Replaces bin/helpers/modify_bias_transformers.sh (which does sed on /opt/venv
# — not possible in a read-only container).
def patch_hf_for_bias():
    import torch.nn as nn
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    _orig = Qwen2MLP.__init__

    def _patched(self, config):
        _orig(self, config)
        old = self.down_proj  # nn.Linear(intermediate, hidden, bias=False)
        self.down_proj = nn.Linear(old.in_features, old.out_features, bias=True)
        # Weight is overwritten by from_pretrained; bias inits to zero (correct start).

    Qwen2MLP.__init__ = _patched
    print("HF: patched Qwen2MLP.__init__ for native down_proj.bias")

    # Same patch for LlamaMLP (config-based __init__, identical shape) --
    # only down_proj gets a bias, gate_proj/up_proj stay bias=config.mlp_bias.
    try:
        from transformers.models.llama.modeling_llama import LlamaMLP
        _orig_llama = LlamaMLP.__init__

        def _patched_llama(self, config):
            _orig_llama(self, config)
            old = self.down_proj
            self.down_proj = nn.Linear(old.in_features, old.out_features, bias=True)

        LlamaMLP.__init__ = _patched_llama
        print("HF: patched LlamaMLP.__init__ for native down_proj.bias")
    except ImportError:
        pass

    # Same patch for Olmo2MLP (config-based __init__, identical shape).
    try:
        from transformers.models.olmo2.modeling_olmo2 import Olmo2MLP
        _orig_olmo2 = Olmo2MLP.__init__

        def _patched_olmo2(self, config):
            _orig_olmo2(self, config)
            old = self.down_proj
            self.down_proj = nn.Linear(old.in_features, old.out_features, bias=True)

        Olmo2MLP.__init__ = _patched_olmo2
        print("HF: patched Olmo2MLP.__init__ for native down_proj.bias")
    except ImportError:
        pass


# ── Runtime patch: add down_proj.bias to vLLM Qwen2MLP ───────────────────────
# (skip_bias_add=False → bias IS applied internally; return_bias=True just also
# returns it to the caller which discards it — one clean addition).
def patch_vllm_for_bias():
    import vllm.model_executor.models.qwen2 as qm
    from vllm.model_executor.layers.linear import RowParallelLinear

    _orig_init = qm.Qwen2MLP.__init__

    def _patched_init(self, hidden_size, intermediate_size, hidden_act,
                      quant_config=None, prefix=""):
        _orig_init(self, hidden_size, intermediate_size, hidden_act,
                   quant_config=quant_config, prefix=prefix)
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=True,
            quant_config=quant_config, prefix=f"{prefix}.down_proj")

    qm.Qwen2MLP.__init__ = _patched_init

    _orig_lw = qm.Qwen2ForCausalLM.load_weights

    def _patched_lw(self, weights):
        loaded = _orig_lw(self, weights)
        for name, p in self.named_parameters():
            if name.endswith("mlp.down_proj.bias"):
                with torch.no_grad():
                    p.zero_()
                loaded.add(name)
        return loaded

    qm.Qwen2ForCausalLM.load_weights = _patched_lw
    print("vLLM: patched Qwen2MLP.__init__ + load_weights for native down_proj.bias")

    # Same patch for vLLM's LlamaMLP -- extra kwargs (bias/reduce_results/
    # disable_tp) vs Qwen2MLP, but down_proj construction pattern is identical.
    try:
        import vllm.model_executor.models.llama as lm

        _orig_llama_init = lm.LlamaMLP.__init__

        def _patched_llama_init(self, hidden_size, intermediate_size, hidden_act,
                                 quant_config=None, bias=False, prefix="",
                                 reduce_results=True, disable_tp=False):
            _orig_llama_init(self, hidden_size, intermediate_size, hidden_act,
                             quant_config=quant_config, bias=bias, prefix=prefix,
                             reduce_results=reduce_results, disable_tp=disable_tp)
            self.down_proj = RowParallelLinear(
                intermediate_size, hidden_size, bias=True,
                quant_config=quant_config, reduce_results=reduce_results,
                disable_tp=disable_tp, prefix=f"{prefix}.down_proj")

        lm.LlamaMLP.__init__ = _patched_llama_init

        _orig_llama_lw = lm.LlamaForCausalLM.load_weights

        def _patched_llama_lw(self, weights):
            loaded = _orig_llama_lw(self, weights)
            for name, p in self.named_parameters():
                if name.endswith("mlp.down_proj.bias"):
                    with torch.no_grad():
                        p.zero_()
                    loaded.add(name)
            return loaded

        lm.LlamaForCausalLM.load_weights = _patched_llama_lw
        print("vLLM: patched LlamaMLP.__init__ + load_weights for native down_proj.bias")
    except ImportError:
        pass

    # Same patch for vLLM's Olmo2MLP -- keyword-only (vllm_config, prefix)
    # signature, different from Qwen2MLP/LlamaMLP's positional style.
    try:
        import vllm.model_executor.models.olmo2 as om

        _orig_olmo2_init = om.Olmo2MLP.__init__

        def _patched_olmo2_init(self, *, vllm_config, prefix=""):
            _orig_olmo2_init(self, vllm_config=vllm_config, prefix=prefix)
            hidden_size = vllm_config.model_config.hf_config.hidden_size
            intermediate_size = vllm_config.model_config.hf_config.intermediate_size
            self.down_proj = RowParallelLinear(
                intermediate_size, hidden_size, bias=True,
                quant_config=vllm_config.quant_config, prefix=f"{prefix}.down_proj")

        om.Olmo2MLP.__init__ = _patched_olmo2_init

        _orig_olmo2_lw = om.Olmo2ForCausalLM.load_weights

        def _patched_olmo2_lw(self, weights):
            loaded = _orig_olmo2_lw(self, weights)
            for name, p in self.named_parameters():
                if name.endswith("mlp.down_proj.bias"):
                    with torch.no_grad():
                        p.zero_()
                    loaded.add(name)
            return loaded

        om.Olmo2ForCausalLM.load_weights = _patched_olmo2_lw
        print("vLLM: patched Olmo2MLP.__init__ + load_weights for native down_proj.bias")
    except ImportError:
        pass


# ── Sync trainable biases from HF model → vLLM ───────────────────────────────
# vLLM fuses Q+K+V into a single qkv_proj; v_proj.bias lives at offset
# q_dim + k_dim inside qkv_proj.bias.  down_proj.bias is in mlp.down_proj.
def sync_bias_to_vllm(llm, hf_model):
    down_proj_by_layer = {}
    v_proj_by_layer    = {}

    for name, p in hf_model.named_parameters():
        if not p.requires_grad:
            continue
        layer_idx = int(name.split(".")[2])
        if name.endswith("mlp.down_proj.bias"):
            down_proj_by_layer[layer_idx] = p.detach().cpu().float()
        elif name.endswith("self_attn.v_proj.bias"):
            v_proj_by_layer[layer_idx] = p.detach().cpu().float()

    v_offset = None
    if v_proj_by_layer:
        cfg      = hf_model.config
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        v_offset = cfg.num_attention_heads * head_dim + cfg.num_key_value_heads * head_dim

    def _do_sync(vllm_model):
        for layer_idx, bias_cpu in down_proj_by_layer.items():
            dp = vllm_model.model.layers[layer_idx].mlp.down_proj
            with torch.no_grad():
                dp.bias.copy_(bias_cpu.to(dp.bias.device, dp.bias.dtype))
        for layer_idx, bias_cpu in v_proj_by_layer.items():
            qkv   = vllm_model.model.layers[layer_idx].self_attn.qkv_proj
            v_dim = bias_cpu.shape[0]
            with torch.no_grad():
                qkv.bias[v_offset:v_offset + v_dim].copy_(
                    bias_cpu.to(qkv.bias.device, qkv.bias.dtype))

    llm.apply_model(_do_sync)


# ── Dataset paths (from refs/TTRL/verl/data/, already harvested to data/ttrl/) ─
DATASET_PATHS = {
    "math":       "data/ttrl/MATH-TTT/train.json",
    "aime":       "data/ttrl/AIME-TTT/train.json",
    "amc":        "data/ttrl/AMC-TTT/train.json",
    "gpqa":       "data/ttrl/GPQA-TTT/train.json",
    "math-l1":    "data/ttrl/MATH-L1-TTT/train.json",
    "math-l2":    "data/ttrl/MATH-L2-TTT/train.json",
    "math-l3":    "data/ttrl/MATH-L3-TTT/train.json",
    "math-l4":    "data/ttrl/MATH-L4-TTT/train.json",
    "math-l5":    "data/ttrl/MATH-L5-TTT/train.json",
    "deepscaler": "data/ttrl/DeepScaleR/train.json",  # Sinii et al. training set
}

# System prompt from both refs: steering-reasoning (train/rl/config.py) and TTRL examples
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def build_prompt_ids(tokenizer, problem: str) -> list:
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user",   "content": problem}],
        tokenize=False, add_generation_prompt=True,
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


# ── Log-prob computation (per-token mean, temperature-corrected) ───────────────
def compute_log_probs(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    resp_ids_list: list,
    mini_batch: int = 4,
    temperature: float = 1.0,
) -> torch.Tensor:
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    P = prompt_ids.shape[0]
    all_lp = []

    for start in range(0, len(resp_ids_list), mini_batch):
        batch = resp_ids_list[start:start + mini_batch]
        bs = len(batch)
        max_R = max(r.shape[0] for r in batch)
        T = P + max_R

        input_ids = torch.full((bs, T), pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros(bs, T,     dtype=torch.long,  device=device)
        resp_mask = torch.zeros(bs, T - 1, dtype=torch.float, device=device)

        for i, r in enumerate(batch):
            r_dev = r.to(device)
            R = r_dev.shape[0]
            end = P + R
            input_ids[i, :end] = torch.cat([prompt_ids, r_dev])
            attn_mask[i, :end] = 1
            resp_mask[i, P - 1:end - 1] = 1.0

        outputs  = model(input_ids=input_ids, attention_mask=attn_mask)
        logits   = outputs.logits[:, :-1] / temperature
        token_lp = (
            F.log_softmax(logits, dim=-1)
            .gather(2, input_ids[:, 1:].unsqueeze(2))
            .squeeze(2)
        )
        n_resp  = resp_mask.sum(dim=1).clamp(min=1)
        seq_lp  = (token_lp * resp_mask).sum(dim=1) / n_resp
        all_lp.append(seq_lp)

    return torch.cat(all_lp, dim=0)  # [G]


# ── Evaluation (greedy, using grading from refs/TTRL) ─────────────────────────
@torch.no_grad()
def evaluate(model, tokenizer, samples, max_new_tokens, batch_size, return_details=False,
             llm=None):
    """Evaluate greedy pass@1. Pass llm= for fast vLLM generation (recommended)."""
    correct = 0
    details = []

    if llm is not None:
        # Fast path: vLLM greedy generation (processes all samples in one call)
        from vllm import SamplingParams
        sync_bias_to_vllm(llm, model)
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user",   "content": s["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            )
            for s in samples
        ]
        sp   = SamplingParams(temperature=0, max_tokens=max_new_tokens, n=1)
        vout = llm.generate(prompts, sp)
        for i, (out, s) in enumerate(zip(vout, samples)):
            resp       = out.outputs[0].text
            predicted  = extract_answer(resp)
            is_correct = _try_grade(predicted, s["answer"])
            if is_correct:
                correct += 1
            if return_details:
                details.append({
                    "id":        s.get("id", i),
                    "prompt":    s["prompt"],
                    "answer":    s["answer"],
                    "predicted": predicted,
                    "response":  resp,
                    "correct":   is_correct,
                })
    else:
        # Slow path: HF batched generation (fallback when vLLM not available)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        device = next(model.parameters()).device
        for start in tqdm(range(0, len(samples), batch_size), desc="eval", leave=False):
            batch = samples[start:start + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user",   "content": s["prompt"]}],
                    tokenize=False, add_generation_prompt=True,
                )
                for s in batch
            ]
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=2048, add_special_tokens=False)
            out = model.generate(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
                do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=pad_id,
            )
            plen = enc["input_ids"].shape[1]
            for i, seq in enumerate(out):
                resp       = tokenizer.decode(seq[plen:], skip_special_tokens=True)
                predicted  = extract_answer(resp)
                is_correct = _try_grade(predicted, batch[i]["answer"])
                if is_correct:
                    correct += 1
                if return_details:
                    details.append({
                        "id":        batch[i].get("id", start + i),
                        "prompt":    batch[i]["prompt"],
                        "answer":    batch[i]["answer"],
                        "predicted": predicted,
                        "response":  resp,
                        "correct":   is_correct,
                    })

    acc = correct / len(samples)
    return (acc, details) if return_details else acc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="TTRL bias-only steering via official repos")
    ap.add_argument("--model_id",       default="Qwen/Qwen2.5-7B")
    ap.add_argument("--dataset",        default="math", choices=sorted(DATASET_PATHS))
    ap.add_argument("--num_steps",      type=int,   default=300)
    ap.add_argument("--G",              type=int,   default=64,   help="rollouts per step")
    ap.add_argument("--lr",             type=float, default=1e-4)
    ap.add_argument("--gen_temp",       type=float, default=0.7)
    ap.add_argument("--max_new_tokens", type=int,   default=3072)
    ap.add_argument("--log_mini_batch", type=int,   default=4)
    ap.add_argument("--eval_batch",     type=int,   default=8)
    ap.add_argument("--eval_every",     type=int,   default=20)
    ap.add_argument("--eval_start",     type=int,   default=0,    help="first step at which periodic eval runs")
    ap.add_argument("--save_every",     type=int,   default=0,    help="save checkpoint every N steps (0 = only at eval points)")
    ap.add_argument("--eval_n",         type=int,   default=100,  help="problems for periodic evals")
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--use_vllm",       action="store_true")
    ap.add_argument("--vllm_gpu_util",  type=float, default=0.55)
    ap.add_argument("--steering_at_layer", type=int, default=None,
                    help="single layer index to train (None = all layers)")
    ap.add_argument("--use_labels",     action="store_true",
                    help="use ground-truth labels from dataset instead of majority-vote pseudo-labels (Sinii et al. style)")
    ap.add_argument("--lr_scheduler",   default="constant", choices=["constant", "cosine"],
                    help="constant (default) or cosine annealing to 0")
    ap.add_argument("--reward_style",   default="grpo", choices=["grpo", "shorter_better", "vpo_k2"],
                    help="grpo=binary correct/incorrect; shorter_better=length-penalised reward; "
                         "vpo_k2=VPO with k=2 objectives [correctness, brevity]")
    ap.add_argument("--trainable_bias", default="down_proj",
                    help="Comma-separated bias types to train: down_proj, v_proj "
                         "(default: down_proj for backward compat)")
    ap.add_argument("--sb_alpha",       type=float, default=0.5,
                    help="ShorterBetter alpha: reward of longest correct response (0..1)")
    ap.add_argument("--step0_cache",    default=None)
    ap.add_argument("--output_dir",     default=None)
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir or
                   Path(_ROOT) / "outputs" / f"ttrl_steer_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    print(f"ARGS CHECK: use_labels={args.use_labels}  seed={args.seed}")

    # 1. Patch HF Qwen2 to add native down_proj.bias (replaces modify_bias_transformers.sh)
    patch_hf_for_bias()

    # 2. Load HF model + tokenizer
    print(f"\nLoading {args.model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        import flash_attn  # noqa: F401
        _attn_impl = "flash_attention_2"
    except ImportError:
        _attn_impl = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=_attn_impl,
    )
    model.eval()

    # down_proj.bias is trainable in every layer (all-layers mode), so
    # gradients must flow through the full depth -- without checkpointing,
    # autograd retains every layer's forward activations for the whole
    # compute_log_probs mini-batch, which is what was blowing past 63GB on
    # even a single mini-batch of 4x~3k-token sequences. use_cache must be
    # off; it's incompatible with checkpointing (and irrelevant here, this
    # is a single forward pass, not autoregressive generation).
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    # Standard PEFT/LoRA-style fix: with a frozen base model, none of the
    # checkpointed segments' *input* activations have requires_grad=True
    # (only the far-downstream down_proj.bias params do), which breaks
    # reentrant-checkpoint's backward graph entirely ("element 0 of tensors
    # does not require grad"). This hooks the embedding output to require
    # grad so checkpointing has a valid path through every frozen layer.
    model.enable_input_require_grads()

    # 3. Freeze everything; selected bias params are zero-inited and made trainable.
    #    --trainable_bias: comma-separated list of "down_proj" and/or "v_proj"
    #    --steering_at_layer: restrict to a single layer (None = all layers)
    bias_types = set(args.trainable_bias.replace(" ", "").split(","))
    # Map user-facing names → parameter name suffixes
    _SUFFIX_MAP = {
        "down_proj": "mlp.down_proj.bias",
        "v_proj":    "self_attn.v_proj.bias",
    }
    trainable_suffixes = {_SUFFIX_MAP[b] for b in bias_types if b in _SUFFIX_MAP}

    trainable = []
    for name, p in model.named_parameters():
        is_trainable = any(name.endswith(sfx) for sfx in trainable_suffixes)
        if args.steering_at_layer is not None and is_trainable:
            is_trainable = f"layers.{args.steering_at_layer}." in name
        if is_trainable:
            # Only zero-init down_proj.bias (it's a new param added by our MLP patch).
            # v_proj.bias exists in pretrained Qwen2 weights — preserve those values.
            if name.endswith("mlp.down_proj.bias"):
                torch.nn.init.zeros_(p)
            p.requires_grad_(True)
            trainable.append(p)
        else:
            p.requires_grad_(False)

    n_layers    = len(model.model.layers)
    n_trainable = sum(p.numel() for p in trainable)
    layer_desc  = (f"layer {args.steering_at_layer}" if args.steering_at_layer is not None
                   else "all layers")
    bias_desc   = " + ".join(sorted(bias_types))
    print(f"  Layers: {n_layers}  |  trainable: {n_trainable:,} params "
          f"({layer_desc} [{bias_desc}] bias)")

    # 4. Optionally load vLLM for fast generation
    #    (replaces modify_bias_vllm.sh + LLMRayActor from refs/steering-reasoning)
    llm = None
    if args.use_vllm:
        patch_vllm_for_bias()
        from vllm import LLM, SamplingParams
        print(f"\nInitializing vLLM (gpu_util={args.vllm_gpu_util}) …")
        llm = LLM(
            args.model_id,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=args.vllm_gpu_util,
            enforce_eager=True,
            max_model_len=args.max_new_tokens + 1024,
            disable_log_stats=True,
        )
        print("  vLLM ready")

    # 5. Optimizer + LR scheduler + GRPO reward processor
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_steps, eta_min=0.0)
        print(f"  LR scheduler: cosine annealing  {args.lr} → 0 over {args.num_steps} steps")
    else:
        scheduler = None
    grpo      = GRPORewardProcessor(num_generations=args.G)

    # VPO k=2 — inlined to avoid scipy/vpo import breaking vLLM serialization.
    # For m=1 (one solution per rollout): raw_score[i] = E_w[w·[c_i, b_i]]
    # Computed via Monte-Carlo with w~Dir(1,1); z-normalized within the group.
    _vpo_rng = None
    if args.reward_style == "vpo_k2":
        import numpy as _np
        _vpo_rng = _np.random.default_rng(args.seed)
        print("  VPO k=2 reward: [correctness, brevity] (inlined, no scipy)")

    def _vpo_k2_advantages(binary_correct, brevities, n_samples=200):
        scores = _np.array([[float(c), b] for c, b in zip(binary_correct, brevities)])
        w = _vpo_rng.dirichlet([1.0, 1.0], size=n_samples)  # (n_samples, 2)
        raw = (scores @ w.T).mean(axis=1)                   # (G,) E_w[w·r]
        std = raw.std()
        if std < 1e-6:
            return torch.zeros(len(binary_correct))
        return torch.tensor((raw - raw.mean()) / (std + 1e-6), dtype=torch.float32)

    # 6. Load dataset (from refs/TTRL/verl/data/, already harvested to data/ttrl/)
    data_path = Path(_ROOT) / DATASET_PATHS[args.dataset]
    samples   = json.loads(data_path.read_text())
    print(f"\nDataset: {args.dataset} — {len(samples)} problems  ({data_path})")

    eval_samples  = samples[:args.eval_n] if args.eval_n < len(samples) else samples
    train_samples = list(samples)
    random.shuffle(train_samples)
    device = next(model.parameters()).device

    # 7. Step-0 evaluation (label from dataset is used for scoring only — never seen in training)
    cache_path = Path(args.step0_cache) if args.step0_cache else None
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        acc0 = cached["acc_step0"]
        print(f"\n★  step 0  pass@1 = {acc0:.4f}  [cached]")
    else:
        print("\nEVAL step 0 (steering bias = 0 → base model) …")
        acc0, details0 = evaluate(model, tokenizer, eval_samples, args.max_new_tokens,
                                  args.eval_batch, return_details=True, llm=llm)
        print(f"★  step 0  pass@1 = {acc0:.4f}  ({acc0 * 100:.2f}%)")
        if cache_path:
            cache_path.write_text(json.dumps({"acc_step0": acc0, "n": len(eval_samples),
                                              "details": details0}, indent=2))

    if args.num_steps == 0:
        print("\nnum_steps=0 — step0 eval done, exiting.")
        return

    # 8. Training loop
    print(f"\n{'═'*64}")
    print(f"TRAIN {args.num_steps} steps | G={args.G} | lr={args.lr} | "
          f"dataset={args.dataset} | {'vLLM' if llm else 'HF'} generation")
    print(f"{'═'*64}")

    training_log = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_loop_start = time.time()

    for step in range(args.num_steps):
        step_start = time.time()
        sample         = train_samples[step % len(train_samples)]
        problem        = sample["prompt"]
        prompt_ids_lst = build_prompt_ids(tokenizer, problem)
        prompt_ids     = torch.tensor(prompt_ids_lst, dtype=torch.long)

        # (a) Generate G rollouts ─────────────────────────────────────────────
        print(f"  step {step+1:3d}: sampling {args.G} rollouts …", flush=True)

        if llm is not None:
            # vLLM fast path: sync bias first (steering-reasoning's weight-update mechanism)
            sync_bias_to_vllm(llm, model)
            sp = SamplingParams(temperature=args.gen_temp,
                                max_tokens=args.max_new_tokens, n=args.G)
            vllm_out = llm.generate([{"prompt_token_ids": prompt_ids_lst}], sp)
            texts    = [o.text for o in vllm_out[0].outputs]
            resp_ids = [torch.tensor(list(o.token_ids), dtype=torch.long)
                        for o in vllm_out[0].outputs]
        else:
            # HF fallback (no vLLM)
            gen_cfg = getattr(model, "generation_config", None)
            raw_eos = getattr(gen_cfg, "eos_token_id", tokenizer.eos_token_id)
            eos_set = set(raw_eos if isinstance(raw_eos, list) else [raw_eos])
            pad_id  = tokenizer.pad_token_id or tokenizer.eos_token_id
            P       = len(prompt_ids_lst)
            texts, resp_ids = [], []
            model.eval()
            with torch.no_grad():
                batch_ids  = prompt_ids.unsqueeze(0).repeat(args.G, 1).to(device)
                batch_mask = torch.ones(args.G, P, dtype=torch.long, device=device)
                out = model.generate(
                    input_ids=batch_ids, attention_mask=batch_mask,
                    do_sample=True, temperature=args.gen_temp,
                    max_new_tokens=args.max_new_tokens, pad_token_id=pad_id,
                )
            for seq in out:
                r = seq[P:].cpu()
                hits = [p[0, 0].item() for eid in eos_set
                        for p in [(r == eid).nonzero(as_tuple=False)] if p.numel()]
                if hits:
                    r = r[:min(hits) + 1]
                resp_ids.append(r)
                texts.append(tokenizer.decode(r, skip_special_tokens=True))

        # (b) Reward computation — labeled (Sinii et al.) or TTRL majority-vote ──
        if args.use_labels:
            true_answer  = sample["answer"]
            rewards = torch.tensor(
                [float(_try_grade(extract_answer(t), true_answer)) for t in texts],
                dtype=torch.float32, device=device,
            )
            pseudo_label   = true_answer
            majority_ratio = rewards.mean().item()
        else:
            # Majority vote — from refs/TTRL/verl/verl/trainer/ppo/ttrl_utils.py
            pseudo_label, majority_ratio = _majority_vote(texts)
            if not pseudo_label or pseudo_label == "None":
                print(f"  step {step+1:3d}: skip — no valid answers extracted")
                training_log.append({"step": step+1, "skip": "no_answer"})
                continue
            binary_correct = [_reward_match(t, pseudo_label) == 1.0 for t in texts]
            if args.reward_style == "shorter_better":
                raw_rewards = shorter_better_reward(texts, binary_correct, alpha=args.sb_alpha)
            else:
                # grpo and vpo_k2 both use binary 0/1 for the skip-check tensor
                raw_rewards = [float(c) for c in binary_correct]
            rewards = torch.tensor(raw_rewards, dtype=torch.float32, device=device)

        n_match = int(rewards.sum().item())

        # (c) GRPO advantages — from refs/steering-reasoning (reward_processor.py, unmodified) ──
        if rewards.std() < 1e-6:
            s1 = step + 1
            print(f"  step {s1:3d}: skip — all {args.G} rewards equal "
                  f"(ratio={majority_ratio:.2f})")
            training_log.append({"step": s1, "skip": "all_same",
                                 "majority_ratio": majority_ratio})
            if args.save_every > 0 and s1 % args.save_every == 0:
                ckpt = {n: p.data.cpu() for n, p in model.named_parameters()
                        if p.requires_grad and any(n.endswith(sfx) for sfx in trainable_suffixes)}
                torch.save(ckpt, out_dir / f"steering_biases_step{s1}.pt")
                torch.save(ckpt, out_dir / "steering_biases_latest.pt")
            do_eval = (args.eval_every > 0
                       and s1 % args.eval_every == 0
                       and s1 >= args.eval_start)
            if do_eval:
                model.eval()
                acc, details_ = evaluate(model, tokenizer, eval_samples,
                                         args.max_new_tokens, args.eval_batch,
                                         return_details=True, llm=llm)
                print(f"★  step {s1:3d}  pass@1 = {acc:.4f}  ({acc*100:.2f}%)  "
                      f"[{len(eval_samples)} problems]")
                ckpt = {n: p.data.cpu() for n, p in model.named_parameters()
                        if p.requires_grad and any(n.endswith(sfx) for sfx in trainable_suffixes)}
                if args.save_every == 0 or s1 % args.save_every != 0:
                    torch.save(ckpt, out_dir / f"steering_biases_step{s1}.pt")
                    torch.save(ckpt, out_dir / "steering_biases_latest.pt")
                eval_record = {"step": s1, "acc": acc, "n": len(eval_samples),
                               "details": details_}
                (out_dir / f"eval_step{s1}.json").write_text(json.dumps(eval_record, indent=2))
            continue

        if args.reward_style == "vpo_k2":
            max_words  = args.max_new_tokens
            brevities  = [1.0 - min(len(t.split()), max_words) / max_words for t in texts]
            advantages = _vpo_k2_advantages(binary_correct, brevities).to(device)
        else:
            # Pass a zero-mask instead of None: GRPORewardProcessor.baseline_rewards
            # has a bug where std_for_metrics is only assigned in the else-branch
            # (invalid_mask is not None), so passing None → UnboundLocalError.
            _no_mask = torch.zeros(args.G, dtype=torch.bool, device=device)
            advantages, _ = grpo.baseline_rewards(rewards, invalid_mask=_no_mask)
            advantages    = advantages.to(device)

        # (d) Log-probs + GRPO loss + backward ───────────────────────────────
        # train() not eval(): HF's gradient_checkpointing_enable() gates
        # checkpointing behind self.training, so eval() here was silently
        # disabling it and retaining full-depth activations regardless of
        # the checkpointing call -- root cause of the ~63GB OOM. No dropout/
        # batchnorm in this model to worry about train-mode side effects on.
        model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        optimizer.zero_grad()
        accum_loss = 0.0
        mb = args.log_mini_batch

        for start in range(0, args.G, mb):
            end       = min(start + mb, args.G)
            batch_lp  = compute_log_probs(
                model, tokenizer, prompt_ids, resp_ids[start:end],
                mini_batch=mb, temperature=args.gen_temp,
            )
            batch_adv  = advantages[start:end]
            batch_loss = -(batch_adv * batch_lp).mean() * (end - start) / args.G
            batch_loss.backward()
            accum_loss += batch_loss.item()

        # (e) Update ──────────────────────────────────────────────────────────
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"]
        sv_rms = (sum(p.pow(2).mean().item() for p in trainable) / len(trainable)) ** 0.5
        len_suffix = ""
        if args.reward_style == "shorter_better":
            lengths = [len(t.split()) for t in texts]
            correct_lengths = [l for l, c in zip(lengths, binary_correct) if c]
            mean_len = sum(lengths) / len(lengths)
            min_correct = min(correct_lengths) if correct_lengths else 0
            len_suffix = f" | len={mean_len:.0f} (min_correct={min_correct})"
        print(
            f"  step {step+1:3d} | loss={accum_loss:.4f} | "
            f"reward={rewards.mean():.3f} ({n_match}/{args.G}) | "
            f"ratio={majority_ratio:.2f} | "
            f"∥grad∥={float(grad_norm):.4f} | sv_rms={sv_rms:.2e} | lr={cur_lr:.2e}"
            + len_suffix
        )
        log_entry = {
            "step": step+1, "loss": accum_loss,
            "reward_mean": rewards.mean().item(), "n_match": n_match,
            "majority_ratio": majority_ratio,
            "grad_norm": float(grad_norm), "sv_rms": sv_rms,
            "step_time_s": time.time() - step_start,
        }
        if args.reward_style == "shorter_better":
            lengths = [len(t.split()) for t in texts]
            correct_lengths = [l for l, c in zip(lengths, binary_correct) if c]
            log_entry["mean_resp_len"]    = sum(lengths) / len(lengths)
            log_entry["mean_correct_len"] = (sum(correct_lengths) / len(correct_lengths)
                                             if correct_lengths else 0.0)
        training_log.append(log_entry)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # (f) Periodic checkpoint (save_every, independent of eval) ──────────
        s1 = step + 1
        if args.save_every > 0 and s1 % args.save_every == 0:
            ckpt = {n: p.data.cpu() for n, p in model.named_parameters()
                    if p.requires_grad and any(n.endswith(sfx) for sfx in trainable_suffixes)}
            torch.save(ckpt, out_dir / f"steering_biases_step{s1}.pt")
            torch.save(ckpt, out_dir / "steering_biases_latest.pt")

        # (g) Periodic eval ───────────────────────────────────────────────────
        do_eval = (args.eval_every > 0
                   and s1 % args.eval_every == 0
                   and s1 >= args.eval_start)
        if do_eval:
            model.eval()
            acc, details_ = evaluate(model, tokenizer, eval_samples,
                                     args.max_new_tokens, args.eval_batch,
                                     return_details=True, llm=llm)
            print(f"★  step {s1:3d}  pass@1 = {acc:.4f}  ({acc*100:.2f}%)  "
                  f"[{len(eval_samples)} problems]")
            ckpt = {n: p.data.cpu() for n, p in model.named_parameters()
                    if p.requires_grad and any(n.endswith(sfx) for sfx in trainable_suffixes)}
            # only write checkpoint here if save_every didn't already do it
            if args.save_every == 0 or s1 % args.save_every != 0:
                torch.save(ckpt, out_dir / f"steering_biases_step{s1}.pt")
                torch.save(ckpt, out_dir / "steering_biases_latest.pt")
            eval_record = {"step": s1, "acc": acc, "n": len(eval_samples),
                           "details": details_}
            (out_dir / f"eval_step{s1}.json").write_text(json.dumps(eval_record, indent=2))

    # 9. Final evaluation — full dataset (all problems, not just eval_n subsample)
    print(f"\n{'═'*64}\nEVAL  final ({len(samples)} problems)\n{'═'*64}")
    model.eval()
    acc_final, details_final = evaluate(model, tokenizer, samples,
                                        args.max_new_tokens, args.eval_batch,
                                        return_details=True, llm=llm)
    print(f"\n★  final pass@1 = {acc_final:.4f}  "
          f"(Δ = {acc_final - acc0:+.4f}  vs step-0 subsample)")
    (out_dir / "eval_final.json").write_text(
        json.dumps({"step": args.num_steps, "acc": acc_final,
                    "n": len(samples), "details": details_final}, indent=2))

    # 10. Save results + steering biases
    train_loop_elapsed = time.time() - train_loop_start
    peak_mem_gb = (torch.cuda.max_memory_allocated() / 1e9
                   if torch.cuda.is_available() else None)
    results = {
        "model_id":    args.model_id,
        "dataset":     args.dataset,
        "num_steps":   args.num_steps,
        "G":           args.G,
        "lr":          args.lr,
        "trainable_bias":      args.trainable_bias,
        "steering_at_layer":   args.steering_at_layer,
        "n_trainable_params":  n_trainable,
        "train_loop_wall_clock_s": train_loop_elapsed,
        "mean_step_time_s":    train_loop_elapsed / max(args.num_steps, 1),
        "peak_gpu_memory_gb":  peak_mem_gb,
        "acc_step0":        acc0,          # subsample (eval_n problems)
        "acc_final":        acc_final,     # full dataset (all problems)
        "delta_vs_subsample": acc_final - acc0,
        "training_log": training_log,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    sv_dict = {n: p.data.cpu() for n, p in model.named_parameters()
               if p.requires_grad}
    torch.save(sv_dict, out_dir / "steering_biases.pt")

    print(f"\n  Results         → {out_dir / 'results.json'}")
    print(f"  Steering biases → {out_dir / 'steering_biases.pt'}")
    print(f"  ({len(sv_dict)} down_proj.bias tensors, "
          f"{sum(v.numel() for v in sv_dict.values()):,} total params)")


if __name__ == "__main__":
    main()
