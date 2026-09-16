#!/usr/bin/env python3
"""TTRL bias-only training on MMAU with Qwen2.5-Omni-7B or Qwen2-Audio-7B-Instruct.

Audio analog of train_ttrl_steer_mathvista.py:
  - Model : Qwen/Qwen2.5-Omni-7B (default, already cached) or Qwen/Qwen2-Audio-7B-Instruct
  - Data  : AudioLLMs/MMAU-mini (1,000 MCQ audio problems)
  - Reward: GRPO binary via majority-vote on extracted option letters (A/B/C/D)
  - Grade : letter extraction — all questions are MCQ
"""
import sys, os
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "refs", "steering-reasoning"))
sys.path.insert(0, os.path.join(_ROOT, "pydeps"))

import argparse, json, math, random, re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, Audio as HFAudio
try:
    import pyarrow as _pa
    _HAS_PA = True
except ImportError:
    _HAS_PA = False
from tqdm import tqdm
from transformers import AutoProcessor

from steering_reasoning.train.rl.reward_processor import GRPORewardProcessor

# VLMEvalKit can_infer, ported for AI2D -- reused unchanged (dataset- and
# modality-agnostic, operates on any {letter: text} choices dict; same reuse
# eval_arc.py already does). See eval_ai2d.py for the verbatim port +
# self-test. Mirrors the grader fix applied to eval_mmau.py.
from eval_ai2d import can_infer


TARGET_SR = 16000  # Whisper requires 16 kHz

COT_INSTRUCTION = (
    "Listen to the audio and answer the multiple-choice question.\n"
    "Choose the best option (A, B, C, or D).\n"
    "Use exactly this format:\n"
    "Reasoning: <brief reasoning>\n"
    "Final answer: <letter>\n\n"
    "Example:\n"
    "Question: What instrument is playing?\n"
    "Options: (A) violin  (B) trumpet  (C) piano  (D) drums\n"
    "Reasoning: The sound has the bright, brassy timbre of a brass instrument.\n"
    "Final answer: B"
)


# ── MMAU dataset helpers ──────────────────────────────────────────────────────

def _resample(arr: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    try:
        from scipy.signal import resample_poly
        gcd = math.gcd(from_sr, to_sr)
        return resample_poly(arr, to_sr // gcd, from_sr // gcd).astype(np.float32)
    except ImportError:
        import resampy
        return resampy.resample(arr, from_sr, to_sr).astype(np.float32)


def get_audio(sample: dict) -> Tuple[np.ndarray, int]:
    a = sample["audio"]
    if isinstance(a, dict) and "array" in a:
        arr = np.array(a["array"], dtype=np.float32)
        sr  = int(a["sampling_rate"])
    elif isinstance(a, dict) and ("bytes" in a or "path" in a):
        import io
        buf = a.get("bytes")
        if not buf:
            with open(a["path"], "rb") as f:
                buf = f.read()
        try:
            import torchaudio
            waveform, sr = torchaudio.load(io.BytesIO(buf))
            arr = waveform.squeeze(0).numpy().astype(np.float32)
        except Exception:
            import scipy.io.wavfile as _wav
            sr, arr = _wav.read(io.BytesIO(buf))
            arr = arr.astype(np.float32)
    else:
        raise ValueError(f"Unknown audio format: {type(a)}")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)  # stereo → mono
    if sr != TARGET_SR:
        arr = _resample(arr, sr, TARGET_SR)
        sr  = TARGET_SR
    return arr, sr


def format_options(options) -> str:
    letters = "ABCD"
    if isinstance(options, dict):
        return "  ".join(f"({k}) {v}" for k, v in sorted(options.items()))
    elif isinstance(options, (list, tuple)):
        return "  ".join(f"({letters[i]}) {v}" for i, v in enumerate(options))
    return str(options)


def get_question_with_options(sample: dict) -> str:
    q   = str(sample.get("question", "")).strip()
    opt = sample.get("options") or sample.get("choices") or []
    return f"{q}\nOptions: {format_options(opt)}" if opt else q


def get_gt_letter(sample: dict) -> str:
    ans = str(sample.get("answer", "")).strip()
    if re.match(r"^[A-D]$", ans, re.IGNORECASE):
        return ans.upper()
    if re.match(r"^[1-4]$", ans):
        return "ABCD"[int(ans) - 1]
    if re.match(r"^[0-3]$", ans):
        return "ABCD"[int(ans)]
    opts = sample.get("options") or sample.get("choices") or []
    if opts:
        norm = ans.lower().strip()
        lst  = list(opts.values()) if isinstance(opts, dict) else list(opts)
        for i, o in enumerate(lst):
            if norm == str(o).lower().strip():
                return "ABCD"[i]
    m = re.search(r"\b([A-D])\b", ans, re.IGNORECASE)
    return m.group(1).upper() if m else ""


# ── Scoring ───────────────────────────────────────────────────────────────────

def build_choices(sample: dict) -> dict:
    """{'A': option_text, 'B': option_text, ...} -- same positional A/B/C/D
    lettering that get_gt_letter() / format_options() above already use for
    MMAU's raw options/choices field (list or dict, letters assigned by
    iteration order). Used as the 'choices' dict for VLMEvalKit's can_infer
    (see eval_ai2d.py) -- MMAU-mini is a 4-option MCQ dataset just like
    AI2D/ARC, can_infer is dataset- and modality-agnostic."""
    opts = sample.get("options") or sample.get("choices") or []
    lst = list(opts.values()) if isinstance(opts, dict) else list(opts)
    return {"ABCD"[i]: v for i, v in enumerate(lst)}


def extract_answer_letter(response: str, sample: dict) -> str:
    """VLMEvalKit can_infer-based extraction (vlmeval/utils/matching_util.py),
    ported verbatim and reused unchanged from eval_ai2d.py -- same reuse
    eval_arc.py already does. Replaces the old bespoke regex parser, which
    only recognized this project's own 'Final answer: X' / 'Answer: X'
    prompt phrasing and silently mis-scored anything else (bare letter,
    '(X)', or narrative responses with no explicit 'answer' cue).
    can_infer() success -> that letter; failure/'Z' sentinel -> 'FAILED'
    (never equals a real gold letter, i.e. graded incorrect, never raises).
    Used both for final/checkpoint grading AND for the majority-vote
    pseudo-labeling that drives GRPO reward during training -- same grader
    everywhere, no train/eval skew."""
    choices = build_choices(sample)
    opt = can_infer(response, choices) if choices else False
    if not opt or opt == "Z":
        return "FAILED"
    return opt


def score_mmau(pred: str, sample: dict, gt_letter: str) -> bool:
    return bool(gt_letter) and extract_answer_letter(pred, sample) == gt_letter.upper()


def majority_vote_mmau(responses: List[str], sample: dict) -> Tuple[Optional[str], float]:
    letters = [extract_answer_letter(r, sample) for r in responses]
    letters = [l for l in letters if l and l != "FAILED"]
    if not letters:
        return None, 0.0
    vote, count = Counter(letters).most_common(1)[0]
    return vote, count / len(responses)


# ── HF patch: add down_proj.bias to Qwen2MLP (text decoder) ──────────────────

def patch_hf_for_bias_audio():
    import torch.nn as nn

    def _add_bias_to_mlp_cls(cls, label):
        _orig = cls.__init__
        def _patched(self, config):
            _orig(self, config)
            if self.down_proj.bias is None:
                old = self.down_proj
                self.down_proj = nn.Linear(old.in_features, old.out_features, bias=True)
        cls.__init__ = _patched
        print(f"HF: patched {label} for down_proj.bias")

    # Qwen2MLP (used by Qwen2-Audio and Qwen2.5-Omni text decoders)
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
        _add_bias_to_mlp_cls(Qwen2MLP, "Qwen2MLP")
    except (ImportError, AttributeError) as e:
        print(f"HF patch warning (Qwen2MLP): {e}")

    # Qwen2.5-Omni may use its own MLP subclass — patch it too if present
    for mod_path in ("transformers.models.qwen2_5_omni.modeling_qwen2_5_omni",):
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            for cls_name in ("Qwen2MLP", "Qwen2_5OmniThinkerMLP"):
                if hasattr(mod, cls_name):
                    _add_bias_to_mlp_cls(getattr(mod, cls_name),
                                         f"{mod_path.split('.')[-1]}.{cls_name}")
        except (ImportError, AttributeError):
            pass


# ── vLLM patch: add down_proj.bias ───────────────────────────────────────────

def patch_vllm_for_bias_audio():
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

    patched = False
    # Try model-specific modules first (Omni, then Audio, then base Qwen2)
    for vllm_mod_name, lm_cls_name in [
        ("vllm.model_executor.models.qwen2_5_omni", "Qwen2_5OmniModel"),
        ("vllm.model_executor.models.qwen2_audio",  "Qwen2AudioForConditionalGeneration"),
    ]:
        try:
            import importlib
            qm_mod = importlib.import_module(vllm_mod_name)
            for cls_name in ("Qwen2MLP",):
                if hasattr(qm_mod, cls_name):
                    getattr(qm_mod, cls_name).__init__ = \
                        _make_mlp_patch(getattr(qm_mod, cls_name).__init__)
                    patched = True
            if hasattr(qm_mod, lm_cls_name):
                getattr(qm_mod, lm_cls_name).load_weights = \
                    _make_lw_patch(getattr(qm_mod, lm_cls_name).load_weights)
        except (ImportError, AttributeError) as e:
            print(f"  vLLM patch ({vllm_mod_name}): {e}")

    if not patched:
        import vllm.model_executor.models.qwen2 as qm
        qm.Qwen2MLP.__init__ = _make_mlp_patch(qm.Qwen2MLP.__init__)
        qm.Qwen2ForCausalLM.load_weights = _make_lw_patch(qm.Qwen2ForCausalLM.load_weights)

    print("vLLM: patched audio model for native down_proj.bias")


# ── Sync HF → vLLM (robust layer-path detection) ─────────────────────────────

_sync_diag_done = False

def sync_bias_to_vllm_audio(llm, hf_model):
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
            # Qwen2.5-Omni thinker paths (checked first — most specific)
            ("x.thinker.model.layers",              lambda x: x.thinker.model.layers),
            ("x.model.thinker.model.layers",        lambda x: x.model.thinker.model.layers),
            ("x.thinker.layers",                    lambda x: x.thinker.layers),
            # Qwen2-Audio / generic paths
            ("x.language_model.model.layers",       lambda x: x.language_model.model.layers),
            ("x.model.language_model.model.layers", lambda x: x.model.language_model.model.layers),
            ("x.model.model.layers",                lambda x: x.model.model.layers),
            ("x.model.layers",                      lambda x: x.model.layers),
            ("x.language_model.layers",             lambda x: x.language_model.layers),
        ]
        for desc, getter in candidates:
            try:
                result = getter(m)
                _path_used[0] = desc
                return result
            except AttributeError:
                continue
        raise AttributeError("Cannot locate decoder layers in vLLM audio model")

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
            dp0     = layers[0].mlp.down_proj
            hf_norm = by_layer[0].norm().item() if 0 in by_layer else float("nan")
            vl_norm = dp0.bias.float().norm().item()
            print(f"  [sync diag] layer 0 bias norm — HF: {hf_norm:.4f}  vLLM: {vl_norm:.4f}",
                  flush=True)

    llm.apply_model(_do_sync)
    if not _sync_diag_done:
        _sync_diag_done = True


# ── Log-prob for one rollout (audio forward pass, gradients enabled) ──────────
#
# Qwen2-Audio merges audio embeddings into the text sequence inside forward(),
# so logits.shape[1] may exceed input_ids.shape[1].  Response tokens are always
# at the END of both sequences, so we index from the back.

def _proc_encode(processor, text, arr, sr, is_omni, **kwargs):
    if is_omni:
        return processor(text=text, audio=[arr], return_tensors="pt", **kwargs)
    return processor(text=text, audios=[arr], sampling_rate=sr, return_tensors="pt", **kwargs)


def _audio_block(is_omni):
    return ({"type": "audio", "audio": "placeholder"} if is_omni
            else {"type": "audio", "audio_url": "placeholder"})


def compute_log_prob_audio(model, processor, user_msg: dict,
                            arr: np.ndarray, sr: int,
                            prompt_text_len: int, response_text: str,
                            temperature: float = 1.0,
                            is_omni: bool = False) -> torch.Tensor:
    device  = next(model.parameters()).device
    messages = [user_msg, {"role": "assistant", "content": response_text}]
    full_text = processor.apply_chat_template(messages, tokenize=False)
    enc = _proc_encode(processor, [full_text], arr, sr, is_omni)

    kwargs = {
        "input_ids":      enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
    }
    for key in ("input_features", "feature_attention_mask"):
        if enc.get(key) is not None:
            kwargs[key] = enc[key].to(device)

    logits = model(**kwargs).logits[0, :-1] / temperature  # [T_exp-1, V]
    ids    = enc["input_ids"][0]                           # [T_text]

    # n_resp: how many response tokens in the text sequence
    n_resp = max(ids.shape[0] - prompt_text_len, 1)
    resp_ids    = ids[-n_resp:].to(device)   # [n_resp]
    resp_logits = logits[-n_resp:]           # [n_resp, V]  (last n_resp in expanded seq)

    tok_lp = F.log_softmax(resp_logits, dim=-1).gather(
        1, resp_ids.unsqueeze(1)).squeeze(1)
    return tok_lp.mean()


# ── Evaluation (vLLM fast path or HF fallback) ───────────────────────────────

@torch.no_grad()
def evaluate_audio(model, processor, llm, samples, max_new_tokens,
                   return_details=False, is_omni=False):
    device = next(model.parameters()).device

    if llm is not None:
        from vllm import SamplingParams
        sp       = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        requests = []
        for s in samples:
            arr, sr  = get_audio(s)
            question = get_question_with_options(s)
            msgs = [{"role": "user", "content": [
                _audio_block(is_omni),
                {"type": "text", "text": f"{COT_INSTRUCTION}\nQuestion: {question}"},
            ]}]
            pt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            requests.append({"prompt": pt, "multi_modal_data": {"audio": (arr, sr)}})
        vllm_outs = llm.generate(requests, sp)
        texts = [o.outputs[0].text for o in vllm_outs]
    else:
        texts = []
        for s in tqdm(samples, desc="eval", leave=False):
            arr, sr  = get_audio(s)
            question = get_question_with_options(s)
            msgs = [{"role": "user", "content": [
                _audio_block(is_omni),
                {"type": "text", "text": f"{COT_INSTRUCTION}\nQuestion: {question}"},
            ]}]
            pt  = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = _proc_encode(processor, [pt], arr, sr, is_omni)
            kw  = {"input_ids": enc["input_ids"].to(device),
                   "attention_mask": enc["attention_mask"].to(device)}
            for k in ("input_features", "feature_attention_mask"):
                if enc.get(k) is not None:
                    kw[k] = enc[k].to(device)
            out  = model.generate(**kw, do_sample=False, max_new_tokens=max_new_tokens,
                                  pad_token_id=processor.tokenizer.eos_token_id)
            n_in = enc["input_ids"].shape[1]
            resp = processor.tokenizer.decode(out[0][n_in:], skip_special_tokens=True)
            texts.append(resp)

    correct = 0
    details = []
    for s, resp in zip(samples, texts):
        gt = get_gt_letter(s)
        ok = score_mmau(resp, s, gt)
        if ok:
            correct += 1
        if return_details:
            details.append({
                "id":        s.get("id", len(details)),
                "question":  get_question_with_options(s),
                "answer":    gt,
                "response":  resp,
                "predicted": extract_answer_letter(resp, s),
                "correct":   ok,
            })

    acc = correct / len(samples)
    return (acc, details) if return_details else acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="TTRL bias-only steering on MMAU (audio)")
    ap.add_argument("--model_id",          default="Qwen/Qwen2.5-Omni-7B")
    ap.add_argument("--num_steps",         type=int,   default=200)
    ap.add_argument("--G",                 type=int,   default=32)
    ap.add_argument("--lr",                type=float, default=1e-3)
    ap.add_argument("--gen_temp",          type=float, default=0.7)
    ap.add_argument("--max_new_tokens",    type=int,   default=512)
    ap.add_argument("--eval_every",        type=int,   default=20)
    ap.add_argument("--eval_start",        type=int,   default=20)
    ap.add_argument("--save_every",        type=int,   default=20)
    ap.add_argument("--eval_n",            type=int,   default=500)
    ap.add_argument("--seed",              type=int,   default=42)
    ap.add_argument("--use_vllm",          action="store_true")
    ap.add_argument("--vllm_gpu_util",     type=float, default=0.35)
    ap.add_argument("--lr_scheduler",      default="constant",
                        choices=["constant", "cosine", "delayed_cosine"])
    ap.add_argument("--lr_warmup_steps",   type=int, default=None,
                    help="For delayed_cosine: flat LR for this many steps, then cosine decay.")
    ap.add_argument("--steering_at_layer", type=int,   default=None)
    ap.add_argument("--output_dir",        default=None)
    # Whisper: 30s → ~750 audio tokens after projector downsampling.
    # 1500 gives headroom for up to ~60s clips.
    ap.add_argument("--max_audio_tokens",  type=int,   default=1500)
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
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir or
                   Path(_ROOT) / "outputs" /
                   f"ttrl_mmau_{args.model_id.replace('/', '_')}_s{args.num_steps}_G{args.G}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Detect model family
    is_omni = "omni" in args.model_id.lower()

    # 1. Patch HF MLP
    patch_hf_for_bias_audio()

    # 2. Load model + processor
    print(f"\nLoading {args.model_id}  ({'Omni' if is_omni else 'Audio'}) …")
    n_visible = torch.cuda.device_count()
    _device_map = "balanced" if n_visible > 1 else {"": 0}
    print(f"GPUs visible: {n_visible}  →  device_map={_device_map!r}")

    if is_omni:
        from transformers import (Qwen2_5OmniThinkerForConditionalGeneration,
                                   Qwen2_5OmniProcessor)
        processor = Qwen2_5OmniProcessor.from_pretrained(args.model_id)
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map=_device_map,
        )
    else:
        from transformers import Qwen2AudioForConditionalGeneration
        processor = AutoProcessor.from_pretrained(args.model_id)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map=_device_map,
        )
    model.eval()

    # 3. Freeze all; zero-init and enable text-decoder down_proj.bias.
    # For Omni the path is thinker.model.layers.*; for Audio it is language_model.model.layers.*.
    # We match by suffix and exclude audio encoder layers (audio_tower / audio_encoder).
    _AUDIO_ENCODER_KEYS = ("audio_tower", "audio_encoder", "whisper", "visual")
    trainable = []
    if args.lora_r:
        # LoRA for MMAU: disable vLLM (audio LoRA vLLM hot-swap not supported)
        if args.use_vllm:
            print("  [WARNING] --lora_r disables vLLM for MMAU (audio LoRA sync not supported)")
            args.use_vllm = False
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
            if not any(k in name.lower() for k in _AUDIO_ENCODER_KEYS):
                p.requires_grad_(True)
                trainable.append(p)
            else:
                p.requires_grad_(False)
        # Enable gradient checkpointing on the text decoder
        lm = getattr(model, "thinker", None) or getattr(model, "language_model", None)
        if lm is not None:
            lm.gradient_checkpointing_enable()
    else:
        for name, p in model.named_parameters():
            is_trainable = (name.endswith("mlp.down_proj.bias")
                            and not any(k in name.lower() for k in _AUDIO_ENCODER_KEYS))
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
        if args.full_ft:
            mode = "full LM fine-tune (audio encoder frozen)"
        else:
            layer_desc = (f"layer {args.steering_at_layer}" if args.steering_at_layer is not None
                          else f"all {len(trainable)} layers")
            mode = f"{layer_desc} [down_proj.bias]"
        print(f"  Trainable: {n_trainable:,} params ({mode})")
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"  Trainable names (first 3): {trainable_names[:3]}")

    # 4. Optionally init vLLM
    llm = None
    lora_sync = None
    if args.use_vllm:
        # Patch vLLM's audio resampler to use scipy (librosa not installed in container)
        try:
            import vllm.multimodal.audio as _vllm_audio
            import scipy.signal as _scipy_signal
            from math import gcd as _gcd

            def _scipy_resample_audio(audio, orig_sr: int, target_sr: int):
                if orig_sr == target_sr:
                    return audio
                g = _gcd(int(orig_sr), int(target_sr))
                return _scipy_signal.resample_poly(
                    audio, target_sr // g, orig_sr // g
                ).astype(audio.dtype)

            _vllm_audio.resample_audio_librosa = _scipy_resample_audio
            print("vLLM: patched audio resampler (scipy fallback)")
        except Exception as _e:
            print(f"vLLM: audio resampler patch skipped ({_e})")
        patch_vllm_for_bias_audio()
        from vllm import LLM, SamplingParams
        print(f"\nInitializing vLLM (gpu_util={args.vllm_gpu_util}) …")
        vllm_max_len = args.max_audio_tokens + args.max_new_tokens + 256
        print(f"  max_audio_tokens={args.max_audio_tokens}, max_model_len={vllm_max_len}")
        llm = LLM(
            args.model_id,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=args.vllm_gpu_util,
            enforce_eager=True,
            max_model_len=vllm_max_len,
            disable_log_stats=True,
            limit_mm_per_prompt={"audio": 1},
        )
        print("  vLLM ready")

    # 5. Optimizer + scheduler + GRPO
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_steps, eta_min=0.0)
        print(f"  LR: cosine {args.lr} → 0 over {args.num_steps} steps")
    elif args.lr_scheduler == "delayed_cosine":
        import math as _math
        T_flat  = args.lr_warmup_steps if args.lr_warmup_steps is not None else args.num_steps // 2
        T_decay = max(1, args.num_steps - T_flat)
        def _delayed_cosine_lambda(step):
            if step < T_flat:
                return 1.0
            progress = (step - T_flat) / T_decay
            return 0.5 * (1.0 + _math.cos(_math.pi * min(progress, 1.0)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _delayed_cosine_lambda)
        print(f"  LR: delayed_cosine — flat {T_flat} steps, then cosine decay over {T_decay} steps")
    else:
        scheduler = None
    grpo = GRPORewardProcessor(num_generations=args.G)

    # 6. Load MMAU-mini
    # torchcodec is unavailable in this container so we can't use load_dataset().
    # Strategy:
    #   1. Load from the cached arrow file via pyarrow — no decoder invoked.
    #   2. Normalise field names: context→audio, instruction→question.
    # The actual HF schema is: context (Audio), instruction (str), choices (list), answer (str).
    print("\nLoading AudioLLMs/MMAU-mini …")
    dataset = None

    # 6a. Try loading from cached arrow file (fast, no torchcodec needed).
    _HF_DS_CACHE = os.environ.get(
        "HF_DATASETS_CACHE",
        os.path.join(os.environ.get("HF_HUB_CACHE", ""), "datasets"))
    _ARROW_CANDIDATES = [
        os.path.join(_HF_DS_CACHE,
                     "AudioLLMs___mmau-mini/MMAU-mini/0.0.0/"
                     "db5fcbf6ceafb0b9e9d0d733f909052e26175ed0/mmau-mini-test.arrow"),
    ]
    for arrow_path in _ARROW_CANDIDATES:
        if not os.path.exists(arrow_path):
            continue
        try:
            import pyarrow as pa
            # HF datasets caches in IPC streaming format; fall back to file format.
            try:
                with pa.ipc.open_stream(arrow_path) as reader:
                    table = reader.read_all()
            except pa.lib.ArrowInvalid:
                with pa.ipc.open_file(arrow_path) as reader:
                    table = reader.read_all()
            raw = table.to_pylist()
            # Normalise field names to match the rest of the script.
            dataset = []
            for r in raw:
                r2 = dict(r)
                if "context" in r2 and "audio" not in r2:
                    r2["audio"] = r2.pop("context")
                if "instruction" in r2 and "question" not in r2:
                    r2["question"] = r2.pop("instruction")
                dataset.append(r2)
            print(f"  Loaded from arrow cache: {len(dataset)} problems "
                  f"(fields: {list(dataset[0].keys())})")
            break
        except Exception as e:
            print(f"  Arrow load failed ({arrow_path}): {e}")

    # 6b. Fallback: HF datasets with decode=False.
    if dataset is None:
        for split in ("test", "validation", "train"):
            try:
                ds = load_dataset("AudioLLMs/MMAU-mini", split=split)
                ds = ds.cast_column("audio", HFAudio(decode=False))
                dataset = list(ds)
                print(f"  Loaded split='{split}', {len(dataset)} problems")
                break
            except Exception as e:
                print(f"  split '{split}' not found: {e}")

    if dataset is None:
        raise RuntimeError("Could not load AudioLLMs/MMAU-mini")

    eval_samples  = dataset[:args.eval_n]
    train_samples = list(dataset)
    random.shuffle(train_samples)
    device = next(model.parameters()).device
    print(f"  eval_n={args.eval_n}")

    # 7. Step-0 eval
    print("\nEVAL step 0 (baseline) …")
    if llm is not None:
        sync_bias_to_vllm_audio(llm, model)
    acc0, details0 = evaluate_audio(model, processor, llm, eval_samples,
                                    args.max_new_tokens, return_details=True,
                                    is_omni=is_omni)
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
        sample   = train_samples[step % len(train_samples)]
        arr, sr  = get_audio(sample)
        question = get_question_with_options(sample)

        user_msg = {"role": "user", "content": [
            _audio_block(is_omni),
            {"type": "text", "text": f"{COT_INSTRUCTION}\nQuestion: {question}"},
        ]}

        # Prompt length in text-token space (used for response masking in log-probs)
        prompt_text = processor.apply_chat_template(
            [user_msg], tokenize=False, add_generation_prompt=True)
        prompt_enc  = _proc_encode(processor, [prompt_text], arr, sr, is_omni)
        prompt_text_len = prompt_enc["input_ids"].shape[1]

        # (a) Generate G rollouts
        print(f"  step {step+1:3d}: sampling {args.G} rollouts …", flush=True)

        if llm is not None:
            sync_bias_to_vllm_audio(llm, model)
            sp = SamplingParams(temperature=args.gen_temp,
                                max_tokens=args.max_new_tokens, n=args.G)
            vllm_out = llm.generate([{
                "prompt":           prompt_text,
                "multi_modal_data": {"audio": (arr, sr)},
            }], sp)
            texts = [o.text for o in vllm_out[0].outputs]
        else:
            enc_dev = {}
            for k in ("input_ids", "attention_mask", "input_features", "feature_attention_mask"):
                if prompt_enc.get(k) is not None:
                    enc_dev[k] = prompt_enc[k].to(device)
            texts = []
            with torch.no_grad():
                for _ in range(args.G):
                    out = model.generate(**enc_dev, do_sample=True,
                                         temperature=args.gen_temp,
                                         max_new_tokens=args.max_new_tokens,
                                         pad_token_id=processor.tokenizer.eos_token_id)
                    n_in = prompt_enc["input_ids"].shape[1]
                    texts.append(processor.tokenizer.decode(
                        out[0][n_in:], skip_special_tokens=True))

        # (b) Pseudo-label: ground-truth or majority-vote
        if args.use_labels:
            pseudo_label = get_gt_letter(sample)
            majority_ratio = sum(1 for t in texts
                                 if extract_answer_letter(t, sample) == pseudo_label) / len(texts)
        else:
            pseudo_label, majority_ratio = majority_vote_mmau(texts, sample)
        s1 = step + 1
        if not pseudo_label:
            print(f"  step {s1:3d}: skip — no valid letters extracted")
            training_log.append({"step": s1, "skip": "no_answer"})
            continue

        correct_flags = [extract_answer_letter(t, sample) == pseudo_label for t in texts]
        raw_rewards   = [1.0 if c else -1.0 for c in correct_flags]
        rewards       = torch.tensor(raw_rewards, dtype=torch.float32, device=device)
        n_match       = int((rewards > 0).sum().item())

        # (c) GRPO advantages
        if rewards.std() < 1e-6:
            print(f"  step {s1:3d}: skip — all {args.G} rewards equal "
                  f"(ratio={majority_ratio:.2f})")
            training_log.append({"step": s1, "skip": "all_same",
                                 "majority_ratio": majority_ratio})
            _do_checkpoint_and_eval(s1, model, processor, llm, eval_samples, args, out_dir,
                                    is_omni=is_omni)
            continue

        _no_mask   = torch.zeros(args.G, dtype=torch.bool, device=device)
        advantages, _ = grpo.baseline_rewards(rewards, invalid_mask=_no_mask)
        advantages = advantages.to(device)

        # (d) Log-probs + GRPO loss via gradient accumulation
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        optimizer.zero_grad()
        accum_loss = 0.0

        for resp_text, adv in zip(texts, advantages):
            lp   = compute_log_prob_audio(model, processor, user_msg, arr, sr,
                                          prompt_text_len, resp_text, args.gen_temp,
                                          is_omni=is_omni)
            loss = -(adv * lp) / args.G
            loss.backward()
            accum_loss += loss.item()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # (e) Optimizer step
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
                                is_omni=is_omni)

    # 9. Final evaluation
    print(f"\n{'═'*64}\nFINAL EVAL ({len(dataset)} problems)\n{'═'*64}")
    model.eval()
    if llm is not None:
        sync_bias_to_vllm_audio(llm, model)
    acc_final, details_final = evaluate_audio(model, processor, llm, dataset,
                                              args.max_new_tokens, return_details=True,
                                              is_omni=is_omni)
    print(f"★  final pass@1 = {acc_final:.4f}  (Δ = {acc_final - acc0:+.4f})")

    results = {
        "model_id": args.model_id, "num_steps": args.num_steps, "G": args.G, "lr": args.lr,
        "acc_step0": acc0, "acc_final": acc_final, "delta": acc_final - acc0,
        "training_log": training_log,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    sv = {n: p.data.cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(sv, out_dir / "steering_biases.pt")
    print(f"\n  Results → {out_dir / 'results.json'}")
    print(f"  Biases  → {out_dir / 'steering_biases.pt'}  "
          f"({sum(v.numel() for v in sv.values()):,} params)")


def _do_checkpoint_and_eval(s1, model, processor, llm, eval_samples, args, out_dir,
                             is_omni=False):
    if args.save_every > 0 and s1 % args.save_every == 0:
        ckpt = {n: p.data.cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(ckpt, out_dir / f"steering_biases_step{s1}.pt")
        torch.save(ckpt, out_dir / "steering_biases_latest.pt")

    do_eval = (args.eval_every > 0 and s1 % args.eval_every == 0
               and s1 >= args.eval_start)
    if not do_eval:
        return

    model.eval()
    if llm is not None:
        sync_bias_to_vllm_audio(llm, model)
    acc, details_ = evaluate_audio(model, processor, llm, eval_samples,
                                   args.max_new_tokens, return_details=True,
                                   is_omni=is_omni)
    print(f"★  step {s1:3d}  pass@1 = {acc:.4f}  ({acc*100:.2f}%)  "
          f"[{len(eval_samples)} problems]")
    if args.save_every == 0 or s1 % args.save_every != 0:
        ckpt = {n: p.data.cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(ckpt, out_dir / f"steering_biases_step{s1}.pt")
        torch.save(ckpt, out_dir / "steering_biases_latest.pt")
    (out_dir / f"eval_step{s1}.json").write_text(
        json.dumps({"step": s1, "acc": acc, "n": len(eval_samples),
                    "details": details_}, indent=2))


if __name__ == "__main__":
    main()
