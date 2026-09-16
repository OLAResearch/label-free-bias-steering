#!/usr/bin/env python3
"""Standalone MathVista eval for a saved bias checkpoint.

Supports two prompt formats:
  cot   — "Reasoning: ...\nFinal answer: X"   (default, our training format)
  boxed — "...put your final answer within \\boxed{}"  (MM-UPT / official style)

Usage:
  python eval_mathvista.py \\
      --bias_path outputs/.../steering_biases_step120.pt \\
      --prompt_format cot \\
      --eval_n 1000 \\
      --output_path eval_step120_full.json
"""
import sys, os
_ROOT = os.path.dirname(os.path.abspath(__file__))

import argparse, difflib, json, re, string
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor

# ── Prompt formats ─────────────────────────────────────────────────────────────

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

# ── Dataset helpers ────────────────────────────────────────────────────────────

def get_image(sample):
    img = sample.get("decoded_image") or sample.get("image")
    if img is None:
        raise ValueError("No image in sample")
    return img.convert("RGB")


def get_question(sample) -> str:
    question = str(sample.get("question", "")).strip()
    choices = sample.get("choices") or []
    if str(sample.get("question_type", "")).strip().lower() == "multi_choice" and choices:
        choice_block = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        return f"{question}\nChoices:\n{choice_block}"
    return question


def get_scoring_kwargs(sample) -> dict:
    return {
        "choices":       sample.get("choices") or [],
        "question_type": str(sample.get("question_type", "")).strip().lower(),
        "answer_type":   str(sample.get("answer_type",  "")).strip().lower(),
        "precision":     sample.get("precision", None),
    }

# ── Graders ────────────────────────────────────────────────────────────────────

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


def _strip_units(text):
    text = _CURRENCY_RE.sub("", text.strip())
    return _UNIT_RE.sub("", text).strip()


def normalize_answer(text):
    text = str(text).lower().strip()
    text = _strip_units(text)
    keep = set("./")
    text = "".join(c for c in text if c not in string.punctuation or c in keep)
    return " ".join(text.split())


def _try_numeric(text):
    try:
        return float(text.strip())
    except ValueError:
        pass
    m = re.match(r"^(-?\d+)\s*/\s*(-?\d+)$", text.strip())
    if m:
        n, d = int(m.group(1)), int(m.group(2))
        return n / d if d != 0 else None
    return None


def extract_final_answer_cot(text: str) -> str:
    m = re.search(r"final answer[:\s]+(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".,;:!? ")
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return (lines[-1] if lines else text.strip()).rstrip(".,;:!? ")


def _last_boxed_only_string(string):
    """Brace-depth-aware \\boxed{} span finder (unlike a plain regex, this
    correctly handles nested braces from \\frac{a}{b}, \\sqrt{}, etc.)."""
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
    # Try \boxed{...} first -- brace-depth aware, not a naive regex (a naive
    # r"\boxed\{([^}]*)\}" truncates at the first inner '}', e.g.
    # \boxed{\frac{5}{3}\pi} would extract as '\frac{5' instead of the full answer).
    boxed = _last_boxed_only_string(text)
    if boxed is not None:
        left = "\\boxed{"
        if boxed.startswith(left) and boxed.endswith("}"):
            return boxed[len(left):-1].strip()
    # Fall back to "Final answer:" or last line
    return extract_final_answer_cot(text)


def normalize_mathvista_prediction(extraction: str, choices: list = None,
                                   question_type: str = "", answer_type: str = "",
                                   precision=None) -> str:
    raw = (extraction or "").strip()
    qt  = (question_type or "").strip().lower()
    at  = (answer_type  or "").strip().lower()
    cvs = list(choices or [])

    if qt == "multi_choice" and cvs:
        labels = [chr(65 + i) for i in range(len(cvs))]
        letter_match = re.findall(r"\(([a-zA-Z])\)", raw)
        if letter_match:
            letter = letter_match[0].upper()
            if letter in labels:
                return normalize_answer(cvs[labels.index(letter)])
        bare = re.match(r"^\s*([A-Z])\s*[.:]?\s*$", raw)
        if bare and bare.group(1) in labels:
            return normalize_answer(cvs[labels.index(bare.group(1))])
        rn = normalize_answer(raw)
        norm_cvs = [normalize_answer(c) for c in cvs]
        if rn in norm_cvs:
            return rn
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


def get_gt_forms(sample) -> List[str]:
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
        extract_fn = extract_final_answer_cot
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


# ── HF / vLLM patches ─────────────────────────────────────────────────────────

def patch_hf_for_bias_vl():
    import torch.nn as nn
    for module_path, cls_name in [
        ("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl", "Qwen2_5_VLMLP"),
        ("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl", "Qwen2MLP"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
            orig = cls.__init__
            # closure to capture orig
            def _make_patch(o, is_mlp_with_bias_kwarg=(cls_name == "Qwen2_5_VLMLP")):
                if is_mlp_with_bias_kwarg:
                    def patched(self, config, bias=False):
                        o(self, config, bias=bias)
                        if self.down_proj.bias is None:
                            old = self.down_proj
                            self.down_proj = nn.Linear(old.in_features, old.out_features, bias=True)
                else:
                    def patched(self, config):
                        o(self, config)
                        if self.down_proj.bias is None:
                            old = self.down_proj
                            self.down_proj = nn.Linear(old.in_features, old.out_features, bias=True)
                return patched
            cls.__init__ = _make_patch(orig)
            print(f"HF: patched {cls_name} for down_proj.bias")
        except (ImportError, AttributeError):
            pass


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

    patched = False
    try:
        import vllm.model_executor.models.qwen2_5_vl as qm_vl
        for cls_name in ("Qwen2_5_VLMLP", "Qwen2MLP"):
            if hasattr(qm_vl, cls_name):
                cls = getattr(qm_vl, cls_name)
                cls.__init__ = _make_mlp_patch(cls.__init__)
                patched = True
        if hasattr(qm_vl, "Qwen2_5_VLForConditionalGeneration"):
            vl_cls = qm_vl.Qwen2_5_VLForConditionalGeneration
            vl_cls.load_weights = _make_lw_patch(vl_cls.load_weights)
    except (ImportError, AttributeError) as e:
        print(f"  vLLM VL patch warning: {e}")

    if not patched:
        import vllm.model_executor.models.qwen2 as qm
        qm.Qwen2MLP.__init__ = _make_mlp_patch(qm.Qwen2MLP.__init__)
        qm.Qwen2ForCausalLM.load_weights = _make_lw_patch(qm.Qwen2ForCausalLM.load_weights)

    print("vLLM: patched for down_proj.bias")


def sync_bias_to_vllm(llm, bias_dict):
    """Sync bias dict (from checkpoint or model) into vLLM in-place."""
    by_layer = {}
    for name, tensor in bias_dict.items():
        if name.endswith("mlp.down_proj.bias"):
            parts = name.split(".")
            try:
                li = parts.index("layers")
                by_layer[int(parts[li + 1])] = tensor.float()
            except (ValueError, IndexError):
                pass
    print(f"  [sync] {len(by_layer)} layers to sync")

    _path_used = [None]

    def _find_layers(m):
        for desc, getter in [
            ("x.language_model.model.layers",       lambda x: x.language_model.model.layers),
            ("x.model.language_model.model.layers",  lambda x: x.model.language_model.model.layers),
            ("x.model.model.layers",                 lambda x: x.model.model.layers),
            ("x.model.layers",                       lambda x: x.model.layers),
        ]:
            try:
                result = getter(m)
                _path_used[0] = desc
                return result
            except AttributeError:
                continue
        raise AttributeError("Cannot locate decoder layers in vLLM model")

    def _do_sync(vllm_model):
        layers = _find_layers(vllm_model)
        print(f"  [sync] layer path: {_path_used[0]}, n={len(layers)}", flush=True)
        for idx, bias_cpu in by_layer.items():
            dp = layers[idx].mlp.down_proj
            if dp.bias is None:
                print(f"  [sync] WARNING: layer {idx} dp.bias is None", flush=True)
                return
            with torch.no_grad():
                dp.bias.copy_(bias_cpu.to(dp.bias.device, dp.bias.dtype))
        norm0 = layers[0].mlp.down_proj.bias.float().norm().item()
        print(f"  [sync] done. Layer 0 bias norm: {norm0:.4f}", flush=True)

    llm.apply_model(_do_sync)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id",       default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--bias_path",      default=None,
                    help="Path to steering_biases_stepN.pt. None = baseline (no bias).")
    ap.add_argument("--lora_path",      default=None,
                    help="Path to LoRA adapter dir (e.g. lora_step100). Mutually exclusive with --bias_path.")
    ap.add_argument("--prompt_format",  default="cot", choices=["cot", "boxed"],
                    help="cot: our 'Final answer:' format; boxed: MM-UPT \\boxed{} format")
    ap.add_argument("--eval_n",         type=int, default=1000)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--vllm_gpu_util",  type=float, default=0.50)
    ap.add_argument("--max_pixels",     type=int, default=262144)
    ap.add_argument("--output_path",    default=None,
                    help="JSON output path. Auto-generated if not set.")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    # ── Patches ──────────────────────────────────────────────────────────────
    patch_hf_for_bias_vl()
    patch_vllm_for_bias_vl()

    # ── vLLM ─────────────────────────────────────────────────────────────────
    from vllm import LLM, SamplingParams
    max_visual_tokens = args.max_pixels // (28 * 28)
    vllm_max_len = max_visual_tokens + args.max_new_tokens + 512
    print(f"\nLoading vLLM model: {args.model_id}")
    print(f"  max_pixels={args.max_pixels} → ~{max_visual_tokens} visual tokens")
    print(f"  max_model_len={vllm_max_len}")
    _lora_kw = ({"enable_lora": True, "max_loras": 1, "max_lora_rank": 64, "max_cpu_loras": 2}
                if args.lora_path else {})
    llm = LLM(
        args.model_id,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.vllm_gpu_util,
        enforce_eager=True,
        max_model_len=vllm_max_len,
        disable_log_stats=True,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"max_pixels": args.max_pixels},
        **_lora_kw,
    )
    print("  vLLM ready")

    processor = AutoProcessor.from_pretrained(args.model_id,
                                              max_pixels=args.max_pixels)

    # ── Apply checkpoint (bias or LoRA) ───────────────────────────────────────
    lora_req = None
    if args.lora_path is not None:
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("adapter", 1, str(args.lora_path))
        print(f"\nLoRA checkpoint: {args.lora_path}")
    elif args.bias_path is not None:
        print(f"\nLoading bias checkpoint: {args.bias_path}")
        bias_dict = torch.load(args.bias_path, map_location="cpu")
        print(f"  {len(bias_dict)} params loaded")
        sync_bias_to_vllm(llm, bias_dict)
    else:
        print("\nNo checkpoint — running baseline (zero bias)")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"\nLoading AI4Math/MathVista testmini …")
    dataset = list(load_dataset("AI4Math/MathVista", split="testmini"))
    samples = dataset[:args.eval_n]
    print(f"  {len(samples)} problems")

    # ── Instruction / extractor ───────────────────────────────────────────────
    if args.prompt_format == "boxed":
        instruction = BOXED_INSTRUCTION
        extract_fn  = extract_final_answer_boxed
        print(f"  Prompt format: boxed (\\boxed{{}})")
    else:
        instruction = COT_INSTRUCTION
        extract_fn  = extract_final_answer_cot
        print(f"  Prompt format: cot (Final answer:)")

    # ── Eval ──────────────────────────────────────────────────────────────────
    print(f"\nRunning eval on {len(samples)} problems …")
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

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

    vllm_outs = llm.generate(requests, sp, lora_request=lora_req)
    texts = [o.outputs[0].text for o in vllm_outs]

    correct = 0
    details = []
    for s, resp in zip(samples, texts):
        gt_forms = get_gt_forms(s)
        ok = score_vqa(resp, gt_forms, extract_fn=extract_fn, **get_scoring_kwargs(s))
        if ok:
            correct += 1
        details.append({
            "id":        s.get("pid", len(details)),
            "question":  get_question(s),
            "answer":    str(s.get("answer", "")),
            "response":  resp,
            "predicted": extract_fn(resp),
            "correct":   ok,
        })

    acc = correct / len(samples)
    print(f"\n{'═'*60}")
    print(f"★  pass@1 = {acc:.4f}  ({acc*100:.2f}%)  [{len(samples)} problems]")
    print(f"   bias_path     : {args.bias_path or 'None (baseline)'}")
    print(f"   prompt_format : {args.prompt_format}")
    print(f"{'═'*60}")

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.output_path is None:
        if args.lora_path:
            m = re.search(r"step(\d+)", str(args.lora_path))
            ckpt_tag = f"lora_step{m.group(1)}" if m else "lora_ckpt"
        elif args.bias_path:
            m = re.search(r"step(\d+)", str(args.bias_path))
            ckpt_tag = f"step{m.group(1)}" if m else "ckpt"
        else:
            ckpt_tag = "baseline"
        out_dir = Path(_ROOT) / "outputs" / "eval_mathvista"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_path = str(out_dir / f"eval_{ckpt_tag}_{args.prompt_format}_n{len(samples)}.json")

    result = {
        "acc":           acc,
        "n":             len(samples),
        "bias_path":     str(args.bias_path),
        "prompt_format": args.prompt_format,
        "model_id":      args.model_id,
        "details":       details,
    }
    Path(args.output_path).write_text(json.dumps(result, indent=2))
    print(f"  Results → {args.output_path}")


if __name__ == "__main__":
    main()
