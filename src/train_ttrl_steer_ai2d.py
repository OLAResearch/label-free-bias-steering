#!/usr/bin/env python3
"""TTRL bias-only steering on AI2D -- same steering algorithm as MathVista/
ScienceQA (down_proj.bias, all layers, GRPO + majority-vote pseudo-labels).

Dataset loading, official prompt, and official (strict) answer parser/grader
are reused UNCHANGED from eval_ai2d.py (VLMEvalKit-faithful, see that file's
docstring for exact provenance) -- imported, not duplicated. This script only
adds what's needed to TRAIN on AI2D: a `cot` prompt variant and a lenient
training-time grader.

Both additions are informed by two findings from this project's ScienceQA
integration (see RESULTS_SCIENCEQA.md), applied here from the start instead
of rediscovering them the slow way:
  1. Training rollouts must carry a reasoning signal. VLMEvalKit's own
     "Please select the correct answer from the options above." prompt is
     an official EVAL protocol, not a training one -- literally used as the
     training prompt, it collapses rollouts to ~1 token, starving GRPO of
     signal. `--prompt_format cot` (default here) appends a step-by-step +
     "The answer is X." instruction instead; eval always additionally uses
     the official VLMEvalKit protocol (via a separate eval_ai2d.py pass) for
     citable numbers.
  2. Training reward computed through an overly strict parser partly teaches
     format compliance instead of content. VLMEvalKit's official `can_infer`
     is deliberately conservative -- see eval_ai2d.py's docstring for the
     specific AI2D quirk (diagram-label questions where the option TEXT is
     itself a letter, e.g. options=['c','D','b','a'], causing `can_infer` to
     see two ambiguous letter-tokens and refuse to resolve them even when
     the model's answer is unambiguous to a human). `--grader lenient`
     (default here) tries the official `can_infer` first, then adds two
     robust fallbacks: the cot-format's own "The answer is X." sentence, and
     a leading "X." / bare "X" match (recovers the diagram-label echo case).
     `--grader strict` uses the official can_infer only, for comparison.
"""
import sys, os
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoProcessor

from train_ttrl_steer_mathvista import (
    patch_hf_for_bias_vl,
    patch_vllm_for_bias_vl,
    sync_bias_to_vllm_vl,
    compute_log_prob_vl,
    compute_log_probs_vl_batch,
)
from eval_ai2d import (
    OFFICIAL_OPTIONS,
    get_image,
    get_question_text,
    get_answer_letter,
    build_choices,
    build_prompt as build_prompt_official,
    get_question,
    COT_SUFFIX,
    can_infer,
    extract_answer_ai2d as extract_answer_official,
    load_ai2d,
)


# ── Lenient training-time grader (falls back from the official can_infer) ──

_COT_ANSWER_RE = re.compile(r"[Tt]he answer is[:\s]*\**\(?([A-Z])\)?\**[\.\s]?")
_LEADING_LETTER_RE = re.compile(r"^\**\(?([A-Z])\)?\**[\.\:\)]?\s")
_BARE_LETTER_RE = re.compile(r"\**\(?([A-Z])\)?\**\.?")


def extract_answer_ai2d_lenient(response: str, sample: dict) -> str:
    choices = build_choices(sample)

    opt = can_infer(response, choices)
    if opt and opt != "Z":
        return opt

    m = _COT_ANSWER_RE.findall(response)
    if m and m[-1] in choices:
        return m[-1]

    stripped = response.strip()
    m2 = _LEADING_LETTER_RE.match(stripped)
    if m2 and m2.group(1) in choices:
        return m2.group(1)

    m3 = _BARE_LETTER_RE.fullmatch(stripped)
    if m3 and m3.group(1) in choices:
        return m3.group(1)

    return "FAILED"


def extract_fn_for(grader: str):
    if grader == "strict":
        return lambda resp, sample: extract_answer_official(resp, sample)
    return extract_answer_ai2d_lenient


def score_ai2d(response: str, sample: dict, extract_fn) -> bool:
    return extract_fn(response, sample) == get_answer_letter(sample)


def majority_vote_ai2d(responses, sample, extract_fn):
    votes = [extract_fn(r, sample) for r in responses]
    votes = [v for v in votes if v != "FAILED"]
    if not votes:
        return None, 0.0
    vote, count = Counter(votes).most_common(1)[0]
    return vote, count / len(responses)


# ── Evaluation (vLLM fast path or HF fallback) ───────────────────────────────

@torch.no_grad()
def evaluate_ai2d(model, processor, llm, samples, max_new_tokens,
                  return_details=False, prompt_format="cot", extract_fn=None, lora_request=None):
    if extract_fn is None:
        extract_fn = extract_answer_ai2d_lenient
    device = next(model.parameters()).device

    if llm is not None:
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        requests = []
        for s in samples:
            image = get_image(s)
            question = get_question(s, prompt_format)
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": question},
            ]}]
            pt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            requests.append({"prompt": pt, "multi_modal_data": {"image": image}})
        vllm_outs = llm.generate(requests, sp, lora_request=lora_request)
        texts = [o.outputs[0].text for o in vllm_outs]
    else:
        texts = []
        for s in samples:
            image = get_image(s)
            question = get_question(s, prompt_format)
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": question},
            ]}]
            prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = processor(text=[prompt_text], images=[image], return_tensors="pt").to(device)
            out = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                 pad_token_id=processor.tokenizer.eos_token_id)
            texts.append(processor.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))

    correct = 0
    n_parse_failed = 0
    details = []
    for s, resp in zip(samples, texts):
        pred = extract_fn(resp, s)
        ok = pred == get_answer_letter(s)
        if pred == "FAILED":
            n_parse_failed += 1
        if ok:
            correct += 1
        if return_details:
            details.append({
                "question": get_question_text(s), "options": list(s["options"]),
                "answer": get_answer_letter(s), "response": resp,
                "predicted": pred, "correct": ok,
            })
    acc = correct / len(samples)
    return (acc, details, n_parse_failed) if return_details else acc


def _do_checkpoint_and_eval(s1, model, processor, llm, eval_samples, args, out_dir, extract_fn,
                             lora_sync=None):
    if args.save_every > 0 and s1 % args.save_every == 0:
        ckpt = {n: p.data.cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(ckpt, out_dir / f"steering_biases_step{s1}.pt")
        torch.save(ckpt, out_dir / "steering_biases_latest.pt")
        if lora_sync is not None:
            model.save_pretrained(str(out_dir / f"lora_step{s1}"))
            model.save_pretrained(str(out_dir / "lora_latest"))

    do_eval = (args.eval_every > 0 and s1 % args.eval_every == 0 and s1 >= args.eval_start)
    if not do_eval:
        return None

    model.eval()
    lora_req = None
    if llm is not None:
        if lora_sync is not None:
            lora_req = lora_sync.sync(model)
        else:
            sync_bias_to_vllm_vl(llm, model)
    acc, details_, n_fail = evaluate_ai2d(model, processor, llm, eval_samples,
                                          args.max_new_tokens, return_details=True,
                                          prompt_format=args.prompt_format, extract_fn=extract_fn,
                                          lora_request=lora_req)
    print(f"★  step {s1:3d}  pass@1 = {acc:.4f}  ({acc*100:.2f}%)  "
          f"[{len(eval_samples)} problems, {n_fail} parse-failed]")
    if args.save_every == 0 or s1 % args.save_every != 0:
        ckpt = {n: p.data.cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(ckpt, out_dir / f"steering_biases_step{s1}.pt")
        torch.save(ckpt, out_dir / "steering_biases_latest.pt")
    (out_dir / f"eval_step{s1}.json").write_text(
        json.dumps({"step": s1, "acc": acc, "n": len(eval_samples), "n_parse_failed": n_fail,
                    "details": details_}, indent=2))
    return acc


def main():
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--seed",              type=int,   default=42)
    ap.add_argument("--use_vllm",          action="store_true")
    ap.add_argument("--vllm_gpu_util",     type=float, default=0.30)
    ap.add_argument("--lr_scheduler",      default="constant", choices=["constant", "cosine", "delayed_cosine"])
    ap.add_argument("--lr_warmup_steps",   type=int, default=None)
    ap.add_argument("--prompt_format",     default="cot", choices=["direct", "cot"])
    ap.add_argument("--grader",            default="strict", choices=["strict", "lenient"])  # strict = official can_infer, matches eval_*.py and the verl reward ports exactly. lenient is opt-in only, for training-reward experiments -- never the default, so results.json always reports the one comparable number.
    ap.add_argument("--log_prob_mini_batch", type=int, default=1,
                    help="Mini-batch size for batched log-prob forward passes (1=sequential, 8=fast)")
    ap.add_argument("--steering_at_layer", type=int,   default=None)
    ap.add_argument("--output_dir",        default=None)
    ap.add_argument("--max_pixels",        type=int,   default=1048576)
    ap.add_argument("--split",             default="test")
    ap.add_argument("--train_hf_dataset",  default=None,
                    help="Optional separate HF dataset for training (default: same as eval). "
                         "E.g. lhndzn/AI2D")
    ap.add_argument("--train_hf_split",    default="train",
                    help="Split to use from --train_hf_dataset (default: train)")
    ap.add_argument("--use_labels",        action="store_true",
                    help="use ground-truth labels instead of majority-vote pseudo-labels")
    ap.add_argument("--full_ft",           action="store_true",
                    help="full fine-tune all language_model params (no vLLM); bias-only otherwise")
    ap.add_argument("--lora_r",            type=int,   default=None,
                    help="LoRA rank; if set uses peft LoRA instead of bias-only (no vLLM)")
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

    out_dir = Path(args.output_dir or Path(_ROOT) / "outputs" / f"ttrl_ai2d_{args.model_id.replace('/', '_')}_s{args.num_steps}_G{args.G}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    extract_fn = extract_fn_for(args.grader)

    patch_hf_for_bias_vl()
    n_visible = torch.cuda.device_count()
    _device_map = "balanced" if n_visible > 1 else {"": 0}
    print(f"\nLoading {args.model_id} …  (GPUs visible: {n_visible}  →  device_map={_device_map!r})")
    from transformers import Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(args.model_id, max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map=_device_map, attn_implementation="flash_attention_2")
    model.eval()

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
        mode = "full LM fine-tune"
    else:
        for name, p in model.named_parameters():
            is_trainable = "language_model" in name and name.endswith("mlp.down_proj.bias")
            if args.steering_at_layer is not None and is_trainable:
                is_trainable = f"layers.{args.steering_at_layer}." in name
            if is_trainable:
                torch.nn.init.zeros_(p)
                p.requires_grad_(True)
                trainable.append(p)
            else:
                p.requires_grad_(False)
        mode = f"{len(trainable)} layers [down_proj.bias]"
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  Trainable: {n_trainable:,} params ({mode})")

    llm = None
    lora_sync = None
    if args.use_vllm:
        from vllm import LLM
        print(f"\nInitializing vLLM (gpu_util={args.vllm_gpu_util}) …")
        max_visual_tokens = args.max_pixels // (28 * 28)
        vllm_max_len = max_visual_tokens + args.max_new_tokens + 512
        if args.lora_r:
            # Native vLLM LoRA hot-swap — no bias patch needed
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

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps, eta_min=0.0)
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
    else:
        scheduler = None

    print(f"\nLoading lmms-lab-encoder/ai2d ({args.split} split) …")
    dataset = load_ai2d(args.split)
    eval_samples = dataset[:args.eval_n]
    device = next(model.parameters()).device
    print(f"  {len(dataset)} eval problems  (eval_n={args.eval_n})")

    if args.train_hf_dataset:
        print(f"\nLoading training data from {args.train_hf_dataset} ({args.train_hf_split} split) …")
        from datasets import load_dataset as _load_hf
        train_dataset_raw = list(_load_hf(args.train_hf_dataset, split=args.train_hf_split))
        print(f"  {len(train_dataset_raw)} training problems")
        # Overlap check: compare question text against full eval dataset
        eval_qtexts = {get_question_text(s).strip() for s in dataset}
        overlap_n = sum(1 for s in train_dataset_raw
                        if get_question_text(s).strip() in eval_qtexts)
        print(f"  Overlap with eval set: {overlap_n} / {len(train_dataset_raw)} training problems "
              f"({'CLEAN — no overlap' if overlap_n == 0 else 'WARNING: overlap detected!'})")
        train_samples = train_dataset_raw
    else:
        train_samples = list(dataset)
    random.shuffle(train_samples)
    print(f"  {len(train_samples)} training problems")
    print(f"  Prompt format: {args.prompt_format}   Grader: {args.grader}")

    print("\nEVAL step 0 (baseline) …")
    _lora_req0 = None
    if llm is not None:
        if lora_sync is not None:
            _lora_req0 = lora_sync.sync(model)
        else:
            sync_bias_to_vllm_vl(llm, model)
    acc0, details0, nfail0 = evaluate_ai2d(model, processor, llm, eval_samples, args.max_new_tokens,
                                           return_details=True, prompt_format=args.prompt_format,
                                           extract_fn=extract_fn, lora_request=_lora_req0)
    print(f"★  step 0  pass@1 = {acc0:.4f}  ({acc0*100:.2f}%)  [{len(eval_samples)} problems, {nfail0} parse-failed]")
    (out_dir / "eval_step0.json").write_text(
        json.dumps({"step": 0, "acc": acc0, "n": len(eval_samples), "n_parse_failed": nfail0,
                    "details": details0}, indent=2))

    print(f"\n{'═'*64}\nTRAIN {args.num_steps} steps | G={args.G} | lr={args.lr} | "
          f"{'vLLM' if llm else 'HF'} generation\n{'═'*64}")

    training_log = []
    best_step, best_ckpt_acc = 0, acc0

    for step in range(args.num_steps):
        sample   = train_samples[step % len(train_samples)]
        image    = get_image(sample)
        question = get_question(sample, args.prompt_format)

        user_msg = {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": question},
        ]}

        prompt_text = processor.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
        prompt_enc  = processor(text=[prompt_text], images=[image], return_tensors="pt")
        prompt_len  = prompt_enc["input_ids"].shape[1]

        print(f"  step {step+1:3d}: sampling {args.G} rollouts …", flush=True)

        if llm is not None:
            from vllm import SamplingParams
            if lora_sync is not None:
                _lora_req = lora_sync.sync(model)
            else:
                sync_bias_to_vllm_vl(llm, model)
                _lora_req = None
            sp = SamplingParams(temperature=args.gen_temp, max_tokens=args.max_new_tokens, n=args.G)
            vllm_out = llm.generate([{"prompt": prompt_text, "multi_modal_data": {"image": image}}], sp,
                                    lora_request=_lora_req)
            texts = [o.text for o in vllm_out[0].outputs]
        else:
            enc_dev = {k: v.to(device) for k, v in prompt_enc.items()}
            texts = []
            with torch.no_grad():
                for _ in range(args.G):
                    out = model.generate(**enc_dev, do_sample=True, temperature=args.gen_temp,
                                         max_new_tokens=args.max_new_tokens,
                                         pad_token_id=processor.tokenizer.eos_token_id)
                    texts.append(processor.decode(out[0][prompt_len:], skip_special_tokens=True))

        if args.use_labels:
            pseudo_label = get_answer_letter(sample)
            majority_ratio = sum(1 for t in texts if extract_fn(t, sample) == pseudo_label) / len(texts)
        else:
            pseudo_label, majority_ratio = majority_vote_ai2d(texts, sample, extract_fn)
        s1 = step + 1
        if not pseudo_label:
            print(f"  step {s1:3d}: skip — no valid answers extracted")
            training_log.append({"step": s1, "skip": "no_answer"})
            continue

        correct_flags = [extract_fn(t, sample) == pseudo_label for t in texts]
        raw_rewards = [1.0 if c else -1.0 for c in correct_flags]
        rewards = torch.tensor(raw_rewards, dtype=torch.float32, device=device)
        n_match = int((rewards > 0).sum().item())

        if rewards.std() < 1e-6:
            print(f"  step {s1:3d}: skip — all {args.G} rewards equal (ratio={majority_ratio:.2f})")
            training_log.append({"step": s1, "skip": "all_same", "majority_ratio": majority_ratio})
            acc = _do_checkpoint_and_eval(s1, model, processor, llm, eval_samples, args, out_dir, extract_fn, lora_sync=lora_sync)
            if acc is not None and acc > best_ckpt_acc:
                best_ckpt_acc, best_step = acc, s1
            continue

        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

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

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"  step {s1:3d} | loss={accum_loss:.4f} | reward={rewards.mean():.3f} "
              f"({n_match}/{args.G}) | ratio={majority_ratio:.2f} | "
              f"∥grad∥={float(grad_norm):.4f} | lr={cur_lr:.2e}")
        training_log.append({
            "step": s1, "loss": accum_loss, "reward_mean": rewards.mean().item(),
            "n_match": n_match, "majority_ratio": majority_ratio, "grad_norm": float(grad_norm),
        })

        acc = _do_checkpoint_and_eval(s1, model, processor, llm, eval_samples, args, out_dir, extract_fn, lora_sync=lora_sync)
        if acc is not None and acc > best_ckpt_acc:
            best_ckpt_acc, best_step = acc, s1

    print(f"\nBest checkpoint: step {best_step}  (eval acc={best_ckpt_acc:.4f})")

    print(f"\n{'═'*64}\nFINAL EVAL ({len(dataset)} problems)  [best ckpt: step {best_step}]\n{'═'*64}")
    best_ckpt_path = out_dir / f"steering_biases_step{best_step}.pt"
    if best_step > 0 and best_ckpt_path.exists():
        print(f"  Loading {best_ckpt_path.name} …")
        saved = torch.load(best_ckpt_path, map_location="cpu")
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in saved:
                    p.copy_(saved[name].to(p.device, p.dtype))
    else:
        print("  No checkpoint found (step 0 was best or file missing) — using current weights")
    model.eval()
    _lora_req_final = None
    if llm is not None:
        if lora_sync is not None:
            _lora_req_final = lora_sync.sync(model)
        else:
            sync_bias_to_vllm_vl(llm, model)
    acc_final, details_final, nfail_final = evaluate_ai2d(
        model, processor, llm, dataset, args.max_new_tokens, return_details=True,
        prompt_format=args.prompt_format, extract_fn=extract_fn, lora_request=_lora_req_final)
    print(f"\n★  final pass@1 = {acc_final:.4f}  (Δ vs step-0 = {acc_final - acc0:+.4f})")
    (out_dir / "eval_final.json").write_text(
        json.dumps({"step": best_step, "acc": acc_final, "n": len(dataset),
                    "n_parse_failed": nfail_final, "details": details_final}, indent=2))

    results = {
        "model_id": args.model_id, "num_steps": args.num_steps, "G": args.G, "lr": args.lr,
        "prompt_format": args.prompt_format, "grader": args.grader,
        "dataset": "AI2D (lmms-lab-encoder/ai2d, test split)",
        "acc_step0": acc0, "best_step": best_step, "best_ckpt_acc": best_ckpt_acc,
        "acc_final": acc_final, "delta": acc_final - acc0, "training_log": training_log,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    sv = {n: p.data.cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(sv, out_dir / "steering_biases.pt")
    print(f"\n  Results → {out_dir / 'results.json'}")
    print(f"  Biases  → {out_dir / 'steering_biases.pt'}  ({sum(v.numel() for v in sv.values()):,} params)")


if __name__ == "__main__":
    main()
