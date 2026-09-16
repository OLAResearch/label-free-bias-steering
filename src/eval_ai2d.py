#!/usr/bin/env python3
"""AI2D eval for a saved bias checkpoint (or baseline) -- follows VLMEvalKit's
official implementation as closely as possible for prompt construction and
answer extraction/grading, rather than inventing a custom evaluator.

Reused verbatim from VLMEvalKit (ported 1:1 from the source, not reinvented):
  - Prompt format (vlmeval/dataset/image_mcq.py, ImageMCQDataset.build_prompt):
        Question: {question}
        Options:
        A. {option}
        B. {option}
        ...
        Please select the correct answer from the options above.
    AI2D has no 'hint' field in the official protocol, so the `Hint: ...` line
    (present in ImageMCQDataset.build_prompt for datasets that do have hints,
    e.g. ScienceQA) never fires for AI2D and is correctly omitted here.
  - Answer extraction (vlmeval/utils/matching_util.py):
        can_infer() = can_infer_option() then can_infer_text() fallback.
        can_infer_option(): strips punctuation, tokenizes, and accepts a
          choice letter only if it's the unique choice-letter token present
          AND appears within the last 5 tokens of the response (VLMEvalKit's
          heuristic for "the model's final answer", not just any mention of
          the letter) -- OR falls back to a regex for "(correct) answer is
          **X**" style phrasing.
        can_infer_text(): fallback for short (<= 2x total option-text length)
          responses that contain exactly one option's literal text verbatim.
        If neither matches: VLMEvalKit's 'exact_matching' policy (no GPT
        judge configured, matching our offline HPC setup with no API key)
        returns opt='Z', which never equals a real A-D/E gold letter --
        i.e. an unparseable response is simply graded incorrect, never
        raises. Ported verbatim below as extract_answer_ai2d() / can_infer().
  - Grading (vlmeval/dataset/utils/multiple_choice.py, eval_vanilla):
        exact match between the extracted option letter and the ground-truth
        letter. AI2D is not a "circular" dataset in VLMEvalKit (no
        mmbench/ccbench-style answer-shuffling protocol), so mcq_vanilla_eval
        applies -- plain per-sample hit/miss, accuracy = mean(hit).

Dataset: lmms-lab-encoder/ai2d (HF mirror). Same question/options/answer/image
semantics as VLMEvalKit's AI2D_TEST TSV (verified against LMMS-Eval's own AI2D
task config, which uses the same HF dataset) -- 3088-problem test split, the
standard AI2D-TEST size cited across VLM papers. Only the hosting differs from
VLMEvalKit's own TSV; prompt construction and grading are VLMEvalKit's, not
LMMS-Eval's (LMMS-Eval's own AI2D prompt/parser -- a fixed-16-token regex
filter -- is intentionally NOT used here per the task's explicit priority).

Steering mechanics (patch_hf_for_bias_vl / patch_vllm_for_bias_vl /
sync_bias_to_vllm) imported UNCHANGED from eval_mathvista.py -- same vLLM
backend, HF<->vLLM sync, generation settings, and output schema as our
existing MathVista/ScienceQA eval scripts. No modification to the steering
implementation.
"""
import sys, os
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import argparse
import json
import re
import string
from pathlib import Path

import torch
from transformers import AutoProcessor

from eval_mathvista import patch_hf_for_bias_vl, patch_vllm_for_bias_vl, sync_bias_to_vllm

OFFICIAL_OPTIONS = list(string.ascii_uppercase)


# ── Official prompt construction (VLMEvalKit ImageMCQDataset.build_prompt) ──

def get_image(sample):
    img = sample["image"]
    if img is None:
        raise ValueError("AI2D sample has no image")
    return img.convert("RGB")


def build_choices(sample) -> dict:
    """{'A': option_text, 'B': option_text, ...} -- AI2D options are a plain
    list in dataset order, VLMEvalKit assigns A/B/C/... by position."""
    return {OFFICIAL_OPTIONS[i]: opt for i, opt in enumerate(sample["options"])}


def get_question_text(sample) -> str:
    return str(sample["question"]).strip()


def get_answer_letter(sample) -> str:
    return OFFICIAL_OPTIONS[int(sample["answer"])]


def build_prompt(sample) -> str:
    """Verbatim port of ImageMCQDataset.build_prompt's text-construction logic
    (image_mcq.py). AI2D has no 'hint' field, so that line never fires here --
    matches the official behavior exactly (not a simplification)."""
    choices = build_choices(sample)
    options_prompt = "Options:\n"
    for key, item in choices.items():
        options_prompt += f"{key}. {item}\n"
    prompt = f"Question: {get_question_text(sample)}\n"
    if choices:
        prompt += options_prompt
        prompt += "Please select the correct answer from the options above. \n"
    return prompt


# Cot prompt variant used by train_ttrl_steer_ai2d.py's --prompt_format cot
# training (see that file's docstring). Kept here (not duplicated there) so
# eval_ai2d.py can score a cot-trained checkpoint under the SAME prompt
# distribution it was trained on -- the official can_infer grader below is
# unchanged either way, only the prompt text differs. Mirrors
# eval_scienceqa.py's --prompt_format direct/cot pattern.
COT_SUFFIX = (
    "Answer step by step, then end your response with the exact sentence "
    '"The answer is X." where X is the option\'s letter.'
)


def get_question(sample: dict, prompt_format: str = "direct") -> str:
    if prompt_format == "direct":
        return build_prompt(sample)
    choices = build_choices(sample)
    options_prompt = "Options:\n"
    for key, item in choices.items():
        options_prompt += f"{key}. {item}\n"
    prompt = f"Question: {get_question_text(sample)}\n"
    if choices:
        prompt += options_prompt
        prompt += COT_SUFFIX
    return prompt


# ── Official answer extraction + grading (VLMEvalKit matching_util.py) ──────
# Ported verbatim from vlmeval/utils/matching_util.py's can_infer_option /
# can_infer_text / can_infer. The 'model is None' (no GPT judge) branch of
# extract_answer_from_item is what our offline setup always takes -- ported
# as the FAILED/'Z' sentinel below.

_VERBOSE_ANSWER_RE = re.compile(r"(?i)(?:correct\s+)?answer\s+is\s+\**([ABCD])\**")


def can_infer_option(answer: str, choices: dict):
    reject_to_answer = [
        "Sorry, I can't help with images of people yet.", "I can't process this file.",
        "I'm sorry, but without the image provided", "Cannot determine the answer",
    ]
    for err in reject_to_answer:
        if err in answer:
            return "Z"

    def count_choice(splits, cands, prefix="", suffix=""):
        return sum(1 for c in cands if prefix + c + suffix in splits)

    answer_mod = answer
    for c in ".()[],:;!*#{}":
        answer_mod = answer_mod.replace(c, " ")
    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if ch in splits and splits.index(ch) > (len(splits) - 5):
                return ch
    elif count == 0 and count_choice(splits, {"Z", ""}) == 1:
        return "Z"

    match = _VERBOSE_ANSWER_RE.search(answer or "")
    if match and match.group(1).upper() in choices:
        return match.group(1).upper()

    return False


def can_infer_text(answer: str, choices: dict):
    answer_l = answer.lower()
    if len(answer_l) > 2 * sum(len(str(v)) for v in choices.values()):
        return False
    cands = [k for k, v in choices.items() if str(v).lower() in answer_l]
    return cands[0] if len(cands) == 1 else False


def can_infer(answer, choices: dict):
    answer = str(answer)
    opt = can_infer_option(answer, choices)
    return opt if opt else can_infer_text(answer, choices)


def extract_answer_ai2d(response: str, sample: dict) -> str:
    """extract_answer_from_item() under VLMEvalKit's 'exact_matching' policy
    (no GPT judge available offline): can_infer() success -> that letter;
    failure -> 'Z' (never equals a real gold letter, i.e. graded incorrect,
    never raises). We report 'FAILED' instead of 'Z' in our own `predicted`
    field for parse-failure-rate diagnostics -- 'Z' is VLMEvalKit's internal
    sentinel, not something we need to preserve verbatim in our output."""
    choices = build_choices(sample)
    opt = can_infer(response, choices)
    if not opt or opt == "Z":
        return "FAILED"
    return opt


def score_ai2d(response: str, sample: dict) -> bool:
    """eval_vanilla(): opt == GT."""
    pred = extract_answer_ai2d(response, sample)
    return pred == get_answer_letter(sample)


# ── Self-test: lock in the ported logic against known VLMEvalKit behavior ──

def _selftest():
    choices = {"A": "Whale population would decrease.", "B": "Fish population would decrease.",
              "C": "Penguin population would increase.", "D": "Seal population will become extinct"}
    cases = [
        # (response, expected_can_infer_result)
        ("B", "B"),
        ("B.", "B"),
        ("The answer is B.", "B"),
        ("The correct answer is **B**.", "B"),
        ("After careful analysis of the food chain, the answer is B.", "B"),
        ("Fish population would decrease.", "B"),  # can_infer_text fallback, exact option text
        ("I'm not sure, could be A or C.", False),  # 2 letter-tokens present -> ambiguous, no match
        ("This is a long rambling paragraph mentioning B in the middle "
         "but then continuing on for many more words after it so B is "
         "not among the last five tokens of the response text here", False),
    ]
    for resp, expected in cases:
        got = can_infer(resp, dict(choices))
        status = "OK" if got == expected else "MISMATCH"
        print(f"  [{status}] can_infer({resp[:50]!r}...) = {got!r}  (expected {expected!r})")
        assert got == expected, f"selftest failed: {resp!r} -> {got!r}, expected {expected!r}"
    print("All self-test cases passed.")


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_ai2d(split: str = "test"):
    from datasets import load_dataset
    return list(load_dataset("lmms-lab-encoder/ai2d", split=split))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id",       default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--bias_path",      default=None,
                    help="Path to steering_biases_stepN.pt. None = baseline (no bias).")
    ap.add_argument("--lora_path",      default=None,
                    help="Path to LoRA adapter dir (e.g. lora_step100). Mutually exclusive with --bias_path.")
    ap.add_argument("--reft_path",      default=None,
                    help="Path to a ReFT checkpoint (steering_biases_stepN.pt-style, "
                         "reft_bank.* keys). Mutually exclusive with --bias_path/--lora_path.")
    ap.add_argument("--prompt_format",  default="direct", choices=["direct", "cot"],
                    help="direct: official VLMEvalKit prompt (bare MCQ, no reasoning "
                         "instruction); cot: step-by-step + official 'The answer is X.' "
                         "closing sentence -- use this to fairly score a checkpoint "
                         "trained with --prompt_format cot. Grading (can_infer) is "
                         "identical either way.")
    ap.add_argument("--split",          default="test")
    ap.add_argument("--eval_n",         type=int, default=3088,
                    help="Full AI2D-TEST is 3088 problems")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--vllm_gpu_util",  type=float, default=0.50)
    ap.add_argument("--max_pixels",     type=int, default=262144)
    ap.add_argument("--output_path",    default=None)
    ap.add_argument("--selftest",       action="store_true",
                    help="Run grading self-test against known cases and exit (no GPU needed).")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    sys.stdout.reconfigure(line_buffering=True)

    patch_hf_for_bias_vl()
    patch_vllm_for_bias_vl()

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

    processor = AutoProcessor.from_pretrained(args.model_id, max_pixels=args.max_pixels)

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
    elif args.reft_path is not None:
        from reft_common_vl import patch_vllm_for_reft_vl, sync_reft_to_vllm_from_state_dict
        print(f"\nLoading ReFT checkpoint: {args.reft_path}")
        patch_vllm_for_reft_vl(llm)
        reft_state = torch.load(args.reft_path, map_location="cpu")
        sync_reft_to_vllm_from_state_dict(llm, reft_state)
        print(f"  {len(reft_state)} params loaded")
    else:
        print("\nNo checkpoint — running baseline (zero bias)")

    print(f"  Prompt format: {args.prompt_format}")
    print(f"\nLoading lmms-lab-encoder/ai2d ({args.split} split) …")
    dataset = load_ai2d(args.split)
    samples = dataset[:args.eval_n]
    print(f"  {len(samples)} / {len(dataset)} problems")

    print(f"\nRunning eval on {len(samples)} problems …")
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    requests = []
    for s in samples:
        image  = get_image(s)
        prompt = get_question(s, args.prompt_format)
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt},
        ]}]
        pt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        requests.append({"prompt": pt, "multi_modal_data": {"image": image}})

    vllm_outs = llm.generate(requests, sp, lora_request=lora_req)
    texts = [o.outputs[0].text for o in vllm_outs]

    correct = 0
    n_parse_failed = 0
    details = []
    for idx, (s, resp) in enumerate(zip(samples, texts)):
        pred = extract_answer_ai2d(resp, s)
        ok = pred == get_answer_letter(s)
        if pred == "FAILED":
            n_parse_failed += 1
        if ok:
            correct += 1
        details.append({
            "id":        idx,
            "question":  get_question_text(s),
            "options":   list(s["options"]),
            "answer":    get_answer_letter(s),
            "response":  resp,
            "predicted": pred,
            "correct":   ok,
        })

    acc = correct / len(samples)
    print(f"\n{'═'*60}")
    print(f"★  pass@1 = {acc:.4f}  ({acc*100:.2f}%)  [{len(samples)} problems]")
    print(f"   parser FAILED (VLMEvalKit 'Z' sentinel): {n_parse_failed}/{len(samples)} "
          f"({n_parse_failed/len(samples)*100:.1f}%)")
    print(f"   bias_path : {args.bias_path or 'None (baseline)'}")
    print(f"{'═'*60}")

    out_path = args.output_path
    if out_path is None:
        if args.lora_path:
            m = re.search(r"step(\d+)", str(args.lora_path))
            step_tag = f"lora_step{m.group(1)}" if m else "lora_ckpt"
        elif args.bias_path:
            step_tag = Path(args.bias_path).stem.replace("steering_biases_", "")
        else:
            step_tag = "baseline"
        out_dir = Path(_ROOT) / "outputs" / "eval_ai2d"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"{step_tag}_{args.prompt_format}_n{len(samples)}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    result = {
        "acc": acc, "n": len(samples), "n_parse_failed": n_parse_failed,
        "bias_path": args.bias_path, "prompt_format": args.prompt_format, "model_id": args.model_id,
        "dataset": "AI2D (lmms-lab-encoder/ai2d, test split)",
        "protocol": "official VLMEvalKit ImageMCQDataset prompt + can_infer answer "
                    "extraction + exact-match grading (vlmeval/dataset/image_mcq.py, "
                    "vlmeval/utils/matching_util.py), exact_matching policy (no GPT judge)",
        "details": details,
    }
    Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
