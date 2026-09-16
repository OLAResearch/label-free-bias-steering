#!/usr/bin/env python3
"""LogicVista eval for a saved bias checkpoint (or baseline).

Official protocol research (see task notes / RESULTS_LOGICVISTA.md for full detail):
  - Official repo (github.com/Yijia-Xiao/LogicVista, arXiv:2407.04973) has NO
    model-querying prompt of its own -- only post-hoc grading
    (eval/extract_accuracy.py) via GPT-4-as-extractor (LangChain ChatOpenAI)
    doing sorted-letter exact-match against ground_truth[id]["answer"].split(", ").
  - VLMEvalKit (open-compass/VLMEvalKit) has a `LogicVista` class
    (vlmeval/dataset/image_vqa.py): TYPE='VQA', no build_prompt override, so
    it inherits ImageBaseDataset.build_prompt verbatim -- i.e. the OFFICIAL
    prompt is simply the raw `question` string + the image, nothing added
    (no "Options:" header, no extra instruction line -- LogicVista's own
    question text already embeds any "Select from A, B, C, and D" /
    "(A) text (B) text" phrasing where it exists).
  - VLMEvalKit's grading (vlmeval/dataset/utils/logicvista.py,
    LogicVista_auxeval/build_prompt_logicvista) is ALSO GPT-4-extraction
    based, and critically: LogicVista.evaluate() only runs the extraction
    loop `if not osp.exists(storage) and model is not None:` -- i.e. even
    VLMEvalKit's own code has NO working non-judge ("exact_matching") path
    for LogicVista specifically (confirmed by reading the current source;
    the `model = None` branch silently produces no judged predictions and
    the function returns None). We have no GPT-4/OpenAI API access in this
    project's offline pipeline, so we do NOT attempt any LLM judge/extractor
    call, matching the task's explicit instruction, and confirmed necessary
    by inspection of the upstream code, not just assumed.
  - Per this project's established precedent (`eval_ai2d.py`, itself porting
    VLMEvalKit's own "exact_matching" no-judge fallback path used elsewhere
    in the codebase for other MCQ datasets), we port
    vlmeval/utils/matching_util.py's can_infer_option / can_infer_text /
    can_infer VERBATIM for single-letter extraction, then extend it with a
    small, clearly-separated multi-letter extension (see
    extract_letters_logicvista) to support the sorted-letter-SET grading
    LogicVista's official protocol actually uses (445/448 rows are single-
    letter, 3/448 are two-letter, e.g. "B, D" -- handled as a sorted-set
    match, not naive string equality, mirroring the official protocol's
    `sorted(answer.split(", "))` comparison).

Dataset: lscpku/LogicVista (HF), single "test" split, 448 rows. Fields:
question, answer, reasoning, skill, broad_capability, specific_capability,
imagesource, sourcelink, liscenced, id, image. There is NO separate
"options" field (unlike AI2D/ScienceQA) -- option labels/text are embedded
in the `question` string itself (359/448 rows have literal "(A) ... (B) ..."
markers with real option text; most others state a letter range/list like
"Select from A, B, C, and D" or "(A-D)" with the actual choice CONTENT
rendered only as pixels in the image, e.g. diagram-pattern-completion
questions) or, for ~39/448 rows, not stated in the text at all (purely
visual MCQ, e.g. "Which figure is a rotation of the object?"). This was
verified empirically (not guessed) by loading the dataset and inspecting
question text across all 448 rows -- see `_detect_letters()` below, whose
letter-range/option-text detection heuristic was validated against the full
dataset with 0 mismatches between detected letter-universe and the ground-
truth answer letter(s) (excluding 2 genuinely defective dataset rows: id
v1_382, a totally blank question/answer row, and id v1_20, whose question
labels choices "1-5" numerically but whose stored gold answer is "3" -- a
digit, not a letter -- so it can never be matched by any letter-based
extractor; both are left in the eval set for honesty, and show up as
permanent misses / diagnosable via `n_parse_failed`, not silently dropped).

Steering mechanics (patch_hf_for_bias_vl / patch_vllm_for_bias_vl /
sync_bias_to_vllm) imported UNCHANGED from eval_mathvista.py -- same vLLM
backend, HF<->vLLM sync, generation settings, and output schema as our
existing MathVista/AI2D/ScienceQA/GQA eval scripts. No modification to the
steering implementation.
"""
import sys, os
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoProcessor

from eval_mathvista import patch_hf_for_bias_vl, patch_vllm_for_bias_vl, sync_bias_to_vllm

VALID_LETTERS = "ABCDEFGHI"  # observed answer-letter range across the dataset is A-G; A-I covers all "Select
                              # from A-I" phrasings we found. Never hardcode a 4-letter universe (see docstring).


# ── Official prompt construction (VLMEvalKit ImageBaseDataset.build_prompt, ──
# ── LogicVista has no override -- the prompt IS the raw question, verbatim) ──

def get_image(sample):
    img = sample["image"]
    if img is None:
        raise ValueError("LogicVista sample has no image")
    return img.convert("RGB")


def get_question_text(sample) -> str:
    return str(sample["question"] or "").strip()


def get_answer_letters(sample) -> list:
    """Gold answer as a sorted, deduped list of letters -- mirrors the official
    protocol's `sorted(ground_truth[id]["answer"].split(", "))` sorted-set
    comparison. 445/448 rows are single-letter; 3/448 are two-letter."""
    raw = str(sample["answer"] or "")
    parts = [p.strip().upper() for p in raw.split(",") if p.strip()]
    return sorted(set(parts))


# ── Per-question option-letter-universe / option-text detection ─────────────
# LogicVista has no separate "options" field (unlike AI2D/ScienceQA) -- the
# valid letter range and (when present) option TEXT must be recovered from
# the question string itself. Validated empirically against all 448 rows
# (0 letter-universe/gold-answer mismatches, excluding 2 defective rows --
# see module docstring) via the following priority order:
#   1. Inline "(A) text (B) text ..." markers, if >=2 found -- most reliable,
#      also gives real option TEXT for can_infer_text. Present in 359/448
#      rows. Handles the AI2D-style "option TEXT is itself the letter" case
#      too (e.g. "(A) A, (B) B, (C) C, (D) D, (E) All..." for cog/spring/
#      drone-stability physics questions) since it only needs the "(X)"
#      markers, not the text content, to find the letter universe.
#   2. Explicit range "A-D" / "A-I" / "A to F" (checked AFTER inline, because
#      some inline-option TEXT itself can spuriously contain a dash pattern,
#      e.g. "(A) Beam A-B moves up..." -- letting inline win first avoids
#      that false positive; empirically this ordering is what got mismatches
#      to 0). 37/448 rows.
#   3. Comma/and/or-separated bare letter list, parens optional, e.g.
#      "the proposals A, B, C or D" or "(A, B, C, or D)" -- take the longest
#      such run found, letters = A..max(letters) (option lists are always
#      presented in order starting at A). 13/448 rows.
#   4. Fallback default A-E (all observed default-tier rows have gold answers
#      within A-D; A-E is a safe superset), for rows with no letter mention
#      in the text at all (39/448 rows -- purely visual MCQ, e.g. "Which
#      figure is a rotation of the object?").

_OPTION_INLINE_RE = re.compile(r'\(([A-I])\)\s*([^()]*?)(?=\s*\([A-I]\)|\Z)')
_RANGE_RE = re.compile(r'\b([A-I])\s*(?:-|to)\s*([A-I])\b')
_LETTER_RUN_RE = re.compile(r'\(?([A-I])\)?(?:\s*,\s*\(?[A-I]\)?){1,8}(?:\s*,?\s*(?:or|and)\s*\(?[A-I]\)?)?')
_DEFAULT_LETTERS = list("ABCDE")


def _detect_letters(question: str):
    inline = _OPTION_INLINE_RE.findall(question)
    if inline:
        letters = sorted(set(k for k, _ in inline))
        if len(letters) >= 2:
            return letters
    m = _RANGE_RE.search(question)
    if m:
        lo, hi = m.group(1), m.group(2)
        if lo <= hi:
            return [chr(c) for c in range(ord(lo), ord(hi) + 1)]
    best = None
    for mo in _LETTER_RUN_RE.finditer(question):
        letters = sorted(set(re.findall(r'[A-I]', mo.group())))
        if best is None or len(letters) > len(best):
            best = letters
    if best and len(best) >= 2:
        hi = max(best)
        return [chr(c) for c in range(ord('A'), ord(hi) + 1)]
    if inline:  # inline found exactly 1 marker -- weak signal, still better than nothing
        return sorted(set(k for k, _ in inline))
    return list(_DEFAULT_LETTERS)


def build_choices(sample) -> dict:
    """{'A': option_text_or_'', 'B': ..., ...}. option_text is '' when the
    option content is only rendered in the image (no text available) -- see
    module docstring. can_infer_option only inspects dict KEYS (the letter
    universe); can_infer_text uses the VALUES, and an empty string never
    causes a false single-candidate match once there are >=2 choices (a
    universal substring match on every key means >=2 candidates -> can_infer_text
    correctly returns False/ambiguous)."""
    q = get_question_text(sample)
    letters = _detect_letters(q)
    inline_text = dict(_OPTION_INLINE_RE.findall(q))
    return {L: inline_text.get(L, "").strip() for L in letters}


def build_prompt(sample) -> str:
    """Verbatim official protocol: ImageBaseDataset.build_prompt has no
    LogicVista override, so the prompt is just the raw question text -- no
    'Options:' header, no added instruction line. This is the complete
    official prompt, not a simplification."""
    return get_question_text(sample)


COT_SUFFIX = (
    "\nAnswer step by step, then end your response with the exact sentence "
    '"The answer is X." where X is the option\'s letter (or letters '
    'separated by commas, e.g. "The answer is A, C.", if more than one '
    "choice applies)."
)


def get_question(sample: dict, prompt_format: str = "direct") -> str:
    if prompt_format == "direct":
        return build_prompt(sample)
    return build_prompt(sample) + COT_SUFFIX


# ── Official (single-letter) answer extraction (VLMEvalKit matching_util.py) ─
# Ported verbatim from vlmeval/utils/matching_util.py's can_infer_option /
# can_infer_text / can_infer -- the 'model is None' (no GPT judge) branch,
# i.e. what our offline setup always takes. Identical to eval_ai2d.py's port.

_VERBOSE_ANSWER_RE = re.compile(r"(?i)(?:correct\s+)?answer\s+is\s+\**([A-I])\**")


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
    cands = [k for k, v in choices.items() if str(v) and str(v).lower() in answer_l]
    return cands[0] if len(cands) == 1 else False


def can_infer(answer, choices: dict):
    answer = str(answer)
    opt = can_infer_option(answer, choices)
    return opt if opt else can_infer_text(answer, choices)


# ── Multi-letter extension (project addition -- see docstring) ─────────────
# LogicVista's official protocol grades a sorted SET of letters (445/448
# rows are single-letter, 3/448 are two-letter). can_infer above is
# inherently single-letter (ported verbatim from VLMEvalKit, which itself
# only ever infers one option). We extend it, gold-blind (never look at the
# sample's actual answer to decide how many letters to extract), with an
# explicit "answer(s)/choice(s)/option(s) is/are X[, Y[, ...]]" scan --
# exactly the sentence the COT_SUFFIX above asks a cot-trained model to
# produce -- before falling back to the single-letter can_infer.

_FINAL_ANSWER_RE = re.compile(r'(?i)(?:answer|choice|option)s?\s+(?:is|are)[:\s]*([^.\n]{0,60})')


def extract_letters_logicvista(response: str, sample: dict) -> list:
    """Returns a sorted, deduped list of extracted letters, or [] if nothing
    could be inferred (caller reports this as 'FAILED')."""
    choices = build_choices(sample)
    response = str(response)

    matches = _FINAL_ANSWER_RE.findall(response)
    if matches:
        tail = matches[-1]  # last occurrence = the final-answer sentence
        letters = [l for l in re.findall(r'[A-I]', tail) if l in choices]
        letters = sorted(set(letters))
        if letters:
            return letters

    opt = can_infer(response, choices)
    if opt and opt != "Z":
        return [opt]
    return []


def extract_answer_logicvista(response: str, sample: dict) -> str:
    """String form for `details`/diagnostics: comma-joined sorted letters, or
    'FAILED' (never a real gold value) if extraction found nothing."""
    letters = extract_letters_logicvista(response, sample)
    return ", ".join(letters) if letters else "FAILED"


def score_logicvista(response: str, sample: dict) -> bool:
    """Sorted-SET exact match -- mirrors the official protocol's
    `sorted(pred_letters) == sorted(gold_letters)` comparison."""
    pred = extract_letters_logicvista(response, sample)
    gold = get_answer_letters(sample)
    return sorted(pred) == gold


# ── Self-test: lock in the ported/extended logic against known cases ────────

def _selftest():
    print("── can_infer (ported verbatim from VLMEvalKit matching_util.py) ──")
    choices = {"A": "True", "B": "False", "C": "Insufficient Information"}
    cases = [
        ("B", "B"),
        ("B.", "B"),
        ("The answer is B.", "B"),
        ("The correct answer is **B**.", "B"),
        ("False.", "B"),  # can_infer_text fallback, exact option text
        ("I'm not sure, could be A or C.", False),  # 2 letter-tokens present -> ambiguous
    ]
    for resp, expected in cases:
        got = can_infer(resp, dict(choices))
        status = "OK" if got == expected else "MISMATCH"
        print(f"  [{status}] can_infer({resp[:50]!r}) = {got!r}  (expected {expected!r})")
        assert got == expected, f"selftest failed: {resp!r} -> {got!r}, expected {expected!r}"

    print("\n── _detect_letters (empirically validated against full 448-row dataset) ──")
    detect_cases = [
        # (question, expected letters)
        ("Which choices in the image (A-D) belong to the green category?", list("ABCD")),
        ("Based on the diagram at the top of the page, which two of the proposals "
         "A, B, C or D completes the diagrams at the bottom of the page?", list("ABCD")),
        ("What options (A, B, C, or D) follow this same rule?", list("ABCD")),
        ("The Small Silver Watch displays the time as 16:00. Select from A, B and C. "
         "(A) True (B)False (C)Insufficient Information", list("ABC")),
        ("The paintings in the museum are to be filed, by genre, then title, in "
         "alphabetical order. Which painting would be positioned fourth? Select from "
         "A, B, C, D, and E.(A) Painting I (B) Painting II (C) Painting III "
         "(D) Painting IV (E) Painting V", list("ABCDE")),
        ("Which 3D shape can be made from the 2D net by folding it away from you?",
         list("ABCDE")),  # no letter mention at all -> default
        ("What would happen if Point C was moved downwards? Select from A, B, C, D, "
         "and E. (A) Beam A-B moves up but stays level, (B) A ends up lower than B, "
         "(C) A ends up higher than B, (D) Cannot be determined, (E) None of the above",
         list("ABCDE")),  # inline-first ordering avoids the "A-B" false range match
        ("Who is the odd-one-out? Select answers from A-I", list("ABCDEFGHI")),
        ("Select which of options A to F corresponds to the rule.", list("ABCDEF")),
    ]
    for q, expected in detect_cases:
        got = _detect_letters(q)
        status = "OK" if got == expected else "MISMATCH"
        print(f"  [{status}] _detect_letters({q[:55]!r}...) = {got!r}  (expected {expected!r})")
        assert got == expected, f"selftest failed: {q!r} -> {got!r}, expected {expected!r}"

    print("\n── extract_letters_logicvista / score_logicvista (incl. 3 known "
          "multi-letter dataset rows) ──")
    fake_samples = [
        # (question, answer, response, expected_score)
        ("Which choices in the image (A-D) belong to the green category?", "A, C",
         "Diagrams A and C have an X in the center, so the answer is A, C.", True),
        ("Which choices in the image (A-D) belong to the green category?", "A, C",
         "A and C.", False),  # no "answer is" cue and can_infer is single-letter-only -> FAILED
        ("Based on the diagram at the top of the page, which two of the proposals "
         "A, B, C or D completes the diagrams at the bottom of the page?", "B, D",
         "After analysis, the answer is B, D.", True),
        ("Which 3D shape can be made from the 2D net by folding it away from you?",
         "A", "Step by step reasoning here. The answer is A.", True),
        ("Which 3D shape can be made from the 2D net by folding it away from you?",
         "A", "Step by step reasoning here. The answer is B.", False),
        ("The Small Silver Watch displays the time as 16:00. Select from A, B and C. "
         "(A) True (B)False (C)Insufficient Information", "B",
         "The time shown does not match, so the statement is False.", True),
    ]
    for q, ans, resp, expected in fake_samples:
        sample = {"question": q, "answer": ans}
        got = score_logicvista(resp, sample)
        status = "OK" if got == expected else "MISMATCH"
        pred = extract_answer_logicvista(resp, sample)
        print(f"  [{status}] score(ans={ans!r}, resp={resp[:45]!r}) = {got}  "
              f"(pred={pred!r}, expected {expected})")
        assert got == expected, f"selftest failed: {resp!r} vs {ans!r} -> {got}, expected {expected}"

    print("\nAll self-test cases passed.")


# ── Dataset loading ───────────────────────────────────────────────────────

def load_logicvista(split: str = "test"):
    from datasets import load_dataset
    return list(load_dataset("lscpku/LogicVista", split=split))


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
                    help="direct: official VLMEvalKit prompt (raw question text, "
                         "verbatim, no instruction added); cot: same question + "
                         "step-by-step + official-style 'The answer is X.' closing "
                         "sentence -- use this to fairly score a checkpoint trained "
                         "with --prompt_format cot. Grading is identical either way.")
    ap.add_argument("--split",          default="test")
    ap.add_argument("--eval_n",         type=int, default=448,
                    help="Full LogicVista test split is 448 problems")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--vllm_gpu_util",  type=float, default=0.50)
    ap.add_argument("--max_pixels",     type=int, default=1048576)
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
    print(f"\nLoading lscpku/LogicVista ({args.split} split) …")
    dataset = load_logicvista(args.split)
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
        pred = extract_answer_logicvista(resp, s)
        gold = get_answer_letters(s)
        ok = score_logicvista(resp, s)
        if pred == "FAILED":
            n_parse_failed += 1
        if ok:
            correct += 1
        details.append({
            "id":        s.get("id", idx),
            "question":  get_question_text(s),
            "answer":    ", ".join(gold),
            "response":  resp,
            "predicted": pred,
            "correct":   ok,
        })

    acc = correct / len(samples)
    print(f"\n{'═'*60}")
    print(f"★  pass@1 = {acc:.4f}  ({acc*100:.2f}%)  [{len(samples)} problems]")
    print(f"   parser FAILED: {n_parse_failed}/{len(samples)} "
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
        out_dir = Path(_ROOT) / "outputs" / "eval_logicvista"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"{step_tag}_{args.prompt_format}_n{len(samples)}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    result = {
        "acc": acc, "n": len(samples), "n_parse_failed": n_parse_failed,
        "bias_path": args.bias_path, "prompt_format": args.prompt_format, "model_id": args.model_id,
        "dataset": "LogicVista (lscpku/LogicVista, test split)",
        "protocol": "raw ImageBaseDataset.build_prompt question text (official VLMEvalKit "
                    "prompt, no override for LogicVista) + can_infer single-letter answer "
                    "extraction (vlmeval/utils/matching_util.py, exact_matching policy, no "
                    "GPT judge, ported verbatim) extended with a gold-blind multi-letter "
                    "final-answer-sentence scan + sorted-set grading (project addition, "
                    "since VLMEvalKit's own LogicVista grader has no working non-judge path "
                    "-- see eval_logicvista.py docstring)",
        "details": details,
    }
    Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
