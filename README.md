# Label-Free Bias-Only Steering

Test-time reinforcement learning that trains only ~100K additive bias
parameters per model, using majority-vote pseudo-labels as the reward
(no ground-truth labels). Same procedure applied across text,
vision-language, and audio reasoning.

## Project structure

```
src/
  train_ttrl_steer.py            # MATH-500 (text)
  train_ttrl_steer_mathvista.py  # MathVista (vision-language)
  train_ttrl_steer_ai2d.py       # AI2D (vision-language)
  train_ttrl_steer_logicvista.py # LogicVista (vision-language)
  train_ttrl_steer_mmau.py       # MMAU (audio)
  eval_ai2d.py                   # standalone checkpoint evaluation
  eval_logicvista.py
  eval_mathvista.py
requirements.txt
```

All training scripts share the same recipe: freeze the backbone, train a
per-layer additive bias vector with GRPO, reward = majority-vote agreement
across rollouts. The vision-language scripts (`train_ttrl_steer_ai2d.py`,
`train_ttrl_steer_logicvista.py`) import shared model/vLLM utilities from
`train_ttrl_steer_mathvista.py`, and their evaluation utilities from the
matching `eval_*.py` file — keep the `src/` directory flat.

## Dependencies

```
pip install -r requirements.txt
```

Two additional repositories are imported at runtime and should be cloned
alongside `src/` (added to `PYTHONPATH`):
- [steering-reasoning](https://github.com/corl-team/steering-reasoning) — GRPO reward processor
- TTRL — majority-vote and MATH grading utilities

## Example launches

Text (MATH-500, Qwen2.5-Math-7B):
```bash
python src/train_ttrl_steer.py \
  --model_id Qwen/Qwen2.5-Math-7B \
  --dataset math --num_steps 300 --G 64 --lr 5e-4 \
  --use_vllm --output_dir outputs/math500_math7b
```

Vision-language (AI2D, Qwen2.5-VL-7B-Instruct):
```bash
python src/train_ttrl_steer_ai2d.py \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --num_steps 200 --G 32 --lr 1e-3 --lr_scheduler cosine \
  --use_vllm --output_dir outputs/ai2d
```

Audio (MMAU, Qwen2.5-Omni-7B):
```bash
python src/train_ttrl_steer_mmau.py \
  --model_id Qwen/Qwen2.5-Omni-7B \
  --num_steps 200 --G 32 --lr 1e-3 --lr_scheduler cosine \
  --use_vllm --output_dir outputs/mmau
```

LoRA comparison (any vision-language script, e.g. MathVista):
```bash
python src/train_ttrl_steer_mathvista.py \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --lora_r 16 --lora_alpha 32 \
  --num_steps 200 --G 32 --use_vllm --output_dir outputs/mathvista_lora16
```

Standalone checkpoint evaluation:
```bash
python src/eval_ai2d.py --bias_path outputs/ai2d/steering_biases.pt
```

## License

MIT
