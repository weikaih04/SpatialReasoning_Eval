# Answer-Only Evaluation Mode

Evaluate a Visual CoT (Latent64) model **without** visual thinking at inference time.

## How It Works

Use the Visual CoT checkpoint with the `thinkmorph_pet_no_thought` (or `thinkmorph_mvc_no_thought`) model config:

```yaml
export MODEL_PATH="ckpt/pet_latent64/run_8gpu/0010000_full_nonema"  # Visual CoT checkpoint
export MODEL_NAME="thinkmorph_pet_no_thought"                       # Answer-only config
```

The config sets `think=False, understanding_output=True, visual_gen=False`, which:
1. Selects `ANSWER_ONLY_SYSTEM_PROMPT`: *"Answer the question directly... Do not think or generate any images."*
2. Runs single-round text generation (skips the visual CoT loop)

## SAT Custom Prompt Fix

SAT_perspective has a custom prompt that says *"You MUST generate a thinking image"*, which conflicts with answer-only mode. Fix: `use_custom_prompt()` in `ThinkMorph.py` returns `False` when `understanding_output=True and think=False`, so SAT falls back to the default MCQ prompt.

## Results (PET Latent64)

| Benchmark | Visual CoT | Answer Only | No Thought (separate model) |
|-----------|:----------:|:-----------:|:---------------------------:|
| AI2Thor PET (278) | 96.76% | **97.84%** | 97.48% |
| Habitat HV (300) | 83.33% | **85.67%** | 82.00% |
| SAT (66) | 57.58% | **65.15%** | 59.09% |
| MindCube (200) | 39.00% | 38.50% | **42.50%** |
