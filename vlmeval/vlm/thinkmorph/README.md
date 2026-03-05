# ThinkMorph Inference Parameters

## Eval Configuration Parameters

These parameters are set in `VLMEvalKit_Thinkmorph/vlmeval/config.py` for each model config.

### Core Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `think` | bool | `True` | Allow `<think>` reasoning in output |
| `understanding_output` | bool | `True` | Text-only output mode. `True`: single text generation, no image loop. `False`: interleaved text+image generation |
| `visual_gen` | bool | `True` | Load VAE weights. Must be `True` if model was trained with visual generation |
| `vae_input` | bool/None | `None` | Whether input images go through VAE encoding (see below) |

### `vae_input` Parameter

Controls whether input images are encoded through VAE (in addition to ViT) during inference.

- **`None` (default):** Auto mode — follows `understanding_output`. When `understanding_output=True`, input images skip VAE (ViT only). When `understanding_output=False`, input images go through both VAE and ViT.
- **`True`:** Force VAE encoding for input images, regardless of `understanding_output`.
- **`False`:** Skip VAE encoding for input images.

#### Why this matters

During training with `visual_gen=True`, input images always go through **both VAE and ViT** pathways. However, during eval with `understanding_output=True`, the default behavior (`vae_input=None`) only sends input images through **ViT**, skipping VAE. This creates a **train-eval mismatch**.

For models trained with visual generation (e.g., Visual CoT, mixed VCoT + answer-only), you should set `vae_input=True` when evaluating in answer-only mode to match the training behavior.

#### Code path

In `inferencer.py`, `update_context_image()` has two branches:
- `vae=True`: input image goes through `prepare_vae_images()` → `forward_cache_update_vae()` (VAE pathway) AND `prepare_vit_images()` → `forward_cache_update_vit()` (ViT pathway)
- `vae=False`: input image only goes through the ViT pathway

The `vae` flag is determined by:
```python
use_vae = vae_input if vae_input is not None else (not understanding_output)
```

### Common Eval Configurations

| Mode | `think` | `understanding_output` | `visual_gen` | `vae_input` |
|---|---|---|---|---|
| Visual CoT | `True` | `False` | `True` | `None` |
| Text CoT | `True` | `True` | `False` | `None` |
| MM CoT | `True` | `False` | `True` | `None` |
| No Thought | `False` | `True` | `False` | `None` |
| Answer-Only (from VCoT ckpt) | `False` | `True` | `True` | `True` |

### System Prompt Selection

Determined by `understanding_output` and `think`:

| `understanding_output` | `think` | System Prompt |
|---|---|---|
| `True` | `False` | `ANSWER_ONLY_SYSTEM_PROMPT` |
| `True` | `True` | `VLM_THINK_SYSTEM_PROMPT` |
| `False` | any | `GEN_THINK_SYSTEM_PROMPT` |
