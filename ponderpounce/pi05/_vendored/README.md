# Vendored π0.5 PyTorch port

This directory contains code ported from
[Physical Intelligence's openpi repository](https://github.com/Physical-Intelligence/openpi)
(via [RoboMME/robomme_policy_learning](https://github.com/RoboMME/robomme_policy_learning) `src/openpi/models_pytorch/`) and from
HuggingFace `transformers` 5.5.0, adapted to support π0.5's joint-attention
architecture inside PonderPounce.

## Why vendor?

`transformers.models.pi0.PI0Model` (added in transformers 5.4.0) implements
**π0 base**: PaliGemma processes the prefix once, the action expert
cross-attends to its KV cache. This is architecturally different from
**π0.5**, which uses *joint attention*: PaliGemma and the action expert
process their tokens in lockstep at every layer, with attention computed
over the concatenated `[paligemma_tokens, expert_tokens]` sequence. π0.5
also adds AdaRMSNorm (Adaptive RMSNorm) to the action expert layers.

The `lerobot/pi05_base` checkpoint was trained with this π0.5 architecture
(see openpi's `pi0_pytorch.py` with `pi05=True`, `use_adarms=[False, True]`),
so loading it requires the same model structure.

Since openpi's PyTorch port targets `transformers==4.53.2`, vendoring with
adaptation to PonderPounce's `transformers 5.x` is needed.

## Files

| Path | Source | Adaptation |
|------|--------|------------|
| `configuration_gemma_adarms.py` | `transformers/models/gemma/configuration_gemma.py` (5.5.0) | Adds `use_adarms: bool` and `adarms_cond_dim: int | None` config fields |
| `modeling_gemma_adarms.py` | `transformers/models/gemma/modeling_gemma.py` (5.5.0) | Replaces `GemmaRMSNorm` with adaptive variant; threads `adarms_cond` through `GemmaDecoderLayer.forward` and `GemmaModel.forward`; layernorm forward returns `(out, gate)` for gated residuals |
| `paligemma_with_expert.py` | `openpi/models_pytorch/gemma_pytorch.py` | Imports adapted to local `modeling_gemma_adarms`; 5.5.0 `GemmaAttention` / `apply_rotary_pos_emb` / `eager_attention_forward` API drift fixes |
| `pi05_pytorch.py` | `openpi/models_pytorch/pi0_pytorch.py` | Imports adapted; weight init paths preserved |

## Caveats

* SigLIP vision tower is stock 5.5.0 (no AdaRMSNorm in vision); its key
  layout matches `lerobot/pi05_base` since vision weights are unchanged.
* `torch.compile` on `sample_actions` is dropped for DeepSpeed compatibility.
* openpi's `_preprocess_observation` is not ported — PonderPounce feeds tensors
  directly to ``Pi05VLA.forward``.

## License

openpi & transformers: Apache 2.0.

## Upstream sync

See [`UPSTREAM.md`](./UPSTREAM.md) for the pinned upstream SHAs, full
diff-stat against the rpl snapshot, categorized list of PonderPounce
adaptations, and the workflow to follow when bumping the vendored
copies or pulling in new upstream optimizations.
