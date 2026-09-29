# Upstream provenance & sync workflow

`_vendored/` is a 2-level fork. This file pins the upstream SHAs we forked from, categorizes our local adaptations, and records what upstream has that we don't. Update it whenever a sync pass lands.

## Provenance chain

```
Physical-Intelligence/openpi (Apache-2.0)
    rpl forked from an unrecorded SHA; closest match is the initial
    PyTorch port at openpi@554657 (2025-09-08).
        │
        ▼
RoboMME/robomme_policy_learning  (third-party submodule, pinned)
    submodule SHA: ecf086c3
    files at:      src/openpi/models_pytorch/
        │
        ▼
worv-ai/PonderPounce (this repo)
    files at:      ponderpounce/models/pi05/_vendored/
```

Treat anything older than the initial port as "best-effort match against rpl's snapshot".

## Files

| File                            | Source                                         | Why we vendored                                                                 |
| ------------------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------- |
| `pi05_pytorch.py`               | rpl `openpi/models_pytorch/pi0_pytorch.py`     | Joint-attention + PonderPounce prepend hooks. Adapted to transformers 5.5+ API.   |
| `paligemma_with_expert.py`      | rpl `openpi/models_pytorch/gemma_pytorch.py`   | Joint-attention `_compute_layer`. Same API drift.                                |
| `configuration_gemma_adarms.py` | transformers `gemma/configuration_gemma.py`    | Adds `use_adarms` / `adarms_cond_dim` config fields.                             |
| `modeling_gemma_adarms.py`      | transformers `gemma/modeling_gemma.py`         | Threads `adarms_cond` through `DecoderLayer.forward`. Stock norm isn't subclass-patchable. |
| `gemma_variants.py`             | rewritten from `openpi/models/gemma.py`        | Stub config registry so we don't pull in all of openpi.                          |

## Categorized adaptations

The diff against rpl is substantial — both core files have grown ~25-30% in length. The new lines fall into four buckets; future sync passes can skip lines in B/C/D when reviewing upstream changes.

- **A. transformers API drift**: paths shifted (`paligemma.language_model` → `paligemma.model.language_model`); attention / rotary / `eager_attention_forward` signatures changed.
- **B. PonderPounce features**: `_prepend_prefix(... extra_embs ...)`, `age_ms` suffix token, extra kwargs on `Pi05VLA.predict_action` (`prepend_prefix_*`, `symbolic_prompt_*`, `pixel_attention_mask`).
- **C. Performance / numerics**: `eager_attention_forward` (faster than SDPA at our seq lengths), per-block gradient checkpointing, ZeRO-3 weight-init bypass, Liger Triton GeGLU, prefix-KV cache.
- **D. DeepSpeed / HF Trainer compat**: upstream's `torch.compile(sample_actions, ...)` is intentionally **not** vendored — it doesn't compose with DeepSpeed-sharded training. Inference-side opt-in is a deferred follow-up (needs the attention-mask dtype path verified first).

## Upstream features

Reviewed against `Physical-Intelligence/openpi/main @ 7b23daa` (2026-03-19).

| Upstream feature                                            | Status here                                                                                                                                          |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Configurable `torch.compile(sample_actions, mode=...)`      | Not adopted. Initial port hit a fp32-attention-mask vs bf16-query dtype rejection in the inductor-compiled SDPA kernel; needs the mask cast in `_prepare_attention_masks_4d` verified end-to-end before we revisit. |
| Prefix-KV cache (`use_cache=True` warmup + suffix-only Euler) | Present. Extra `DynamicCache.crop(prefix_len)` between Euler steps to work around 5.5+ cache mutation under `use_cache=False`. Bit-exact verified.   |
| `_attn_implementation = "eager"` switch at sample_actions   | N/A — our `_compute_layer` already calls `eager_attention_forward` directly.                                                                          |

## Optimization references (not yet in upstream openpi)

| Work          | Effect on Pi0.5                                          | Code                                                                                                              | Adoption cost                                                                                                       |
| ------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| realtime-vla  | Reports ~29 ms full-inference on RTX 4090 via 24 fused Triton kernels | [dexmal/realtime-vla](https://github.com/dexmal/realtime-vla) (MIT)                                              | Largest measured single-model lever — `~20×` over our path. Cost: hand-fused kernels, separate inference path. Worth a focused spike. |
| FASTER        | ~10× TTFA via Horizon-Aware Schedule                      | [innovator-zero/FASTER](https://github.com/innovator-zero/FASTER)                                                  | Inference-only HAS is small but degrades SR on a constant-schedule ckpt; full benefit needs HAS-mixed fine-tuning.   |
| RTC           | Robust to inference delay (async + inpaint)              | [Physical-Intelligence/real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) (sim only; algorithm not in openpi yet) | Lives in our inference dispatcher (was `inference/engine/scheduler.py`), not the model. Sim-eval value is low.                                  |
| SnapFlow      | 1-NFE, 9.6× denoise speedup, ≥ SR                         | Not released as of 2026-05.                                                                                       | Distillation training; new ckpt required.                                                                            |

## Sync workflow

1. Bump the rpl submodule SHA in this repo.
2. Diff `openpi/models_pytorch/{pi0_pytorch,gemma_pytorch}.py` against the previous pin. Decide per change.
3. Open a focused PR per portable change, citing the upstream SHA.
4. Refresh the **Upstream features** table when anything in it changes status.

## Sync triggers

- Upstream openpi commit subject hits any of: `compile`, `cache`, `attention`, `scheduler`, `sampling`.
- Upstream pins a new `transformers` version.
- A new optimization paper above ships code.

## See also

- `_vendored/README.md` — per-file provenance summary (everyday reference).
