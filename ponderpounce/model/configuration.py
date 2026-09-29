"""PonderPounce VLA configuration.

Organized as nested sub-configs:
    PonderPounceConfig
    ├── s2: System 2 config (deliberative VLM; e.g. S2QwenConfig)
    ├── s1: System 1 config (a concrete arch config, polymorphic)
    ├── schedule: ScheduleConfig    — async S1/S2 timing (float-time sampler)
    ├── flow: FlowMatchingConfig    — S1 flow matching action denoising
    └── top-level flags: s1_grad_to_s2 (cross boundaries)

``s1`` is polymorphic: a YAML names the concrete binding config via
``class_path``, and the HF-serialized form carries its ``model_type``, which
``_coerce`` resolves back to the binding class via ``S1_REGISTRY``
(``ponderpounce.model.configuration``). ``s2``/``schedule``/``flow`` are
fixed-type — reconstructed as their default class (a serialized ``model_type``
is stripped).
"""

import warnings
from typing import TYPE_CHECKING, ClassVar

# S1 arch registry: model_type → binding config class (lazy import paths). Lets
# _coerce / from_pretrained rebuild the polymorphic s1 sub-config from a
# serialized model_type. DEFAULT_S1_MODEL_TYPE lives there too, so this module
# names no concrete arch.
from lazyregistry import Registry
from transformers import PretrainedConfig

# S1 arch registry: model_type -> binding config class (lazy import path).
S1_REGISTRY: Registry[str, type] = Registry(name="ponderpounce_s1")
S1_REGISTRY.update({"ponderpounce_s1_pi05": "ponderpounce.model.s1_pi05:PonderPounceS1Pi05Config"})
DEFAULT_S1_MODEL_TYPE = "ponderpounce_s1_pi05"
# model_type strings written by earlier (pre-release) checkpoints.
_S1_MODEL_TYPE_ALIASES = {"reflex_dual_s1_pi05": "ponderpounce_s1_pi05"}

if TYPE_CHECKING:
    from lazyregistry import Registry

    from ponderpounce.model.system2 import System2


_LEGACY_S1_ARCH_MODEL_TYPES = {
    "pi05": "ponderpounce_s1_pi05",
    "pi0": "ponderpounce_s1_pi0",
    "simvla": "ponderpounce_s1_simvla",
    "starvla": "ponderpounce_s1_starvla",
}


def _migrate_legacy_s1_arch(fields: dict) -> tuple[str, dict]:
    """Translate pre-polymorphic ``reflex_s1_arch`` configs.

    Older checkpoints serialized the S1 sub-config as ``model_type:
    reflex_s1_arch`` plus an ``arch`` discriminator. New checkpoints serialize
    the concrete binding model_type directly. Keep old ckpts loadable without
    weakening the "fields but no discriminator" guard for ad-hoc flat configs.
    """
    migrated = dict(fields)
    arch = str(migrated.pop("arch", "pi05")).lower()
    mt = _LEGACY_S1_ARCH_MODEL_TYPES.get(arch)
    if mt is None:
        raise ValueError(f"unknown legacy S1 arch {arch!r} in model_type 'reflex_s1_arch'")
    if mt in {"ponderpounce_s1_pi05", "ponderpounce_s1_pi0"}:
        if "action_chunk" in migrated and "action_horizon" not in migrated:
            migrated["action_horizon"] = migrated["action_chunk"]
        if "proprio_dim" in migrated and "state_dim" not in migrated:
            migrated["state_dim"] = migrated["proprio_dim"]
        migrated.pop("action_chunk", None)
        migrated.pop("proprio_dim", None)
    return mt, migrated


def _coerce(
    sub: PretrainedConfig | dict | None,
    *,
    registry: "Registry | None" = None,
    default_model_type: str | None = None,
    default_cls: type[PretrainedConfig] | None = None,
) -> PretrainedConfig:
    """Reconstruct a sub-config from an instance, a jsonargparse
    ``{class_path, ...}`` dict, or an HF-serialized ``{model_type, ...}`` dict.

    HF ``from_dict`` does ``cls(**dict)`` and does not rebuild polymorphic
    children, so subtype reconstruction lives here. A polymorphic ``s1`` is
    resolved through ``registry`` by ``model_type`` (falling back to
    ``default_model_type``); fixed-type ``s2``/``schedule``/``flow`` use
    ``default_cls``.
    """
    if isinstance(sub, PretrainedConfig):
        return sub
    sub = sub or {}
    if "class_path" in sub:
        return _from_class_path(sub)
    fields = {k: v for k, v in sub.items() if k != "model_type"}
    if registry is not None:
        # A polymorphic sub-config dict must declare its type. An empty dict means
        # "use the default arch"; a dict WITH fields but no discriminator is a stale
        # flat config (pre-decouple ``s1: {arch: ...}``) that would otherwise silently
        # build the default arch and swallow the real fields — fail loud instead.
        if fields and "model_type" not in sub:
            raise ValueError(
                f"polymorphic sub-config has fields {sorted(fields)} but no `model_type`/`class_path` "
                "discriminator; migrate a legacy flat `s1: {arch: ...}` config to a `class_path:` block "
                "or carry `model_type`"
            )
        mt = sub.get("model_type") or default_model_type
        if mt is None:
            raise ValueError("polymorphic sub-config requires a `model_type` (or a default)")
        if mt == "reflex_s1_arch":
            mt, fields = _migrate_legacy_s1_arch(fields)
        mt = _S1_MODEL_TYPE_ALIASES.get(mt, mt)
        try:
            cls = registry[mt]
        except KeyError:
            raise ValueError(
                f"unknown model_type {mt!r} for the {registry.name!r} sub-config registry; register it there"
            ) from None
        return cls(**fields)
    assert default_cls is not None
    return default_cls(**fields)


def _from_class_path(sub: dict) -> PretrainedConfig:
    """Instantiate a ``{class_path, init_args}`` dict by importing the class."""
    import importlib

    module_path, _, cls_name = sub["class_path"].rpartition(".")
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(**(sub.get("init_args") or {}))


# ── Sub-configs ───────────────────────────────────────────────────────────


class S2QwenConfig(PretrainedConfig):
    """System 2: deliberative Qwen3.5 VLM architecture."""

    model_type = "ponderpounce_s2_qwen"

    def __init__(
        self,
        vlm_model_id: str = "Qwen/Qwen3.5-4B",
        num_cognition_tokens: int = 1,
        # Opt-in custom S2 attention mask. Existing ckpts pin the value in
        # their yaml; the default just stays out of the way.
        isolated_cognition_attention: bool = False,
        # Demo conditioning
        max_demo_frames: int = -1,  # debug-only hard cap; -1 disables (production), >0 forces a cap
        demo_fps: float = 1.0,  # sample demo at this fps; 0.0 = keep every frame
        # Subgoal conditioning
        # if True, emit "Subgoal: ...\n" after frames at each transition and supervise those text tokens with CE.
        # Gate supervision at cog positions (``COGNITION_TOKEN_STR`` as CE target) is a separate, architectural
        # requirement and runs regardless of this flag — the autoregressive gate is what lets System2 decide
        # transition vs non-transition at inference, so disabling it would break inference. If you need to fully
        # zero the S2 CE loss (for an ablation), set ``s2_loss_weight=0`` on ``PonderPounceConfig`` instead.
        inject_subgoal: bool = True,
        # emit the dataset's reasoning_text at transitions as ``TRIGGER reasoning TRIGGER subgoal``
        inject_reasoning: bool = False,
        subgoal_grounded: bool = True,  # parse "at <y, x>" in the subgoal text and emit Qwen-style "at [x, y]" with coords normalized to [0, 1000]
        bbox_noise_pixels: int = 8,  # max ±pixel jitter on grounded bbox coords (truncated Gaussian, std=n/2.5) — matches robomme's training-time ``add_grounding_augmentation`` with noise_range=8
        subgoal_log_interval: int = 0,  # if > 0, log decoded GT/pred subgoal text from rank 0 every N forward calls
        label_smoothing: float = 0.0,  # CE label smoothing for subgoal loss
        # Use Liger's fused linear cross-entropy for the subgoal supervision path.
        # Avoids materializing the (B, L, V) logits tensor (~10–30 GB at p95 length on
        # Qwen3.5). When True, ``liger-kernel`` must be importable; when False, the
        # eager ``lm_head + cross_entropy`` path is used and the dependency is optional.
        use_fused_ce: bool = True,
        # Inference-only hard cap on S2 cumulative token count. Long video-history tasks
        # can grow the S2 prompt past the VLM's effective context (Qwen3.5 hybrid attention
        # remains numerically stable but spends increasing per-fire compute on the prefill).
        # When the cap is exceeded at inference, the append-context S2 path raises
        # :class:`ponderpounce.model.system2.ContextOverflowError`, which the
        # dispatcher converts to an explicit episode failure instead of letting the harness
        # hit its act timeout. Training is unaffected — the check lives on inference paths only.
        max_context_tokens: int = 16384,
        # Current-observation-only S2 context (execution-memory control). Every
        # S2 query sees ``[task, demo, obs_t]`` and nothing from earlier queries —
        # no past observations, no emitted subgoal text, no past cognition.
        # Training builds one independent sequence per query (``prepare.py``);
        # inference rewinds the append context to the prefix before each fire.
        current_obs_only: bool = False,
        # Memory-only knob for ``current_obs_only`` training: rows are pushed
        # through the backbone in chunks of at most this many (padded) tokens,
        # each under an outer activation checkpoint. Rows are independent
        # sequences, so the loss is unchanged; only peak memory / recompute move.
        current_obs_chunk_tokens: int = 4096,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if current_obs_chunk_tokens <= 0:
            raise ValueError(f"current_obs_chunk_tokens must be > 0, got {current_obs_chunk_tokens}")
        if bbox_noise_pixels < 0:
            raise ValueError(f"bbox_noise_pixels must be >= 0, got {bbox_noise_pixels}")
        if subgoal_log_interval < 0:
            raise ValueError(f"subgoal_log_interval must be >= 0, got {subgoal_log_interval}")
        if max_context_tokens <= 0:
            raise ValueError(f"max_context_tokens must be > 0, got {max_context_tokens}")
        self.vlm_model_id = vlm_model_id
        self.num_cognition_tokens = num_cognition_tokens
        self.isolated_cognition_attention = isolated_cognition_attention
        self.max_demo_frames = max_demo_frames
        self.demo_fps = demo_fps
        self.inject_subgoal = inject_subgoal
        self.inject_reasoning = inject_reasoning
        self.subgoal_grounded = subgoal_grounded
        self.bbox_noise_pixels = bbox_noise_pixels
        self.subgoal_log_interval = subgoal_log_interval
        self.label_smoothing = label_smoothing
        self.use_fused_ce = use_fused_ce
        self.max_context_tokens = max_context_tokens
        self.current_obs_only = current_obs_only
        self.current_obs_chunk_tokens = current_obs_chunk_tokens

    def build_system2(self) -> "System2":
        """Construct the System2 deliberative VLM module from this config."""
        from ponderpounce.model.system2 import System2

        return System2(
            vlm_model_id=self.vlm_model_id,
            num_cognition_tokens=self.num_cognition_tokens,
            isolated_cognition_attention=self.isolated_cognition_attention,
            max_context_tokens=self.max_context_tokens,
            current_obs_only=self.current_obs_only,
            current_obs_chunk_tokens=self.current_obs_chunk_tokens,
        )


class ScheduleConfig(PretrainedConfig):
    """Async S1/S2 schedule (float-time sampler).

    T (number of S2 queries) is derived from the episode duration at the
    configured ``s2_interval_mean_ms`` cadence. ``s2_max_seq_length`` is
    not used by ``schedule.py`` — it is enforced after tokenization by a
    hard check in ``prepare_inputs``.

    Intervals and compute delays are drawn lognormal per-tick and clipped at
    ``±K·σ`` in log-space (``schedule.LOGNORMAL_CLIP_K``, default 3). Setting
    ``sigma=0`` collapses to a deterministic schedule. Defaults:
    ``s1/s2_interval_sigma=0.05`` (narrow — the two intervals model a periodic
    schedule), ``s2_compute_delay_sigma=0.3`` (moderate — GPU/token variance
    is real but bounded; p99 lands at ~580 ms for mean=300 ms).
    """

    model_type = "ponderpounce_schedule"

    def __init__(
        self,
        s2_max_seq_length: int = 8192,
        # S1 interval (between successive S1 firings)
        s1_interval_mean_ms: float = 100.0,
        s1_interval_sigma: float = 0.05,
        # S2 interval (between successive S2 queries)
        s2_interval_mean_ms: float = 1000.0,
        s2_interval_sigma: float = 0.05,
        # S2 compute delay (query → ready)
        s2_compute_delay_mean_ms: float = 300.0,
        s2_compute_delay_sigma: float = 0.3,
        # Augmentation + filtering
        cognition_dropout: float = 0.15,
        s1_min_loss_interval_ms: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if s1_interval_mean_ms <= 0 or s2_interval_mean_ms <= 0:
            raise ValueError("interval means must be > 0")
        if s2_interval_mean_ms < s1_interval_mean_ms:
            warnings.warn(
                f"s2_interval_mean_ms ({s2_interval_mean_ms}) < s1_interval_mean_ms "
                f"({s1_interval_mean_ms}) — unusual (S2 faster than S1). Proceeding.",
                stacklevel=2,
            )
        if not 0 <= cognition_dropout < 1:
            raise ValueError(f"cognition_dropout must be in [0, 1), got {cognition_dropout}")
        if s2_max_seq_length < 1024:
            raise ValueError(f"s2_max_seq_length must be >= 1024, got {s2_max_seq_length}")
        for sigma in (s1_interval_sigma, s2_interval_sigma, s2_compute_delay_sigma):
            if sigma < 0:
                raise ValueError(f"sigma must be >= 0, got {sigma}")
        if s1_min_loss_interval_ms < 0:
            raise ValueError(f"s1_min_loss_interval_ms must be >= 0, got {s1_min_loss_interval_ms}")

        self.s2_max_seq_length = s2_max_seq_length
        self.s1_interval_mean_ms = s1_interval_mean_ms
        self.s1_interval_sigma = s1_interval_sigma
        self.s2_interval_mean_ms = s2_interval_mean_ms
        self.s2_interval_sigma = s2_interval_sigma
        self.s2_compute_delay_mean_ms = s2_compute_delay_mean_ms
        self.s2_compute_delay_sigma = s2_compute_delay_sigma
        self.cognition_dropout = cognition_dropout
        # ``s1_min_loss_interval_ms``: drop S1 ticks within this window
        # of the previously kept tick. ``schedule.sample_schedule``
        # filters greedily; ``prepare_inputs`` never materializes the
        # dropped ticks (S1 forward + loss both skipped). ``0`` (default)
        # keeps every tick the lognormal sampler emits.
        self.s1_min_loss_interval_ms = s1_min_loss_interval_ms


class FlowMatchingConfig(PretrainedConfig):
    """Flow matching parameters for S1 action denoising."""

    model_type = "ponderpounce_flow"

    def __init__(
        self,
        alpha: float = 1.5,
        beta: float = 1.0,
        t_min: float = 0.001,
        t_max: float = 0.999,
        max_pos_len: int = 512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not (0 <= t_min < t_max <= 1):
            raise ValueError(f"Must have 0 <= t_min < t_max <= 1, got t_min={t_min}, t_max={t_max}")
        self.alpha = alpha
        self.beta = beta
        self.t_min = t_min
        self.t_max = t_max
        self.max_pos_len = max_pos_len


# ── Top-level config ──────────────────────────────────────────────────────


class PonderPounceConfig(PretrainedConfig):
    """PonderPounce VLA configuration with nested sub-configs.

    ``s1`` is polymorphic (any registered S1 binding config, resolved via
    ``S1_REGISTRY``); ``s2`` is fixed-type (``S2QwenConfig``). The dual core
    depends only on the uniform S1 contract (``build_adapter`` /
    ``build_s1_image_transform`` / ``s1_hard_prompt_spec`` / ``proprio_dim`` /
    ``action_chunk`` / ``max_s1_batch`` / ``img_size`` / ``delta_action_mask``)
    and the S2 contract (``build_system2``).
    """

    model_type = "ponderpounce"
    sub_configs: ClassVar[dict[str, type[PretrainedConfig]]] = {
        "s2": PretrainedConfig,
        "s1": PretrainedConfig,
        "schedule": ScheduleConfig,
        "flow": FlowMatchingConfig,
    }

    def __init__(
        self,
        s2: PretrainedConfig | dict | None = None,
        s1: PretrainedConfig | dict | None = None,
        schedule: ScheduleConfig | dict | None = None,
        flow: FlowMatchingConfig | dict | None = None,
        # Top-level training strategy (crosses S1/S2 boundary).
        # Controls gradient flow from fm_loss back through S2 backbone.
        # 0.0 = detach (no gradient), 1.0 = full (unstable at ~step 200),
        # 0<x<1 = fixed scale, -1 = auto (data-dependent: w2/(w1*N_ticks_per_query)).
        s1_grad_to_s2: float = -1.0,
        # Loss weights for total = s1_loss_weight * fm_loss + s2_loss_weight * subgoal_loss.
        # s2 default lives in the docstring's recommended [0.05, 0.5] starting range
        # (per-token CE on the subgoal text runs at a very different scale from the
        # per-step flow-matching MSE; tune per run).
        s1_loss_weight: float = 1.0,
        s2_loss_weight: float = 0.1,
        # L2-SP: penalize S2 backbone drift from pretrained init.
        # total_loss += s2_l2sp_weight * ||θ_s2 - θ_s2_init||²
        # 0 = disabled, typical range: 1e-4 to 1e-6.
        s2_l2sp_weight: float = 0.0,
        # S1 read-out aux loss: InfoNCE forcing S1's trunk to decode the injected
        # cognition into the frame's ground-truth subgoal text. 0 = off (no-op);
        # total_loss += s1_readout_aux_weight * readout_aux_loss.
        s1_readout_aux_weight: float = 0.0,
        # Scalar gain applied to the projected cognition embedding at the S1
        # injection point (before S1 attention). 1.0 = no-op; >1 amplifies the
        # single cognition token against the many image/text prefix tokens.
        cognition_gain: float = 1.0,
        # Additive pre-softmax attention-logit bias on the cognition KEY
        # positions for all S1 queries (age token unbiased). 0.0 = no-op;
        # log(3)..log(5) multiplies attention mass toward cognition by ~3-5x.
        s1_cognition_attn_bias: float = 0.0,
        # Cognition-swap contrastive loss: run a second S1 forward with each
        # frame's cognition swapped for another frame's (distinct subgoal) and
        # hinge on the flow-loss gap so predictions must DEGRADE under the
        # wrong cognition. 0.0 = off (no second forward).
        s1_cognition_swap_weight: float = 0.0,
        # Hinge margin in flow-loss units (base fm loss runs ~0.02, so 0.05
        # forces a clear gap; tune per run).
        s1_cognition_swap_margin: float = 0.05,
        # Optional path to a full PonderPounceVLA checkpoint to load weights
        # from after __init__. Used to chain stage 3 → stage 4 (and any
        # downstream resume that keeps weights but not optimizer state).
        # ``s1.pretrained`` / ``s2.vlm_model_id`` still drive the initial
        # build; this overwrites the assembled state at the very end.
        pretrained: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if s1_loss_weight < 0 or s2_loss_weight < 0:
            raise ValueError(f"loss weights must be >= 0, got s1={s1_loss_weight}, s2={s2_loss_weight}")
        if s1_readout_aux_weight < 0:
            raise ValueError(f"s1_readout_aux_weight must be >= 0, got {s1_readout_aux_weight}")
        if s1_cognition_swap_weight < 0:
            raise ValueError(f"s1_cognition_swap_weight must be >= 0, got {s1_cognition_swap_weight}")
        if s1_cognition_swap_margin < 0:
            raise ValueError(f"s1_cognition_swap_margin must be >= 0, got {s1_cognition_swap_margin}")
        # Coerce dict → sub-config instances (for from_pretrained / jsonargparse /
        # direct construction on raw YAML). s1/s2 are polymorphic; schedule/flow
        # are fixed types.
        self.s2: PretrainedConfig = _coerce(s2, default_cls=S2QwenConfig)
        self.s1: PretrainedConfig = _coerce(s1, registry=S1_REGISTRY, default_model_type=DEFAULT_S1_MODEL_TYPE)
        # Fixed-type sub-configs: route through _coerce so a serialized dict's
        # ``model_type`` is stripped before construction.
        self.schedule = _coerce(schedule, default_cls=ScheduleConfig)
        self.flow = _coerce(flow, default_cls=FlowMatchingConfig)
        self.s1_grad_to_s2 = s1_grad_to_s2
        self.s1_loss_weight = s1_loss_weight
        self.s2_loss_weight = s2_loss_weight
        self.s2_l2sp_weight = s2_l2sp_weight
        self.s1_readout_aux_weight = s1_readout_aux_weight
        self.cognition_gain = cognition_gain
        self.s1_cognition_attn_bias = s1_cognition_attn_bias
        self.s1_cognition_swap_weight = s1_cognition_swap_weight
        self.s1_cognition_swap_margin = s1_cognition_swap_margin
        self.pretrained = pretrained
