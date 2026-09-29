"""Shared constants for VLA training."""

# Sentinel value used in input_ids to mark where image features should be spliced in.
# Used by CogACT and GR00T model families (collator writes, model reads).
IMAGE_TOKEN_INDEX = -200

# Carrier token for cognition slots: must be a real vocab token (not a negative
# sentinel) so the LM head can target it in cross-entropy — the model emits it at
# the gate position before a cognition chunk to mark a transition.
# ``<|fim_pad|>`` (Qwen Fill-In-Middle pad byte): single-piece across Qwen2.5 /
# Qwen3 / Qwen3.5 (ids 151662 / 248063) and near-absent from pretraining + VLA
# data, so its embedding / LM-head row carry negligible prior to contaminate the
# signal. Resolved via ``resolve_cognition_token_id()`` (id not baked in).
COGNITION_TOKEN_STR = "<|fim_pad|>"


# Subgoal-body opener. A text prefix like ``"Subgoal: "`` is multi-piece BPE whose
# first id also appears inside subgoal bodies (``"Sub"`` opens both), polluting the
# transition-vs-content split; a vocab-resident single-piece special token whose
# ``token → id → token`` round-trip is bit-exact removes that collision class.
# ``<|fim_prefix|>`` partners ``<|fim_pad|>``: same FIM family, single-piece across
# Qwen2.5 / Qwen3 / Qwen3.5, negligible prior outside code.
SUBGOAL_TRIGGER_STR = "<|fim_prefix|>"


def _resolve_special_token_id(tokenizer, token_str: str, role: str) -> int:
    """Resolve ``token_str`` via ``tokenizer``, raising at init (not as a ghost
    training run) on either silent-corruption mode:

    1. unk / ``None`` id — string isn't a real token, so CE would supervise a
       bogus class.
    2. ``id → token`` round-trip mismatch — string tokenised into multiple pieces.
    """
    tid = tokenizer.convert_tokens_to_ids(token_str)
    if tid is None or (tokenizer.unk_token_id is not None and tid == tokenizer.unk_token_id):
        raise ValueError(
            f"Tokenizer does not own {role} token {token_str!r}. "
            f"{role.capitalize()} training requires this string to be a real vocab "
            f"token. Pick a different carrier or use a VLM whose tokenizer knows it."
        )
    roundtrip = tokenizer.convert_ids_to_tokens(tid)
    if roundtrip != token_str:
        raise ValueError(
            f"Tokenizer round-trip inconsistent for {role} carrier: "
            f"{token_str!r} → id {tid} → {roundtrip!r}. Cannot "
            f"safely use this carrier with this tokenizer."
        )
    return int(tid)


def resolve_cognition_token_id(tokenizer) -> int:
    """Resolve :data:`COGNITION_TOKEN_STR` to its vocab id with fast-fail."""
    return _resolve_special_token_id(tokenizer, COGNITION_TOKEN_STR, "cognition carrier")


def resolve_subgoal_trigger_id(tokenizer) -> int:
    """Resolve :data:`SUBGOAL_TRIGGER_STR` to its vocab id with fast-fail."""
    return _resolve_special_token_id(tokenizer, SUBGOAL_TRIGGER_STR, "subgoal trigger")


# PyTorch's default cross-entropy ignore index. Unified here so the
# subgoal-label collator and the subgoal CE loss cannot drift apart.
IGNORE_INDEX = -100
