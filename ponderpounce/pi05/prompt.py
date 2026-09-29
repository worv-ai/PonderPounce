"""Reference Pi0 prompt format — single source of truth.

Mirrors ``mme_vla_suite/training/config.py:142``. Every consumer of the
Pi05 PaliGemma hard channel must turn task text into prompt token ids
through ``tokenize_pi0_prompt`` here — that is what keeps the S1 input
byte-identical across the train and inference paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import BatchEncoding, PreTrainedTokenizerBase


def _clean(s: str) -> str:
    return (s or "").strip().replace("_", " ").replace("\n", " ")


def build_pi0_prompt(task: str, subgoal: str | None) -> str:
    """Pi05 PaliGemma prefix prompt.

    - ``subgoal is None`` → ``"Task: {task};\\nAction: "`` (Stage-2 soft-only, paper-aligned).
    - ``subgoal`` str (incl. ``""``) → ``"Task: {task};\\nCurrent Subgoal: {sg};\\nAction: "``
      (Stage-1 byte-equivalent).
    """
    if subgoal is None:
        return f"Task: {_clean(task)};\nAction: "
    return f"Task: {_clean(task)};\nCurrent Subgoal: {_clean(subgoal)};\nAction: "


def tokenize_pi0_prompt(
    tokenizer: PreTrainedTokenizerBase,
    *,
    tasks: Sequence[str],
    subgoals: Sequence[str | None],
    max_token_len: int,
) -> BatchEncoding:
    """Build + tokenize Pi05 hard-channel prompts — the one sanctioned path.

    ``tasks`` and ``subgoals`` are zipped element-wise (lengths must match);
    pass ``subgoal=None`` for the soft-only template. Right padding is set
    per call (the action expert was trained right-padded), so the tokenizer
    object itself is never mutated.
    """
    prompts = [build_pi0_prompt(task, subgoal) for task, subgoal in zip(tasks, subgoals, strict=True)]
    return tokenizer(
        prompts,
        padding="max_length",
        padding_side="right",
        truncation=True,
        max_length=max_token_len,
        add_special_tokens=True,
        return_tensors="pt",
    )
