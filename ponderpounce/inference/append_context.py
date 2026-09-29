"""Append-only S2 inference contexts.

The training path still owns the canonical full-sequence builder.  This module
is the inference-only runtime: a session keeps the committed KV/cache state on
the S2 server and appends only new observations, generated subgoal text, and
cognition/terminator tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch
from PIL import Image as PILImage

from ponderpounce.constants import SUBGOAL_TRIGGER_STR, resolve_cognition_token_id
from ponderpounce.inference.session import Session
from ponderpounce.model.subgoal_text import format_subgoal
from ponderpounce.model.system2 import (
    ContextOverflowError,
    _restore_cache_state,
    _snapshot_cache_state,
    _static_layer_max_cache_len,
)

CognitionKind = Literal["transition", "nontransition"]
"""Whether the S2 fire generated a new subgoal (its gate transitioned) or only a cognition token."""


@dataclass(frozen=True)
class S2ContextResult:
    subgoal_tokens: list[int] | None
    subgoal_text: str
    cognition: torch.Tensor
    kind: CognitionKind
    gate_score: float | None
    n_input_frames: int
    reasoning_text: str = ""


def split_generated_block(decoded: str) -> tuple[str, str]:
    """Split a decoded generation block into ``(reasoning, subgoal)``.

    Blocks are ``TRIGGER subgoal`` (plain transition) or ``TRIGGER reasoning
    TRIGGER subgoal`` (demo-parse reasoning ckpts, first exec transition).
    The last trigger-delimited part is always the subgoal.
    """
    parts = decoded.split(SUBGOAL_TRIGGER_STR)
    if len(parts) >= 3:
        return SUBGOAL_TRIGGER_STR.join(parts[1:-1]).strip(), parts[-1]
    return "", parts[-1] if len(parts) > 1 else decoded


class S2SessionContext(Protocol):
    def fire(self, obs_pils: list[PILImage.Image]) -> S2ContextResult: ...


@dataclass(frozen=True)
class _Segment:
    input_ids: torch.Tensor
    mm_token_type_ids: torch.Tensor
    cognition_mask: torch.Tensor
    s2_query_ids: torch.Tensor
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return int(self.input_ids.numel())


@dataclass
class _BranchSnapshot:
    cache_state: list
    context_len: int
    position: int
    query_index: int
    n_obs_frames: int
    last_hidden: torch.Tensor | None
    task_prefix_pending: bool
    cognition_mask: torch.Tensor | None = None
    s2_query_ids: torch.Tensor | None = None


class _SegmentCompiler:
    """Build byte-level S2 sequence segments without rebuilding history."""

    def __init__(
        self,
        processor: Any,
        *,
        grounded: bool,
        num_cognition_tokens: int,
        spatial_merge_size: int | None = None,
    ) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.grounded = bool(grounded)
        self.num_cognition_tokens = int(num_cognition_tokens)
        self.cognition_token_id = resolve_cognition_token_id(self.tokenizer)
        self.vision_start_id = int(self.tokenizer.convert_tokens_to_ids("<|vision_start|>"))
        self.image_pad_id = int(self.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        self.vision_end_id = int(self.tokenizer.convert_tokens_to_ids("<|vision_end|>"))
        if spatial_merge_size is None:
            spatial_merge_size = int(getattr(processor.image_processor, "merge_size", 2))
        self.spatial_merge_size = int(spatial_merge_size)

    def prefix(self, task: str, demo_images: list[PILImage.Image]) -> _Segment:
        if not demo_images:
            return self.empty()
        preamble = self._encode_text(f"<|im_start|>user\n{task}")
        pieces = [self._text_segment(preamble, qid=-1)]
        pieces.append(self.images(demo_images, qid=-1))
        return self.concat(pieces)

    def obs(self, images: list[PILImage.Image], *, qid: int, task_prefix: str | None = None) -> _Segment:
        pieces: list[_Segment] = []
        if task_prefix is not None:
            pieces.append(self._text_segment(self._encode_text(f"<|im_start|>user\n{task_prefix}"), qid=qid))
        pieces.append(self.images(images, qid=qid))
        return self.concat(pieces)

    def subgoal_text(self, text: str, *, qid: int) -> _Segment:
        toks = self._encode_text(format_subgoal(text, grounded=self.grounded, bbox_noise_pixels=0, rng=None))
        return self._text_segment(toks, qid=qid)

    def generated_tokens(self, tokens: list[int], *, qid: int) -> _Segment:
        return self._text_segment(tokens, qid=qid)

    def cog(self, *, qid: int) -> _Segment:
        ids = torch.full((self.num_cognition_tokens,), self.cognition_token_id, dtype=torch.long)
        return _Segment(
            input_ids=ids,
            mm_token_type_ids=torch.zeros_like(ids),
            cognition_mask=torch.ones_like(ids, dtype=torch.bool),
            s2_query_ids=torch.full_like(ids, qid),
        )

    def images(self, images: list[PILImage.Image], *, qid: int) -> _Segment:
        if not images:
            z = torch.zeros(0, dtype=torch.long)
            return _Segment(z, z.clone(), torch.zeros(0, dtype=torch.bool), z.clone())
        processed = self.processor.image_processor(images=images, return_tensors="pt")
        grid = processed["image_grid_thw"]
        ids: list[int] = []
        mm: list[int] = []
        for g in grid:
            n_image_tokens = int(g.prod().item()) // (self.spatial_merge_size**2)
            ids.append(self.vision_start_id)
            mm.append(0)
            ids.extend([self.image_pad_id] * n_image_tokens)
            mm.extend([1] * n_image_tokens)
            ids.append(self.vision_end_id)
            mm.append(0)
        input_ids = torch.tensor(ids, dtype=torch.long)
        return _Segment(
            input_ids=input_ids,
            mm_token_type_ids=torch.tensor(mm, dtype=torch.long),
            cognition_mask=torch.zeros(len(ids), dtype=torch.bool),
            s2_query_ids=torch.full((len(ids),), int(qid), dtype=torch.long),
            pixel_values=processed["pixel_values"],
            image_grid_thw=grid,
        )

    def concat(self, segments: list[_Segment]) -> _Segment:
        segments = [s for s in segments if s.length > 0]
        if not segments:
            return self.empty()
        pixel_values = [s.pixel_values for s in segments if s.pixel_values is not None and s.pixel_values.numel()]
        grids = [s.image_grid_thw for s in segments if s.image_grid_thw is not None and s.image_grid_thw.numel()]
        return _Segment(
            input_ids=torch.cat([s.input_ids for s in segments]),
            mm_token_type_ids=torch.cat([s.mm_token_type_ids for s in segments]),
            cognition_mask=torch.cat([s.cognition_mask for s in segments]),
            s2_query_ids=torch.cat([s.s2_query_ids for s in segments]),
            pixel_values=torch.cat(pixel_values, dim=0) if pixel_values else None,
            image_grid_thw=torch.cat(grids, dim=0) if grids else None,
        )

    @staticmethod
    def empty() -> _Segment:
        z = torch.zeros(0, dtype=torch.long)
        return _Segment(z, z.clone(), torch.zeros(0, dtype=torch.bool), z.clone())

    def _encode_text(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False).input_ids)

    def _text_segment(self, tokens: list[int] | torch.Tensor, *, qid: int) -> _Segment:
        ids = torch.as_tensor(tokens, dtype=torch.long)
        return _Segment(
            input_ids=ids,
            mm_token_type_ids=torch.zeros_like(ids),
            cognition_mask=torch.zeros(ids.shape[0], dtype=torch.bool),
            s2_query_ids=torch.full(ids.shape, int(qid), dtype=torch.long),
        )


class _MRopeCursor:
    def __init__(self, model: Any) -> None:
        self._model = model
        self._merge = int(model.config.vision_config.spatial_merge_size)
        self.position = 0

    def positions_for(self, segment: _Segment, *, device: torch.device) -> torch.Tensor:
        out: list[torch.Tensor] = []
        grid_iter = iter(segment.image_grid_thw) if segment.image_grid_thw is not None else iter(())
        ids = segment.input_ids
        mm = segment.mm_token_type_ids
        i = 0
        while i < int(ids.numel()):
            if int(mm[i].item()) == 0:
                j = i + 1
                while j < int(ids.numel()) and int(mm[j].item()) == 0:
                    j += 1
                n = j - i
                pos = torch.arange(n, device=device, dtype=torch.long).view(1, -1).expand(3, -1) + self.position
                out.append(pos)
                self.position += n
                i = j
                continue
            j = i + 1
            while j < int(ids.numel()) and int(mm[j].item()) == 1:
                j += 1
            grid = next(grid_iter)
            out.append(
                self._model.get_vision_position_ids(
                    self.position,
                    grid.to(device),
                    1,
                    self._merge,
                    device=device,
                )
            )
            self.position += int(max(int(grid[1].item()), int(grid[2].item())) // self._merge)
            i = j
        if not out:
            return torch.zeros(3, 1, 0, device=device, dtype=torch.long)
        return torch.cat(out, dim=1).unsqueeze(1)

    def snapshot(self) -> int:
        return int(self.position)

    def restore(self, position: int) -> None:
        self.position = int(position)


class _BaseAppendContext:
    def __init__(
        self,
        *,
        session: Session,
        device: torch.device,
        dtype: torch.dtype,
        max_new_tokens: int,
        convert_s2_coords: bool,
    ) -> None:
        self.session = session
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_new_tokens = int(max_new_tokens)
        self.convert_s2_coords = bool(convert_s2_coords)
        self.query_index = 0
        self.n_obs_frames = 0
        self._last_hidden: torch.Tensor | None = None
        self._task_prefix_pending = not bool(session.demo_images)

    def _score_gate(self, logits: torch.Tensor, trigger_id: int) -> float:
        return float(torch.softmax(logits[0].float(), dim=-1)[trigger_id].item())


class SoftS2SessionContext(_BaseAppendContext):
    def __init__(
        self,
        *,
        session: Session,
        system2: Any,
        device: torch.device,
        dtype: torch.dtype,
        max_new_tokens: int,
    ) -> None:
        super().__init__(
            session=session, device=device, dtype=dtype, max_new_tokens=max_new_tokens, convert_s2_coords=False
        )
        self.s2 = system2
        self.compiler = _SegmentCompiler(
            session.s2_processor,
            grounded=session.subgoal_grounded,
            num_cognition_tokens=int(system2.num_cognition_tokens),
            spatial_merge_size=int(system2.vlm.config.vision_config.spatial_merge_size),
        )
        self._prefix = self.compiler.prefix(session.task_text, session.demo_images)
        self.reset()

    def reset(self) -> None:
        self.query_index = 0
        self.n_obs_frames = 0
        self._last_hidden = None
        self._task_prefix_pending = not bool(self.session.demo_images)
        self.rope = _MRopeCursor(self.s2.vlm.model)
        self.cache = self.s2._new_static_cache()
        self._context_len = 0
        self._cum_cm = torch.zeros(1, 0, dtype=torch.bool, device=self.device)
        self._cum_qid = torch.zeros(1, 0, dtype=torch.long, device=self.device)
        if self._prefix.length:
            self._append(self._prefix, total_queries=0)
        # ``current_obs_only``: every fire restarts from this post-prefix state,
        # mirroring the training layout ``[task, demo, obs_t, (subgoal_t), cog_t]``
        # with no earlier query in context.
        self._prefix_snapshot = self._snapshot() if getattr(self.s2, "current_obs_only", False) else None

    @torch.no_grad()
    def fire(self, obs_pils: list[PILImage.Image]) -> S2ContextResult:
        if self._prefix_snapshot is not None:
            query_index, n_obs_frames = self.query_index, self.n_obs_frames
            self._restore(self._prefix_snapshot)
            self.query_index, self.n_obs_frames = query_index, n_obs_frames
        fire_start = self._snapshot()
        try:
            qid = self.query_index
            task_prefix = self.session.task_text if self._task_prefix_pending else None
            self._append(self.compiler.obs(obs_pils, qid=qid, task_prefix=task_prefix), total_queries=0)
            self._task_prefix_pending = False
            logits = self.s2.vlm.lm_head(self._last_hidden[:, -1, :])  # type: ignore[index]
            next_id = int(logits.argmax(dim=-1).item())
            trigger_id = int(self.s2._subgoal_trigger_id)
            cog_id = int(self.s2._cognition_token_id)
            gate_score = self._score_gate(logits, trigger_id)
            sg_tokens: list[int] | None = None
            sg_text = ""
            reasoning_text = ""
            kind: CognitionKind = "nontransition"
            if next_id == trigger_id:
                branch = self._snapshot()
                try:
                    inner: list[int] = []
                    self._append(self.compiler.generated_tokens([trigger_id], qid=qid), total_queries=0)
                    committed = False
                    for _ in range(max(0, self.max_new_tokens - 1)):
                        logits = self.s2.vlm.lm_head(self._last_hidden[:, -1, :])  # type: ignore[index]
                        tok = int(logits.argmax(dim=-1).item())
                        if tok == cog_id:
                            committed = True
                            break
                        inner.append(tok)
                        self._append(self.compiler.generated_tokens([tok], qid=qid), total_queries=0)
                    if committed:
                        sg_tokens = [trigger_id, *inner]
                        decoded = self.session.s2_processor.tokenizer.decode(sg_tokens, skip_special_tokens=False)
                        reasoning_text, sg_text = split_generated_block(decoded)
                        kind = "transition" if sg_text else "nontransition"
                    else:
                        self._restore(branch)
                except Exception:
                    self._restore(branch)
                    raise
            cog = self._append(self.compiler.cog(qid=qid), total_queries=1)
            cognition = cog[0]
            self.query_index += 1
            self.n_obs_frames += 1
            return S2ContextResult(
                subgoal_tokens=sg_tokens,
                subgoal_text=sg_text,
                cognition=cognition.detach().to("cpu", torch.float32).contiguous(),
                kind=kind,
                gate_score=gate_score,
                n_input_frames=self.n_obs_frames,
                reasoning_text=reasoning_text,
            )
        except Exception:
            self._restore(fire_start)
            raise

    @torch.no_grad()
    def _append(self, segment: _Segment, *, total_queries: int) -> torch.Tensor:
        if segment.length == 0:
            assert self._last_hidden is not None
            return self._last_hidden
        new_len = self._context_len + segment.length
        if new_len > int(self.s2.effective_context_cap):
            raise ContextOverflowError(
                length=new_len,
                cap=int(self.s2.effective_context_cap),
            )
        ids = segment.input_ids.to(self.device).unsqueeze(0)
        cm = segment.cognition_mask.to(self.device).unsqueeze(0).bool()
        qid = segment.s2_query_ids.to(self.device).unsqueeze(0)
        cache_state = _snapshot_cache_state(self.cache)
        rope_pos = self.rope.snapshot()
        new_cm = torch.cat([self._cum_cm, cm], dim=1)
        new_qid = torch.cat([self._cum_qid, qid], dim=1)
        try:
            pos = self.rope.positions_for(segment, device=self.device)
            embeds = self.s2._build_inputs_embeds(
                input_ids=ids,
                cognition_mask=cm,
                pixel_values=(
                    segment.pixel_values.to(self.device, self.dtype) if segment.pixel_values is not None else None
                ),
                image_grid_thw=(
                    segment.image_grid_thw.to(self.device) if segment.image_grid_thw is not None else None
                ),
                s2_total_queries=total_queries,
            )
            static_len = _static_layer_max_cache_len(self.cache)
            with self.s2._isolation_stash(
                cm,
                qid,
                cumulative_cognition_mask=new_cm,
                cumulative_s2_query_ids=new_qid,
                static_cache_len=static_len,
            ):
                outputs = self.s2._run_cached_language_model(
                    position_ids=pos,
                    attention_mask=torch.ones((1, new_len), dtype=torch.long, device=self.device),
                    inputs_embeds=embeds,
                    past_key_values=self.cache,
                    compile_ok=segment.pixel_values is None,
                )
        except Exception:
            _restore_cache_state(self.cache, cache_state)
            self.rope.restore(rope_pos)
            raise
        self._context_len = new_len
        self._cum_cm = new_cm
        self._cum_qid = new_qid
        self._last_hidden = outputs.last_hidden_state
        return self._last_hidden

    def _snapshot(self) -> _BranchSnapshot:
        return _BranchSnapshot(
            cache_state=_snapshot_cache_state(self.cache),
            context_len=self._context_len,
            position=self.rope.snapshot(),
            query_index=self.query_index,
            n_obs_frames=self.n_obs_frames,
            last_hidden=self._last_hidden,
            task_prefix_pending=self._task_prefix_pending,
            cognition_mask=self._cum_cm.clone(),
            s2_query_ids=self._cum_qid.clone(),
        )

    def _restore(self, snap: _BranchSnapshot) -> None:
        _restore_cache_state(self.cache, snap.cache_state)
        assert snap.cognition_mask is not None and snap.s2_query_ids is not None
        self._context_len = snap.context_len
        self._cum_cm = snap.cognition_mask
        self._cum_qid = snap.s2_query_ids
        self.rope.restore(snap.position)
        self.query_index = snap.query_index
        self.n_obs_frames = snap.n_obs_frames
        self._last_hidden = snap.last_hidden
        self._task_prefix_pending = snap.task_prefix_pending
