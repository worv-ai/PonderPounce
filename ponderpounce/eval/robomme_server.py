"""PonderPounce as a ``vla-eval`` model server for RoboMME (sequential inference).

One process serves one policy. Everything runs inline on the harness's
observation callback, on a simulated clock that advances one control step
(``1 / sim_freq_hz``) per observation, so the schedule is deterministic and
identical to the one the checkpoint trained under:

* **Ponder (System 2)** fires every ``s2_period_ms`` of simulated time,
  appends the current observation to its episode context and emits one
  cognition token (plus, when its gate transitions, a subgoal text). The
  cognition becomes *visible* to System 1 only after the simulated compute
  delay the checkpoint trained with (``schedule.s2_compute_delay_mean_ms``).
* **Pounce (System 1)** fires every ``s1_period_ms``; each fire predicts an
  action chunk from the current observation, proprioception, the task text
  and the newest *visible* cognition together with that cognition's age.
  By default it does not start until the first cognition is visible
  (``wait_first_cognition``); until then the arm holds its pose.
* **Dispatch (System 0)** plays the chunk back one action per control step
  and repeats the last action if a chunk runs out before the next fire.

Launch (the harness talks WebSocket to it)::

    python -m ponderpounce.eval.robomme_server \\
        --args.checkpoint_path /path/to/checkpoint --port 8000

then point a ``vla-eval`` run config at ``ws://localhost:8000`` (see
``configs/robomme.yaml`` and ``scripts/eval_robomme.sh``).
"""

from __future__ import annotations

import logging
import os
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoTokenizer
from vla_eval.model_servers.base import ModelServer, SessionContext
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, Observation

from ponderpounce.inference.append_context import S2ContextResult, SoftS2SessionContext
from ponderpounce.inference.local import LocalSystem1
from ponderpounce.inference.options import InferenceOpts
from ponderpounce.inference.session import Session
from ponderpounce.model.configuration import PonderPounceConfig
from ponderpounce.model.loading import load_model
from ponderpounce.model.system2 import ContextOverflowError
from ponderpounce.normalize import NormStats, apply_absolute_action_mask, load_stats, normalize_state, unnormalize
from ponderpounce.pi05.images import pad_camera_slots
from ponderpounce.pi05.prompt import tokenize_pi0_prompt

logger = logging.getLogger(__name__)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
NS_PER_MS = 1_000_000


def resolve_checkpoint(path_or_repo: str) -> str:
    """A local checkpoint directory, or a Hub repo id (``worv-ai/ponderpounce-9b-robomme``)
    that is downloaded into the Hub cache (``HF_HOME``) on first use."""
    if os.path.isdir(path_or_repo):
        return path_or_repo
    from huggingface_hub import snapshot_download

    logger.info("downloading %s from the Hugging Face Hub (first run only)", path_or_repo)
    return snapshot_download(path_or_repo, allow_patterns=["config.json", "model.safetensors", "norm_stats.json"])


# ── Observation adapter ──────────────────────────────────────────────


@dataclass
class ObsAdapter:
    """Harness observation dict -> model-side inputs.

    ``demo_pils`` subsamples the conditioning video the benchmark streams at the
    env frame rate down to the ``demo_fps`` the checkpoint trained with.
    """

    camera_keys: tuple[str, ...]
    proprio_dim: int
    max_demo_frames: int
    norm_stats: NormStats | None
    demo_fps: float
    env_fps: float

    def camera_pils(self, obs: Observation) -> list[PILImage.Image]:
        images = obs.get("images", {})
        pils: list[PILImage.Image] = []
        for cam in self.camera_keys:
            arr = images.get(cam)
            if arr is None:
                raise KeyError(f"observation is missing camera {cam!r}; have {list(images)}")
            pils.append(PILImage.fromarray(np.asarray(arr)).convert("RGB"))
        return pils

    def raw_proprio_array(self, obs: Observation) -> np.ndarray | None:
        state = obs.get("states")
        if state is None:
            state = obs.get("state")
        if state is None:
            return None
        arr = np.asarray(state, dtype=np.float32).flatten()
        if arr.shape[0] > self.proprio_dim:
            arr = arr[: self.proprio_dim]
        elif arr.shape[0] < self.proprio_dim:
            arr = np.pad(arr, (0, self.proprio_dim - arr.shape[0]))
        return arr

    def proprio_tensor(self, obs: Observation) -> torch.Tensor | None:
        arr = self.raw_proprio_array(obs)
        if arr is None:
            return None
        if self.norm_stats is not None:
            arr = normalize_state(arr, self.norm_stats)
        return torch.as_tensor(arr, dtype=torch.float32)

    def demo_pils(self, raw_frames: Any) -> list[PILImage.Image]:
        if not raw_frames:
            return []
        pils: list[PILImage.Image] = []
        for frame in raw_frames:
            if isinstance(frame, PILImage.Image):
                pils.append(frame.convert("RGB"))
            elif isinstance(frame, np.ndarray):
                pils.append(PILImage.fromarray(frame).convert("RGB"))
        n_raw = len(pils)
        if self.demo_fps > 0 and self.env_fps > 0 and n_raw > 1:
            pils = pils[:: max(1, round(self.env_fps / self.demo_fps))]
        if self.max_demo_frames > 0 and len(pils) > self.max_demo_frames:
            keep = np.linspace(0, len(pils) - 1, self.max_demo_frames, dtype=int)
            pils = [pils[i] for i in keep]
        if len(pils) != n_raw:
            logger.info("demo: %d raw frames -> %d (demo_fps=%.3g)", n_raw, len(pils), self.demo_fps)
        return pils


# ── Per-episode state ────────────────────────────────────────────────


@dataclass
class Cognition:
    tensor: torch.Tensor  # (K, H) CPU float32
    obs_at_ns: int  # timestamp of the observation Ponder looked at
    produced_at_ns: int
    ready_at_ns: int  # visible to Pounce from here on (simulated compute delay)
    subgoal: str
    kind: str


@dataclass
class Chunk:
    actions: torch.Tensor  # (H, action_dim) model space (normalized, deltas)
    raw_state: np.ndarray | None  # proprio at the S1 fire, for the delta -> absolute rebase
    cursor: int = 0


@dataclass
class Episode:
    session: Session
    s2_context: SoftS2SessionContext | None = None
    tick: int = 0  # control steps seen so far
    cognitions: list[Cognition] = field(default_factory=list)
    chunk: Chunk | None = None
    s1_started: bool = False
    s1_next_fire_ns: int | None = None
    s2_next_fire_ns: int | None = None
    noise_rng: torch.Generator | None = None
    n_s1_fires: int = 0
    n_s2_fires: int = 0
    active_subgoal: str = ""


# ── Server ───────────────────────────────────────────────────────────


class PonderPounceRoboMMEServer(ModelServer):
    """Sequential PonderPounce policy server (one GPU, one process)."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        sim_freq_hz: float = 20.0,
        s2_period_ms: int = 1000,
        s1_period_ms: int = 1000,
        s2_compute_delay_ms: float | None = None,
        wait_first_cognition: bool = True,
        s1_num_inference_steps: int = 10,
        camera_keys: list[str] | tuple[str, ...] = ("agentview", "wrist"),
        max_demo_frames: int | None = None,
        s2_static_cache_len: int | None = None,
        seed: int = 0,
    ) -> None:
        """
        Args:
            checkpoint_path: directory with ``config.json``, ``model.safetensors``
                and ``norm_stats.json``, or the Hub id of a released checkpoint
                (``worv-ai/ponderpounce-9b-robomme``).
            device, dtype: where both systems live. bfloat16 on one 80 GB GPU.
            sim_freq_hz: the benchmark's control rate (RoboMME: 20 Hz). One
                observation advances the simulated clock by ``1 / sim_freq_hz``.
            s2_period_ms, s1_period_ms: Ponder / Pounce fire periods in simulated
                time. The RoboMME checkpoints trained at 1000 / 1000.
            s2_compute_delay_ms: how long a cognition stays invisible to Pounce
                after Ponder fires. ``None`` reads the value the checkpoint trained
                with from its config (``schedule.s2_compute_delay_mean_ms``).
            wait_first_cognition: hold the arm until the first cognition is
                visible instead of starting Pounce on the null cognition.
            s1_num_inference_steps: flow-matching Euler steps for the action chunk.
            camera_keys: observation image keys, in the order the model expects.
            max_demo_frames: cap on conditioning-demo frames after subsampling;
                ``None`` uses the checkpoint config (``s2.max_demo_frames``).
            s2_static_cache_len: KV cache capacity for Ponder's context; ``None``
                tracks the checkpoint's ``s2.max_context_tokens``.
            seed: base seed of the per-episode flow-matching noise.
        """
        super().__init__()
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")
        if sim_freq_hz <= 0:
            raise ValueError("sim_freq_hz must be > 0")
        self._device = torch.device(device)
        self._dtype = _DTYPES[dtype]
        self._dt_ns = int(round(1e9 / sim_freq_hz))
        self._s2_period_ns = int(s2_period_ms) * NS_PER_MS
        self._s1_period_ns = int(s1_period_ms) * NS_PER_MS
        self._wait_first_cognition = bool(wait_first_cognition)
        self._camera_keys = tuple(camera_keys)
        self._seed = int(seed)
        self._episodes: dict[str, Episode] = {}
        self._episode_counts: dict[str, int] = {}

        checkpoint_path = resolve_checkpoint(checkpoint_path)
        config = PonderPounceConfig.from_pretrained(checkpoint_path)
        self._s2_delay_ns = int(
            (
                float(config.schedule.s2_compute_delay_mean_ms)
                if s2_compute_delay_ms is None
                else float(s2_compute_delay_ms)
            )
            * NS_PER_MS
        )
        logger.info(
            "loading %s (S1 %s, S2 %s) -> %s %s",
            checkpoint_path,
            config.s1.model_type,
            config.s2.vlm_model_id,
            device,
            dtype,
        )
        self._model = load_model(
            checkpoint_path,
            device=self._device,
            dtype=self._dtype,
            inference_opts=InferenceOpts(s2_static_cache_len=s2_static_cache_len),
        )
        norm_stats_path = os.path.join(checkpoint_path, "norm_stats.json")
        self._norm_stats: NormStats | None = load_stats(norm_stats_path) if os.path.exists(norm_stats_path) else None
        if self._norm_stats is None:
            logger.warning("norm_stats.json not found in %s - actions are served un-normalized", checkpoint_path)

        self._s1 = LocalSystem1(
            self._model, device=self._device, dtype=self._dtype, num_inference_steps=s1_num_inference_steps
        )
        self._s2 = self._model.system2.to(self._device, self._dtype).eval()
        self._s2_processor = AutoProcessor.from_pretrained(config.s2.vlm_model_id)
        self._s2_max_new_tokens = 320 if bool(getattr(config.s2, "inject_reasoning", False)) else 40
        self._subgoal_grounded = bool(config.s2.subgoal_grounded)
        self._s1_transform = config.s1.build_s1_image_transform().s1_transform
        self._adapter = ObsAdapter(
            camera_keys=self._camera_keys,
            proprio_dim=int(config.s1.proprio_dim),
            max_demo_frames=int(config.s2.max_demo_frames) if max_demo_frames is None else int(max_demo_frames),
            norm_stats=self._norm_stats,
            demo_fps=float(config.s2.demo_fps),
            env_fps=float(sim_freq_hz),
        )
        # Pi0.5 text channel: the task prompt, tokenized once per episode.
        self._s1_tokenizer: Any = None
        self._s1_max_token_len = 0
        self._s1_num_camera_slots = 0
        hard = config.s1.s1_hard_prompt_spec()
        if hard is not None:
            if bool(hard["subgoal_enabled"]):
                raise NotImplementedError(
                    "checkpoints with hard_subgoal_enabled=True are not supported by this server"
                )
            self._s1_tokenizer = AutoTokenizer.from_pretrained(hard["tokenizer_id"])
            self._s1_tokenizer.padding_side = "right"
            self._s1_max_token_len = int(hard["max_token_len"])
            self._s1_num_camera_slots = int(hard["num_camera_slots"])
        logger.info(
            "ready: chunk=%d actions, action_dim=%d, S2 period %d ms, S1 period %d ms, S2 delay %d ms, cameras=%s",
            self._s1.action_chunk_size,
            self._s1.action_dim,
            s2_period_ms,
            s1_period_ms,
            self._s2_delay_ns // NS_PER_MS,
            self._camera_keys,
        )

    # ── vla-eval surface ─────────────────────────────────────────

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"images": IMAGE_RGB, "states": RAW, "task_description": LANGUAGE}

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        sid = ctx.session_id
        task_payload = config.get("task", config) if isinstance(config, dict) else {}
        task_text = str(task_payload.get("task_description") or config.get("task_description") or "")
        demo = self._adapter.demo_pils(config.get("demo_frames")) if isinstance(config, dict) else []
        session = Session(
            sid=sid,
            obs_adapter=self._adapter,
            task_text=task_text,
            demo_images=demo,
            subgoal_grounded=self._subgoal_grounded,
            s2_processor=self._s2_processor,
            s1_transform=self._s1_transform,
            num_cognition_tokens=int(self._s2.num_cognition_tokens),
            s1_num_camera_slots=self._s1_num_camera_slots,
        )
        if task_text:
            session.s1_prompt_ids, session.s1_prompt_mask = self._prompt_ids(task_text)
        # Deterministic flow-matching noise per (session, episode), independent of
        # wall time; the episode counter folds in so a reused session id gets a
        # fresh stream each episode.
        n = self._episode_counts.get(sid, 0)
        self._episode_counts[sid] = n + 1
        rng = torch.Generator().manual_seed(zlib.crc32(f"{self._seed}:{sid}:{n}".encode()))
        self._episodes[sid] = Episode(session=session, noise_rng=rng)
        logger.info("[%s] episode start: task=%r demo_frames=%d", sid, task_text, len(demo))

    async def on_observation(self, obs: Observation, ctx: SessionContext) -> None:
        ep = self._episodes.get(ctx.session_id)
        if ep is None:
            raise RuntimeError(f"observation for unknown session {ctx.session_id!r}; on_episode_start not called")
        self._ingest_metadata(ep, obs)
        ep.tick += 1
        now = ep.tick * self._dt_ns  # the first observation sits at t = dt, as in training

        # Ponder: every s2_period from the first observation on.
        if ep.s2_next_fire_ns is None:
            ep.s2_next_fire_ns = now
        if now >= ep.s2_next_fire_ns:
            self._fire_s2(ep, obs, now)
            ep.s2_next_fire_ns += self._s2_period_ns

        # Pounce: starts once a cognition is visible (or right away without the
        # wait gate), then every s1_period on its own grid.
        if not ep.s1_started and (not self._wait_first_cognition or self._visible_cognition(ep, now) is not None):
            ep.s1_started = True
            ep.s1_next_fire_ns = now
        if ep.s1_started and now >= ep.s1_next_fire_ns:  # type: ignore[operator]
            self._fire_s1(ep, obs, now)
            ep.s1_next_fire_ns += self._s1_period_ns  # type: ignore[operator]

        await ctx.send_action(self._dispense(ep, obs))

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        ep = self._episodes.pop(ctx.session_id, None)
        if ep is not None:
            logger.info(
                "[%s] episode end: %d steps, %d Ponder fires, %d Pounce fires, last subgoal=%r",
                ctx.session_id,
                ep.tick,
                ep.n_s2_fires,
                ep.n_s1_fires,
                ep.active_subgoal,
            )

    # ── Systems ──────────────────────────────────────────────────

    def _ingest_metadata(self, ep: Episode, obs: Observation) -> None:
        """RoboMME sends the task text on every observation and the conditioning
        video on the first one; pick both up before Ponder's context exists."""
        s = ep.session
        task = obs.get("task_description") if isinstance(obs, dict) else None
        if task and not s.task_text:
            s.task_text = str(task)
            s.s1_prompt_ids, s.s1_prompt_mask = self._prompt_ids(s.task_text)
        video = obs.get("video_history") if isinstance(obs, dict) else None
        if video and not s.demo_images and ep.s2_context is None:
            s.demo_images = self._adapter.demo_pils(video)

    def _prompt_ids(self, task: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self._s1_tokenizer is None or not task:
            return None, None
        tok = tokenize_pi0_prompt(
            self._s1_tokenizer, tasks=[task], subgoals=[None], max_token_len=self._s1_max_token_len
        )
        return tok["input_ids"].squeeze(0), tok["attention_mask"].squeeze(0)

    def _fire_s2(self, ep: Episode, obs: Observation, now: int) -> None:
        if ep.s2_context is None:
            ep.s2_context = SoftS2SessionContext(
                session=ep.session,
                system2=self._s2,
                device=self._device,
                dtype=self._dtype,
                max_new_tokens=self._s2_max_new_tokens,
            )
        try:
            result: S2ContextResult = ep.s2_context.fire(self._adapter.camera_pils(obs))
        except ContextOverflowError as exc:
            # Fail the episode loudly rather than serving a policy whose memory
            # silently stopped growing.
            raise RuntimeError(f"Ponder context overflow at step {ep.tick}: {exc}") from exc
        ep.n_s2_fires += 1
        if result.subgoal_text:
            ep.active_subgoal = result.subgoal_text
            logger.info("[%s] t=%.2fs subgoal -> %r", ep.session.sid, now / 1e9, result.subgoal_text)
        ep.cognitions.append(
            Cognition(
                tensor=result.cognition,
                obs_at_ns=now,
                produced_at_ns=now,
                ready_at_ns=now + self._s2_delay_ns,
                subgoal=result.subgoal_text,
                kind=result.kind,
            )
        )
        del ep.cognitions[:-8]  # only the newest few can ever be selected

    @staticmethod
    def _visible_cognition(ep: Episode, now: int) -> Cognition | None:
        for cog in reversed(ep.cognitions):
            if cog.ready_at_ns <= now:
                return cog
        return None

    def _fire_s1(self, ep: Episode, obs: Observation, now: int) -> None:
        s = ep.session
        pils = self._adapter.camera_pils(obs)
        images = torch.stack([s.s1_transform(im) for im in pils], dim=0).unsqueeze(0)  # (1, n_cams, 3, H, W)
        pixel_mask: torch.Tensor | None = None
        if self._s1_num_camera_slots > 0:
            images, pixel_mask = pad_camera_slots(images, self._s1_num_camera_slots)
        proprio = self._adapter.proprio_tensor(obs)
        raw_state = self._adapter.raw_proprio_array(obs)
        cog = self._visible_cognition(ep, now)
        if cog is None:
            cognition, age_ms = self._s1.null_cognition, 0.0
        else:
            cognition, age_ms = cog.tensor, max(0.0, (now - cog.obs_at_ns) / NS_PER_MS)
        kwargs: dict[str, Any] = {
            "images": images,
            "cognition": cognition.unsqueeze(0),
            "age_ms": torch.tensor([age_ms], dtype=torch.float32),
            "proprio": None if proprio is None else proprio.unsqueeze(0),
            "pixel_attention_mask": pixel_mask,
        }
        if s.s1_prompt_ids is not None:
            kwargs["symbolic_prompt_ids"] = s.s1_prompt_ids.unsqueeze(0)
            kwargs["symbolic_prompt_mask"] = s.s1_prompt_mask.unsqueeze(0)
        spec = self._s1.noise_spec
        if spec is not None:
            kwargs["noise"] = torch.randn(1, *spec, generator=ep.noise_rng, dtype=torch.float32)
        chunk = self._s1.predict_action(**kwargs)[0]  # (H, action_dim) CPU float32
        ep.chunk = Chunk(actions=chunk, raw_state=raw_state)
        ep.n_s1_fires += 1

    def _dispense(self, ep: Episode, obs: Observation) -> Action:
        """One env-space action per control step: the next chunk row (the last
        row again once the chunk is exhausted), or a hold before the first chunk."""
        if ep.chunk is None:
            return {"actions": self._hold(self._adapter.raw_proprio_array(obs))}
        c = ep.chunk
        idx = min(c.cursor, int(c.actions.shape[0]) - 1)
        c.cursor = min(c.cursor + 1, int(c.actions.shape[0]))
        action = c.actions[idx : idx + 1].numpy()
        if self._norm_stats is not None:
            action = unnormalize(action, self._norm_stats)
            if self._norm_stats.delta_action_mask is not None and c.raw_state is not None:
                action = apply_absolute_action_mask(action, c.raw_state, mask=self._norm_stats.delta_action_mask)
        return {"actions": action}

    def _hold(self, raw_state: np.ndarray | None) -> np.ndarray:
        """Env-space 'stay where you are': zero on delta channels (rebased onto the
        current joints), the current proprio value on absolute channels."""
        ns = self._norm_stats
        dim = (
            int(ns.q01.shape[0])
            if ns is not None and ns.q01 is not None
            else (0 if raw_state is None else raw_state.shape[-1])
        )
        hold = np.zeros((1, dim), dtype=np.float32)
        if raw_state is None:
            return hold
        rs = np.asarray(raw_state, dtype=np.float32).reshape(1, -1)
        mask = ns.delta_action_mask if ns is not None else None
        if mask is None:
            w = min(dim, rs.shape[1])
            hold[0, :w] = rs[0, :w]
            return hold
        hold = apply_absolute_action_mask(hold, rs, mask=mask)
        m = np.asarray(mask, dtype=bool)
        w = min(m.shape[0], dim, rs.shape[1])
        hold[0, np.flatnonzero(~m[:w])] = rs[0, np.flatnonzero(~m[:w])]
        return hold


if __name__ == "__main__":  # pragma: no cover
    from vla_eval.model_servers.serve import run_server

    run_server(PonderPounceRoboMMEServer)
