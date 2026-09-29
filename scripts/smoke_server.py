#!/usr/bin/env python3
"""Drive the RoboMME server through one synthetic episode (no simulator).

Checks that the checkpoint loads, that Ponder and Pounce fire on the trained
schedule, and that env-space actions come out with the right shape:

    python scripts/smoke_server.py --ckpt /path/to/checkpoint [--steps 60]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

import numpy as np

from ponderpounce.eval.robomme_server import PonderPounceRoboMMEServer


class FakeCtx:
    def __init__(self) -> None:
        self.session_id = "smoke"
        self.actions: list[np.ndarray] = []

    async def send_action(self, action) -> None:
        self.actions.append(np.asarray(action["actions"]))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--demo-frames", type=int, default=40, help="synthetic conditioning video length at 20 fps")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    t0 = time.perf_counter()
    server = PonderPounceRoboMMEServer(a.ckpt, device=a.device)
    print(f"loaded in {time.perf_counter() - t0:.0f}s")
    ctx = FakeCtx()
    rng = np.random.default_rng(0)
    await server.on_episode_start({"task": {"task_description": "pick up the red cube twice"}}, ctx)
    demo = [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(a.demo_frames)]
    per_step = []
    for k in range(a.steps):
        obs = {
            "images": {
                "agentview": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
                "wrist": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
            },
            "states": rng.standard_normal(8).astype(np.float32),
            "task_description": "pick up the red cube twice",
        }
        if k == 0:
            obs["video_history"] = demo
        t = time.perf_counter()
        await server.on_observation(obs, ctx)
        per_step.append(time.perf_counter() - t)
    ep = server._episodes["smoke"]
    print(
        f"steps={a.steps} ponder_fires={ep.n_s2_fires} pounce_fires={ep.n_s1_fires} action_shape={ctx.actions[-1].shape}"
    )
    print(f"step wall ms: median={1e3 * float(np.median(per_step)):.0f} max={1e3 * max(per_step):.0f}")
    await server.on_episode_end({}, ctx)
    assert ep.n_s2_fires == 1 + (a.steps - 1) // 20, "Ponder should fire every 20 steps from the first observation"
    assert ep.n_s1_fires >= 1 and ctx.actions[-1].shape == (1, server._s1.action_dim)
    print("OK")


if __name__ == "__main__":
    asyncio.run(main())
