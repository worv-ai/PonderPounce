#!/usr/bin/env python3
"""Per-task success rates over the recording databases of one or more shards.

python scripts/aggregate_results.py runs/robomme            # every recording-*.sqlite below the directory
python scripts/aggregate_results.py runs/robomme --json runs/robomme/results.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory searched recursively for recording-*.sqlite")
    ap.add_argument("--json", default=None, help="also write the table here")
    a = ap.parse_args()
    dbs = sorted(Path(a.root).rglob("recording-*.sqlite"))
    if not dbs:
        raise SystemExit(f"no recording-*.sqlite under {a.root}")
    per_task: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "fail": 0, "error": 0})
    for db in dbs:
        uri = f"file:{db}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            for task, status in conn.execute("SELECT task_name, status FROM episode_results"):
                per_task[task][status if status in ("success", "fail", "error") else "error"] += 1
    rows = []
    for task in sorted(per_task):
        c = per_task[task]
        n = c["success"] + c["fail"] + c["error"]
        rows.append(
            {
                "task": task,
                "episodes": n,
                "success": c["success"],
                "error": c["error"],
                "success_rate": 100.0 * c["success"] / n if n else 0.0,
            }
        )
    total = sum(r["episodes"] for r in rows)
    succ = sum(r["success"] for r in rows)
    mean_task = sum(r["success_rate"] for r in rows) / len(rows) if rows else 0.0
    width = max(len(r["task"]) for r in rows)
    print(f"{'task':<{width}}  episodes  success  error  SR%")
    for r in rows:
        print(
            f"{r['task']:<{width}}  {r['episodes']:>8}  {r['success']:>7}  {r['error']:>5}  {r['success_rate']:5.1f}"
        )
    print(
        f"{'MEAN over tasks':<{width}}  {total:>8}  {succ:>7}  {'':>5}  {mean_task:5.2f}   (episode-weighted {100.0 * succ / total if total else 0.0:.2f})"
    )
    if a.json:
        Path(a.json).write_text(
            json.dumps(
                {
                    "tasks": rows,
                    "mean_task_success_rate": mean_task,
                    "episodes": total,
                    "databases": [str(d) for d in dbs],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
