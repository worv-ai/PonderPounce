#!/usr/bin/env bash
# Run RoboMME-16 against a PonderPounce checkpoint on one machine.
#
#   scripts/eval_robomme.sh --ckpt <dir-or-hub-id> [--out runs/robomme] [--episodes 50] [--gpus 0,1,...] [--sims-per-gpu 4]
#
# One server per GPU (each holds the whole model, ~27 GB of weights in bf16),
# the 16 tasks split round-robin across the servers, and --sims-per-gpu
# simulator containers per server (the server interleaves their sessions; the
# simulators render on CPU with lavapipe, about one core each). Per-task success
# rates go to <out>/results.json; per-episode videos and the step recording land under
# <out>/shard_<gpu>_<sim>/.
#
# Needs: docker with the RoboMME image (see README), a venv with this package
# (`uv sync`); commands go through `uv run` unless PYTHON / VLA_EVAL are set.
set -euo pipefail

CKPT=""; OUT="runs/robomme"; EPISODES=50; MAX_STEPS=1300; GPUS="0"; BASE_PORT=8000; SIMS=4
IMAGE="ghcr.io/worv-ai/vla-evaluation-harness-public/robomme:latest"
TASKS=(BinFill PickXtimes SwingXtimes StopCube VideoUnmask VideoUnmaskSwap ButtonUnmask ButtonUnmaskSwap
       PickHighlight VideoRepick VideoPlaceButton VideoPlaceOrder MoveCube InsertPeg PatternLock RouteStick)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) CKPT="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --episodes) EPISODES="$2"; shift 2;;
    --max-steps) MAX_STEPS="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --sims-per-gpu) SIMS="$2"; shift 2;;
    --base-port) BASE_PORT="$2"; shift 2;;
    --image) IMAGE="$2"; shift 2;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \?//'; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[[ -n "$CKPT" ]] || { echo "--ckpt is required" >&2; exit 2; }
command -v docker >/dev/null || { echo "docker not found" >&2; exit 1; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"

# Interpreter + harness CLI: `uv run` from this repo by default; override with PYTHON / VLA_EVAL.
PY="${PYTHON:-uv run python}"
VLA_EVAL="${VLA_EVAL:-uv run vla-eval}"
mkdir -p "$OUT"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
N=${#GPU_LIST[@]}
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do [[ -n "$p" ]] && kill -INT "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

# 1. one server per GPU
for i in "${!GPU_LIST[@]}"; do
  port=$((BASE_PORT + i))
  CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" $PY -m ponderpounce.eval.robomme_server \
      --args.checkpoint_path "$CKPT" --args.device cuda:0 --port "$port" \
      > "$OUT/server_$i.log" 2>&1 &
  PIDS+=($!)
  echo "server $i: gpu ${GPU_LIST[$i]} port $port (log $OUT/server_$i.log)"
done
for i in "${!GPU_LIST[@]}"; do
  port=$((BASE_PORT + i))
  until curl -sf "http://localhost:$port/health" >/dev/null; do
    kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "server $i died, see $OUT/server_$i.log" >&2; exit 1; }
    sleep 5
  done
  echo "server $i healthy"
done

# 2. per server: its task subset, split over SIMS simulator containers (episode shards)
write_config() {  # <path> <port> <out_dir> <task>...
  local path=$1 port=$2 out=$3; shift 3
  {
    echo "server:"
    echo "  url: ws://localhost:$port"
    echo "  timeout: 300.0"
    echo "docker:"
    echo "  image: $IMAGE"
    echo "  env: [ROBOMME_USE_LAVAPIPE=1, NVIDIA_VISIBLE_DEVICES=void, CUDA_VISIBLE_DEVICES=]"
    echo "  user: host"
    echo "render: cpu"
    echo "output_dir: $out"
    echo "benchmarks:"
    echo "  - benchmark: vla_eval.benchmarks.robomme.benchmark:RoboMMEBenchmark"
    echo "    subname: ponderpounce"
    echo "    episodes_per_task: $EPISODES"
    echo "    params:"
    echo "      action_space: joint_angle"
    echo "      max_steps: $MAX_STEPS"
    echo "      tasks:"
    for t in "$@"; do echo "        - $t"; done
    echo "    recording:"
    echo "      output_dir: $out/episodes"
    echo "      record_step: true"
    echo "      record_video: true"
  } > "$path"
}

SHARD_PIDS=()
for i in "${!GPU_LIST[@]}"; do
  port=$((BASE_PORT + i))
  shard_tasks=()
  for t in "${!TASKS[@]}"; do [[ $((t % N)) -eq $i ]] && shard_tasks+=("${TASKS[$t]}"); done
  for ((j = 0; j < SIMS; j++)); do
    shard_out="$OUT/shard_${i}_${j}"
    mkdir -p "$shard_out"
    write_config "$shard_out/config.yaml" "$port" "$shard_out" "${shard_tasks[@]}"
    $VLA_EVAL run -c "$shard_out/config.yaml" --render cpu --yes --shard-id "$j" --num-shards "$SIMS" > "$shard_out/run.log" 2>&1 &
    SHARD_PIDS+=($!)
  done
  echo "server $i: ${#shard_tasks[@]} tasks over $SIMS simulators -> $OUT/shard_${i}_*"
done
fail=0
for p in "${SHARD_PIDS[@]}"; do wait "$p" || fail=1; done

# 3. per-task success rates over every shard
$PY "$(dirname "${BASH_SOURCE[0]}")/aggregate_results.py" "$OUT" --json "$OUT/results.json" || true
echo "done: $OUT (exit $fail)"
exit $fail
