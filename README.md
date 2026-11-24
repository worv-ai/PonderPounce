# PonderPounce: A Pretrained MLLM as an Episode Context Engine for Robot Control

Suhwan Choi<sup>\*1,2</sup>, Jaeyoon Jung<sup>\*1</sup>, Sungkyung Kim<sup>1,2</sup>, Yunsung Lee<sup>†1</sup>, Youngjae Yu<sup>†2</sup>

<sup>1</sup>MAUM.AI &nbsp;&nbsp; <sup>2</sup>Seoul National University  
<sup>\*</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Co-corresponding authors

[![Project Page](https://img.shields.io/badge/🌐_Project_Page-PonderPounce-5868F9?style=flat-square)](https://worv-ai.github.io/ponderpounce/)
[![arXiv](https://img.shields.io/badge/📄_arXiv-2608.24115-B31B1B?style=flat-square)](https://arxiv.org/abs/2608.24115)
[![Code](https://img.shields.io/badge/💻_Code-GitHub-1F2430?style=flat-square)](https://github.com/worv-ai/PonderPounce)
[![Checkpoint](https://img.shields.io/badge/🤗_Checkpoint-PonderPounce--9B--RoboMME-FFD21E?style=flat-square)](https://huggingface.co/worv-ai/PonderPounce-9B-RoboMME)
[![Checkpoint](https://img.shields.io/badge/🤗_Checkpoint-PonderPounce--WM--9B--RoboMME-FFD21E?style=flat-square)](https://huggingface.co/worv-ai/PonderPounce-WM-9B-RoboMME)
[![Status](https://img.shields.io/badge/Status-Under_Review-FFBD17?style=flat-square)](https://worv-ai.github.io/ponderpounce/)

<p align="center">
  <a href="https://worv-ai.github.io/ponderpounce/">
    <img src="https://worv-ai.github.io/ponderpounce/static/images/PP-Title.png" alt="PonderPounce overview" width="100%">
  </a>
</p>

> **Ponder** remembers and reasons over the episode. **Pounce** acts quickly from the latest cognition.

> [!NOTE]
> **PonderPounce-WM is out.** [`worv-ai/PonderPounce-WM-9B-RoboMME`](https://huggingface.co/worv-ai/PonderPounce-WM-9B-RoboMME) adds a future-observation objective to Ponder and reaches **67.42%** on RoboMME (60.83% for the base model on the same data). It runs with the same code: see [PonderPounce-WM](#ponderpounce-wm).

## News

- **[2026/11/24]** **PonderPounce-WM** checkpoint is out: [`worv-ai/PonderPounce-WM-9B-RoboMME`](https://huggingface.co/worv-ai/PonderPounce-WM-9B-RoboMME), PonderPounce trained with an auxiliary future-observation objective on Ponder (**67.42%** on RoboMME, against 60.83% for the base model on the same data). The same server and scripts run it (see [PonderPounce-WM](#ponderpounce-wm)).
- **[2026/09/29]** Inference code and the RoboMME checkpoint are out: [`worv-ai/PonderPounce-9B-RoboMME`](https://huggingface.co/worv-ai/PonderPounce-9B-RoboMME) on Hugging Face, with a sequential model server and scripts to run the full RoboMME-16 evaluation (see [Evaluation on RoboMME](#evaluation-on-robomme)).
- **[2026/08/25]** We released the [paper](https://arxiv.org/abs/2608.24115) and [project page](https://worv-ai.github.io/ponderpounce/).

## Overview

Robotic manipulation often depends on information that is no longer visible in the current observation: an occluded object, an earlier event, the identity of a referent, or a demonstrated procedure. Existing memory-dependent policies typically introduce task-specific samplers, stores, retrievers, or compressors.

**PonderPounce instead reuses a pretrained multimodal large language model's native causal context as robot memory.**

- **Ponder (System 2)** retains observations, demonstrations, and prior cognition in an append-only MLLM context. It reasons over the episode and produces continuous cognition at sparse queries.
- **Pounce (System 1)** is a fast vision-language-action controller conditioned on the current observation, instruction, proprioception, and the latest cognition.
- **Asynchronous interface** routes only one continuous cognition token and its age from Ponder to Pounce, keeping long-context processing off the action path.
- **End-to-end training** jointly adapts both pretrained systems without a purpose-built memory module or separate bridge pretraining.

## Architecture

<p align="center">
  <img src="https://worv-ai.github.io/ponderpounce/static/images/PP-Arch.png" alt="PonderPounce architecture" width="100%">
</p>

Ponder accumulates the instruction, demonstrations, and episode observations in its native context. Transition tokens gate optional demonstration reasoning and subgoal text, while carrier states form continuous cognition. Pounce combines the newest cognition with the current observation to predict an action chunk. The two systems run on decoupled clocks, enabling **20 Hz action playback** while the context engine updates asynchronously.

## Results

### RoboMME

PonderPounce substantially improves memory-dependent robot control while keeping the Pounce architecture and interface fixed.

| Method | Episode memory | 1× data | 9× data |
|:--|:--|--:|--:|
| $\pi_{0.5}$ | None | 17.93% | — |
| $\pi_{0.5}$ + past actions | Action history | 19.73% | — |
| SAM2Act+ | SAM2 bank + attention | 21.37% | — |
| MemER | Keyframe selector / tracker | 42.38% | — |
| FrameSamp+Modul | Frame tokens + modulator | 44.51% | 57.88% |
| **PonderPounce (ours)** | **Native MLLM context** | **60.83%** | **75.54%** |

Key findings:

- Scaling Ponder from a pretrained **0.8B** MLLM to **9B** improves RoboMME success from **50.04% to 60.83%** under the same controller and interface.
- Optimized serving reaches **78 ms p50** for cognition refresh and **25 ms p50** for Pounce invocation.
- With 9× training data, PonderPounce reaches **75.54%**, compared with **57.88%** for FrameSamp+Modul.

### RoboCasa-DC

The same interface transfers to demonstration-conditioned manipulation using action supervision alone.

| Method | Average success |
|:--|--:|
| Vid2Robot | 8.8% |
| UniSkill | 11.2% |
| ViVLA | 8.0% |
| SeeTraceAct | 11.6% |
| **PonderPounce (ours)** | **12.5 ± 0.9%** |
| PonderPounce, cognition disabled | 8.6 ± 0.4% |

## Memory in Action

<p align="center">
  <img src="https://worv-ai.github.io/ponderpounce/static/images/PP-Demo.png" alt="PonderPounce remembers transient target markers in RoboMME PickHighlight" width="100%">
</p>

In RoboMME **PickHighlight**, target markers appear briefly and then disappear. A current-observation policy searches and times out, whereas PonderPounce retains the earlier cue in episode context and grasps both highlighted targets.

## PonderPounce-WM

<p align="center">
  <img src="https://huggingface.co/worv-ai/PonderPounce-WM-9B-RoboMME/resolve/main/architecture.png" alt="PonderPounce-WM architecture" width="100%">
</p>

PonderPounce-WM keeps the two systems and the single-token interface, and adds a world-model objective to Ponder. At every query Ponder's carrier block holds the cognition token followed by three **future slots**, and during training a linear head on each slot regresses how the third-person view will change 0.5 s, 1 s and 2 s ahead, as the difference of pooled features from Pounce's own image encoder. The objective is auxiliary: no pixels are generated, and Pounce still receives only the cognition token, so the action path is unchanged. On RoboMME the model reaches **67.42%** (three evaluation repeats: 68.75 / 66.25 / 67.25) against 60.83% for the base checkpoint trained on the same data.

Every command in this README takes either checkpoint: replace the Hub id with `worv-ai/PonderPounce-WM-9B-RoboMME`. The future slots and their head are part of the checkpoint and are described by its `config.json`; nothing else has to be configured.

## Installation

Prerequisites:

- [uv](https://docs.astral.sh/uv/) (dependency management and `uv run`); Python 3.11 or 3.12 is installed by `uv sync` if needed.
- An NVIDIA GPU with at least 32 GB of free memory for one policy server (~27 GB of bf16 weights for both systems plus 1-2 GB of Ponder KV cache per simulator session; a 40-80 GB card serves several sessions at once) and a CUDA driver for the CUDA 13 build of PyTorch that `uv sync` installs.
- Docker, for the RoboMME simulator image (evaluation only; the container needs no GPU).
- About 30 GB of disk for the checkpoint (27 GB) plus the tokenizers / processors of the two backbones (`Qwen/Qwen3.5-9B`, `google/paligemma-3b-pt-224`), downloaded from the Hugging Face Hub on first use. The backbone weights themselves are not downloaded; every weight comes from the checkpoint.

```bash
git clone https://github.com/worv-ai/PonderPounce && cd PonderPounce
uv sync                                                # Python 3.11/3.12, CUDA GPU
uv pip install flash-linear-attention causal-conv1d    # optional Qwen3.5 kernels, see below
```

Every command below is run with `uv run` from the repository root, so nothing needs to be activated (`pip install -e .` into your own environment works too).

`flash-linear-attention` and `causal-conv1d` are the Qwen3.5 hybrid-attention kernels Ponder uses (`causal-conv1d` builds against your CUDA toolkit, or take a prebuilt wheel from its releases page for your torch/CUDA pair); without them transformers falls back to its PyTorch implementation, which is correct but several times slower.

## Checkpoints

| Model | Ponder (System 2) | Pounce (System 1) | Training data | RoboMME |
|:--|:--|:--|:--|--:|
| [worv-ai/PonderPounce-9B-RoboMME](https://huggingface.co/worv-ai/PonderPounce-9B-RoboMME) | Qwen3.5-9B | π0.5 (PaliGemma 3B + 300M action expert) | RoboMME 1× | 60.83% |
| [worv-ai/PonderPounce-WM-9B-RoboMME](https://huggingface.co/worv-ai/PonderPounce-WM-9B-RoboMME) | Qwen3.5-9B + future slots | π0.5 (PaliGemma 3B + 300M action expert) | RoboMME 1× | 67.42% |

Pass the Hub id as `checkpoint_path` and the server downloads it into the Hub cache (`HF_HOME`) on first use, or point it at a local directory. A checkpoint directory holds `config.json`, `model.safetensors` (bf16, ~27 GB, every weight of both systems) and `norm_stats.json`; only the tokenizers/processors of the two backbones (`Qwen/Qwen3.5-9B`, `google/paligemma-3b-pt-224`) are fetched from the Hugging Face Hub on first use.

## Evaluation on RoboMME

The evaluation harness is [vla-eval](https://github.com/worv-ai/vla-evaluation-harness-public) (installed as a dependency); the RoboMME simulator ships as a Docker image, so `docker` must be available on the machine:

```bash
docker pull ghcr.io/worv-ai/vla-evaluation-harness-public/robomme:latest
```

The policy runs as a model server that the harness queries at every control step. `ponderpounce/eval/robomme_server.py` is a **sequential** server: one process, one GPU, Ponder and Pounce run inline on a simulated clock that advances one control step per observation, so the fire schedule is the one the checkpoint trained with (Ponder every 1 s, Pounce every 1 s, cognition visible to Pounce 300 ms after Ponder fires, 20-action chunks played back at 20 Hz). No asynchronous machinery is needed for the benchmark, which makes the server easy to read and to adapt.

```bash
# 1. start the server
uv run python -m ponderpounce.eval.robomme_server --args.checkpoint_path worv-ai/PonderPounce-9B-RoboMME --port 8000

# 2. run the 16 tasks (50 episodes each, one simulator container)
uv run vla-eval run -c configs/robomme.yaml
uv run scripts/aggregate_results.py runs/robomme      # per-task success rates over the recording databases
```

`scripts/eval_robomme.sh --ckpt worv-ai/PonderPounce-9B-RoboMME --gpus 0,1,2,3,4,5,6,7 --sims-per-gpu 8` does the same with one server per GPU, the tasks split across the servers and several simulator containers per server (the server interleaves their sessions; the simulators of one server claim episodes from a shared recording database under `<out>/server_<n>/`, so each episode runs once), and writes the per-task table to `<out>/results.json`; the full benchmark (16 tasks × 50 episodes × up to 1300 steps) takes about 2 hours on 4 H100s with 8 simulators each, and the run above with one server and one simulator would take a day, so sharding is the practical way to run it. Server options (`--args.<name>`) are documented in the server's constructor; the defaults reproduce the paper setting.

`uv run scripts/smoke_server.py --ckpt worv-ai/PonderPounce-9B-RoboMME` drives one synthetic episode through the server without a simulator, to check an installation.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{choi2026ponderpounce,
  title={PonderPounce: A Pretrained MLLM as an Episode Context Engine for Robot Control},
  author={Choi, Suhwan and Jung, Jaeyoon and Kim, Sungkyung and Lee, Yunsung and Yu, Youngjae},
  journal={arXiv preprint arXiv:2608.24115},
  year={2026}
}

@article{jung2026ponderpounce,
  title={PonderPounce-WM: When Does Predicting the Future Help Context-Dependent Robot Control?},
  author={Jung, Jaeyoon and Choi, Suhwan and Lee, Yunsung and Yu, Youngjae},
  journal={arXiv preprint arXiv:2611.26996},
  year={2026}
}
```

## Acknowledgements

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2026-25522885, Development of a World Foundation Model for Training and Deployment of Physical AI Systems).

This project is a collaboration between [MAUM.AI](https://maum.ai/) and [Seoul National University](https://www.snu.ac.kr/). Visit [WoRV Research](https://worv-ai.github.io/) for more work on robotic foundation models.
