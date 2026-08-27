# PonderPounce: A Pretrained MLLM as an Episode Context Engine for Robot Control

Suhwan Choi<sup>\*1,2</sup>, Jaeyoon Jung<sup>\*1</sup>, Sungkyung Kim<sup>1,2</sup>, Yunsung Lee<sup>†1</sup>, Youngjae Yu<sup>†2</sup>

<sup>1</sup>MAUM.AI &nbsp;&nbsp; <sup>2</sup>Seoul National University  
<sup>\*</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Co-corresponding authors

[![Project Page](https://img.shields.io/badge/🌐_Project_Page-PonderPounce-5868F9?style=flat-square)](https://worv-ai.github.io/ponderpounce/)
[![arXiv](https://img.shields.io/badge/📄_arXiv-2608.24115-B31B1B?style=flat-square)](https://arxiv.org/abs/2608.24115)
[![Code](https://img.shields.io/badge/💻_Code-Coming_Soon-1F2430?style=flat-square)](https://github.com/worv-ai/PonderPounce)
[![Status](https://img.shields.io/badge/Status-Under_Review-FFBD17?style=flat-square)](https://worv-ai.github.io/ponderpounce/)

<p align="center">
  <a href="https://worv-ai.github.io/ponderpounce/">
    <img src="https://worv-ai.github.io/ponderpounce/static/images/PP-Title.png" alt="PonderPounce overview" width="100%">
  </a>
</p>

> **Ponder** remembers and reasons over the episode. **Pounce** acts quickly from the latest cognition.

## News

- **[2026/08/25]** We released the [paper](https://arxiv.org/abs/2608.24115) and [project page](https://worv-ai.github.io/ponderpounce/). Code and checkpoints are coming soon.

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

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{choi2026ponderpounce,
  title   = {PonderPounce: A Pretrained MLLM as an Episode Context Engine for Robot Control},
  author  = {Choi, Suhwan and Jung, Jaeyoon and Kim, Sungkyung and Lee, Yunsung and Yu, Youngjae},
  journal = {arXiv preprint arXiv:2608.24115},
  year    = {2026}
}
```

## Acknowledgements

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2026-25522885, Development of a World Foundation Model for Training and Deployment of Physical AI Systems).

This project is a collaboration between [MAUM.AI](https://maum.ai/) and [Seoul National University](https://www.snu.ac.kr/). Visit [WoRV Research](https://worv-ai.github.io/) for more work on robotic foundation models.
