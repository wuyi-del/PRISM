# PRISM: Prioritized Instructive Signal Modulation for On-Policy Distillation

[![arXiv](https://img.shields.io/badge/arXiv-PRISM-red.svg)](https://arxiv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **When to Trust the Teacher?** PRISM introduces a three-level modulation framework that dynamically adjusts how much a student model should trust its teacher during on-policy distillation.

## Overview

On-Policy Distillation (OPD) transfers reasoning capabilities from a teacher to a student by using the student's own rollouts. However, **not all teacher signals are equally trustworthy** — blindly following the teacher can propagate errors from incorrect rollouts or over-constrain the student on trajectories it already handles well.

**PRISM** addresses this with three complementary levels of teacher-trust modulation:

| Level | Component | Mechanism |
|-------|-----------|-----------|
| **Level 1** | **Outcome Gate** | Trajectory-level binary gating: full teacher weight (λ_w=1.0) for incorrect rollouts, reduced weight (λ_r=0.2) for correct ones |
| **Level 2** | **NLL-based Quality Weighting** | Differentiates incorrect trajectories by teacher perplexity — high-PPL (confusing) errors get more correction, low-PPL (near-correct) errors get less |
| **Level 3** | **Token Gate** | Token-level entropy-based gating: masks teacher signal on tokens where the student is already confident |

PRISM is built on top of the [veRL](https://github.com/volcengine/verl) framework and extends the G-OPD/ExOPD distillation paradigm.

## Key Results

### Strong-to-Weak Distillation (30B → 4B)

| Method | AIME24 | AIME25 | HMMT | MATH-500 | AMC23 |
|--------|--------|--------|------|----------|-------|
| GRPO (no teacher) | 46.88 | 45.31 | 35.00 | 78.60 | 69.17 |
| OPD | 56.77 | 51.98 | 42.50 | 79.80 | 72.50 |
| ExOPD | 56.25 | 51.04 | 42.50 | 79.60 | 72.50 |
| **PRISM** | **60.63** | **54.69** | **43.75** | **80.20** | **72.50** |

*All results reported as Mean@32 (32 samples per problem). PRISM achieves consistent gains on the hardest benchmarks (AIME24: +3.86, AIME25: +2.72 over the strongest non-PRISM baseline).*

### Same-Size Distillation (4B → 4B)

| Method | AIME24 | AIME25 | HMMT | MATH-500 | AMC23 |
|--------|--------|--------|------|----------|-------|
| OPD | 60.42 | 53.33 | 42.50 | 80.20 | 72.50 |
| ExOPD | 60.63 | 52.92 | 42.50 | 80.40 | 72.50 |
| **PRISM** | **62.71** | **56.67** | **45.00** | **80.60** | **72.50** |

## Installation

Our code is based on [veRL](https://github.com/volcengine/verl) (v0.6.1). To set up the environment:

```bash
conda create -n prism python==3.10
conda activate prism
cd verl/
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math-verify
```

## Quick Start

### 1. Download Training Data

Training data is available on HuggingFace:
```bash
git clone https://huggingface.co/datasets/Keven16/G-OPD-Training-Data ../G-OPD-Training-Data
```

### 2. Run PRISM (30B → 4B Strong-to-Weak)

```bash
cd PRISM
bash examples/run_prism_30b_to_4b.sh
```

Modify the model paths in the script to point to your local models:
- `STUDENT_MODEL`: Path to Qwen3-4B-Non-Thinking
- `BASE_MODEL`: Path to Qwen3-4B (base)
- `TEACHER_MODEL`: Path to Qwen3-30B-A3B-Instruct-2507

### 3. Run PRISM (4B → 4B Same-Size)

```bash
cd PRISM
bash examples/run_prism_4b_to_4b.sh
```

### 4. Run OPD Baseline

```bash
cd PRISM
bash examples/run_opd_baseline.sh
```

## PRISM Configuration Parameters

PRISM is activated by adding the following Hydra overrides to the standard OPD training command:

```bash
# Core PRISM parameters
actor_rollout_ref.actor.policy_loss.rgopd_alpha=1.0          # Fusion strength (0-1)
actor_rollout_ref.actor.policy_loss.rgopd_lambda_wrong=1.0   # Weight for incorrect trajectories
actor_rollout_ref.actor.policy_loss.rgopd_lambda_right=0.2   # Weight for correct trajectories
actor_rollout_ref.actor.policy_loss.rgopd_ppl_weighted_error=True  # Enable NLL-based weighting
actor_rollout_ref.actor.policy_loss.rgopd_ppl_tau=5.0        # Temperature for PPL softmax
actor_rollout_ref.actor.policy_loss.disable_token_gate=True   # Disable token gate (optional)
```

### Advanced Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rgopd_dynamic_lambda_right` | `False` | Auto-adjust λ_r based on batch accuracy |
| `rgopd_rank_based_ppl` | `False` | Use rank-based PPL (eliminates cross-problem incomparability) |
| `rgopd_ppl_filter_sigma` | `0.0` | Sigma threshold for filtering extreme degenerate trajectories |
| `rgopd_tau` | `1.0` | Temperature for token-level entropy gate |
| `rgopd_teacher_entropy` | `0.1` | Teacher entropy threshold for token gate |

## Evaluation

### Math Reasoning

```bash
cd math_eval/
bash run_eval_math.sh
```

### Code Generation

```bash
# EvalPlus
CUDA_VISIBLE_DEVICES=0 bash code_eval/scripts/run_evalplus.sh humaneval <MODEL_PATH> 0 1.0 1.0 4

# LiveCodeBench
bash code_eval/scripts/run_lcb_gen.sh --model <MODEL_NAME> --local_model_path <MODEL_PATH>
```

## Repository Structure

```
PRISM/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── download_eval_data.py        # Evaluation data download script
├── examples/                    # Training scripts
│   ├── run_prism_30b_to_4b.sh   # PRISM strong-to-weak (30B→4B)
│   ├── run_prism_4b_to_4b.sh    # PRISM same-size (4B→4B)
│   └── run_opd_baseline.sh      # OPD baseline
├── data/                        # Math evaluation data
│   ├── aime24/
│   ├── aime25/
│   ├── amc23/
│   ├── hmmt25_feb/
│   ├── hmmt25_nov/
│   └── math500/
├── math_eval/                   # Math evaluation code
├── code_eval/                   # Code evaluation code (EvalPlus + LiveCodeBench)
└── verl/                        # Training framework (modified verl v0.6.1)
    └── verl/
        ├── trainer/
        │   ├── main_ppo.py              # Training entry point
        │   └── ppo/
        │       ├── ray_trainer.py       # Distributed training orchestrator
        │       ├── core_algos.py        # Core PPO algorithms
        │       └── rollout_corr_helper.py
        └── workers/
            ├── actor/
            │   └── dp_actor.py          # ★ PRISM core logic (RGOPD-v2)
            └── config/
                └── actor.py             # PolicyLossConfig
```

## Core Algorithm

PRISM's core logic is implemented in `verl/verl/workers/actor/dp_actor.py` (labeled as `RGOPD-v2: ExOPD-RL adaptive fusion`). The key computation flow:

1. **Compute ExOPD advantage**: `exopd_adv = -(log_prob_actor - log_prob_ref)` (reverse KL)
2. **Outcome Gate**: `g_y = λ_w * (1 - reward) + λ_r * reward` (trajectory-level)
3. **PPL-Weighted Error**: Replace uniform `λ_w` with softmax(-PPL / τ) for wrong trajectories
4. **Token Gate**: `s_t = sigmoid((H_student - H_teacher) / τ)` (token-level)
5. **Fusion**: `w_bar = clip(α * g_y * s_t / λ_w, 0, 1)`
6. **Scale Alignment**: Match ExOPD magnitude to GRPO magnitude
7. **Final Advantage**: `A = (1 - w_bar) * A_grpo + w_bar * A_exopd_scaled`

## Citation

If you find PRISM helpful, please cite our work:

```bibtex
@article{prism2026,
  title={PRISM: When to Trust the Teacher? Prioritized Instructive Signal Modulation for On-Policy Distillation},
  author={PRISM Authors},
  journal={arXiv preprint},
  year={2026}
}
```

## Acknowledgments

Our training code is based on [veRL](https://github.com/volcengine/verl). Our evaluation code is based on [Absolute-Zero-Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner), which builds upon [EvalPlus](https://github.com/evalplus/evalplus) and [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details. The vendored verl framework retains its original Apache 2.0 license.
