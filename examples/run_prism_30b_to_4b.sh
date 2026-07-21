#!/bin/bash
# ============================================================
#  PRISM: Prioritized Instructive Signal Modulation
#  Strong-to-Weak Distillation (30B → 4B)
#
#  Method:
#    - GRPO + PRISM Three-Level Modulation:
#      Level 1: Outcome Gate (trajectory-level binary gating)
#      Level 2: NLL-based Quality Weighting (teacher PPL softmax)
#      Level 3: Token Gate (entropy-based token-level gating)
#    - Scale Alignment (GRPO ↔ ExOPD magnitude matching)
#
#  Configuration:
#    - Teacher: Qwen3-30B-A3B-Instruct-2507
#    - Student: Qwen3-4B (Non-Thinking mode)
#    - Data: DeepMath-103K (level6 filtered)
#    - Training: 60 steps, batch=1024, lr=1e-5
#    - Evaluation: AIME24 + AIME25, Mean@32
# ============================================================
set -x
export PYTHONUNBUFFERED=1

export WANDB_API_KEY=""
export WANDB_MODE=online
export USED_MODEL="no_api"

# --- Paths (modify these) ---
STUDENT_MODEL=/path/to/Qwen3-4B-Non-Thinking
BASE_MODEL=/path/to/Qwen3-4B
TEACHER_MODEL=/path/to/Qwen3-30B-A3B-Instruct-2507
TRAIN_DATA=../G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet
CHECKPOINT_DIR=/path/to/checkpoints/PRISM-30Bto4B

aime24_test_path=../G-OPD-Training-Data/AIME2024/test.parquet
aime25_test_path=../G-OPD-Training-Data/AIME2025/test.parquet
test_files="['$aime24_test_path', '$aime25_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=5.0 \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.bypass_mode=false \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    data.seed=42 \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$STUDENT_MODEL" \
    +actor_rollout_ref.model.base_model_path="$BASE_MODEL" \
    +actor_rollout_ref.ref.model.path="$TEACHER_MODEL" \
    +actor_rollout_ref.ref.model.base_model_path="$BASE_MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
    actor_rollout_ref.actor.policy_loss.rgopd_alpha=1.0 \
    actor_rollout_ref.actor.policy_loss.rgopd_lambda_wrong=1.0 \
    actor_rollout_ref.actor.policy_loss.rgopd_lambda_right=0.2 \
    actor_rollout_ref.actor.policy_loss.rgopd_ppl_weighted_error=True \
    actor_rollout_ref.actor.policy_loss.rgopd_ppl_tau=5.0 \
    actor_rollout_ref.actor.policy_loss.disable_token_gate=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger='["console","wandb"]' \
    trainer.log_val_generations=10 \
    trainer.project_name='PRISM' \
    trainer.experiment_name='PRISM_30Bto4B_DeepMath_seed42' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.test_freq=10 \
    trainer.total_epochs=60 \
    trainer.total_training_steps=60 $@
