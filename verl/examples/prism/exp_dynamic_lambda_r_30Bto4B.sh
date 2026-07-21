#!/bin/bash
# ============================================================
#  R12: Dynamic Lambda Right (Adaptive Soft Gate)
#  基于 R6 Outcome Only，将固定 λr=0.2 改为根据 batch 正确率自适应调整
#  
#  核心改动：rgopd_dynamic_lambda_right=True
#  物理含义：
#    acc=0.50 → λr≈0.40（模型还在学，多听 teacher）
#    acc=0.75 → λr≈0.10（有进步了，弱化 teacher）
#    acc=0.85 → λr≈0.02（很自信，几乎不听）
#    acc=0.90 → λr≈0.08（高度自信，极弱 teacher）
#  
#  其余配置与 R6 完全一致（30B→4B, T=1.0, token gate 关, 60步, from scratch）
# ============================================================
set -x
export PYTHONUNBUFFERED=1

export WANDB_API_KEY=""
export WANDB_MODE=online
export USED_MODEL="no_api"

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
        data.train_files=../G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet \
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
        # ===== 模型配置 =====
        actor_rollout_ref.model.path=/path/to/Qwen3-4B-Non-Thinking \
        +actor_rollout_ref.model.base_model_path=/path/to/Qwen3-4B \
        +actor_rollout_ref.ref.model.path=/path/to/Qwen3-30B-A3B-Instruct-2507 \
        +actor_rollout_ref.ref.model.base_model_path=/path/to/Qwen3-4B \
        # ===== 优化器 =====
        actor_rollout_ref.actor.optim.lr=1e-5 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
        actor_rollout_ref.model.use_remove_padding=True \
        # ===== PPO / GRPO 配置 =====
        actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
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
        # ===== Rollout (vLLM) =====
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
        # ===== Reward =====
        algorithm.use_kl_in_reward=False \
        reward_model.reward_manager=naive \
        # ===== RGOPD-v2 核心参数 =====
        actor_rollout_ref.actor.policy_loss.rgopd_alpha=1.0 \
        actor_rollout_ref.actor.policy_loss.rgopd_lambda_wrong=1.0 \
        # ↓↓↓ 唯一差异：Dynamic λ_r 替代固定 λr=0.2 ↓↓↓
        actor_rollout_ref.actor.policy_loss.rgopd_dynamic_lambda_right=True \
        actor_rollout_ref.actor.policy_loss.rgopd_tau=1.0 \
        # Token gate 关闭（与R6一致，已证实有害）
        actor_rollout_ref.actor.policy_loss.disable_token_gate=True \
        # ===== 训练控制 =====
        trainer.critic_warmup=0 \
        trainer.val_before_train=True \
        trainer.logger='["console","wandb"]' \
        trainer.log_val_generations=10 \
        trainer.project_name='on-policy-distillation' \
        trainer.experiment_name='dynamic_lambda_r_30Bto4B_OutcomeOnly' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=50 \
        trainer.default_local_dir=/G-OPD-checkpoints/DynamicLambdaR_30B-to-4B \
        trainer.test_freq=10 \
        trainer.total_epochs=60 $@
