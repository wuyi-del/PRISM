#!/bin/bash
# ============================================================
#  ExOPD 30B→4B 强到弱蒸馏（G-OPD 论文 Table 3 复现）
#
#  配置：纯 ExOPD (α=0, 无 GRPO 分支)
#    - Teacher: Qwen3-30B-A3B-Instruct-2507 (30B, 已训练)
#    - Student: Qwen3-4B-Non-Thinking (4B 基座)
#    - T=1.0, λ=1.25 (rgopd_lambda_wrong/right = 1.25)
#    - seed=42, total_epochs=2 (~110 steps ≈ 100 steps)
#
#  对标 G-OPD 论文 Table 3 (Exp 8c):
#    AIME24 mean@32=58.7%, AIME25 mean@32=50.8%
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
        # ===== 模型配置 (30B → 4B, 强到弱) =====
        actor_rollout_ref.model.path=/data/local_disk3/Qwen3-4B-Non-Thinking \
        +actor_rollout_ref.model.base_model_path=/data/local_disk3/Qwen3-4B \
        +actor_rollout_ref.ref.model.path=/data/local_disk3/Qwen3-30B-A3B-Instruct-2507 \
        +actor_rollout_ref.ref.model.base_model_path=/data/local_disk3/Qwen3-30B-A3B \
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
        # ===== ExOPD 核心配置 (α=0 → 纯 ExOPD, 关闭 GRPO 分支) =====
        actor_rollout_ref.actor.policy_loss.rgopd_alpha=0.0 \
        actor_rollout_ref.actor.policy_loss.rgopd_lambda_wrong=1.25 \
        actor_rollout_ref.actor.policy_loss.rgopd_lambda_right=1.25 \
        # 纯 ExOPD 不需要 PPL 加权 / token gate / dynamic tau
        actor_rollout_ref.actor.policy_loss.rgopd_ppl_weighted_error=False \
        actor_rollout_ref.actor.policy_loss.disable_token_gate=True \
        # ===== 训练控制 =====
        trainer.critic_warmup=0 \
        trainer.val_before_train=True \
        trainer.logger='["console","wandb"]' \
        trainer.log_val_generations=10 \
        trainer.project_name='on-policy-distillation' \
        trainer.experiment_name='ExOPD_30Bto4B_T1.0_lambda1.25' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=20 \
        trainer.test_freq=10 \
        trainer.default_local_dir=/G-OPD-checkpoints/ExOPD-30Bto4B \
        trainer.total_epochs=2 $@
