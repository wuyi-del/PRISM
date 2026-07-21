#!/bin/bash
# ============================================================
#  R8: P1-5b — S2 4B→4B 同规模蒸馏实验
#
#  目的：验证 S2 Dynamic Tau 在 teacher-student 同规模时是否有效
#        (同规模蒸馏是常见场景, 如 RL 模型蒸馏到原始基座)
#
#  配置：S2 完全相同参数, Teacher=Qwen3-4B-RL-Math (已训练)
#    - Teacher: Qwen3-4B-RL-Math (30B→4B 的 RL 产物)
#    - Student: Qwen3-4B-Non-Thinking (原始基座)
#    - ppl_weighted_error=True, τ=5.0, λw=1.0, λr=0.2
#    - disable_token_gate=True
#    - T=1.0, seed=42, 60步
#
#  预期：S2 在同规模下仍有效, 证明方法不依赖教师优势
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
        # ===== 模型配置 (4B → 4B, 同规模) =====
        actor_rollout_ref.model.path=/path/to/Qwen3-4B-Non-Thinking \
        +actor_rollout_ref.model.base_model_path=/path/to/Qwen3-4B \
        +actor_rollout_ref.ref.model.path=/path/to/Qwen3-4B-RL-Math \
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
        # ===== RGOPD-v2 核心参数 (S2 Dynamic Tau) =====
        actor_rollout_ref.actor.policy_loss.rgopd_alpha=1.0 \
        actor_rollout_ref.actor.policy_loss.rgopd_lambda_wrong=1.0 \
        actor_rollout_ref.actor.policy_loss.rgopd_lambda_right=0.2 \
        # S1/S2 核心: PPL 加权错误轨迹
        actor_rollout_ref.actor.policy_loss.rgopd_ppl_weighted_error=True \
        # S2: 固定 τ=5.0
        actor_rollout_ref.actor.policy_loss.rgopd_ppl_tau=5.0 \
        # Token gate 关闭
        actor_rollout_ref.actor.policy_loss.disable_token_gate=True \
        # ===== 训练控制 =====
        trainer.critic_warmup=0 \
        trainer.val_before_train=True \
        trainer.logger='["console","wandb"]' \
        trainer.log_val_generations=10 \
        trainer.project_name='on-policy-distillation' \
        trainer.experiment_name='S2_DynamicTau_4Bto4B_seed42_samescale' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=20 \
        trainer.test_freq=10 \
        trainer.default_local_dir=/G-OPD-checkpoints/S2-DynamicTau-4Bto4B-R8 \
        trainer.total_epochs=60 $@
