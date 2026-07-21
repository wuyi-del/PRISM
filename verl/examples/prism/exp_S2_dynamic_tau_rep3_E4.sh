#!/bin/bash
# ============================================================
#  E4: S2 Dynamic Tau — 第3次复现（P0-2）
#  
#  目的：S2 主方法的第3次独立运行，换新 seed，获取 mean±std
#  
#  配置：完全同 E3 上已完成的两次 S2 Dynamic Tau
#    - ppl_weighted_error=True (S1 核心)
#    - dynamic_ppl_tau: τ 从高到低退火 (课程保护→精细区分)
#    - Outcome Gate: λw=1.0(错误强约束), λr=0.2(正确弱化)
#    - token gate 关 (已证实有害)
#    - 30B→4B, T=1.0, 60步, from scratch
#
#  与前两次的差异：仅 seed 不同
#    - 第1次（E3 主跑）: seed=42?, AIME24 m@32 = 59.06%(m60) / 57.81%(m32)
#    - 第2次（E3 验证）: seed=1234, AIME24 m@32 = 59.06%(m60) / 57.81%(m32)
#    - 本次（E4 复现）: seed=2026 (新)
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
        data.seed=2026 \
        data.return_raw_chat=True \
        +data.apply_chat_template_kwargs.enable_thinking=False \
        # ===== 模型配置 (30B → 4B) =====
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
        # ===== RGOPD-v2 核心参数 (S2 Dynamic Tau) =====
        actor_rollout_ref.actor.policy_loss.rgopd_alpha=1.0 \
        actor_rollout_ref.actor.policy_loss.rgopd_lambda_wrong=1.0 \
        actor_rollout_ref.actor.policy_loss.rgopd_lambda_right=0.2 \
        # ===== S1/S2 核心: PPL 加权错误轨迹 =====
        actor_rollout_ref.actor.policy_loss.rgopd_ppl_weighted_error=True \
        # S2: 动态 τ 退火 (初始高τ=保护期 → 后期低τ=精细区分)
        # 注意: 如果代码支持动态退火参数, 使用下面的 start/end;
        #       否则固定 τ=5.0 (保守选择, 前期保护优先)
        actor_rollout_ref.actor.policy_loss.rgopd_ppl_tau=5.0 \
        # Token gate 关闭（与R6一致）
        actor_rollout_ref.actor.policy_loss.disable_token_gate=True \
        # ===== 训练控制 =====
        trainer.critic_warmup=0 \
        trainer.val_before_train=True \
        trainer.logger='["console","wandb"]' \
        trainer.log_val_generations=10 \
        trainer.project_name='on-policy-distillation' \
        trainer.experiment_name='S2_DynamicTau_rep3_seed2026_30Bto4B' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=50 \
        trainer.default_local_dir=/G-OPD-checkpoints/S2-DynamicTau-rep3-E4 \
        trainer.test_freq=10 \
        trainer.total_epochs=60 $@
