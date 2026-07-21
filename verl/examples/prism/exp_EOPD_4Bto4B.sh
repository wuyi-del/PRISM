#!/bin/bash
# ============================================================
#  Exp 1e: EOPD (Entropy-Aware OPD) 4B→4B
#
#  论文：Entropy-Aware On-Policy Distillation (2603.07079)
#  核心改动：高熵(teacher不确定的) token → 不学 teacher → mask 掉 teacher signal
#
#  我们的实现：在 dp_actor.py 的 exopd_adv 上乘以低熵 mask
#    - eopd_tau=0.8 (与原论文一致)
#    - 高熵 token (H > 0.8): teacher signal 置 0 → 不惩罚学生偏差
#    - 低熵 token (H ≤ 0.8): 正常用 reverse KL
#
#  公平对比配置（与 ExOPD Exp 1d 完全一致，除了 eopd_enabled=True）：
#    - Teacher: Qwen3-4B-RL-Math (同 ExOPD)
#    - Student: Qwen3-4B-Non-Thinking (同 ExOPD)
#    - T=1.0, λ=1.25, Batch=1024, LR=1e-5, 50 steps
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
        actor_rollout_ref.model.path=/data/local_disk3/Qwen3-4B-Non-Thinking \
        +actor_rollout_ref.model.base_model_path=/data/local_disk3/Qwen3-4B \
        +actor_rollout_ref.ref.model.path=/data/local_disk3/Qwen3-4B-RL-Math \
        +actor_rollout_ref.ref.model.base_model_path=/data/local_disk3/Qwen3-4B \
        actor_rollout_ref.actor.optim.lr=1e-5 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
        actor_rollout_ref.actor.policy_loss.lambda_vals=1.25 \
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
        actor_rollout_ref.actor.policy_loss.rgopd_alpha=0.0 \
        +actor_rollout_ref.actor.policy_loss.eopd_enabled=True \
        +actor_rollout_ref.actor.policy_loss.eopd_tau=0.8 \
        trainer.critic_warmup=0 \
        trainer.val_before_train=True \
        trainer.logger='["console","wandb"]' \
        trainer.log_val_generations=10 \
        trainer.project_name='on-policy-distillation' \
        trainer.experiment_name='EOPD_4Bto4B_T1.0_lambda1.25_tau0.8' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=20 \
        trainer.test_freq=10 \
        trainer.default_local_dir=/G-OPD-checkpoints/EOPD-4Bto4B \
        trainer.total_epochs=1 $@
