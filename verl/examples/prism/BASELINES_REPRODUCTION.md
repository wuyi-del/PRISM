# OPD Baseline 复现文档

> **目的**：为 RGOPD-v2 论文提供公平的 baseline 对比  
> **创建日期**：2026.4.30  
> **最后更新**：2026.4.30

---

## 一、方法全景与选择依据

当前 OPD 方向主要的改进论文如下：

| 方法 | 论文 | 核心思想 | 是否复现 | 原因 |
|------|------|---------|:--------:|------|
| **EOPD** | *Entropy-Aware On-Policy Distillation* (2603.07079) | 高熵 token → teacher 不确定 → 忽略其信号 | ✅ | 直接可比；与 RGOPD-v2 entropy gate 强相关 |
| **RLAD** | *RLAD* (Chen et al., 2025) | 信任区域约束 + 加法融合 GRPO 与 teacher 信号 | ✅ | 同样是 GRPO+OPD 融合路线，对比有意义 |
| **REOPOLD** | *Relaxed On-Policy Distillation* (2603.11137) | 三组件稳定：reward clipping + 高熵 token 采样 + 两阶段课程 | ✅ | 与 EOPD 形成方向对立（高熵 vs 低熵），叙事价值高 |
| **OPSD** | *Self-Distilled Reasoner* (2601.18734) | 单模型自蒸馏，model 既做 teacher 又做 student | ❌ | 需要 full-vocab logits；自蒸馏范式与外部 teacher 对比无意义 |

---

## 二、三方法对比一览

三种方法都是在 ExOPD（反向 KL）基础上做改进，但核心信念截然不同：

```
ExOPD 基础：exopd_adv = -(log π_student - λ · log π_teacher + (λ-1) · log π_ref)

         EOPD：去除高熵（不确定）token
               exopd_adv = exopd_adv × 𝕀[H_teacher ≤ τ]
               → "teacher 不确定的 token，不要学"

      REOPOLD：保留高熵（多样性大）token
               exopd_adv = exopd_adv × 𝕀[H_student ≥ τ_β]  (top-β 分位数)
               再加 reward 下界截断防止极端负梯度
               → "有多样性的 token，值得学；teacher 信号不要太极端"

         RLAD：放弃纯 teacher 路线，改为 GRPO + teacher 加法融合
               advantages = A_grpo + α · clip(ρ, 1/c, c) · teacher_signal
               → "同时利用 reward 信号和 teacher 信号，用 trust-region 防过度模仿"
```

**EOPD vs REOPOLD 的本质分歧**：
- EOPD 用的是 **teacher 熵**（teacher 的不确定性），高熵 → 移除信号
- REOPOLD 用的是 **student 熵**（student 的多样性），高熵 → 保留信号
- 二者关于"高熵 token 要不要学"的结论完全相反，放在同一张表格里论文价值极高

---

## 三、EOPD 复现

### 3.1 原论文方法

**论文**：*Entropy-Aware On-Policy Distillation* (arXiv 2603.07079)  
**核心**：teacher 分布熵 > τ 的 token，认为 teacher 在该 token 上不确定，将 teacher 信号置零

原论文实际实现是切换到 **forward KL + top-16 renormalize**（需要 full vocab logits），但语义等价于：
> 高熵 token → 不信任 teacher → 移除该 token 的 teacher 梯度信号

### 3.2 我们的实现（等效简化版）

由于 verl 框架只传递 per-token scalar log_prob（没有 full vocab logits），我们实现**语义等效的熵掩码版本**：

```python
# fsdp_workers.py: compute_ref_log_prob
eopd_enabled = self.config.actor.policy_loss.get("eopd_enabled", False)
output, ref_entropy = self.ref_policy.compute_log_prob(data=data, calculate_entropy=eopd_enabled)
if eopd_enabled and ref_entropy is not None:
    output = DataProto.from_dict(tensors={"ref_log_prob": output, "ref_entropy": ref_entropy})

# dp_actor.py: select_keys
if "ref_entropy" in data.batch.keys():
    select_keys.append("ref_entropy")

# dp_actor.py: advantage computation (after exopd_adv = -reverse_kl)
eopd_enabled = bool(policy_loss_cfg.get("eopd_enabled", False))
if eopd_enabled:
    eopd_tau = float(policy_loss_cfg.get("eopd_tau", 0.8))
    teacher_ent_map = model_inputs.get("ref_entropy", None)  # teacher entropy
    if teacher_ent_map is not None:
        low_ent_mask = (teacher_ent_map <= eopd_tau).float()  # 低熵 token 保留
        exopd_adv = exopd_adv * low_ent_mask
```

**与原论文的差异**：
- 原论文：高熵 token → forward KL（top-16 renormalized）
- 我们：高熵 token → teacher signal = 0（直接置零）
- 语义完全等价：都是"不让 teacher 的不确定性污染 student 更新"

### 3.3 超参选择

| 超参 | 值 | 来源 |
|------|---|------|
| `eopd_tau` | **0.8** | 论文默认值（nats，对应中等不确定性） |
| `lambda_vals` | **1.25** | 与 ExOPD 对齐（公平对比） |

> **τ=0.8 的含义**：token entropy 最大值约为 log(vocab_size) ≈ log(150k) ≈ 11.9 nats，τ=0.8 相当于过滤掉约 top-15% 最高熵的 token。

### 3.4 训练脚本

| 配置 | 脚本 |
|------|------|
| 4B→4B（同尺寸） | `exp_EOPD_4Bto4B.sh` |
| 30B→4B（强到弱） | `exp_EOPD_30Bto4B.sh` |

**关键命令行参数**：
```bash
actor_rollout_ref.actor.policy_loss.rgopd_alpha=0.0 \
+actor_rollout_ref.actor.policy_loss.eopd_enabled=True \
+actor_rollout_ref.actor.policy_loss.eopd_tau=0.8 \
actor_rollout_ref.actor.policy_loss.lambda_vals=1.25
```

---

## 四、RLAD 复现

### 4.1 原论文方法

**论文**：*RLAD: Reinforcement Learning Aware Distillation* (Chen et al., 2025)  
**核心公式**：
```
A_t = A_grpo + α · clip(ρ_t, 1/c, c) · log(π_teacher / π_ref_base)
ρ_t = π_student / π_ref = exp(log_prob_student - log_prob_ref)
```

**设计动机**：
- 纯 OPD（反向 KL）只有 teacher 信号，没有 reward 信号，不能区分 teacher 错误的情况
- 纯 GRPO 只有 reward 信号，没有方向指导
- RLAD **加法融合**：两者互补，trust-region 约束防止 student 过度偏离 ref

**clip(ρ, 1/c, c) 的作用**：
- ρ > c：student 已经超过 ref 太多 → 截断，防止过度追随 teacher
- ρ < 1/c：student 已经远低于 ref → 截断，防止退化
- c=1.5 是原论文默认值

### 4.2 我们的实现

```python
# dp_actor.py: RLAD block (after EOPD block, before RGOPD-v2 block)
rlad_enabled = bool(policy_loss_cfg.get("rlad_enabled", False))
if rlad_enabled:
    rlad_alpha = float(policy_loss_cfg.get("rlad_alpha", 1.0))
    rlad_c = float(policy_loss_cfg.get("rlad_c", 1.5))
    grpo_adv_rlad = model_inputs["advantages"]

    # Trust-region ratio: ρ_t = π_student / π_ref
    rho = torch.exp(old_log_prob - model_inputs["ref_log_prob"])
    rho_clipped = torch.clamp(rho, 1.0 / rlad_c, rlad_c)

    # Teacher advantage: log(π_teacher / π_ref_base)
    teacher_signal = model_inputs["ref_log_prob"] - model_inputs["base_ref_log_prob"]

    # Scale alignment: map teacher signal to GRPO magnitude
    resp_len = response_mask.sum(dim=-1).clamp(min=1)
    with torch.no_grad():
        g_resp = (grpo_adv_rlad * response_mask).sum(dim=-1) / resp_len
        ts_resp = (teacher_signal * rho_clipped * response_mask).sum(dim=-1) / resp_len
        g_scale = g_resp.abs().mean().clamp(min=1e-6)
        ts_scale = ts_resp.abs().mean().clamp(min=1e-6)
        scale_ratio = g_scale / ts_scale

    # Additive fusion
    advantages = grpo_adv_rlad + rlad_alpha * rho_clipped * teacher_signal * scale_ratio
```

**注意事项**：
1. `lambda_vals=1.0`（取消 ExOPD 外插，RLAD 有自己的 teacher 增强机制）
2. `teacher_signal = ref_log_prob - base_ref_log_prob`：teacher 相对于其基座的优势，表示"teacher RL 训练带来的收益"
3. RLAD 设置后，`elif not rlad_enabled: advantages = exopd_adv` 的 elif 保护防止覆盖

### 4.3 超参选择

| 超参 | 值 | 来源 |
|------|---|------|
| `rlad_alpha` | **1.0** | 论文默认值 |
| `rlad_c` | **1.5** | 论文默认值 |
| `lambda_vals` | **1.0** | 取消 ExOPD 外插（RLAD 有独立 teacher signal） |

> **关于原论文温度**：RLAD 原论文用 T=0.6，但为公平对比我们统一使用 T=1.0。RLAD 的 trust-region 机制本身就提供选择性（clip(ρ,1/c,c) 限制学习步长），不强依赖低温。

### 4.4 训练脚本

| 配置 | 脚本 |
|------|------|
| 4B→4B（同尺寸） | `exp_RLAD_4Bto4B.sh` |
| 30B→4B（强到弱） | `exp_RLAD_30Bto4B.sh` |

**关键命令行参数**：
```bash
actor_rollout_ref.actor.policy_loss.rgopd_alpha=0.0 \
actor_rollout_ref.actor.policy_loss.lambda_vals=1.0 \
+actor_rollout_ref.actor.policy_loss.rlad_enabled=True \
+actor_rollout_ref.actor.policy_loss.rlad_alpha=1.0 \
+actor_rollout_ref.actor.policy_loss.rlad_c=1.5
```

---

## 五、REOPOLD 复现

### 5.1 原论文方法

**论文**：*Scaling Reasoning Efficiently via Relaxed On-Policy Distillation* (arXiv 2603.11137)  
**核心思想**：把 RL 训练中的稳定技巧（reward clipping、entropy sampling、课程学习）移植到 on-policy 蒸馏

完整目标函数（Eq. 4）：
```
J_REOPOLD(θ) = E[ 1/Σ M_{i,t}  · Σ ρ_{i,t}(θ) · R̂_{i,t}(θ) · M_{i,t} ]
```

其中 `ρ_{i,t}` 是重要性采样权重，`R̂` 是截断后的 reward，`M` 是动态掩码。

#### 组件一：Mixture-Based Reward Clipping（下界截断）

**公式（Eq. 7）**：
```
R̂_{i,t} = max(sg(R_{i,t}), log(λ/(1-λ)))
```
- `R_{i,t} = log π_teacher / π_student`（token 级别的 teacher/student 对数似然比）
- `sg(·)` = stop-gradient（切断反向传播）
- `λ` = mixture coefficient ∈ (0,1)，控制 clipping 的松紧

**作用**：当 student 对某个 token 概率远高于 teacher 时，`R_{i,t} → -∞`，产生极端负梯度 → clipping 给一个有限下界

#### 组件二：高熵 Token 选择（Entropy-Guided Sampling）

**公式（Eq. 8）**：
```
mask = 𝕀[H_t^student ≥ τ_β]   # 保留 top-β% 高熵 token
```
- `H_t^student` = **student** 在 token t 的预测熵（注意：用 student 熵，非 teacher 熵）
- `τ_β` = batch 内 top-β 分位数（动态阈值，随训练自适应）

**作用**：student 高熵 token = student 仍在探索的位置 → 这里最值得学 teacher；  
低熵 token = student 已经确定 → teacher 信号几乎为零，不必浪费计算

⚠️ **与 EOPD 的方向对立**：
- EOPD：过滤高熵 ← teacher 高熵 token 不学（teacher 不确定）  
- REOPOLD：保留高熵 ← student 高熵 token 重点学（student 自己不确定）

#### 组件三：两阶段训练（Exploration-to-Refinement）

**Exploration Phase**（前 T_switch 步）：
```
M_{i,t} = 𝕀[R_{i,t} ≥ log(λ/(1-λ))]   # 只保留正向 reward（不学负向 token）
```
**Refinement Phase**（T_switch 步之后）：
```
M_{i,t} = 𝕀[H_t^student ≥ τ_β]         # 切换为 entropy 掩码
```

**设计动机**：早期 student 还不确定哪里该学，先做容易的（正向 reward）；  
训练稳定后用 entropy 掩码精细化，集中在 student 真正需要帮助的位置。

### 5.2 超参分析

原论文未在正文明确列出所有超参，根据论文上下文和消融实验推断：

| 超参 | 推荐值 | 推断依据 |
|------|--------|---------|
| `λ` (reward clipping) | **0.2** | 原论文 Figure 12 敏感性分析，0.2 附近最优 |
| `β` (entropy percentile) | **0.5** (top-50%) | 原论文 Figure 9：β=0.5 比 0.2 更保守但更稳定 |
| `T_switch` | **训练总步数的 50%** | 探索/精炼各半，常见课程学习策略 |
| 温度 | **0.6**（论文原始） → 公平对比用 **1.0** | 与其他 baseline 统一 |

> **关于 λ=0.2 的含义**：
> `log(λ/(1-λ)) = log(0.2/0.8) = log(0.25) ≈ -1.386`
> 即 reward 下界约为 -1.4，防止极端负梯度

### 5.3 我们的实现方案

**简化版本（推荐先跑）**：只实现组件一 + 组件二，跳过两阶段切换。  
理由：消融实验显示两阶段对最终性能提升约 0.3pt（Table 8 最后一行），边际贡献小，但实现复杂度高。

#### 代码实现

**`dp_actor.py` 改动**（在 RLAD block 之前，或并列作为独立 block）：

```python
# ===== REOPOLD: Relaxed On-Policy Distillation =====
# 组件1：reward 下界截断（防极端负梯度）
# 组件2：student 高熵 token 选择（保留 student 不确定的位置）
reopold_enabled = bool(policy_loss_cfg.get("reopold_enabled", False))
if reopold_enabled:
    reopold_lambda = float(policy_loss_cfg.get("reopold_lambda", 0.2))
    reopold_beta = float(policy_loss_cfg.get("reopold_beta", 0.5))

    # === 组件一：reward clipping (下界截断) ===
    reward_lower_bound = math.log(reopold_lambda / (1.0 - reopold_lambda))  # ≈ -1.386 for λ=0.2
    # exopd_adv = -(log π_student - λ·log π_teacher + (λ-1)·log π_ref) = R_{teacher/student}
    # 注意：exopd_adv = -reverse_kl = log(π_teacher/π_student)（当 λ=1.0 时）
    exopd_adv_clipped = torch.clamp(exopd_adv, min=reward_lower_bound)

    # === 组件二：student 高熵 token 掩码（top-β 分位数）===
    if entropy is not None:
        # entropy shape: (bsz, response_len)
        # 在 response token 内计算 top-β 分位数
        flat_ent = (entropy * response_mask).reshape(-1)
        nonzero_ent = flat_ent[flat_ent > 0]
        if nonzero_ent.numel() > 0:
            tau_beta = torch.quantile(nonzero_ent, 1.0 - reopold_beta)
        else:
            tau_beta = torch.tensor(0.0, device=entropy.device)
        high_ent_mask = (entropy >= tau_beta).float() * response_mask
    else:
        high_ent_mask = response_mask.float()

    exopd_adv = exopd_adv_clipped * high_ent_mask

    # monitoring
    mask_sum_re = response_mask.sum().clamp(min=1)
    micro_batch_metrics["reopold/reward_lower_bound"] = reward_lower_bound
    if entropy is not None:
        micro_batch_metrics["reopold/tau_beta"] = tau_beta.detach().item()
        micro_batch_metrics["reopold/high_ent_token_ratio"] = (high_ent_mask.sum() / mask_sum_re).detach().item()
# ===== End REOPOLD =====
```

**注意事项**：
1. REOPOLD 用 **student entropy**（`entropy` 变量，已在 student forward pass 时计算）
2. EOPD 用 **teacher entropy**（`ref_entropy` 变量，从 teacher forward pass 获取）
3. 因此 REOPOLD **不需要** `fsdp_workers.py` 改动（student entropy 已有），只需 `dp_actor.py`
4. 需要确保 `calculate_entropy=True`：当 `reopold_enabled=True` 时加入 entropy 计算条件

**`dp_actor.py` calculate_entropy 条件更新**：
```python
eopd_enabled_flag = bool(policy_loss_cfg.get("eopd_enabled", False))
reopold_enabled_flag = bool(policy_loss_cfg.get("reopold_enabled", False))
calculate_entropy = entropy_coeff != 0 or rgopd_enabled or eopd_enabled_flag or reopold_enabled_flag
```

**代码位置**：REOPOLD block 插在 EOPD block 和 RLAD block 之间（均在 `exopd_adv` 上操作）：
```
exopd_adv = -reverse_kl               # ExOPD 基础
     ↓
[EOPD block]   exopd_adv *= low_ent_mask(teacher)     # 互斥：EOPD 或 REOPOLD 只开一个
[REOPOLD block] exopd_adv = clamp(exopd_adv, lb) * high_ent_mask(student)
     ↓
[RLAD block]   advantages = A_grpo + α·ρ·teacher_signal  # 独立路径
     ↓
[RGOPD-v2 block] advantages = (1-w)·A_grpo + w·exopd_adv
elif not rlad_enabled: advantages = exopd_adv          # 纯 ExOPD/EOPD/REOPOLD
```

### 5.4 训练脚本

需要新建两个脚本（`exp_REOPOLD_4Bto4B.sh` 和 `exp_REOPOLD_30Bto4B.sh`）：

**关键命令行参数**：
```bash
actor_rollout_ref.actor.policy_loss.rgopd_alpha=0.0 \
actor_rollout_ref.actor.policy_loss.lambda_vals=1.25 \   # 与 ExOPD / EOPD 对齐
+actor_rollout_ref.actor.policy_loss.reopold_enabled=True \
+actor_rollout_ref.actor.policy_loss.reopold_lambda=0.2 \
+actor_rollout_ref.actor.policy_loss.reopold_beta=0.5
```

---

## 六、公平对比配置（所有方法统一）

| 参数 | 值 | 备注 |
|------|---|------|
| Student | Qwen3-4B-Non-Thinking | 固定 |
| Teacher (4B→4B) | Qwen3-4B-RL-Math | 固定 |
| Teacher (30B→4B) | Qwen3-30B-A3B-Instruct-2507 | 固定 |
| 训练数据 | DeepMath-103K (level6, 57K) | 固定 |
| Batch size | 1024 | 固定 |
| LR | 1e-5 | 固定 |
| 采样温度 | 1.0 | 固定（统一，与 RLAD 原论文 0.6 不同） |
| rollout n | 1 | 固定 |
| 评测 | AIME24/25, Mean@32 | 固定 |
| Steps (4B→4B) | ~50 (total_epochs=1) | 固定 |
| Steps (30B→4B) | ~100 (total_epochs=2) | 固定 |

**各方法差异化超参**：

| 方法 | `lambda_vals` | 额外参数 |
|------|:---:|---------|
| ExOPD (baseline) | 1.25 | — |
| EOPD | 1.25 | `eopd_tau=0.8` |
| REOPOLD | 1.25 | `reopold_lambda=0.2`, `reopold_beta=0.5` |
| RLAD | 1.0 | `rlad_alpha=1.0`, `rlad_c=1.5` |
| RGOPD-v2 | 1.25 | `rgopd_alpha=1.0`, gate params |

> **为什么 RLAD 的 lambda_vals=1.0**：RLAD 有自己的 teacher 增强机制（`alpha * rho * teacher_signal`），不需要 ExOPD 的 λ>1 外插。若同时保留 λ=1.25，会导致两套 teacher 增强叠加，不公平。

---

## 七、代码改动总览

### 已完成

| 文件 | 改动 | 用途 |
|------|------|------|
| `verl/workers/fsdp_workers.py` | `compute_ref_log_prob` 中条件计算并传递 `ref_entropy` | EOPD |
| `verl/workers/actor/dp_actor.py` | `select_keys` 加入 `ref_entropy` | EOPD |
| `verl/workers/actor/dp_actor.py` | `calculate_entropy` 条件加入 `eopd_enabled_flag` | EOPD |
| `verl/workers/actor/dp_actor.py` | EOPD masking block（~Line 550-570） | EOPD |
| `verl/workers/actor/dp_actor.py` | RLAD trust-ratio block（~Line 572-608） | RLAD |
| `verl/workers/actor/dp_actor.py` | `elif not rlad_enabled` 保护（~Line 755） | RLAD |
| `examples/g_opd/exp_EOPD_4Bto4B.sh` | 训练脚本 | EOPD |
| `examples/g_opd/exp_EOPD_30Bto4B.sh` | 训练脚本 | EOPD |
| `examples/g_opd/exp_RLAD_4Bto4B.sh` | 训练脚本 | RLAD |
| `examples/g_opd/exp_RLAD_30Bto4B.sh` | 训练脚本 | RLAD |

### 待完成（REOPOLD）

| 文件 | 改动 | 状态 |
|------|------|------|
| `verl/workers/actor/dp_actor.py` | `calculate_entropy` 条件加入 `reopold_enabled_flag` | ⬜ 待实现 |
| `verl/workers/actor/dp_actor.py` | REOPOLD block（reward clipping + high-entropy mask） | ⬜ 待实现 |
| `examples/g_opd/exp_REOPOLD_4Bto4B.sh` | 训练脚本 | ⬜ 待创建 |
| `examples/g_opd/exp_REOPOLD_30Bto4B.sh` | 训练脚本 | ⬜ 待创建 |

---

## 八、结果记录表（待填）

### 同尺寸蒸馏（4B→4B, T=1.0, ~50 steps, Mean@32）

| 方法 | AIME24 | AIME25 | Best Step | vs ExOPD AIME24 | vs ExOPD AIME25 | 状态 |
|------|:------:|:------:|:---------:|:---------------:|:---------------:|:----:|
| ExOPD (Exp 1d) | **62.81%** | **56.98%** | S40 | baseline | baseline | ✅ |
| EOPD (τ=0.8) | — | — | — | — | — | 🔄 待跑 |
| REOPOLD (λ=0.2, β=0.5) | — | — | — | — | — | ⬜ 待实现 |
| RLAD (α=1.0, c=1.5) | — | — | — | — | — | 🔄 待跑 |
| **RGOPD-v2** | **62.71%** | **57.0%** | S10~S20 | -0.1pt | +0.0pt | ✅ |

### 强到弱蒸馏（30B→4B, T=1.0, ~100 steps, Mean@32）

| 方法 | AIME24 | AIME25 | Best Step | vs ExOPD AIME24 | vs ExOPD AIME25 | 状态 |
|------|:------:|:------:|:---------:|:---------------:|:---------------:|:----:|
| ExOPD (Exp 8c) | **58.7%** | **50.8%** | 论文 | baseline | baseline | ✅ 引用 |
| EOPD (τ=0.8) | — | — | — | — | — | 🔄 待跑 |
| REOPOLD (λ=0.2, β=0.5) | — | — | — | — | — | ⬜ 待实现 |
| RLAD (α=1.0, c=1.5) | — | — | — | — | — | 🔄 待跑 |
| **RGOPD-v2** | **58.02%** | **52.92%** | S50~S60 | -0.68pt | **+2.12pt** ✅ | ✅ |

---

## 九、运行优先级

```
P0 (尽快跑，结果直接影响论文):
  1. exp_EOPD_4Bto4B.sh     (~3-4h, 8×H20)
  2. exp_EOPD_30Bto4B.sh    (~6-8h, 8×H20)

P1 (实现完成后跑):
  3. exp_REOPOLD_4Bto4B.sh  (~3-4h)   ← 先完成代码实现
  4. exp_REOPOLD_30Bto4B.sh (~6-8h)

P2 (补充对比):
  5. exp_RLAD_4Bto4B.sh     (~3-4h)
  6. exp_RLAD_30Bto4B.sh    (~6-8h)
```

---

## 十、论文叙事价值

这三个 baseline 的对比在论文中的叙事框架：

```
ExOPD (基础)
  │
  ├── EOPD：  teacher高熵→删除   "不学teacher不确定的"
  │           → 问题：student自己高熵（不确定）时怎么办？
  │
  ├── REOPOLD：student高熵→保留  "student不确定的重点学"
  │           → 问题：teacher信号质量如何区分？
  │
  ├── RLAD：  GRPO+teacher加法   "同时用reward和teacher信号"
  │           → 问题：两信号如何自适应融合权重？
  │
  └── RGOPD-v2：
              outcome gate × token gate
              → outcome: reward信号决定"这道题值不值得学teacher"
              → token:   student熵×teacher熵联合决定"这个token值不值得学"
              → 统一了以上所有方法的关切：reward质量 + 双熵自适应
```

RGOPD-v2 的 dual gate 可以理解为 EOPD（teacher熵）× REOPOLD（student熵）× RLAD（reward门控）的统一泛化，论文中这是一个有力的叙事角度。
