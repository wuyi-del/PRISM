# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        # ===== S2/S3 shared: update step counter for curriculum scheduling =====
        self._update_step_counter = 0
        # ===== End S2/S3 =====

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            # reset input_ids, attention_mask, position_ids to ref model inputs if ref model input_ids is different from actor input_ids
            if "ref_input_ids" in micro_batch.keys():
                input_ids = micro_batch["ref_input_ids"]
                attention_mask = micro_batch["ref_attention_mask"]
                position_ids = micro_batch["ref_position_ids"]
                batch_size, seqlen = input_ids.shape

            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_ref_input_ids = "ref_input_ids" in data.batch.keys() # handle when ref input_ids is different from actor input_ids
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if has_ref_input_ids:
            select_keys.extend(["ref_input_ids", "ref_attention_mask", "ref_position_ids"])
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        # ===== S2/S3: increment step counter =====
        self._update_step_counter += 1
        current_step = self._update_step_counter
        # ===== End S2/S3 =====

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
         # Include base model log probs for corrected reward computation
        # These are computed when actor_rollout_ref.model.base_model_path and
        # actor_rollout_ref.ref.model.base_model_path are both specified
        if "base_log_prob" in data.batch.keys():
            select_keys.append("base_log_prob")
        if "base_ref_log_prob" in data.batch.keys():
            select_keys.append("base_ref_log_prob")
        # Include ref_log_prob for only_reverse_kl_advantages mode
        if self.config.policy_loss.only_reverse_kl_advantages and "ref_log_prob" in data.batch.keys():
            if "ref_log_prob" not in select_keys:
                select_keys.append("ref_log_prob")
        # Include ref_entropy for EOPD (teacher entropy for masking high-uncertainty tokens)
        if "ref_entropy" in data.batch.keys():
            select_keys.append("ref_entropy")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        # Include opd_teacher for multi-teacher distillation
        if "opd_teacher" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("opd_teacher")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    policy_loss_cfg = self.config.policy_loss
                    rgopd_alpha = float(policy_loss_cfg.get("rgopd_alpha", 0.0))
                    rgopd_enabled = bool(policy_loss_cfg.get("only_reverse_kl_advantages", False) and rgopd_alpha > 0)
                    eopd_enabled_flag = bool(policy_loss_cfg.get("eopd_enabled", False))
                    reopold_enabled_flag = bool(policy_loss_cfg.get("reopold_enabled", False))

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    # reopold needs student entropy for high-entropy token selection
                    calculate_entropy = entropy_coeff != 0 or rgopd_enabled or eopd_enabled_flag or reopold_enabled_flag
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = policy_loss_cfg.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # only use reverse KL for advantages if only_reverse_kl_advantages is True
                    if policy_loss_cfg.only_reverse_kl_advantages:
                        # Corrected reverse KL with base model normalization if base log probs are available
                        # Formula: (log_prob_actor - log_prob_ref) - (log_prob_actor_base - log_prob_ref_base)
                        # This removes the base model bias from both actor and ref models
                        if "base_log_prob" in model_inputs and "base_ref_log_prob" in model_inputs:
                            lambda_vals = policy_loss_cfg.lambda_vals

                            if policy_loss_cfg.multi_teacher_distill:
                                #### multi-teacher distillation ####
                                if "opd_teacher" in model_inputs:
                                    opd_teacher = model_inputs["opd_teacher"]
                                    batch_size = old_log_prob.shape[0]

                                    reverse_kl = torch.zeros_like(old_log_prob)

                                    for i in range(batch_size):
                                        teacher_type = opd_teacher[i] if isinstance(opd_teacher, (list, tuple)) else opd_teacher
                                        # TODO: need to improve the logic here
                                        if teacher_type == "math":
                                            if lambda_vals == 1.0:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["ref_log_prob"][i]
                                            else:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["base_log_prob"][i] - (model_inputs["ref_log_prob"][i] - model_inputs["base_log_prob"][i]) * lambda_vals
                                        elif teacher_type == "code":
                                            if lambda_vals == 1.0:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["base_ref_log_prob"][i]
                                            else:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["base_log_prob"][i] - (model_inputs["base_ref_log_prob"][i] - model_inputs["base_log_prob"][i]) * lambda_vals
                                        else:
                                            reverse_kl[i] = old_log_prob[i] - model_inputs["ref_log_prob"][i]
                                else:
                                    reverse_kl = old_log_prob - model_inputs["ref_log_prob"]
                                #### multi-teacher distillation ####
                            else:
                                #### single-teacher distillation ####
                                reverse_kl = old_log_prob - model_inputs["base_log_prob"]
                                reward_correction = model_inputs["ref_log_prob"] - model_inputs["base_log_prob"]

                                if lambda_vals == 1.0:
                                    reverse_kl = old_log_prob - model_inputs["ref_log_prob"]
                                else:
                                    reverse_kl = reverse_kl - reward_correction * lambda_vals
                                #### single-teacher distillation ####
                        else:
                            # Standard reverse KL: log(π_actor / π_ref) = log_prob_actor - log_prob_ref
                            reverse_kl = old_log_prob - model_inputs["ref_log_prob"]

                        exopd_adv = (-reverse_kl)

                        # ===== EOPD: Entropy-Aware Token Masking =====
                        # High-entropy (uncertain) teacher tokens → zero out teacher signal
                        # (equivalent to forward KL with top-k renormalize in original EOPD paper,
                        #  but implemented as a simpler mask using teacher entropy threshold)
                        eopd_enabled = bool(policy_loss_cfg.get("eopd_enabled", False))
                        if eopd_enabled:
                            eopd_tau = float(policy_loss_cfg.get("eopd_tau", 0.8))
                            if "ref_entropy" in model_inputs:
                                # Use teacher (ref) entropy: shape (bsz, seq_len) or (bsz, response_len)
                                teacher_ent_map = model_inputs["ref_entropy"]
                            elif entropy is not None:
                                # Fallback: use student entropy as proxy (less accurate but functional)
                                teacher_ent_map = entropy.detach()
                            else:
                                teacher_ent_map = None
                            if teacher_ent_map is not None:
                                # Low-entropy tokens: trust teacher (keep reverse KL)
                                # High-entropy tokens: zero out teacher signal (don't penalize student)
                                low_ent_mask = (teacher_ent_map <= eopd_tau).float()  # (bsz, seq_len)
                                exopd_adv = exopd_adv * low_ent_mask
                        # ===== End EOPD =====

                        # ===== REOPOLD: Relaxed On-Policy Distillation =====
                        # Ko et al. (2025), arXiv 2603.11137
                        # Two components (simplified, no 2-stage curriculum):
                        #   1. Mixture-based reward clipping: R̂ = max(R, log(λ/(1-λ)))
                        #      → prevents extreme negative gradients when student >> teacher
                        #   2. Student high-entropy token selection: keep top-β% tokens by student entropy
                        #      → focus teacher signal on tokens where student is still uncertain
                        # ⚠️  Direction is OPPOSITE to EOPD:
                        #      EOPD:    keep low-entropy (certain) TEACHER tokens
                        #      REOPOLD: keep high-entropy (uncertain) STUDENT tokens
                        reopold_enabled = bool(policy_loss_cfg.get("reopold_enabled", False))
                        if reopold_enabled:
                            import math as _math
                            reopold_lambda = float(policy_loss_cfg.get("reopold_lambda", 0.2))
                            reopold_beta = float(policy_loss_cfg.get("reopold_beta", 0.5))

                            # --- Component 1: reward lower-bound clipping ---
                            # log(λ/(1-λ)) ≈ -1.386 for λ=0.2
                            reward_lower_bound = _math.log(reopold_lambda / (1.0 - reopold_lambda))
                            exopd_adv = torch.clamp(exopd_adv, min=reward_lower_bound)

                            # --- Component 2: student high-entropy token mask ---
                            if entropy is not None:
                                # entropy: (bsz, response_len), zero-padded outside response
                                # compute top-β percentile over response tokens in batch
                                flat_ent = entropy.reshape(-1)
                                mask_flat = response_mask.reshape(-1).bool()
                                valid_ent = flat_ent[mask_flat]
                                if valid_ent.numel() > 0:
                                    # quantile(1 - beta) gives the threshold: tokens above it are top-beta%
                                    tau_beta = torch.quantile(valid_ent.float(), 1.0 - reopold_beta)
                                    high_ent_mask = ((entropy >= tau_beta) & response_mask.bool()).float()
                                else:
                                    tau_beta = torch.tensor(0.0, device=entropy.device)
                                    high_ent_mask = response_mask.float()
                                exopd_adv = exopd_adv * high_ent_mask

                                # monitoring
                                mask_sum_re = response_mask.sum().clamp(min=1)
                                micro_batch_metrics["reopold/reward_lb"] = reward_lower_bound
                                micro_batch_metrics["reopold/tau_beta"] = tau_beta.detach().item()
                                micro_batch_metrics["reopold/high_ent_ratio"] = (high_ent_mask.sum() / mask_sum_re).detach().item()
                            else:
                                # entropy not computed (shouldn't happen if calculate_entropy is set correctly)
                                micro_batch_metrics["reopold/reward_lb"] = reward_lower_bound
                        # ===== End REOPOLD =====

                        # ===== RLAD: Trust-Region-Ratio On-Policy Distillation =====
                        # Formula (Chen et al., 2025): A_t = A_grpo + alpha * clip(rho_t, 1/c, c) * log(π_t / π_ref)
                        # rho_t = π_student / π_ref = exp(log_prob_student - log_prob_ref)
                        # This is an additive (not multiplicative) fusion: GRPO signal + teacher signal
                        rlad_enabled = bool(policy_loss_cfg.get("rlad_enabled", False))
                        if rlad_enabled:
                            rlad_alpha = float(policy_loss_cfg.get("rlad_alpha", 1.0))
                            rlad_c = float(policy_loss_cfg.get("rlad_c", 1.5))
                            grpo_adv_rlad = model_inputs["advantages"]  # (bsz, response_len)

                            # Trust-region ratio: ρ_t = π_student / π_ref
                            rho = torch.exp(old_log_prob - model_inputs["ref_log_prob"])  # (bsz, response_len)
                            rho_clipped = torch.clamp(rho, 1.0 / rlad_c, rlad_c)

                            # Teacher advantage signal: log(π_teacher / π_ref)
                            teacher_signal = model_inputs["ref_log_prob"] - model_inputs["base_ref_log_prob"] \
                                if "base_ref_log_prob" in model_inputs \
                                else model_inputs["ref_log_prob"]  # fallback: just log π_teacher

                            # Scale teacher signal to GRPO magnitude
                            resp_len_rlad = response_mask.sum(dim=-1).clamp(min=1)
                            with torch.no_grad():
                                g_resp_rlad = (grpo_adv_rlad * response_mask).sum(dim=-1) / resp_len_rlad
                                ts_resp = (teacher_signal * rho_clipped * response_mask).sum(dim=-1) / resp_len_rlad
                                g_scale_rlad = g_resp_rlad.abs().mean().clamp(min=1e-6)
                                ts_scale = ts_resp.abs().mean().clamp(min=1e-6)
                                rlad_scale_ratio = g_scale_rlad / ts_scale

                            # Additive fusion: A_t = A_grpo + alpha * clip(rho, 1/c, c) * teacher_signal
                            advantages = grpo_adv_rlad + rlad_alpha * rho_clipped * teacher_signal * rlad_scale_ratio

                            # monitoring
                            mask_sum_rlad = response_mask.sum().clamp(min=1)
                            micro_batch_metrics["rlad/rho_mean"] = ((rho_clipped * response_mask).sum() / mask_sum_rlad).detach().item()
                            micro_batch_metrics["rlad/teacher_signal_mean"] = ((teacher_signal.abs() * response_mask).sum() / mask_sum_rlad).detach().item()
                            micro_batch_metrics["rlad/scale_ratio"] = rlad_scale_ratio.detach().item()
                        # ===== End RLAD =====

                        # ===== RGOPD-v2: ExOPD-RL adaptive fusion =====
                        if rgopd_enabled:
                            grpo_adv = model_inputs["advantages"]

                            # --- binary rewards for outcome gate ---
                            if "token_level_rewards" in model_inputs:
                                scores = model_inputs["token_level_rewards"].sum(dim=-1)
                                rewards = (scores > 0).float()
                            else:
                                resp_len = response_mask.sum(dim=-1).clamp(min=1)
                                grpo_seq = (grpo_adv * response_mask).sum(dim=-1) / resp_len
                                rewards = (grpo_seq > 0).float()

                            # --- dual gate ---
                            lw = float(policy_loss_cfg.get("rgopd_lambda_wrong", 1.0))
                            # ===== Dynamic Lambda Right (Adaptive Soft Gate) =====
                            # 根据当前 batch 正确率动态调整 λr：
                            #   准确率越高 → 模型越自信 → 越少听 teacher（λr↓）
                            #   准确率越低 → 模型还在学习 → 多听 teacher（λr↑）
                            # 消除手动超参 λr=0.2，让模型自己决定何时信任 teacher
                            use_dynamic_lr = bool(policy_loss_cfg.get("rgopd_dynamic_lambda_right", False))
                            if use_dynamic_lr:
                                acc = rewards.mean().clamp(min=0.10, max=0.99)
                                lr_ = float(max(0.05, 1.0 - acc.item() * 1.2))
                                micro_batch_metrics["rgopd/dyn_lambda_r"] = lr_
                                micro_batch_metrics["rgopd/dyn_batch_acc"] = acc.item()
                            else:
                                lr_ = float(policy_loss_cfg.get("rgopd_lambda_right", 0.2))
                            # ===== End Dynamic Lambda Right =====
                            tau = max(float(policy_loss_cfg.get("rgopd_tau", 1.0)), 1e-6)
                            denom = max(lw, 1e-6)

                            # outcome gate (trajectory-level)
                            g_y = lw * (1.0 - rewards) + lr_ * rewards

                            # ===== S1: PPL-Weighted Error Path (SCOPE-style) =====
                            # 对错误轨迹按 teacher PPL 做差异化加权：
                            #   高 PPL → teacher 对此路径越"困惑/不确定" → 给更高 ExOPD 权重（需要更多纠偏）
                            #   低 PPL → teacher 认为此路径接近正确 → 可降低 ExOPD 权重（减少过度纠偏）
                            # 正确轨迹不受影响（保持 lr_*rewards）
                            ppl_weighted_error = bool(policy_loss_cfg.get("rgopd_ppl_weighted_error", False))
                            # ===== F: Rank-based PPL (消除跨题不可比性) =====
                            # 用 batch 内排名替代绝对值：低PPL(高质量)→高rank→高权重
                            # 解决 n=1 下不同题目间 PPL 绝对值不可比的问题
                            rank_based_ppl = bool(policy_loss_cfg.get("rgopd_rank_based_ppl", False))
                            # ===== End F =====
                            if ppl_weighted_error:
                                ppl_tau_val = max(float(policy_loss_cfg.get("rgopd_ppl_tau", 1.0)), 1e-6)
                                if "ref_log_prob" in model_inputs:
                                    ref_lp = model_inputs["ref_log_prob"]  # (bsz, response_len)
                                    ppl_resp_len = response_mask.sum(dim=-1).clamp(min=1)
                                    # 序列级 PPL proxy: -mean(ref_log_prob)，越高表示 teacher 越不认可此输出
                                    seq_ppl = -(ref_lp * response_mask).sum(dim=-1) / ppl_resp_len  # (bsz,)
                                    wrong_mask = (rewards < 0.5).bool()  # 错误轨迹 mask
                                    if wrong_mask.any():
                                        wrong_ppl = seq_ppl[wrong_mask]
                                        # ===== P2-7: PPL Filter (过滤极端退化轨迹) =====
                                        ppl_filter_sigma = float(policy_loss_cfg.get("rgopd_ppl_filter_sigma", 0.0))
                                        filter_handled = False
                                        if ppl_filter_sigma > 0 and wrong_ppl.numel() >= 2:
                                            wp_mean = wrong_ppl.mean()
                                            wp_std = wrong_ppl.std().clamp(min=1e-6)
                                            filter_threshold = wp_mean + ppl_filter_sigma * wp_std
                                            filtered_mask = wrong_ppl > filter_threshold
                                            micro_batch_metrics["rgopd/filter_threshold"] = filter_threshold.detach().item()
                                            micro_batch_metrics["rgopd/filter_kept_ratio"] = (filtered_mask.sum().float() / filtered_mask.numel()).detach().item()
                                            if filtered_mask.any():
                                                filtered_ppl = wrong_ppl[filtered_mask]
                                                if rank_based_ppl:
                                                    ranks_filtered = torch.argsort(torch.argsort(filtered_ppl)).float()
                                                    inv_ranks_f = ranks_filtered.max() - ranks_filtered
                                                    ppl_weights_f = torch.softmax(inv_ranks_f / ppl_tau_val, dim=0)
                                                    micro_batch_metrics["rgopd/f_rank_mean"] = ranks_filtered.mean().detach().item()
                                                else:
                                                    ppl_weights_f = torch.softmax(-filtered_ppl / ppl_tau_val, dim=0)
                                                full_weights = torch.full_like(wrong_ppl, 1.0)
                                                full_weights[filtered_mask] = ppl_weights_f
                                                ppl_weights = full_weights
                                            else:
                                                ppl_weights = torch.ones_like(wrong_ppl)
                                            filter_handled = True
                                        # ===== End P2-7 PPL Filter =====
                                        
                                        if not filter_handled:
                                            if rank_based_ppl:
                                                ranks = torch.argsort(torch.argsort(wrong_ppl)).float()
                                                inv_ranks = ranks.max() - ranks
                                                ppl_weights = torch.softmax(inv_ranks / ppl_tau_val, dim=0)
                                                micro_batch_metrics["rgopd/f_rank_mean"] = ranks.mean().detach().item()
                                                micro_batch_metrics["rgopd/f_inv_rank_max"] = inv_ranks.max().detach().item()
                                            else:
                                                ppl_weights = torch.softmax(-wrong_ppl / ppl_tau_val, dim=0)
                                        # 用 ppl_weights 替换错误轨迹的均匀 lw
                                        g_y_base = g_y.clone()
                                        g_y_base[wrong_mask] = lw * ppl_weights
                                        g_y = g_y_base
                                    micro_batch_metrics["rgopd/s1_seq_ppl_mean"] = seq_ppl.mean().detach().item()
                                    micro_batch_metrics["rgopd/s1_wrong_ppl_mean"] = seq_ppl[wrong_mask].mean().detach().item() if wrong_mask.any() else 0.0
                                    micro_batch_metrics["rgopd/s1_ppl_weight_max"] = ppl_weights.max().detach().item() if wrong_mask.any() else 0.0
                                    micro_batch_metrics["rgopd/s1_ppl_weight_min"] = ppl_weights.min().detach().item() if wrong_mask.any() else 0.0
                                else:
                                    if torch.distributed.get_rank() == 0:
                                        print("[WARN] S1 ppl_weighted_error enabled but ref_log_prob not available, skipping.")
                            # ===== End S1 =====

                            # token gate (token-level)
                            if entropy is None:
                                raise RuntimeError("RGOPD requires token entropy, but entropy was not computed.")
                            teacher_ent_val = float(policy_loss_cfg.get("rgopd_teacher_entropy", 0.1))
                            teacher_entropy = torch.full_like(entropy, teacher_ent_val)
                            delta_h = torch.clamp((entropy - teacher_entropy) / tau, -10.0, 10.0)
                            s_t = torch.sigmoid(delta_h)

                            # w_bar keeps alpha as a true global fusion strength.
                            w_raw = g_y.unsqueeze(1) * s_t / denom
                            w_bar = torch.clamp(rgopd_alpha * w_raw, 0.0, 1.0)

                            # --- scale alignment ---
                            # n=1 时 GRPO 是 response-level scalar 广播到 token，
                            # ExOPD 是 token-wise reverse-KL，两者量纲不同。
                            # 不做 z-score（会在 micro_batch=1 时把 GRPO 减成 0），
                            # 而是用 response-level 平均绝对值做纯缩放对齐。
                            resp_len = response_mask.sum(dim=-1).clamp(min=1)
                            with torch.no_grad():
                                # response-level 聚合：每条 response → 一个标量
                                g_resp = (grpo_adv * response_mask).sum(dim=-1) / resp_len
                                e_resp = (exopd_adv * response_mask).sum(dim=-1) / resp_len
                                # 用 mean(|x|) 作为尺度度量（比 std 更鲁棒于 micro_batch=1）
                                g_scale = g_resp.abs().mean().clamp(min=1e-6)
                                e_scale = e_resp.abs().mean().clamp(min=1e-6)
                                scale_ratio = g_scale / e_scale

                            # 把 ExOPD 缩放到 GRPO 的量纲，再做加权融合
                            exopd_scaled = exopd_adv * scale_ratio
                            advantages = (1.0 - w_bar) * grpo_adv + w_bar * exopd_scaled

                            # --- monitoring metrics ---
                            mask_sum = response_mask.sum().clamp(min=1)
                            micro_batch_metrics["rgopd/w_bar_mean"] = ((w_bar * response_mask).sum() / mask_sum).detach().item()
                            micro_batch_metrics["rgopd/w_bar_max"] = w_bar.max().detach().item()
                            micro_batch_metrics["rgopd/binary_reward_mean"] = rewards.mean().detach().item()
                            micro_batch_metrics["rgopd/student_entropy_mean"] = ((entropy * response_mask).sum() / mask_sum).detach().item()
                            micro_batch_metrics["rgopd/scale_ratio"] = scale_ratio.detach().item()
                            micro_batch_metrics["rgopd/grpo_scale"] = g_scale.detach().item()
                            micro_batch_metrics["rgopd/exopd_scale"] = e_scale.detach().item()
                        elif not rlad_enabled:
                            # Pure ExOPD / EOPD path: use teacher-only advantage
                            advantages = exopd_adv
                        # (RLAD: advantages already set in the RLAD block above)
                        # ===== End RGOPD-v2 =====
                   
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
