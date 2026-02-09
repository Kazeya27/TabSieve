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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager, find_latest_ckpt_path
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, config=None):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            # Get length from the initial response mask
            response_length = grpo_calculation_mask.size(1)
            # This mask is the one intended for GRPO
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # ---------------------------------------------------------------------
        # Correctness task-weighting (categorical vs numerical)
        #
        # Motivation:
        #   In our mixed dataset (classification + regression), the regression
        #   correctness reward can produce denser / smoother signals early on.
        #   This often yields larger |A_correctness| for numerical tasks, which
        #   may dominate the policy gradient update.
        #
        # We maintain EMA statistics of mean(|A_correctness|) for:
        #   - categorical tasks
        #   - numerical tasks
        # and derive a *task weight* to either:
        #   (a) suppress regression (down-weight numerical correctness advantage),
        # or
        #   (b) boost classification (up-weight categorical correctness advantage).
        #
        # IMPORTANT: to better reflect the *effective* optimization signal, the
        # EMA update is driven by actor-side statistics computed *after* the PPO
        # ratio is known (so we can filter tokens outside the PPO clip band).
        # This introduces a 1-step lag: the weight computed from step t is
        # applied starting from step t+1.
        # ---------------------------------------------------------------------
        self._correctness_adv_abs_ema = {"categorical": None, "numerical": None}
        self._correctness_adv_abs_ema_steps = {"categorical": 0, "numerical": 0}
        # Per-task weights applied to the correctness advantage.
        #   - categorical: classification-like tasks
        #   - numerical: regression-like tasks
        # NOTE: we keep a legacy scalar (`_correctness_adv_reg_weight`) for
        # backward-compatible logging, but the actual scaling uses this dict.
        self._correctness_adv_task_weights = {"categorical": 1.0, "numerical": 1.0}
        self._correctness_adv_reg_weight = 1.0


    def _update_correctness_adv_reg_weight(
        self,
        *,
        # Preferred path: provide already-aggregated abs-mean values.
        # These are expected to be computed on the actor side *after* the PPO ratio
        # is known, so we can filter out tokens outside the PPO clip band.
        abs_mean_overrides: Optional[dict] = None,
        # Fallback path (kept for backward compatibility / debugging):
        correctness_adv: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        task_types=None,
    ):
        """Update EMA(|A_correctness|) and compute the correctness task-weight.

        The weight is used to balance the optimization strength between
        categorical and numerical *correctness* signals.

        Three modes are supported (selected by `algorithm.correctness_adv_task_weight_mode`):
          - "suppress_regression": down-weight numerical correctness advantage.
          - "boost_classification": up-weight categorical correctness advantage.
          - "suppress_and_boost" (or "both"): apply both (suppress numerical + boost categorical)
            using a *direct* `/2` attenuation on the ratio-derived coefficient (NOT log-space).

        IMPORTANT (clip-aware EMA):
          - If `abs_mean_overrides` is provided, we treat those values as the
            *effective* per-task mean(|A_correctness|), typically computed on the
            actor side using only tokens with PPO ratio within the clip band.
          - This is the recommended pathway for the user's request.
          - If overrides are missing, we fall back to computing abs-mean from
            `correctness_adv` and `response_mask` (no clip filtering).
        """

        metrics: dict = {}

        algo_cfg = self.config.algorithm

        # -------------------------
        # Hyper-parameters
        # -------------------------
        ema_beta = float(algo_cfg.get("correctness_adv_reg_weight_ema_beta", 0.99))
        eps = float(algo_cfg.get("correctness_adv_reg_weight_eps", 1e-6))
        # NOTE: the original YAML used `..._use_bias_correction`; we support both.
        bias_correction = bool(
            algo_cfg.get(
                "correctness_adv_reg_weight_bias_correction",
                algo_cfg.get("correctness_adv_reg_weight_use_bias_correction", True),
            )
        )

        mode = str(algo_cfg.get("correctness_adv_task_weight_mode", "suppress_regression")).lower()
        suppress_modes = {
            "suppress_regression",
            "downweight_regression",
            "downweight_numerical",
            "suppress_numerical",
        }
        boost_modes = {
            "boost_classification",
            "boost_categorical",
            "upweight_categorical",
            "upweight_classification",
        }
        both_modes = {
            # Apply BOTH: boost categorical AND suppress numerical.
            "suppress_and_boost",
            "boost_and_suppress",
            "suppress_boost",
            "boost_suppress",
            "both",
        }
        is_both = mode in both_modes
        is_boost = (mode in boost_modes) and (not is_both)
        is_suppress = (mode in suppress_modes) and (not is_both) and (not is_boost)
        if not (is_both or is_boost or is_suppress):
            # Be conservative: unknown mode -> suppress regression.
            is_suppress = True

        # clamp range defaults depend on mode
        if is_suppress:
            default_w_min, default_w_max = 0.1, 1.0
        elif is_boost:
            default_w_min, default_w_max = 1.0, 10.0
        else:
            # both: we need to support weights on both sides
            default_w_min, default_w_max = 0.1, 10.0
        w_min = float(algo_cfg.get("correctness_adv_reg_weight_min", default_w_min))
        w_max = float(algo_cfg.get("correctness_adv_reg_weight_max", default_w_max))

        # Optional constraints (kept for compatibility).
        only_downweight = bool(algo_cfg.get("correctness_adv_reg_weight_only_down", True))
        only_upweight = bool(algo_cfg.get("correctness_adv_reg_weight_only_up", True))

        # -------------------------
        # 1) Obtain per-task abs-mean values for this step
        # -------------------------
        abs_means: dict = {}
        if abs_mean_overrides is not None:
            # Expected keys: {"categorical": float, "numerical": float}
            for k in ("categorical", "numerical"):
                v = abs_mean_overrides.get(k)
                if v is None:
                    continue
                try:
                    v_f = float(v)
                except Exception:
                    continue
                if np.isfinite(v_f) and v_f >= 0.0:
                    abs_means[k] = v_f
        else:
            # Fallback (no clip filtering): compute token-level abs-mean from the batch.
            if correctness_adv is None or response_mask is None or task_types is None:
                metrics["training/correctness_adv/reg_weight"] = float(self._correctness_adv_reg_weight)
                return float(self._correctness_adv_reg_weight), metrics, None

            try:
                task_types_arr = np.asarray(task_types, dtype=object)
            except Exception:
                metrics["training/correctness_adv/reg_weight"] = float(self._correctness_adv_reg_weight)
                return float(self._correctness_adv_reg_weight), metrics, None

            if task_types_arr.shape[0] != correctness_adv.shape[0]:
                metrics["training/correctness_adv/reg_weight"] = float(self._correctness_adv_reg_weight)
                return float(self._correctness_adv_reg_weight), metrics, None

            device = correctness_adv.device
            resp_mask = response_mask.to(device=device).bool()
            with torch.no_grad():
                for tname in ("categorical", "numerical"):
                    type_mask_1d = torch.as_tensor(task_types_arr == tname, device=device)
                    if not torch.any(type_mask_1d):
                        continue
                    type_mask = resp_mask & type_mask_1d[:, None]
                    vals = torch.masked_select(correctness_adv, type_mask)
                    if vals.numel() == 0:
                        continue
                    abs_means[tname] = float(torch.mean(torch.abs(vals)).item())

        # If we have nothing to update, keep the previous weight.
        if not abs_means:
            metrics["training/correctness_adv/reg_weight"] = float(self._correctness_adv_reg_weight)
            return float(self._correctness_adv_reg_weight), metrics, None

        # -------------------------
        # 2) Update EMA for each task type
        # -------------------------
        with torch.no_grad():
            for tname, abs_mean in abs_means.items():
                metrics[f"training/correctness_adv/abs_mean/{tname}"] = float(abs_mean)

                prev = self._correctness_adv_abs_ema.get(tname)
                if prev is None:
                    ema = float(abs_mean)
                else:
                    ema = ema_beta * float(prev) + (1.0 - ema_beta) * float(abs_mean)
                self._correctness_adv_abs_ema[tname] = float(ema)
                self._correctness_adv_abs_ema_steps[tname] = int(self._correctness_adv_abs_ema_steps.get(tname, 0)) + 1

                if bias_correction:
                    step = self._correctness_adv_abs_ema_steps[tname]
                    ema_hat = float(ema) / (1.0 - (ema_beta**step))
                    metrics[f"training/correctness_adv/abs_mean_ema_hat/{tname}"] = float(ema_hat)
                metrics[f"training/correctness_adv/abs_mean_ema/{tname}"] = float(ema)

        ema_cat = self._correctness_adv_abs_ema.get("categorical")
        ema_num = self._correctness_adv_abs_ema.get("numerical")
        if ema_cat is None or ema_num is None:
            # Need both sides to compute a ratio.
            metrics["training/correctness_adv/reg_weight"] = float(self._correctness_adv_reg_weight)
            return float(self._correctness_adv_reg_weight), metrics, None

        # Bias-corrected EMAs if enabled.
        if bias_correction:
            step_cat = self._correctness_adv_abs_ema_steps["categorical"]
            step_num = self._correctness_adv_abs_ema_steps["numerical"]
            ema_cat_hat = float(ema_cat) / (1.0 - (ema_beta**step_cat))
            ema_num_hat = float(ema_num) / (1.0 - (ema_beta**step_num))
        else:
            ema_cat_hat = float(ema_cat)
            ema_num_hat = float(ema_num)

        # -------------------------
        # 3) Compute ratio and derive the task weight
        # -------------------------
        raw_ratio = ema_cat_hat / (ema_num_hat + eps)
        metrics["training/correctness_adv/reg_weight_raw_ratio"] = float(raw_ratio)

        # NOTE:
        #   raw_ratio > 1 means categorical has larger abs-adv than numerical.
        #   raw_ratio < 1 means numerical dominates.
        # We always produce per-task weights (categorical & numerical).
        # For backward compatibility, we also keep a legacy scalar metric
        # `training/correctness_adv/reg_weight` that depends on the selected mode.
        if is_both:
            # -----------------------------------------------------------------
            # BOTH (suppress + boost): apply symmetric reweighting.
            #
            # If we were to BOTH suppress numerical by `raw_ratio` and boost
            # categorical by `1/raw_ratio`, the *relative* scaling between tasks
            # would be squared (too aggressive).
            #
            # The user requests a *direct* "divide-by-2" attenuation (NOT
            # log-space). We therefore derive a ratio-based coefficient and
            # halve it in the linear domain:
            #
            #   boost_factor = EMA_num / EMA_cat  (>= 0)
            #   eta = boost_factor / 2
            #   w_cat = eta
            #   w_num = 1 / eta
            #
            # To guarantee the intended direction without requiring additional
            # config flags (no "only_up/only_down" needed), we clamp:
            #   w_cat in [1, w_max]  (never suppress categorical)
            #   w_num in [w_min, 1]  (never boost numerical)
            # -----------------------------------------------------------------
            boost_factor = ema_num_hat / (ema_cat_hat + eps)
            boost_factor = max(float(boost_factor), eps)

            # Direct linear attenuation: eta = boost_factor / 2.
            eta = float(boost_factor) / 2.0
            eta = max(eta, eps)
            w_cat = float(eta)
            w_num = float(1.0 / eta)

            # Mode-specific clamping to preserve direction:
            #   categorical: only boost (>=1)
            #   numerical: only suppress (<=1)
            w_cat = float(np.clip(w_cat, 1.0, max(1.0, w_max)))
            w_num = float(np.clip(w_num, min(w_min, 1.0), 1.0))

            # Store per-task weights.
            self._correctness_adv_task_weights = {"categorical": w_cat, "numerical": w_num}
            # Legacy scalar: keep categorical weight for historical dashboards.
            self._correctness_adv_reg_weight = float(w_cat)

            metrics["training/correctness_adv/task_weight/categorical"] = float(w_cat)
            metrics["training/correctness_adv/task_weight/numerical"] = float(w_num)
            metrics["training/correctness_adv/reg_weight"] = float(w_cat)
            metrics["training/correctness_adv/reg_weight_mode"] = 2.0

        elif is_boost:
            # Boost categorical: inverse ratio.
            w_cat = ema_num_hat / (ema_cat_hat + eps)
            if only_upweight:
                w_cat = max(1.0, float(w_cat))
            w_cat = float(np.clip(float(w_cat), w_min, w_max))
            w_num = 1.0

            self._correctness_adv_task_weights = {"categorical": float(w_cat), "numerical": float(w_num)}
            self._correctness_adv_reg_weight = float(w_cat)

            metrics["training/correctness_adv/task_weight/categorical"] = float(w_cat)
            metrics["training/correctness_adv/task_weight/numerical"] = float(w_num)
            metrics["training/correctness_adv/reg_weight"] = float(w_cat)
            metrics["training/correctness_adv/reg_weight_mode"] = 1.0

        else:
            # Default: suppress regression (numerical)
            w_num = raw_ratio
            if only_downweight:
                w_num = min(1.0, float(w_num))
            w_num = float(np.clip(float(w_num), w_min, w_max))
            w_cat = 1.0

            self._correctness_adv_task_weights = {"categorical": float(w_cat), "numerical": float(w_num)}
            self._correctness_adv_reg_weight = float(w_num)

            metrics["training/correctness_adv/task_weight/categorical"] = float(w_cat)
            metrics["training/correctness_adv/task_weight/numerical"] = float(w_num)
            metrics["training/correctness_adv/reg_weight"] = float(w_num)
            metrics["training/correctness_adv/reg_weight_mode"] = 0.0

        # We do NOT return a scaled tensor here. Scaling for the current batch is
        # applied separately using the *previous* stored weight(s) (1-step lag).
        return float(self._correctness_adv_reg_weight), metrics, None


    def _apply_correctness_adv_task_weight(
        self,
        *,
        correctness_adv: torch.Tensor,
        task_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the *current* correctness task weight to a correctness advantage tensor.

        This method does NOT update EMA or the weight; it only applies the stored
        per-task weights from `self._correctness_adv_task_weights` (typically
        computed from previous steps).

        Args:
            correctness_adv: (bs, T) correctness advantage tensor.
            task_type_ids: (bs,) int tensor. 0=categorical, 1=numerical.
        """

        w_cat = float(self._correctness_adv_task_weights.get("categorical", 1.0))
        w_num = float(self._correctness_adv_task_weights.get("numerical", 1.0))
        if w_cat == 1.0 and w_num == 1.0:
            return correctness_adv

        scaled = correctness_adv.clone()
        cat_mask = task_type_ids == 0
        num_mask = task_type_ids == 1
        if w_cat != 1.0 and torch.any(cat_mask):
            scaled[cat_mask] = scaled[cat_mask] * w_cat
        if w_num != 1.0 and torch.any(num_mask):
            scaled[num_mask] = scaled[num_mask] * w_num

        return scaled

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            assert n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0, f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            megatron_dp = n_gpus // (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _log_rollout_sample(self, batch: DataProto):
        """Print a single rollout sample for GRPO to keep logs lightweight."""
        if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            return
        if not self.config.trainer.get("log_rollout_sample", False):
            return
        if "prompts" not in batch.batch or "responses" not in batch.batch:
            return
        if len(batch.batch["prompts"]) == 0:
            return

        try:
            prompt_ids = batch.batch["prompts"][0].detach().cpu().tolist()
            response_ids = batch.batch["responses"][0].detach().cpu().tolist()
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            max_chars = self.config.trainer.get("rollout_sample_max_chars", None)
            if max_chars is not None:
                prompt_text = prompt_text[:max_chars]
                response_text = response_text[:max_chars]

            reward_text = ""
            if "token_level_scores" in batch.batch:
                reward_val = batch.batch["token_level_scores"][0].sum().item()
                reward_text = f"\n[GRPO rollout][step {self.global_steps}] reward: {reward_val:.4f}"

            log_msg = (
                f"[GRPO rollout][step {self.global_steps}] prompt: {prompt_text}\n"
                f"[GRPO rollout][step {self.global_steps}] response: {response_text}{reward_text}"
            )
            print(log_msg)
        except Exception as e:
            print(f"[GRPO rollout] failed to log sample: {e}")

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, "worker_nsight_options must be set when profile_steps is set"
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(OmegaConf.select(self.config.trainer, "worker_nsight_options"))

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.workers.rollout.async_server import AsyncLLMServerManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        BaseCheckpointManager.local_mkdir(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    @staticmethod
    def _compute_reward_info_metrics(reward_extra_infos: dict[str, list]) -> dict[str, float]:
        """Aggregate custom reward components for logging."""
        reward_info_metrics: dict[str, float] = {}
        task_types = reward_extra_infos.get("task_type")
        task_types_arr = None
        if task_types is not None:
            try:
                task_types_arr = np.asarray(task_types, dtype=object)
            except Exception:
                task_types_arr = None
        def add_stats(prefix: str, values: np.ndarray, mean_values: Optional[np.ndarray] = None) -> None:
            if values.size == 0:
                return
            if np.all(np.isnan(values)):
                return
            mean_src = mean_values if mean_values is not None else values
            if mean_src.size == 0 or np.all(np.isnan(mean_src)):
                return
            reward_info_metrics[f"{prefix}/mean"] = float(np.nanmean(mean_src))
            reward_info_metrics[f"{prefix}/max"] = float(np.nanmax(values))
            reward_info_metrics[f"{prefix}/min"] = float(np.nanmin(values))
        for key, values in reward_extra_infos.items():
            if key == "task_type":
                continue
            if len(values) == 0:
                continue
            try:
                arr = np.asarray(values, dtype=np.float32)
            except Exception:
                continue
            if arr.size == 0:
                continue
            mean_values = np.clip(arr, None, 1.0) if key == "mape" else None
            add_stats(f"reward/{key}", arr, mean_values=mean_values)
            if task_types_arr is not None and len(task_types_arr) == len(arr):
                for task_type in ("categorical", "numerical"):
                    mask = task_types_arr == task_type
                    if not np.any(mask):
                        continue
                    type_vals = arr[mask]
                    if key == "mape" and task_type != "numerical":
                        continue
                    mean_values = np.clip(type_vals, None, 1.0) if key == "mape" else None
                    add_stats(f"reward/{task_type}/{key}", type_vals, mean_values=mean_values)
        if task_types_arr is not None:
            correctness_vals = reward_extra_infos.get("correctness_reward")
            if correctness_vals is not None:
                try:
                    correctness_arr = np.asarray(correctness_vals, dtype=np.float32)
                except Exception:
                    correctness_arr = None
                if correctness_arr is not None and len(correctness_arr) == len(task_types_arr):
                    cat_mask = task_types_arr == "categorical"
                    if np.any(cat_mask):
                        cat_vals = correctness_arr[cat_mask]
                        if cat_vals.size > 0 and not np.all(np.isnan(cat_vals)):
                            reward_info_metrics["reward/categorical/accuracy"] = float(np.nanmean(cat_vals))
        return reward_info_metrics

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = self.global_steps in self.config.trainer.profile_steps if self.config.trainer.profile_steps is not None else False
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}
                timing_raw = {}
                reward_extra_infos_dict: dict[str, list] = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                            metrics.update(self._compute_reward_info_metrics(reward_extra_infos_dict))

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        self._log_rollout_sample(batch)

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        # -----------------------------------------------------------------
                        # Component-wise GRPO advantages (structure / select / correctness)
                        # + weighted sum of advantages + single PPO clipping ("sum-then-clip").
                        #
                        # Why this design?
                        #   1) GRPO standardizes rewards within each prompt-group (subtract mean / divide std).
                        #      Therefore any constant scaling on reward (e.g., lambda_s, lambda_f, lambda_c)
                        #      is largely cancelled by the standardization.
                        #   2) To preserve the intended optimization weighting, we compute GRPO advantages
                        #      *per reward component* and re-apply the lambdas at the advantage level:
                        #
                        #        A_final = λ_s A_struct + λ_f A_select + λ_c A_corr_scaled
                        #
                        #      where A_corr_scaled optionally down-weights numerical-task correctness using
                        #      an EMA ratio of mean(|A_corr|) between categorical and numerical tasks.
                        #   3) We then feed A_final into the *vanilla* verl actor loss (one PPO clipping),
                        #      i.e., we do NOT clip each component separately.
                        #
                        # This mode is enabled by:
                        #   algorithm.use_component_advantages=True
                        # and currently supports outcome-style GRPO (adv_estimator=grpo) with
                        # algorithm.use_kl_in_reward=False.
                        # -----------------------------------------------------------------
                        use_component_adv = bool(self.config.algorithm.get("use_component_advantages", True))
                        enable_reg_weight = bool(self.config.algorithm.get("enable_correctness_adv_reg_weight", True))

                        can_use_component_adv = (
                            use_component_adv
                            and self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO
                            and not bool(self.config.algorithm.use_kl_in_reward)
                        )

                        structure_vals = batch.non_tensor_batch.get("structure_reward")
                        select_vals = batch.non_tensor_batch.get("reward_select")
                        correctness_vals = batch.non_tensor_batch.get("correctness_reward")

                        if can_use_component_adv and structure_vals is not None and correctness_vals is not None:
                            try:
                                structure_arr = np.asarray(structure_vals, dtype=np.float32)
                                correctness_arr = np.asarray(correctness_vals, dtype=np.float32)
                                if select_vals is None:
                                    select_arr = None
                                else:
                                    select_arr = np.asarray(select_vals, dtype=np.float32)
                            except Exception:
                                structure_arr = None
                                correctness_arr = None
                                select_arr = None

                            batch_size = batch.batch["token_level_rewards"].shape[0]

                            # Fallback to vanilla advantage if any parsing / shape mismatch happens.
                            if (
                                structure_arr is None
                                or correctness_arr is None
                                or structure_arr.shape[0] != batch_size
                                or correctness_arr.shape[0] != batch_size
                            ):
                                batch = compute_advantage(
                                    batch,
                                    adv_estimator=self.config.algorithm.adv_estimator,
                                    gamma=self.config.algorithm.gamma,
                                    lam=self.config.algorithm.lam,
                                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                    multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                                    config=self.config.algorithm,
                                )
                            else:
                                if select_arr is None or select_arr.shape[0] != batch_size:
                                    select_arr = np.zeros_like(structure_arr)

                                # Read reward mixing coefficients from the custom reward config.
                                # NOTE: These lambdas must be applied on advantages (not rewards) to
                                # preserve their intended effect under GRPO standardization.
                                custom_reward_cfg = self.config.get("custom_reward_function") or {}
                                reward_kwargs = custom_reward_cfg.get("reward_kwargs") or {}
                                lambda_s = float(reward_kwargs.get("lambda_s", 0.1))
                                # For reward f (select), default to 0.0 to avoid accidental weighting
                                # when using reward functions without this component.
                                lambda_f = float(reward_kwargs.get("lambda_f", 0.0))
                                lambda_c = float(reward_kwargs.get("lambda_c", 0.9))

                                response_mask = batch.batch.get("response_mask")
                                if response_mask is None:
                                    response_mask = compute_response_mask(batch)
                                response_lengths = response_mask.sum(-1).long()

                                if torch.any(response_lengths <= 0):
                                    # Degenerate sequences; fall back to vanilla advantage.
                                    batch = compute_advantage(
                                        batch,
                                        adv_estimator=self.config.algorithm.adv_estimator,
                                        gamma=self.config.algorithm.gamma,
                                        lam=self.config.algorithm.lam,
                                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                        multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                                        config=self.config.algorithm,
                                    )
                                else:
                                    device = batch.batch["token_level_rewards"].device
                                    batch_idx = torch.arange(batch_size, device=device)

                                    def _build_token_rewards(per_sample_arr: np.ndarray) -> torch.Tensor:
                                        """Place a per-sample scalar reward on the final response token."""
                                        token_rewards = torch.zeros_like(batch.batch["token_level_rewards"], dtype=torch.float32)
                                        per_sample = torch.as_tensor(per_sample_arr, dtype=torch.float32, device=device)
                                        token_rewards[batch_idx, response_lengths - 1] = per_sample
                                        return token_rewards

                                    def _build_component_batch(token_rewards: torch.Tensor) -> DataProto:
                                        """Minimal DataProto required by `compute_advantage` for GRPO."""
                                        tensors = {
                                            "token_level_rewards": token_rewards,
                                            "token_level_scores": token_rewards,
                                            "response_mask": response_mask,
                                        }
                                        # Multi-turn GRPO uses loss_mask as the calculation mask.
                                        if self.config.actor_rollout_ref.rollout.multi_turn.enable and "loss_mask" in batch.batch:
                                            tensors["loss_mask"] = batch.batch["loss_mask"]
                                        if "reward_baselines" in batch.batch:
                                            tensors["reward_baselines"] = batch.batch["reward_baselines"]

                                        non_tensors = {}
                                        if "uid" in batch.non_tensor_batch:
                                            non_tensors["uid"] = batch.non_tensor_batch["uid"]
                                        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

                                    def _compute_component_adv(per_sample_arr: np.ndarray) -> Optional[torch.Tensor]:
                                        comp = _build_component_batch(_build_token_rewards(per_sample_arr))
                                        comp = compute_advantage(
                                            comp,
                                            adv_estimator=self.config.algorithm.adv_estimator,
                                            gamma=self.config.algorithm.gamma,
                                            lam=self.config.algorithm.lam,
                                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                                            config=self.config.algorithm,
                                        )
                                        return comp.batch.get("advantages")

                                    # 1) Per-component GRPO advantages
                                    adv_s = _compute_component_adv(structure_arr)
                                    adv_f = _compute_component_adv(select_arr)
                                    adv_c = _compute_component_adv(correctness_arr)

                                    if adv_s is None or adv_f is None or adv_c is None:
                                        # Safety fallback
                                        batch = compute_advantage(
                                            batch,
                                            adv_estimator=self.config.algorithm.adv_estimator,
                                            gamma=self.config.algorithm.gamma,
                                            lam=self.config.algorithm.lam,
                                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                                            config=self.config.algorithm,
                                        )
                                    else:
                                        # 2) Optional: correctness task-weighting (categorical vs numerical).
                                        #
                                        # IMPORTANT:
                                        #   We only *apply* the stored weight here (1-step lag).
                                        #   The EMA/weight update itself is done later, using actor-side
                                        #   clip-aware statistics (see below after actor update).
                                        adv_c_scaled = adv_c
                                        if enable_reg_weight:
                                            task_types = batch.non_tensor_batch.get("task_type")
                                            if task_types is not None:
                                                # Encode task types as a tensor so the actor can compute
                                                # clip-aware per-task stats even when it splits TensorDict batches.
                                                try:
                                                    task_types_arr = np.asarray(task_types, dtype=object)
                                                except Exception:
                                                    task_types_arr = None

                                                if task_types_arr is not None and task_types_arr.shape[0] == batch_size:
                                                    # 0 = categorical, 1 = numerical
                                                    task_type_ids = torch.from_numpy((task_types_arr == "numerical").astype(np.int8)).to(device)
                                                    batch.batch["task_type_ids"] = task_type_ids
                                                    # Provide raw correctness component advantage for actor-side stats.
                                                    batch.batch["advantages_correctness_component"] = adv_c.detach()

                                                    # Apply the *current* stored weight to this batch.
                                                    adv_c_scaled = self._apply_correctness_adv_task_weight(
                                                        correctness_adv=adv_c,
                                                        task_type_ids=task_type_ids,
                                                    )

                                                # Log the task weights used for this batch (before update).
                                                # NOTE: 1-step lag - these are the weights computed from the previous steps.
                                                w_cat_used = float(self._correctness_adv_task_weights.get("categorical", 1.0))
                                                w_num_used = float(self._correctness_adv_task_weights.get("numerical", 1.0))
                                                metrics["training/correctness_adv/task_weight_used/categorical"] = w_cat_used
                                                metrics["training/correctness_adv/task_weight_used/numerical"] = w_num_used
                                                # Backward-compatible scalar (kept for historical dashboards).
                                                metrics["training/correctness_adv/reg_weight_used"] = float(self._correctness_adv_reg_weight)

                                                mode = str(self.config.algorithm.get("correctness_adv_task_weight_mode", "suppress_regression")).lower()
                                                if mode in (
                                                    "suppress_and_boost",
                                                    "boost_and_suppress",
                                                    "suppress_boost",
                                                    "boost_suppress",
                                                    "both",
                                                ):
                                                    mode_id = 2.0
                                                elif mode in (
                                                    "boost_classification",
                                                    "boost_categorical",
                                                    "upweight_categorical",
                                                    "upweight_classification",
                                                ):
                                                    mode_id = 1.0
                                                else:
                                                    mode_id = 0.0
                                                metrics["training/correctness_adv/reg_weight_mode"] = mode_id
                                            else:
                                                # Without task types we cannot apply per-task weighting.
                                                metrics["training/correctness_adv/reg_weight_used"] = float(self._correctness_adv_reg_weight)

                                        # 3) Weighted sum of advantages. Actor will do a single PPO clipping.
                                        adv_final = (lambda_s * adv_s) + (lambda_f * adv_f) + (lambda_c * adv_c_scaled)
                                        batch.batch["advantages"] = adv_final
                                        # For GRPO, verl sets returns == advantages; keep consistent.
                                        batch.batch["returns"] = adv_final

                                        # ------------------------------
                                        # W&B diagnostics (token-weighted, consistent with token-mean loss)
                                        #
                                        # Log:
                                        #   1) Raw component advantages A_s, A_f, A_c (split by task_type)
                                        #   2) Correctness advantage after regression down-weight (A_c_scaled)
                                        #   3) Lambda-weighted component contributions (λ_s A_s, λ_f A_f, λ_c A_c_scaled)
                                        #   4) Final combined advantage A_final
                                        #
                                        # We log mean(|A|) because GRPO advantages are approximately zero-mean
                                        # after whitening; abs-mean better reflects update magnitude.
                                        # ------------------------------
                                        with torch.no_grad():
                                            # Float mask so masked_mean works (and matches token-mean aggregation).
                                            resp_mask_f = response_mask.to(device=device).float()

                                            def _masked_abs_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
                                                return float(masked_mean(torch.abs(x), mask).item())

                                            # Prepare tensors to log.
                                            raw = {
                                                "structure": adv_s,
                                                "select": adv_f,
                                                "correctness": adv_c,
                                            }
                                            # correctness after EMA down-weight for numerical tasks (before λ_c)
                                            scaled = {"correctness": adv_c_scaled}
                                            weighted = {
                                                "structure": lambda_s * adv_s,
                                                "select": lambda_f * adv_f,
                                                "correctness": lambda_c * adv_c_scaled,
                                            }

                                            def _log_for_tag(tag: str, mask: torch.Tensor):
                                                # Raw component advantages
                                                for name, tens in raw.items():
                                                    metrics[f"training/adv_component/raw_abs_mean/{name}/{tag}"] = _masked_abs_mean(tens, mask)
                                                # Scaled correctness advantage (regression down-weight)
                                                for name, tens in scaled.items():
                                                    metrics[f"training/adv_component/scaled_abs_mean/{name}/{tag}"] = _masked_abs_mean(tens, mask)
                                                # Lambda-weighted component contributions
                                                for name, tens in weighted.items():
                                                    metrics[f"training/adv_component/weighted_abs_mean/{name}/{tag}"] = _masked_abs_mean(tens, mask)
                                                # Final combined advantage
                                                metrics[f"training/adv_component/final_abs_mean/{tag}"] = _masked_abs_mean(adv_final, mask)

                                            # Log for all samples in the batch.
                                            _log_for_tag("all", resp_mask_f)

                                            # Log split by task type if available.
                                            task_types = batch.non_tensor_batch.get("task_type")
                                            if task_types is not None:
                                                try:
                                                    task_types_arr = np.asarray(task_types, dtype=object)
                                                except Exception:
                                                    task_types_arr = None
                                                if task_types_arr is not None and task_types_arr.shape[0] == batch_size:
                                                    for tname in ("categorical", "numerical"):
                                                        tmask_1d = torch.as_tensor(task_types_arr == tname, device=device).float()
                                                        _log_for_tag(tname, resp_mask_f * tmask_1d[:, None])

                                            # Backward-compatible aggregate keys (kept for existing dashboards).
                                            metrics["training/adv_component/abs_mean/structure"] = metrics[
                                                "training/adv_component/raw_abs_mean/structure/all"
                                            ]
                                            metrics["training/adv_component/abs_mean/select"] = metrics["training/adv_component/raw_abs_mean/select/all"]
                                            metrics["training/adv_component/abs_mean/correctness"] = metrics[
                                                "training/adv_component/raw_abs_mean/correctness/all"
                                            ]
                                            metrics["training/adv_component/abs_mean/final"] = metrics["training/adv_component/final_abs_mean/all"]

                                            metrics["training/adv_component/lambda_s"] = float(lambda_s)
                                            metrics["training/adv_component/lambda_f"] = float(lambda_f)
                                            metrics["training/adv_component/lambda_c"] = float(lambda_c)

                        else:
                            # Vanilla verl behavior: compute a single advantage from total reward.
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                                config=self.config.algorithm,
                            )
                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                        # -----------------------------------------------------------------
                        # Correctness task-weight EMA update (clip-aware)
                        #
                        # The actor is the only place where we know the PPO ratio
                        # (log_prob - old_log_prob), so we compute clip-filtered
                        # mean(|A_correctness|) on the actor side and feed it back
                        # here to update the EMA and the task weight.
                        #
                        # NOTE: This introduces a 1-step lag: the weight updated from
                        # this batch will be applied starting from the next batch.
                        # -----------------------------------------------------------------
                        if bool(self.config.algorithm.get("enable_correctness_adv_reg_weight", True)):
                            abs_cat = actor_output_metrics.get(
                                "actor/correctness_adv/abs_mean_in_clip/categorical", None
                            )
                            abs_num = actor_output_metrics.get(
                                "actor/correctness_adv/abs_mean_in_clip/numerical", None
                            )
                            if abs_cat is not None or abs_num is not None:
                                _, reg_metrics, _ = self._update_correctness_adv_reg_weight(
                                    abs_mean_overrides={
                                        "categorical": abs_cat,
                                        "numerical": abs_num,
                                    }
                                )
                                metrics.update(reg_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
