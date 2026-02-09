# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.utils.debug import marked_timer
from verl.utils.torch_functional import masked_mean


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

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

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = self.global_steps in (self.config.trainer.profile_steps or [])
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result["reward_extra_info"]
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch["token_level_scores"].sum(dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items() if std > 0 or len(prompt_uid2metric_vals[uid]) == 1]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                continue
                            else:
                                raise ValueError(f"{num_gen_batches=} >= {max_num_gen_batches=}." + " Generated too many. Please check if your data are too difficult." + " You could also try set max_num_gen_batches=0 to enable endless trials.")
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # === Updating ===

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

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        # -----------------------------------------------------------------
                        # Component-wise GRPO advantages (structure / select / correctness)
                        # + weighted sum of advantages + single PPO clipping ("sum-then-clip").
                        #
                        # NOTE: RayDAPOTrainer overrides `fit()`, so we replicate the same logic
                        # as in `verl/trainer/ppo/ray_trainer.py` here.
                        #
                        # Enabled by:
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

                                custom_reward_cfg = self.config.get("custom_reward_function") or {}
                                reward_kwargs = custom_reward_cfg.get("reward_kwargs") or {}
                                lambda_s = float(reward_kwargs.get("lambda_s", 0.1))
                                lambda_f = float(reward_kwargs.get("lambda_f", 0.0))
                                lambda_c = float(reward_kwargs.get("lambda_c", 0.9))

                                response_mask = batch.batch.get("response_mask")
                                if response_mask is None:
                                    response_mask = compute_response_mask(batch)
                                response_lengths = response_mask.sum(-1).long()

                                if torch.any(response_lengths <= 0):
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
                                        token_rewards = torch.zeros_like(batch.batch["token_level_rewards"], dtype=torch.float32)
                                        per_sample = torch.as_tensor(per_sample_arr, dtype=torch.float32, device=device)
                                        token_rewards[batch_idx, response_lengths - 1] = per_sample
                                        return token_rewards

                                    def _build_component_batch(token_rewards: torch.Tensor) -> DataProto:
                                        tensors = {
                                            "token_level_rewards": token_rewards,
                                            "token_level_scores": token_rewards,
                                            "response_mask": response_mask,
                                        }
                                        if self.config.actor_rollout_ref.rollout.multi_turn.enable and "loss_mask" in batch.batch:
                                            tensors["loss_mask"] = batch.batch["loss_mask"]
                                        if "reward_baselines" in batch.batch:
                                            tensors["reward_baselines"] = batch.batch["reward_baselines"]

                                        non_tensors = {}
                                        if "uid" in batch.non_tensor_batch:
                                            non_tensors["uid"] = batch.non_tensor_batch["uid"]
                                        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

                                    def _compute_component_adv(per_sample_arr: np.ndarray):
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

                                    adv_s = _compute_component_adv(structure_arr)
                                    adv_f = _compute_component_adv(select_arr)
                                    adv_c = _compute_component_adv(correctness_arr)

                                    if adv_s is None or adv_f is None or adv_c is None:
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
                                        adv_c_scaled = adv_c
                                        if enable_reg_weight:
                                            task_types = batch.non_tensor_batch.get("task_type")
                                            if task_types is not None:
                                                _, reg_metrics, scaled = self._update_correctness_adv_reg_weight(
                                                    correctness_adv=adv_c,
                                                    response_mask=response_mask,
                                                    task_types=task_types,
                                                )
                                                metrics.update(reg_metrics)
                                                if scaled is not None:
                                                    adv_c_scaled = scaled
                                            else:
                                                metrics["training/correctness_adv/reg_weight"] = 1.0

                                        adv_final = (lambda_s * adv_s) + (lambda_f * adv_f) + (lambda_c * adv_c_scaled)
                                        batch.batch["advantages"] = adv_final
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
                                            resp_mask_f = response_mask.to(device=device).float()

                                            def _masked_abs_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
                                                return float(masked_mean(torch.abs(x), mask).item())

                                            raw = {
                                                "structure": adv_s,
                                                "select": adv_f,
                                                "correctness": adv_c,
                                            }
                                            scaled = {"correctness": adv_c_scaled}
                                            weighted = {
                                                "structure": lambda_s * adv_s,
                                                "select": lambda_f * adv_f,
                                                "correctness": lambda_c * adv_c_scaled,
                                            }

                                            def _log_for_tag(tag: str, mask: torch.Tensor):
                                                for name, tens in raw.items():
                                                    metrics[f"training/adv_component/raw_abs_mean/{name}/{tag}"] = _masked_abs_mean(tens, mask)
                                                for name, tens in scaled.items():
                                                    metrics[f"training/adv_component/scaled_abs_mean/{name}/{tag}"] = _masked_abs_mean(tens, mask)
                                                for name, tens in weighted.items():
                                                    metrics[f"training/adv_component/weighted_abs_mean/{name}/{tag}"] = _masked_abs_mean(tens, mask)
                                                metrics[f"training/adv_component/final_abs_mean/{tag}"] = _masked_abs_mean(adv_final, mask)

                                            _log_for_tag("all", resp_mask_f)

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
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

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

                progress_bar.update(1)
                self.global_steps += 1
