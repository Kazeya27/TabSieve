# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x

# -----------------------------------------------------------------------------
# Correctness task-weighting mode (dynamic EMA):
#   - suppress_regression  : down-weight numerical-task correctness advantage
#   - boost_classification : up-weight categorical-task correctness advantage
#   - suppress_and_boost   : do BOTH (suppress numerical + boost categorical)
#       using half-strength in log-space (eta/=2)
# Override by setting env var CORR_TASK_WEIGHT_MODE.
# -----------------------------------------------------------------------------
CORR_TASK_WEIGHT_MODE=${CORR_TASK_WEIGHT_MODE:-suppress_regression}
EXP_ID=${EXP_ID:-0731}


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.enable_correctness_adv_reg_weight=True \
    algorithm.correctness_adv_task_weight_mode=${CORR_TASK_WEIGHT_MODE} \
    algorithm.correctness_adv_reg_weight_ema_beta=0.9 \
    algorithm.correctness_adv_reg_weight_min=0.8 \
    algorithm.correctness_adv_reg_weight_max=1.0 \
    data.train_files=/workspace/wyy/data/rl/rl_samples_42k_filter-4k.parquet \
    data.val_files=/workspace/wyy/data/rl/rl_samples_42k_filter-4k.parquet \
    custom_reward_function.path=config/rl/reward.py \
    trainer.project_name=tabsieve \
    trainer.experiment_name="${EXP_ID}" \
    custom_reward_function.name=F_beta_reward \
    +custom_reward_function.reward_kwargs.lambda_c=0.7 \
    +custom_reward_function.reward_kwargs.lambda_f=0.2 \
    +custom_reward_function.reward_kwargs.alpha=3.0 \
    +custom_reward_function.reward_kwargs.beta=1.0 \
    data.train_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=5120 \
    actor_rollout_ref.rollout.max_num_batched_tokens=9216 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/workspace/wyy/ckpt/TabSieve/SFT \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model'] \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.68 \
    actor_rollout_ref.rollout.n=8 \
    +actor_rollout_ref.rollout.repetition_penalty=1.02 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.rollout_data_dir="/workspace/wyy/ckpt/TabSieve/RL/${EXP_ID}/rollouts" \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0.03 \
    trainer.logger=['console','wandb'] \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=157 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.default_local_dir="/workspace/wyy/ckpt/TabSieve/RL/${EXP_ID}" \
    trainer.total_epochs=4 $@

