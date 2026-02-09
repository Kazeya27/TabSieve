# TabSieve Training Code

This repository contains the official training code for “TabSieve: Explicit In-Table Evidence Selection for Tabular Prediction.”

## Method
![method](./docs/method.jpg?raw=true)

## Training Framework
Training is implemented on top of the [VeRL](https://github.com/volcengine/verl) framework.

## Training Command
Run the following from the repository root:

```bash
cd train
bash config/run_qwen3-8b_f2-reward.sh
```

The script launches training via `python3 -m verl.trainer.main_ppo ...`.
