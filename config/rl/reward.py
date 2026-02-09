"""
Custom reward helpers for RL runs.

The base_reward function combines two signals:
- Structure reward: output must contain exactly one <select>...</select> pair.
- Correctness reward: parsed from \box{answer}; categorical requires exact
  match; numerical uses an exponential decay of MAPE.
"""

import math
import re
from typing import Optional
from pdb import set_trace

_BOX_PATTERN = re.compile(r'\\boxed\s*\{\s*([^}]*)\s*\}')
_SELECT_PATTERN = re.compile(r"<select>\s*(?P<nums>\d+(?:\s*,\s*\d+)*)?\s*</select>")


def _extract_box_answer(text: str) -> Optional[str]:
    """Return the first answer captured in \\box{...}, or None if missing."""
    match = _BOX_PATTERN.search(text)
    return match.group(1).strip() if match else None


def _structure_reward(text: str) -> float:
    """
    Give 1 only when there is a parsable \\box{...} answer AND exactly one
    <select>...</select> pair; otherwise 0.
    """
    if _extract_box_answer(text) is None:
        return 0.0
    matches = _SELECT_PATTERN.findall(text)
    return 1.0 if len(matches) == 1 else 0.0


def _numerical_reward(pred_str: str, gt_str: str, alpha: float) -> float:
    """
    Compute exp(-alpha * MAPE) for numerical tasks. Returns 0 on parse failure.
    """
    mape = _numerical_mape(pred_str, gt_str)
    if mape is None:
        return 0.0
    return math.exp(-alpha * mape)


def _numerical_mape(pred_str: str, gt_str: str) -> Optional[float]:
    """Return MAPE for numerical tasks. Returns None on parse failure."""
    try:
        pred_val = float(pred_str)
        gt_val = float(gt_str)
    except (TypeError, ValueError):
        return None

    denom = abs(gt_val) + 1e-8  # avoid division by zero
    return abs(pred_val - gt_val) / denom


def _parse_select_indices(text: str) -> Optional[set[int]]:
    """
    Extract integer indices from a single <select>...</select> block.
    Returns None on any parsing failure.
    """
    matches = _SELECT_PATTERN.findall(text)
    # set_trace()
    if len(matches) != 1:
        return None

    tokens = re.split(r"[\\s,，]+", matches[0].strip())
    indices = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        try:
            indices.append(int(token))
        except ValueError:
            return None
    return set(indices)


def _coerce_to_index_set(values) -> set[int]:
    """Convert raw chosen/rejected payloads into a set of ints."""
    if values is None:
        return set()
    if isinstance(values, str):
        values = [v for v in re.split(r"[\\s,，]+", values) if v]
    elif not isinstance(values, (list, tuple, set)):
        values = [values]

    indices: set[int] = set()
    for item in values:
        if isinstance(item, (list, tuple, set)):
            indices.update(_coerce_to_index_set(item))
            continue
        if item is None:
            continue
        try:
            idx = int(str(item).strip())
        except (TypeError, ValueError):
            continue
        indices.add(idx)
    return indices


def _compute_f_beta(pred: set[int], pos: set[int], neg: set[int], beta: float) -> float:
    """
    Compute F_beta for the positive class.
    Treat predictions outside the known positive/negative pool as false positives.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    if not pos:
        return 1.0 if len(pred) == 0 else 0.0

    tp = len(pred & pos)
    fp = len(pred & neg)
    # fp_unknown = len(pred - (pos | neg))
    # fp = fp_known + fp_unknown
    fn = len(pos - pred)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision == 0.0 or recall == 0.0:
        return 0.0

    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    return (1 + beta_sq) * precision * recall / denom if denom else 0.0


def base_reward(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: Optional[dict] = None,
    lambda_s: float = 0.1,
    lambda_c: float = 0.9,
    alpha: float = 4.0,
):
    """
    Compute reward with structure + correctness components.

    Signature follows default_compute_score: (data_source, solution_str, ground_truth, extra_info, ...).
    alpha can be passed via reward_kwargs.
    """
    _ = data_source  # unused but kept for signature compatibility
    task_type = None
    if isinstance(extra_info, dict):
        task_type = extra_info.get("task_type")
    if task_type is None:
        # Default to categorical if task_type is not provided
        task_type = "categorical"
    structure_reward = _structure_reward(solution_str)
    answer = _extract_box_answer(solution_str)

    correctness_reward = 0.0
    mape = float("nan")
    if answer is not None:
        if task_type == "categorical":
            correctness_reward = 1.0 if answer == str(ground_truth).strip() else 0.0
        elif task_type == "numerical":
            mape_val = _numerical_mape(answer, ground_truth)
            if mape_val is None:
                correctness_reward = 0.0
            else:
                correctness_reward = math.exp(-alpha * mape_val)
                mape = float(mape_val)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")
    # print(f"pred={answer}, ground_truth={ground_truth}")
    total = lambda_s * structure_reward + lambda_c * correctness_reward
    return {
        "score": float(total),
        "structure_reward": float(structure_reward),
        "correctness_reward": float(correctness_reward),
        "task_type": task_type,
        "mape": float(mape),
    }


def F_beta_reward(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: Optional[dict] = None,
    alpha: float = 4.0,
    beta: float = 2.0,
    lambda_s: float = 0.1,
    lambda_f: float = 0.2,
    lambda_c: float = 0.7,
):
    """
    Reward based on parsing <select>...</select> indices and computing F_beta
    against positives in extra_info["chosen"] (negatives in extra_info["rejected"]).

    reward_select is the positive-class F_beta; score blends structure and F_beta.
    """
    task_type = None
    if isinstance(extra_info, dict):
        task_type = extra_info.get("task_type")
    if task_type is None:
        # Default to categorical if task_type is not provided
        task_type = "categorical"
    structure_reward = _structure_reward(solution_str)
    answer = _extract_box_answer(solution_str)
    
    
    predicted_indices = _parse_select_indices(solution_str)
    structure_reward = structure_reward if predicted_indices is not None else 0.0

    chosen = set()
    rejected = set()
    if isinstance(extra_info, dict):
        chosen = _coerce_to_index_set(extra_info.get("chosen"))
        rejected = _coerce_to_index_set(extra_info.get("rejected"))

    reward_select = 0.0
    if structure_reward:
        reward_select = _compute_f_beta(predicted_indices, chosen, rejected, beta)
    correctness_reward = 0.0
    mape = float("nan")
    if answer is not None:
        if task_type == "categorical":
            correctness_reward = 1.0 if answer == str(ground_truth).strip() else 0.0
        elif task_type == "numerical":
            mape_val = _numerical_mape(answer, ground_truth)
            if mape_val is None:
                correctness_reward = 0.0
            else:
                correctness_reward = math.exp(-alpha * mape_val)
                mape = float(mape_val)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")
        
    total = lambda_s * structure_reward + lambda_f * reward_select + lambda_c * correctness_reward
    # print(f"select_reward={reward_select}, pred={answer}, ground_truth={ground_truth}")
    return {
        "score": float(total),
        "structure_reward": float(structure_reward),
        "reward_select": float(reward_select),
        "correctness_reward": float(correctness_reward),
        "task_type": task_type,
        "mape": float(mape),
    }

def _ablation_structure_reward(text: str) -> float:
    """
    Give 1 only when there is a parsable \\box{...} answer AND exactly one
    <select>...</select> pair; otherwise 0.
    """
    if _extract_box_answer(text) is None:
        return 0.0
    return 1.0

def ablation_reward(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: Optional[dict] = None,
    lambda_f: float = 0.1,
    lambda_c: float = 0.9,
    alpha: float = 3.0,
):
    """
    Compute reward with structure + correctness components.

    Signature follows default_compute_score: (data_source, solution_str, ground_truth, extra_info, ...).
    alpha can be passed via reward_kwargs.
    """
    _ = data_source  # unused but kept for signature compatibility
    task_type = None
    if isinstance(extra_info, dict):
        task_type = extra_info.get("task_type")
    if task_type is None:
        # Default to categorical if task_type is not provided
        task_type = "categorical"
    structure_reward = _ablation_structure_reward(solution_str)
    answer = _extract_box_answer(solution_str)

    correctness_reward = 0.0
    mape = float("nan")
    if answer is not None:
        if task_type == "categorical":
            correctness_reward = 1.0 if answer == str(ground_truth).strip() else 0.0
        elif task_type == "numerical":
            mape_val = _numerical_mape(answer, ground_truth)
            if mape_val is None:
                correctness_reward = 0.0
            else:
                correctness_reward = math.exp(-alpha * mape_val)
                mape = float(mape_val)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")
    # print(f"pred={answer}, ground_truth={ground_truth}")
    total = lambda_f * structure_reward + lambda_c * correctness_reward
    return {
        "score": float(total),
        "structure_reward": float(structure_reward),
        "correctness_reward": float(correctness_reward),
        "task_type": task_type,
        "mape": float(mape),
    }

if __name__ == "__main__":
    print(F_beta_reward("1", "<think> xxx <select> </select> ddd </think> \\boxed{3}", 2, {"task_type": "numerical", "chosen": [], "rejected":[3,4]}))

