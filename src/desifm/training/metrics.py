"""Training and evaluation metrics."""

from __future__ import annotations

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    valid = targets != -100
    if valid.sum() == 0:
        return {"overall": 0.0, "z": 0.0, "spec": 0.0}
    overall = (pred[valid] == targets[valid]).float().mean().item()
    z_ok = (pred[:, 0] == targets[:, 0]).float().mean().item()
    spec_mask = targets[:, 1:] != -100
    if spec_mask.any():
        spec_ok = (pred[:, 1:][spec_mask] == targets[:, 1:][spec_mask]).float().mean().item()
    else:
        spec_ok = 0.0
    return {"overall": overall, "z": z_ok, "spec": spec_ok}


def masked_spec_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, mask_pos: torch.Tensor | None
) -> float:
    if mask_pos is None or not mask_pos.any():
        return float("nan")
    pred = logits.argmax(dim=-1)
    correct, total = 0, 0
    for b in range(targets.shape[0]):
        for t in range(mask_pos.shape[1]):
            if mask_pos[b, t]:
                tgt_i = 1 + t
                if pred[b, tgt_i] == targets[b, tgt_i]:
                    correct += 1
                total += 1
    return correct / max(total, 1)
