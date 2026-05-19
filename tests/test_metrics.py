import torch
from desifm.training.metrics import accuracy, masked_spec_accuracy


def test_accuracy_perfect():
    B, T, V = 2, 5, 32
    targets = torch.randint(0, V, (B, T))
    logits = torch.nn.functional.one_hot(targets, V).float() * 10
    m = accuracy(logits, targets)
    assert m["overall"] == 1.0


def test_masked_spec_accuracy_selective():
    B, T, V = 1, 4, 16
    targets = torch.tensor([[10, 11, 12, 13]])
    logits = torch.zeros(B, T, V)
    for t in range(T):
        logits[0, t, targets[0, t]] = 5.0
    mask_pos = torch.tensor([[True, False, True, False]])
    acc = masked_spec_accuracy(logits, targets, mask_pos)
    assert acc == 1.0
