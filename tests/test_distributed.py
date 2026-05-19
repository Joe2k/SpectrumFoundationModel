from desifm.training.distributed import is_main_process, unwrap
import torch.nn as nn


def test_is_main_and_unwrap():
    assert is_main_process(0)
    m = nn.Linear(2, 2)
    assert unwrap(m) is m
