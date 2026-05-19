from desifm.training.distributed import all_ranks_agree_skip
import torch


def test_all_ranks_agree_skip_single_process():
    assert all_ranks_agree_skip(False, torch.device("cpu")) is False
    assert all_ranks_agree_skip(True, torch.device("cpu")) is True
