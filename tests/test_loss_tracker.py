from desifm.training.loss_tracker import LossTracker


def test_window_mean_and_ema():
    t = LossTracker(window=3, ema_decay=0.5, max_loss=100.0)
    assert t.update(10.0)
    assert t.update(20.0)
    assert abs(t.window_mean() - 15.0) < 1e-6
    assert abs(t.window_median() - 15.0) < 1e-6
    assert t.ema is not None


def test_skip_outlier():
    t = LossTracker(window=3, max_loss=50.0)
    assert t.update(1.0)
    assert not t.update(1e6)
