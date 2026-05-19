from desifm.training.loss_tracker import LossTracker


def test_window_mean_and_ema():
    t = LossTracker(window=3, ema_decay=0.5)
    t.update(10.0)
    t.update(20.0)
    assert abs(t.window_mean() - 15.0) < 1e-6
    assert t.ema is not None
