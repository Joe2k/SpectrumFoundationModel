"""Running loss statistics for stable logging and checkpointing."""

from __future__ import annotations

from collections import deque


class LossTracker:
    def __init__(self, window: int = 10, ema_decay: float = 0.98):
        self.window: deque[float] = deque(maxlen=window)
        self.ema_decay = ema_decay
        self.ema: float | None = None

    def update(self, loss: float) -> None:
        self.window.append(loss)
        if self.ema is None:
            self.ema = loss
        else:
            self.ema = self.ema_decay * self.ema + (1.0 - self.ema_decay) * loss

    def window_mean(self) -> float:
        if not self.window:
            return float("inf")
        return sum(self.window) / len(self.window)
