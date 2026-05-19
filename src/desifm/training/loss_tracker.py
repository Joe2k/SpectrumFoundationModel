"""Running loss statistics for stable logging and checkpointing."""

from __future__ import annotations

import math
from collections import deque


class LossTracker:
    def __init__(self, window: int = 10, ema_decay: float = 0.98, max_loss: float = 50.0):
        self.window: deque[float] = deque(maxlen=window)
        self.ema_decay = ema_decay
        self.max_loss = max_loss
        self.ema: float | None = None

    def update(self, loss: float) -> bool:
        """Record loss if finite and below max_loss. Returns False to skip the step."""
        if not math.isfinite(loss) or loss > self.max_loss:
            return False
        self.window.append(loss)
        if self.ema is None:
            self.ema = loss
        else:
            self.ema = self.ema_decay * self.ema + (1.0 - self.ema_decay) * loss
        return True

    def window_mean(self) -> float:
        if not self.window:
            return float("inf")
        return sum(self.window) / len(self.window)

    def window_median(self) -> float:
        if not self.window:
            return float("inf")
        s = sorted(self.window)
        n = len(s)
        mid = n // 2
        if n % 2:
            return s[mid]
        return 0.5 * (s[mid - 1] + s[mid])
