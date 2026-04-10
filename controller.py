"""
controller.py
Converts activation scalar + threshold into clean trigger events.

Implements:
- Threshold crossing detection (activation > threshold)
- Refractory period (debounce): ignores re-triggers for N ms after a trigger
- Hysteresis: requires signal to drop below (threshold * release_ratio) before re-arming
"""

import time


class TriggerController:
    """
    Call update(activation) each frame.
    Returns True exactly once per muscle contraction.

    Args:
        threshold:        activation µV above which a trigger fires
        refractory_ms:    minimum ms between triggers (debounce)
        release_ratio:    signal must drop below threshold*release_ratio to re-arm
                          (hysteresis, prevents rapid re-fire on plateau signals)
    """

    def __init__(self, threshold: float = 50.0, refractory_ms: int = 300,
                 release_ratio: float = 0.7):
        self.threshold = threshold
        self.refractory_ms = refractory_ms
        self.release_ratio = release_ratio

        self._last_trigger_time: float = 0.0
        self._armed: bool = True        # ready to fire
        self._triggered: bool = False   # single-frame trigger flag

    def update(self, activation: float) -> bool:
        """
        Feed current activation value.
        Returns True on the frame a trigger fires, False otherwise.
        """
        now = time.monotonic()
        self._triggered = False

        refractory_elapsed = (now - self._last_trigger_time) * 1000 >= self.refractory_ms

        # Re-arm when signal drops below hysteresis level after a trigger
        if not self._armed:
            if activation < self.threshold * self.release_ratio:
                self._armed = True

        # Fire if armed, above threshold, and outside refractory period
        if self._armed and refractory_elapsed and activation >= self.threshold:
            self._triggered = True
            self._last_trigger_time = now
            self._armed = False  # disarm until signal drops

        return self._triggered

    def is_active(self, activation: float) -> bool:
        """True while activation is above threshold (for visual feedback)."""
        return activation >= self.threshold

    def reset(self):
        self._last_trigger_time = 0.0
        self._armed = True
        self._triggered = False

    def set_threshold(self, threshold: float):
        self.threshold = max(1.0, threshold)
