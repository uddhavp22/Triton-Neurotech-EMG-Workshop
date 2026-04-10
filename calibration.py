"""
calibration.py
Records rest and flex windows, computes a suggested threshold.

Suggested threshold blends two guards:
- noise floor protection: rest_mean + 3 * rest_std
- separation target: rest_mean + (flex_mean - rest_mean) * sensitivity
"""

import numpy as np


class Calibration:
    def __init__(self):
        self.rest_values: np.ndarray | None = None
        self.flex_values: np.ndarray | None = None
        self.rest_mean: float = 0.0
        self.rest_std: float = 0.0
        self.flex_mean: float = 0.0
        self.flex_std: float = 0.0
        self.threshold: float = 50.0  # default fallback µV

    def record_rest(self, activations: np.ndarray):
        """Store rest-phase activation values."""
        self.rest_values = activations
        self.rest_mean = float(np.mean(activations))
        self.rest_std = float(np.std(activations))

    def record_flex(self, activations: np.ndarray):
        """Store flex-phase activation values."""
        self.flex_values = activations
        self.flex_mean = float(np.mean(activations))
        self.flex_std = float(np.std(activations))

    def compute_threshold(self, sensitivity: float = 0.4) -> float:
        """
        Compute suggested threshold.
        sensitivity in [0,1]: 0 = triggers very easily, 1 = requires hard flex.
        """
        if self.rest_values is None or self.flex_values is None:
            return self.threshold
        gap = self.flex_mean - self.rest_mean
        noise_floor = self.rest_mean + 3.0 * self.rest_std
        if gap <= 0:
            # Flex not stronger than rest — rely on the rest noise floor.
            self.threshold = noise_floor + 8.0
        else:
            separation_target = self.rest_mean + gap * sensitivity
            self.threshold = max(noise_floor, separation_target)
        return self.threshold

    def is_calibrated(self) -> bool:
        return self.rest_values is not None and self.flex_values is not None

    def summary(self) -> str:
        if not self.is_calibrated():
            return "Not calibrated"
        return (
            f"Rest: {self.rest_mean:.1f} µV  |  "
            f"Flex: {self.flex_mean:.1f} µV  |  "
            f"Threshold: {self.threshold:.1f} µV"
        )
