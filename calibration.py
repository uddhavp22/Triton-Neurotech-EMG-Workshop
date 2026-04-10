"""
calibration.py
Records rest and flex windows, computes a suggested threshold.

Suggested threshold uses robust activation percentiles from REST and FLEX.
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
        self.activation_gate: float = 15.0
        self.rest_p95: float = 0.0
        self.rest_p99: float = 0.0
        self.flex_p25: float = 0.0
        self.flex_p50: float = 0.0
        self.quality_margin: float = 0.0

    def record_rest(self, activations: np.ndarray):
        """Store rest-phase activation values."""
        self.rest_values = activations
        self.rest_mean = float(np.mean(activations))
        self.rest_std = float(np.std(activations))
        self.rest_p95 = float(np.percentile(activations, 95))
        self.rest_p99 = float(np.percentile(activations, 99))

    def record_flex(self, activations: np.ndarray):
        """Store flex-phase activation values."""
        self.flex_values = activations
        self.flex_mean = float(np.mean(activations))
        self.flex_std = float(np.std(activations))
        self.flex_p25 = float(np.percentile(activations, 25))
        self.flex_p50 = float(np.percentile(activations, 50))

    def compute_threshold(self, sensitivity: float = 0.4) -> float:
        """
        Compute suggested threshold.
        sensitivity in [0,1]: 0 = triggers very easily, 1 = requires hard flex.
        """
        if self.rest_values is None or self.flex_values is None:
            return self.threshold
        noise_floor = max(self.rest_p99, self.rest_mean + 2.5 * self.rest_std)
        self.activation_gate = max(10.0, self.rest_p95)
        self.quality_margin = self.flex_p25 - noise_floor

        if self.flex_p25 <= noise_floor:
            # Weak separation: bias low enough to still detect, but stay above rest peaks.
            self.threshold = noise_floor + max(5.0, 0.15 * max(noise_floor, 1.0))
        else:
            midpoint = noise_floor + (self.flex_p25 - noise_floor) * (0.35 + 0.25 * sensitivity)
            self.threshold = min(midpoint, self.flex_p50)
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

    def reset(self):
        self.rest_values = None
        self.flex_values = None
        self.rest_mean = 0.0
        self.rest_std = 0.0
        self.flex_mean = 0.0
        self.flex_std = 0.0
        self.threshold = 50.0
        self.activation_gate = 15.0
        self.rest_p95 = 0.0
        self.rest_p99 = 0.0
        self.flex_p25 = 0.0
        self.flex_p50 = 0.0
        self.quality_margin = 0.0
