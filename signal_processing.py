"""
signal_processing.py
Converts NEW raw EMG samples into a single activation scalar.

Pipeline: raw (µV) → IIR DC removal → abs() rectify → moving average → activation scalar

NOTE: only call push() with new samples from emg.pull().
Display data comes from emg.peek() — no display buffer here.
"""

import numpy as np
from collections import deque


class EMGProcessor:
    def __init__(self, sampling_rate: int = 500, smooth_ms: int = 150, channel: int = 0):
        self.sampling_rate = sampling_rate
        self.channel = channel
        self.smooth_ms = smooth_ms
        self.smooth_samples = max(1, int(sampling_rate * smooth_ms / 1000))
        self._rect_window: deque[float] = deque([0.0] * self.smooth_samples,
                                                maxlen=self.smooth_samples)
        self._activation = 0.0
        self._baseline = 0.0   # long-term IIR DC tracker

    def push(self, samples: np.ndarray):
        """
        Feed new samples (n_channels, n_new). May be empty (0 columns) — safe to call.

        DC removal uses a slow IIR baseline tracker (τ ≈ 5 s at 500 Hz).
        Do NOT use per-chunk median: a 10-sample flex chunk has median ≈ flex value,
        so after subtraction the activation collapses to ~0 µV.
        """
        if samples.shape[1] == 0:
            return
        ch = self.channel if self.channel < samples.shape[0] else 0
        sig = np.asarray(samples[ch], dtype=float)
        for v in sig:
            # α = 0.998 → time constant ≈ 500 samples ≈ 1 s  (fast enough to remove
            # slow electrode drift, slow enough not to eat the EMG envelope)
            self._baseline = 0.998 * self._baseline + 0.002 * float(v)
            self._rect_window.append(abs(float(v) - self._baseline))
        self._activation = float(np.mean(self._rect_window))

    def activation(self) -> float:
        return self._activation

    def set_channel(self, ch: int):
        self.channel = ch
        # Reset so old channel data doesn't bleed into new channel's activation
        self._rect_window = deque([0.0] * self.smooth_samples, maxlen=self.smooth_samples)
        self._activation = 0.0
        self._baseline = 0.0

    @staticmethod
    def compute_activation_from_window(display_data: np.ndarray, channel: int,
                                       smooth_samples: int) -> float:
        """
        One-shot activation from a display peek window.
        Used by the calibration background thread without touching the shared processor.

        Uses the same IIR-style approach: subtract a long-window mean as the DC estimate,
        then rectify. This avoids the per-chunk median bug.
        """
        if display_data.shape[1] == 0:
            return 0.0
        ch = channel if channel < display_data.shape[0] else 0
        window = np.asarray(display_data[ch, -smooth_samples:], dtype=float)
        # Use a longer-term mean from the full available buffer as the DC estimate
        long_window = np.asarray(display_data[ch], dtype=float)
        dc = float(np.mean(long_window)) if len(long_window) > 0 else 0.0
        return float(np.mean(np.abs(window - dc)))
