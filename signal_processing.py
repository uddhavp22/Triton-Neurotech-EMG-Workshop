"""
signal_processing.py
Converts NEW raw EMG samples into a single activation scalar.

Pipeline: raw (µV) → abs() rectify → moving average → activation scalar

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

    def push(self, samples: np.ndarray):
        """
        Feed new samples (n_channels, n_new). May be empty (0 columns) — safe to call.
        """
        if samples.shape[1] == 0:
            return
        ch = self.channel if self.channel < samples.shape[0] else 0
        sig = np.asarray(samples[ch], dtype=float)
        centered = sig - float(np.median(sig))
        for v in centered:
            self._rect_window.append(abs(float(v)))
        self._activation = float(np.mean(self._rect_window))

    def activation(self) -> float:
        return self._activation

    def set_channel(self, ch: int):
        self.channel = ch
        # Reset window so old channel data doesn't bleed in
        self._rect_window = deque([0.0] * self.smooth_samples, maxlen=self.smooth_samples)
        self._activation = 0.0

    @staticmethod
    def compute_activation_from_window(display_data: np.ndarray, channel: int,
                                       smooth_samples: int) -> float:
        """
        One-shot activation from a display peek window.
        Used by calibration thread without touching the shared processor.
        """
        if display_data.shape[1] == 0:
            return 0.0
        ch = channel if channel < display_data.shape[0] else 0
        window = np.asarray(display_data[ch, -smooth_samples:], dtype=float)
        sig = np.abs(window - float(np.median(window)))
        return float(np.mean(sig))
