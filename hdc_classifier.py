"""
hdc_classifier.py
Small hyperdimensional classifier for binary EMG state detection.

The classifier uses simple time-domain features from a short EMG window,
projects them into a high-dimensional bipolar space, and builds class
prototypes for REST and FLEX.
"""

from __future__ import annotations

import math
import numpy as np


class HDClassifier:
    def __init__(self, sampling_rate: int = 500, window_ms: int = 200,
                 dims: int = 4000, seed: int = 7):
        self.sampling_rate = sampling_rate
        self.window_ms = window_ms
        self.window_samples = max(16, int(sampling_rate * window_ms / 1000))
        self.dims = dims
        self.seed = seed

        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self._proj: np.ndarray | None = None
        self.prototype_rest: np.ndarray | None = None
        self.prototype_flex: np.ndarray | None = None

        self.decision_threshold: float = 0.65
        self.rest_score_p95: float = 0.0
        self.flex_score_p05: float = 1.0
        self.trained: bool = False

    def fit(self, rest_windows: list[np.ndarray], flex_windows: list[np.ndarray]) -> bool:
        if not rest_windows or not flex_windows:
            self.trained = False
            return False

        rest_features = np.vstack([self._extract_features(w) for w in rest_windows])
        flex_features = np.vstack([self._extract_features(w) for w in flex_windows])
        features = np.vstack([rest_features, flex_features])

        self.feature_mean = features.mean(axis=0)
        self.feature_std = np.maximum(features.std(axis=0), 1e-6)

        rng = np.random.default_rng(self.seed)
        self._proj = rng.choice((-1.0, 1.0), size=(self.dims, features.shape[1]))

        rest_hv = np.vstack([self._encode_from_features(f) for f in rest_features])
        flex_hv = np.vstack([self._encode_from_features(f) for f in flex_features])

        self.prototype_rest = self._normalize(rest_hv.sum(axis=0))
        self.prototype_flex = self._normalize(flex_hv.sum(axis=0))

        rest_scores = np.array([self.score_window(w) for w in rest_windows], dtype=float)
        flex_scores = np.array([self.score_window(w) for w in flex_windows], dtype=float)

        self.rest_score_p95 = float(np.percentile(rest_scores, 95))
        self.flex_score_p05 = float(np.percentile(flex_scores, 5))
        midpoint = 0.5 * (self.rest_score_p95 + self.flex_score_p05)
        self.decision_threshold = float(np.clip(midpoint, 0.55, 0.9))
        self.trained = True
        return True

    def score_window(self, window: np.ndarray) -> float:
        if not self.trained or self.prototype_rest is None or self.prototype_flex is None:
            return 0.0

        hv = self._encode(window)
        rest_sim = float(np.dot(hv, self.prototype_rest))
        flex_sim = float(np.dot(hv, self.prototype_flex))
        margin = flex_sim - rest_sim
        return 1.0 / (1.0 + math.exp(-4.0 * margin))

    def predict(self, window: np.ndarray) -> tuple[bool, float]:
        score = self.score_window(window)
        return score >= self.decision_threshold, score

    def summary(self) -> str:
        if not self.trained:
            return "Classifier not trained"
        return (
            f"REST p95 {self.rest_score_p95*100:.0f}%  |  "
            f"FLEX p05 {self.flex_score_p05*100:.0f}%  |  "
            f"threshold {self.decision_threshold*100:.0f}%"
        )

    def _encode(self, window: np.ndarray) -> np.ndarray:
        return self._encode_from_features(self._extract_features(window))

    def _encode_from_features(self, features: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None or self._proj is None:
            return np.zeros(self.dims, dtype=float)
        norm = (features - self.feature_mean) / self.feature_std
        hv = self._proj @ norm
        return np.where(hv >= 0.0, 1.0, -1.0)

    def _extract_features(self, window: np.ndarray) -> np.ndarray:
        sig = np.asarray(window, dtype=float)
        if sig.size == 0:
            return np.zeros(6, dtype=float)

        centered = sig - float(np.median(sig))
        abs_sig = np.abs(centered)
        diff = np.diff(centered) if centered.size > 1 else np.zeros(1)

        rms = math.sqrt(float(np.mean(centered ** 2)))
        waveform_len = float(np.sum(np.abs(diff)))
        zero_cross = float(np.mean(centered[:-1] * centered[1:] < 0)) if centered.size > 1 else 0.0

        return np.array([
            float(np.mean(abs_sig)),
            float(np.std(abs_sig)),
            float(np.max(abs_sig)),
            rms,
            waveform_len / max(1, centered.size),
            zero_cross,
        ], dtype=float)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-9:
            return vec.astype(float)
        return vec.astype(float) / norm
