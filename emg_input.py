"""
emg_input.py — MindRove EMG input with threaded pull/peek buffering.

Preferred backend order:
1. LSL stream published by MindRove Connect
2. Direct MindRove SDK
3. Synthetic fallback

pull()  → returns only NEW samples since last call (used for activation processing)
peek(n) → returns last n samples from display buffer (used for plotting, non-destructive)

Threaded acquisition keeps the incoming stream fresh even when the UI renders
at 60 FPS, which reduces the "chunky" feel from polling hardware in the main loop.
"""

import os
import time
import threading
import numpy as np

DISPLAY_SECS = 5  # how many seconds to keep in the display ring buffer

try:
    from pylsl import StreamInlet, resolve_byprop, resolve_streams, proc_clocksync, proc_dejitter
    _LSL_AVAILABLE = True
except ImportError:
    _LSL_AVAILABLE = False

try:
    from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False


class EMGInput:
    def __init__(self, synthetic: bool = False, preferred_backend: str = "lsl"):
        self.synthetic = synthetic
        self.preferred_backend = preferred_backend

        self.board = None
        self.board_id = None
        self.emg_channels: list[int] = []
        self.sampling_rate: int = 500
        self.backend: str = "synthetic" if synthetic else "uninitialized"
        self.source_name: str = "synthetic"

        self._running = False
        self._acq_thread: threading.Thread | None = None

        self._disp_buf: np.ndarray | None = None
        self._disp_lock = threading.Lock()

        self._pending_buf: np.ndarray | None = None
        self._pending_lock = threading.Lock()

        # LSL state
        self._lsl_inlet = None

        # Synthetic state
        self._synth_t = 0.0

    # ── Public API ──────────────────────────────────────────────────────────────

    def start(self):
        if self.synthetic:
            self._start_synthetic()
            return

        started = False
        backends = ["lsl", "sdk"] if self.preferred_backend == "lsl" else ["sdk", "lsl"]

        for backend in backends:
            if backend == "lsl" and _LSL_AVAILABLE:
                started = self._start_lsl()
            elif backend == "sdk" and _SDK_AVAILABLE:
                started = self._start_sdk()
            if started:
                break

        if not started:
            if not _LSL_AVAILABLE:
                print("[EMGInput] pylsl not installed — cannot use LSL backend.")
            if not _SDK_AVAILABLE:
                print("[EMGInput] mindrove SDK not installed — cannot use direct backend.")
            print("[EMGInput] Falling back to synthetic mode.")
            self._start_synthetic()

    def pull(self) -> np.ndarray:
        """
        Returns ONLY new samples since last call as (n_channels, n_new).
        May return a 0-column array if no new data arrived.
        """
        with self._pending_lock:
            if self._pending_buf is None:
                return np.zeros((self.n_channels, 0))
            pending = self._pending_buf.copy()
            self._pending_buf = np.zeros((self.n_channels, 0))
        return pending

    def peek(self, n: int | None = None) -> np.ndarray:
        """
        Non-destructive read of the display buffer.
        Returns last n samples (all channels) as (n_channels, n).
        """
        with self._disp_lock:
            if self._disp_buf is None:
                return np.zeros((self.n_channels, n or 1))
            if n is None:
                return self._disp_buf.copy()
            return self._disp_buf[:, -n:].copy()

    def stop(self):
        self._running = False

        if self._acq_thread is not None and self._acq_thread.is_alive():
            self._acq_thread.join(timeout=0.5)

        if self.board is not None and self.backend == "sdk":
            try:
                self.board.stop_stream()
                self.board.release_session()
            except Exception:
                pass

        self._lsl_inlet = None
        print(f"[EMGInput] Stopped ({self.backend}).")

    @property
    def n_channels(self) -> int:
        return len(self.emg_channels)

    # ── Backend setup ───────────────────────────────────────────────────────────

    def _start_lsl(self) -> bool:
        try:
            stream = self._resolve_lsl_stream()
            if stream is None:
                print("[EMGInput] No MindRove LSL stream found.")
                return False

            self._lsl_inlet = StreamInlet(
                stream,
                max_buflen=DISPLAY_SECS + 1,
                processing_flags=proc_clocksync | proc_dejitter,
            )
            self.emg_channels = list(range(int(stream.channel_count())))
            self.sampling_rate = int(stream.nominal_srate()) if stream.nominal_srate() > 0 else 500
            self.synthetic = False
            self.backend = "lsl"
            self.source_name = f"{stream.name()} ({stream.type()})"

            self._init_buffers()
            self._running = True
            self._acq_thread = threading.Thread(target=self._lsl_loop, daemon=True)
            self._acq_thread.start()

            # Drain any already-buffered startup chunk.
            time.sleep(0.05)
            self.pull()
            print(
                f"[EMGInput] Connected via LSL — {self.n_channels} ch @ "
                f"{self.sampling_rate} Hz from {self.source_name}"
            )
            return True
        except Exception as e:
            print(f"[EMGInput] LSL connection failed: {e}")
            self._lsl_inlet = None
            return False

    def _start_sdk(self) -> bool:
        try:
            BoardShim.disable_board_logger()
            params = MindRoveInputParams()
            self.board_id = BoardIds.MINDROVE_WIFI_BOARD
            self.board = BoardShim(self.board_id, params)
            self.board.prepare_session()
            self.board.start_stream()
            self.emg_channels = list(BoardShim.get_emg_channels(self.board_id))
            self.sampling_rate = int(BoardShim.get_sampling_rate(self.board_id))
            self.synthetic = False
            self.backend = "sdk"
            self.source_name = "MindRove SDK"

            self._init_buffers()
            self._running = True
            self._acq_thread = threading.Thread(target=self._sdk_loop, daemon=True)
            self._acq_thread.start()

            time.sleep(0.05)
            self.pull()
            print(f"[EMGInput] Connected via SDK — {self.n_channels} ch @ {self.sampling_rate} Hz")
            return True
        except Exception as e:
            print(f"[EMGInput] SDK connection failed: {e}")
            if self.board is not None:
                try:
                    self.board.release_session()
                except Exception:
                    pass
            self.board = None
            return False

    def _start_synthetic(self):
        self.emg_channels = list(range(4))
        self.sampling_rate = 500
        self.synthetic = True
        self.backend = "synthetic"
        self.source_name = "Synthetic"

        self._init_buffers()
        self._running = True
        self._acq_thread = threading.Thread(target=self._synth_loop, daemon=True)
        self._acq_thread.start()
        print(f"[EMGInput] Synthetic — {self.n_channels} ch @ {self.sampling_rate} Hz")

    def _init_buffers(self):
        n = DISPLAY_SECS * self.sampling_rate
        with self._disp_lock:
            self._disp_buf = np.zeros((self.n_channels, n))
        with self._pending_lock:
            self._pending_buf = np.zeros((self.n_channels, 0))

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _append_chunk(self, new_data: np.ndarray):
        if new_data.size == 0:
            return

        n = new_data.shape[1]
        with self._pending_lock:
            if self._pending_buf is not None:
                self._pending_buf = np.hstack([self._pending_buf, new_data])

        with self._disp_lock:
            if self._disp_buf is None:
                return
            if n >= self._disp_buf.shape[1]:
                self._disp_buf[:, :] = new_data[:, -self._disp_buf.shape[1]:]
            else:
                self._disp_buf = np.roll(self._disp_buf, -n, axis=1)
                self._disp_buf[:, -n:] = new_data

    def _resolve_lsl_stream(self):
        preferred_type = os.getenv("MINDROVE_LSL_TYPE", "EMG")
        preferred_name = os.getenv("MINDROVE_LSL_NAME", "").strip().lower()
        preferred_source = os.getenv("MINDROVE_LSL_SOURCE_ID", "").strip().lower()

        candidates = []
        try:
            candidates.extend(resolve_byprop("type", preferred_type, timeout=1.5))
        except Exception:
            pass

        if not candidates:
            try:
                candidates.extend(resolve_streams(wait_time=1.5))
            except Exception:
                return None

        if not candidates:
            return None

        def score(stream):
            name = stream.name().lower()
            stype = stream.type().lower()
            source = stream.source_id().lower()
            score_val = 0
            if "mindrove" in name or "mindrove" in source:
                score_val += 8
            if stype == preferred_type.lower():
                score_val += 4
            if stream.channel_count() >= 4:
                score_val += 2
            if stream.nominal_srate() > 0:
                score_val += 1
            if preferred_name and preferred_name in name:
                score_val += 20
            if preferred_source and preferred_source in source:
                score_val += 20
            return score_val

        return max(candidates, key=score)

    def _lsl_loop(self):
        idle_sleep = 0.002
        while self._running and self._lsl_inlet is not None:
            try:
                chunk, _timestamps = self._lsl_inlet.pull_chunk(timeout=0.05, max_samples=64)
            except Exception:
                time.sleep(0.05)
                continue

            if not chunk:
                time.sleep(idle_sleep)
                continue

            arr = np.asarray(chunk, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            else:
                arr = arr.T

            if arr.shape[0] >= self.n_channels:
                self._append_chunk(arr[:self.n_channels, :])

    def _sdk_loop(self):
        idle_sleep = 0.005
        while self._running and self.board is not None:
            try:
                raw = self.board.get_board_data()
            except Exception:
                time.sleep(0.05)
                continue

            if raw.shape[1] == 0:
                time.sleep(idle_sleep)
                continue

            valid = [ch for ch in self.emg_channels if ch < raw.shape[0]]
            if valid:
                self._append_chunk(raw[valid, :])

    def _synth_loop(self):
        chunk = 10
        dt = chunk / self.sampling_rate
        while self._running:
            t = self._synth_t
            self._synth_t += dt

            # Baseline noise ~8µV; burst every 5s to simulate flex.
            burst = 0.0
            phase = t % 6.0
            if 2.5 < phase < 3.0:
                burst = 200.0

            data = np.random.randn(4, chunk) * 8.0
            data[0] += burst * np.clip(np.random.randn(chunk), 0, None)
            data[1] += burst * 0.3 * np.clip(np.random.randn(chunk), 0, None)

            self._append_chunk(data)
            time.sleep(dt)
