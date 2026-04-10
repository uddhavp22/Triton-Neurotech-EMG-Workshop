"""
reaction.py — Reaction time challenge state machine.
All rendering is in main.py (PyQt6 widgets).
"""
import time
import random


class ReactionLogic:
    MAX_DELAY_S = 4.0
    MIN_DELAY_S = 1.5

    def __init__(self):
        self.state: str = 'enter_name'
        self.name_input: str = ""
        self.leaderboard: list[tuple[str, float]] = []
        self._go_time: float = 0.0
        self._wait_end: float = 0.0
        self._result_ms: float = 0.0
        self._current_name: str = ""
        self._false_start: bool = False

    # ── Name entry ─────────────────────────────────────────────────────────────

    def add_char(self, c: str):
        if len(self.name_input) < 14 and c.isprintable():
            self.name_input += c

    def backspace(self):
        self.name_input = self.name_input[:-1]

    def submit_name(self):
        name = self.name_input.strip()
        if name:
            self._current_name = name
            self.name_input = ""
            self._start_round()

    # ── Round logic ────────────────────────────────────────────────────────────

    def _start_round(self):
        delay = random.uniform(self.MIN_DELAY_S, self.MAX_DELAY_S)
        self._wait_end = time.monotonic() + delay
        self._false_start = False
        self.state = 'waiting'

    def trigger_flex(self) -> bool:
        """Returns True if a valid (non-false-start) flex was recorded."""
        if self.state == 'waiting':
            self._false_start = True
            self.state = 'result'
            self._result_ms = -1.0
            return False
        elif self.state == 'go':
            self._result_ms = (time.monotonic() - self._go_time) * 1000
            self.leaderboard.append((self._current_name, self._result_ms))
            self.leaderboard.sort(key=lambda x: x[1])
            self.state = 'result'
            return True
        return False

    def update(self):
        """Call each frame; auto-transitions waiting → go."""
        if self.state == 'waiting' and time.monotonic() >= self._wait_end:
            self._go_time = time.monotonic()
            self.state = 'go'

    def next_participant(self):
        self.state = 'leaderboard'

    def new_participant(self):
        self.state = 'enter_name'
        self.name_input = ""

    def get_rank_label(self) -> str:
        if not self.leaderboard:
            return ""
        idx = next(
            (i for i, (n, ms) in enumerate(self.leaderboard)
             if n == self._current_name and abs(ms - self._result_ms) < 1),
            -1,
        )
        if idx == 0:  return "NEW RECORD!"
        if idx == 1:  return "2nd place"
        if idx == 2:  return "3rd place"
        if idx >= 3:  return f"#{idx + 1} place"
        return ""

    @property
    def current_name(self) -> str:
        return self._current_name

    @property
    def result_ms(self) -> float:
        return self._result_ms

    @property
    def false_start(self) -> bool:
        return self._false_start

    @property
    def time_until_go(self) -> float:
        """Seconds remaining before GO fires. Only valid in 'waiting' state."""
        return max(0.0, self._wait_end - time.monotonic())

    @property
    def wait_total(self) -> float:
        return self.MAX_DELAY_S - self.MIN_DELAY_S
