"""
reaction.py
Reaction time challenge: wait for GO cue, flex as fast as possible.
Tracks per-participant times and shows a leaderboard.
"""

import time
import random
import pygame


BG       = ( 11,  15,  26)
PANEL    = ( 20,  27,  46)
PANEL2   = ( 28,  38,  62)
BORDER   = ( 42,  58,  94)
TEXT_C   = (232, 236, 248)
DIM_C    = ( 90, 106, 148)
GO_C     = (  0, 210, 110)
WAIT_C   = (255,  72,  72)
ACCENT   = ( 77, 159, 255)
GOLD     = (255, 210,  50)
SILVER   = (192, 192, 192)
BRONZE   = (205, 127,  50)

SCREEN_W, SCREEN_H = 900, 600


class ReactionChallenge:
    """
    States:
      'enter_name'  - type participant name
      'waiting'     - random delay before GO
      'go'          - GO cue shown, waiting for flex
      'result'      - show reaction time
      'leaderboard' - show top scores
    """

    MAX_DELAY_S = 4.0
    MIN_DELAY_S = 1.5

    def __init__(self, surf: pygame.Surface, font_large, font_med, font_small):
        self.surf = surf
        self.font_large = font_large
        self.font_med = font_med
        self.font_small = font_small

        self.state = 'enter_name'
        self.name_input = ""
        self.leaderboard: list[tuple[str, float]] = []  # (name, ms)
        self._go_time: float = 0.0
        self._wait_end: float = 0.0
        self._result_ms: float = 0.0
        self._current_name: str = ""
        self._false_start: bool = False
        self._dots_frame = 0

    def handle_key(self, event: pygame.event.Event):
        """Process keyboard input for name entry."""
        if self.state != 'enter_name':
            return
        if event.key == pygame.K_RETURN and self.name_input.strip():
            self._current_name = self.name_input.strip()[:14]
            self.name_input = ""
            self._start_round()
        elif event.key == pygame.K_BACKSPACE:
            self.name_input = self.name_input[:-1]
        elif event.unicode.isprintable() and len(self.name_input) < 14:
            self.name_input += event.unicode

    def _start_round(self):
        delay = random.uniform(self.MIN_DELAY_S, self.MAX_DELAY_S)
        self._wait_end = time.monotonic() + delay
        self._false_start = False
        self.state = 'waiting'

    def trigger_flex(self) -> bool:
        """
        Called when EMG trigger fires. Returns True if it counted as a valid response.
        """
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

    def run_frame(self, activation: float, threshold: float) -> str:
        """Update logic, draw frame, return state."""
        self._dots_frame += 1

        # Auto-transition from waiting to go
        if self.state == 'waiting' and time.monotonic() >= self._wait_end:
            self._go_time = time.monotonic()
            self.state = 'go'

        self._draw(activation, threshold)
        return self.state

    def next_participant(self):
        """Called after showing result — go to leaderboard or next name."""
        self.state = 'leaderboard'

    def new_participant(self):
        """Start fresh for next person."""
        self.state = 'enter_name'
        self.name_input = ""

    def _draw(self, activation: float, threshold: float):
        s = self.surf
        s.fill(BG)

        if self.state == 'enter_name':
            self._draw_name_entry()
        elif self.state == 'waiting':
            self._draw_waiting()
        elif self.state == 'go':
            self._draw_go()
        elif self.state == 'result':
            self._draw_result()
        elif self.state == 'leaderboard':
            self._draw_leaderboard()

        # Always show EMG activation bar at bottom
        self._draw_emg_strip(activation, threshold)

    def _draw_name_entry(self):
        s = self.surf
        t = self.font_large.render("Reaction Challenge", True, TEXT_C)
        s.blit(t, t.get_rect(center=(SCREEN_W // 2, 120)))

        inst = self.font_med.render("Enter your name and press ENTER", True, DIM_C)
        s.blit(inst, inst.get_rect(center=(SCREEN_W // 2, 190)))

        # Name box
        box = pygame.Rect(SCREEN_W // 2 - 180, 240, 360, 56)
        pygame.draw.rect(s, (30, 30, 50), box, border_radius=8)
        pygame.draw.rect(s, (80, 120, 200), box, 2, border_radius=8)
        cursor = "|" if (self._dots_frame // 30) % 2 == 0 else ""
        name_txt = self.font_med.render(self.name_input + cursor, True, TEXT_C)
        s.blit(name_txt, name_txt.get_rect(center=box.center))

        hint = self.font_small.render("When GO appears — flex as fast as possible!", True, DIM_C)
        s.blit(hint, hint.get_rect(center=(SCREEN_W // 2, 340)))

    def _draw_waiting(self):
        s = self.surf
        t = self.font_large.render("GET READY...", True, WAIT_C)
        s.blit(t, t.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 60)))

        dots = "." * (1 + (self._dots_frame // 20) % 3)
        d = self.font_med.render(f"Wait for it{dots}", True, DIM_C)
        s.blit(d, d.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 20)))

        name = self.font_small.render(f"Participant: {self._current_name}", True, DIM_C)
        s.blit(name, (20, 20))

    def _draw_go(self):
        s = self.surf
        # Big flash
        pygame.draw.rect(s, (0, 40, 0), (0, 0, SCREEN_W, SCREEN_H))
        t = self.font_large.render("GO!", True, GO_C)
        # Scale up
        scale_surf = pygame.transform.scale(t, (t.get_width() * 2, t.get_height() * 2))
        s.blit(scale_surf, scale_surf.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 40)))

        hint = self.font_med.render("FLEX NOW!", True, (180, 255, 180))
        s.blit(hint, hint.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 80)))

        name = self.font_small.render(f"Participant: {self._current_name}", True, DIM_C)
        s.blit(name, (20, 20))

    def _draw_result(self):
        s = self.surf
        if self._false_start:
            t = self.font_large.render("FALSE START!", True, WAIT_C)
            s.blit(t, t.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 60)))
            sub = self.font_med.render("Flexed before GO!", True, DIM_C)
            s.blit(sub, sub.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 10)))
        else:
            t = self.font_large.render(f"{self._result_ms:.0f} ms", True, GO_C)
            s.blit(t, t.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 60)))

            name_t = self.font_med.render(self._current_name, True, TEXT_C)
            s.blit(name_t, name_t.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 - 110)))

            # Rank label
            rank_label = self._get_rank_label()
            sub = self.font_med.render(rank_label, True, DIM_C)
            s.blit(sub, sub.get_rect(center=(SCREEN_W // 2, SCREEN_H // 2 + 20)))

        lb = self.font_small.render("Press L for leaderboard  |  N for next participant", True, DIM_C)
        s.blit(lb, lb.get_rect(center=(SCREEN_W // 2, SCREEN_H - 80)))

    def _get_rank_label(self) -> str:
        if not self.leaderboard:
            return ""
        idx = next((i for i, (n, _) in enumerate(self.leaderboard)
                    if n == self._current_name and abs(_ - self._result_ms) < 1), -1)
        if idx == 0:
            return "NEW RECORD!"
        elif idx == 1:
            return "2nd place"
        elif idx == 2:
            return "3rd place"
        else:
            return f"#{idx+1} place"

    def _draw_leaderboard(self):
        s = self.surf
        t = self.font_large.render("LEADERBOARD", True, GOLD)
        s.blit(t, t.get_rect(center=(SCREEN_W // 2, 60)))

        colors = [GOLD, SILVER, BRONZE]
        for i, (name, ms) in enumerate(self.leaderboard[:8]):
            color = colors[i] if i < 3 else TEXT_C
            rank = f"#{i+1}"
            row = self.font_med.render(f"{rank:>3}  {name:<16}  {ms:.0f} ms", True, color)
            s.blit(row, row.get_rect(center=(SCREEN_W // 2, 130 + i * 46)))

        hint = self.font_small.render("Press N to add next participant", True, DIM_C)
        s.blit(hint, hint.get_rect(center=(SCREEN_W // 2, SCREEN_H - 60)))

    def _draw_emg_strip(self, activation: float, threshold: float):
        s = self.surf
        bar_x, bar_y, bar_w, bar_h = 20, SCREEN_H - 32, 240, 18
        pygame.draw.rect(s, PANEL2, (bar_x, bar_y, bar_w, bar_h), border_radius=bar_h // 2)
        max_v = max(threshold * 2.5, activation * 1.1, 50.0)
        fill_w = int(bar_w * min(1.0, activation / max_v))
        is_active = activation >= threshold
        color = GO_C if is_active else ACCENT
        if fill_w > 2:
            pygame.draw.rect(s, color, (bar_x, bar_y, fill_w, bar_h),
                             border_radius=bar_h // 2)
        if threshold > 0:
            thr_x = bar_x + int(bar_w * threshold / max_v)
            pygame.draw.line(s, GOLD, (thr_x, bar_y - 4), (thr_x, bar_y + bar_h + 4), 2)
        pygame.draw.rect(s, BORDER, (bar_x, bar_y, bar_w, bar_h), 1, border_radius=bar_h // 2)
        lbl = self.font_small.render(f"EMG  {activation:.0f}µV", True, DIM_C)
        s.blit(lbl, (bar_x, bar_y - 20))
