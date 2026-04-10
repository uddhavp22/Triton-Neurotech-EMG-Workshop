"""
main.py — EMG BCI Workshop · MindRove 4-Channel Armband / LSL
Triton Neurotech

Keys:
  C         calibrate
  G         jump game
  R         reaction challenge
  M         main menu
  1-4       select active EMG channel
  SPACE     manual trigger (menu/game/reaction)
  ESC       quit
"""

import sys, time, threading
import pygame
import numpy as np

from emg_input import EMGInput
from signal_processing import EMGProcessor
from calibration import Calibration
from controller import TriggerController
from game import JumpGame, SCREEN_W, SCREEN_H, FPS
from reaction import ReactionChallenge

# ── Palette ────────────────────────────────────────────────────────────────────
BG       = ( 11,  15,  26)   # near-black navy
PANEL    = ( 20,  27,  46)   # card background
PANEL2   = ( 28,  38,  62)   # slightly lighter panel
BORDER   = ( 42,  58,  94)   # subtle border
TEXT     = (232, 236, 248)   # main text
DIM      = ( 90, 106, 148)   # secondary text
ACCENT   = ( 77, 159, 255)   # blue accent
GREEN    = (  0, 210, 110)   # active / triggered
RED      = (255,  72,  72)   # threshold / danger
YELLOW   = (255, 210,  50)   # smoothed signal
PURPLE   = (186, 104, 255)   # highlight

CH_COLORS = [
    ( 77, 159, 255),  # CH1 blue
    (  0, 210, 110),  # CH2 green
    (255, 210,  50),  # CH3 yellow
    (255, 128, 171),  # CH4 pink
    (186, 104, 255),  # CH5 purple
    (255, 165,  50),  # CH6 orange
    ( 64, 224, 208),  # CH7 teal
    (200, 200, 200),  # CH8 light grey
]

CAL_DURATION_S = 4
DISPLAY_N = 500 * 4   # 4 seconds of display samples
DISABLED_CHANNELS = {1}  # CH2 has unstable/noisy data on this workshop rig


# ── Fonts ──────────────────────────────────────────────────────────────────────
def make_fonts():
    pygame.font.init()
    try:
        return {
            'xl':    pygame.font.SysFont('SF Pro Display,Helvetica Neue,Arial', 56, bold=True),
            'large': pygame.font.SysFont('SF Pro Display,Helvetica Neue,Arial', 38, bold=True),
            'med':   pygame.font.SysFont('Helvetica Neue,Arial',                26),
            'small': pygame.font.SysFont('Helvetica Neue,Arial',                17),
            'tiny':  pygame.font.SysFont('Helvetica Neue,Arial',                13),
        }
    except Exception:
        return {
            'xl':    pygame.font.SysFont('Arial', 56, bold=True),
            'large': pygame.font.SysFont('Arial', 38, bold=True),
            'med':   pygame.font.SysFont('Arial', 26),
            'small': pygame.font.SysFont('Arial', 17),
            'tiny':  pygame.font.SysFont('Arial', 13),
        }


# ── Drawing helpers ────────────────────────────────────────────────────────────
def draw_panel(surf, rect, radius=10, color=None, border=True):
    c = color or PANEL
    pygame.draw.rect(surf, c, rect, border_radius=radius)
    if border:
        pygame.draw.rect(surf, BORDER, rect, 1, border_radius=radius)


def draw_pill(surf, cx, cy, w, h, color, label, font, text_color=BG):
    r = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
    pygame.draw.rect(surf, color, r, border_radius=h // 2)
    t = font.render(label, True, text_color)
    surf.blit(t, t.get_rect(center=(cx, cy)))


def draw_label(surf, text, x, y, font, color=DIM, anchor='left'):
    t = font.render(text, True, color)
    if anchor == 'center':
        surf.blit(t, t.get_rect(center=(x, y)))
    elif anchor == 'right':
        surf.blit(t, t.get_rect(right=x, centery=y))
    else:
        surf.blit(t, (x, y))
    return t.get_width()


# ── Multi-channel EMG plot ─────────────────────────────────────────────────────
class MultiChannelPlot:
    """
    Stacked 4-channel EMG display. Each channel auto-scales independently.
    Active channel shows threshold line and thicker trace.
    """

    def __init__(self, x, y, w, h, n_channels: int = 4):
        self.rect = pygame.Rect(x, y, w, h)
        self.n_channels = n_channels

    def draw(self, surf, data: np.ndarray, threshold: float,
             active_ch: int, fonts: dict):
        r = self.rect
        n_ch = min(self.n_channels, data.shape[0])
        ch_h = r.h // n_ch

        draw_panel(surf, r, radius=8)

        for i in range(n_ch):
            ch_rect = pygame.Rect(r.x + 1, r.y + i * ch_h + 1, r.w - 2, ch_h - 1)
            is_active = (i == active_ch)
            self._draw_channel(surf, data[i], ch_rect, CH_COLORS[i],
                               threshold if is_active else 0.0,
                               f"CH{i+1}", is_active, fonts['tiny'])

        # Dividers between channels
        for i in range(1, n_ch):
            dy = r.y + i * ch_h
            pygame.draw.line(surf, BORDER, (r.x + 4, dy), (r.right - 4, dy), 1)

    def _draw_channel(self, surf, signal: np.ndarray, rect: pygame.Rect,
                      color: tuple, threshold: float, label: str,
                      is_active: bool, font):
        # Auto-scale: use 99th percentile of non-zero values
        nonzero = np.abs(signal[signal != 0.0])
        if len(nonzero) > 10:
            scale = max(30.0, float(np.percentile(nonzero, 99)) * 2.2)
        else:
            scale = 150.0

        n = len(signal)
        mid_y = rect.centery

        # Center line
        pygame.draw.line(surf, BORDER, (rect.x, mid_y), (rect.right, mid_y), 1)

        # Signal trace
        if n > 1:
            step = max(1, n // rect.w)   # decimate if too many samples
            indices = range(0, n, step)
            pts = []
            for i in indices:
                sx = rect.x + int(i * rect.w / n)
                sv = float(signal[i])
                sy = int(mid_y - (sv / scale) * (rect.h // 2 - 3))
                sy = max(rect.top + 1, min(rect.bottom - 1, sy))
                pts.append((sx, sy))
            if len(pts) > 1:
                width = 2 if is_active else 1
                alpha_color = color if is_active else tuple(max(0, c - 60) for c in color)
                pygame.draw.lines(surf, alpha_color, False, pts, width)

        # Threshold line (active channel only)
        if threshold > 0 and scale > 0:
            ty = int(mid_y - (threshold / scale) * (rect.h // 2 - 3))
            if rect.top < ty < rect.bottom:
                pygame.draw.line(surf, RED, (rect.x + 2, ty), (rect.right - 2, ty), 1)
                draw_label(surf, f"{threshold:.0f}µV", rect.right - 4, ty - 8,
                           font, RED, anchor='right')

        # Channel label
        lbl_color = color if is_active else DIM
        draw_label(surf, label, rect.x + 6, rect.y + 4, font, lbl_color)

        # Scale hint (dimmed)
        draw_label(surf, f"±{scale/2:.0f}µV", rect.right - 6, rect.y + 4,
                   font, BORDER, anchor='right')


# ── Activation bar ─────────────────────────────────────────────────────────────
class ActivationBar:
    """Horizontal bar showing smoothed EMG activation vs threshold."""

    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)

    def draw(self, surf, activation: float, threshold: float, fonts: dict):
        r = self.rect
        draw_panel(surf, r, radius=r.h // 2, color=PANEL2)

        max_v = max(threshold * 2.5, activation * 1.2, 50.0)
        ratio = min(1.0, activation / max_v)
        fill_w = int(r.w * ratio)

        is_active = activation >= threshold
        color = GREEN if is_active else ACCENT

        if fill_w > 2:
            fill_r = pygame.Rect(r.x, r.y, fill_w, r.h)
            pygame.draw.rect(surf, color, fill_r, border_radius=r.h // 2)

        # Threshold marker
        if threshold > 0 and max_v > 0:
            tx = r.x + int(r.w * threshold / max_v)
            pygame.draw.line(surf, YELLOW, (tx, r.y - 4), (tx, r.bottom + 4), 2)

        pygame.draw.rect(surf, BORDER, r, 1, border_radius=r.h // 2)

        # Value label
        draw_label(surf, f"{activation:.1f} µV", r.right + 12,
                   r.centery, fonts['small'], TEXT if is_active else DIM)


# ── Channel selector ───────────────────────────────────────────────────────────
class ChannelSelector:
    def __init__(self, x, y, n=4, disabled: set[int] | None = None):
        self.x, self.y = x, y
        self.n = min(n, len(CH_COLORS))
        self.disabled = set(disabled or set())
        # Shrink buttons to fit all channels within ~450px from x
        max_total = 450
        self.gap = 6
        self.bw = min(44, (max_total - self.gap * (self.n - 1)) // self.n)
        self.bh = 26

    def draw(self, surf, active: int, fonts: dict):
        draw_label(surf, "CHANNEL", self.x, self.y - 16, fonts['tiny'], DIM)
        for i in range(self.n):
            bx = self.x + i * (self.bw + self.gap)
            r = pygame.Rect(bx, self.y, self.bw, self.bh)
            is_sel = (i == active)
            is_disabled = i in self.disabled
            bg = CH_COLORS[i] if is_sel and not is_disabled else PANEL2
            bd = CH_COLORS[i] if is_sel and not is_disabled else BORDER
            pygame.draw.rect(surf, bg, r, border_radius=5)
            pygame.draw.rect(surf, bd, r, 1, border_radius=5)
            tc = (70, 78, 104) if is_disabled else (BG if is_sel else DIM)
            label = f"CH{i+1}" if not is_disabled else f"{i+1}X"
            t = fonts['tiny'].render(label, True, tc)
            surf.blit(t, t.get_rect(center=r.center))
        if self.disabled:
            disabled_text = "disabled: " + ", ".join(f"CH{i+1}" for i in sorted(self.disabled))
            draw_label(surf, disabled_text, self.x + 170, self.y - 16, fonts['tiny'], BORDER)

    def get_rects(self):
        return [pygame.Rect(self.x + i * (self.bw + self.gap), self.y, self.bw, self.bh)
                for i in range(self.n)]

    def hit_test(self, pos) -> int | None:
        for i, r in enumerate(self.get_rects()):
            if r.collidepoint(pos) and i not in self.disabled:
                return i
        return None


# ── Button ─────────────────────────────────────────────────────────────────────
class Button:
    def __init__(self, x, y, w, h, label, sub="", color=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.sub = sub
        self.color = color or ACCENT

    def draw(self, surf, fonts, hovered=False):
        alpha = 230 if hovered else 180
        c = tuple(min(255, int(v * (1.15 if hovered else 1.0))) for v in self.color)
        draw_panel(surf, self.rect, radius=10, color=c, border=False)
        if hovered:
            pygame.draw.rect(surf, (255, 255, 255, 60), self.rect, 1, border_radius=10)
        t = fonts['med'].render(self.label, True, BG)
        surf.blit(t, t.get_rect(center=(self.rect.centerx, self.rect.centery - (8 if self.sub else 0))))
        if self.sub:
            s = fonts['tiny'].render(self.sub, True, tuple(max(0, v - 60) for v in BG))
            surf.blit(s, s.get_rect(center=(self.rect.centerx, self.rect.centery + 12)))

    def hovered(self, pos):
        return self.rect.collidepoint(pos)


# ── Calibration screen ─────────────────────────────────────────────────────────
class CalibrationScreen:
    SLIDER_MIN = 5.0
    SLIDER_MAX = 600.0
    STEPS = ["Instructions", "Record REST", "Record FLEX", "Set Threshold"]

    def __init__(self, surf, fonts, emg: EMGInput, proc: EMGProcessor,
                 cal: Calibration, ctrl: TriggerController):
        self.surf = surf
        self.fonts = fonts
        self.emg = emg
        self.proc = proc
        self.cal = cal
        self.ctrl = ctrl

        self.plot = MultiChannelPlot(30, 90, SCREEN_W - 60, 220, emg.n_channels)
        self.act_bar = ActivationBar(30, 328, SCREEN_W - 180, 22)
        self.ch_sel = ChannelSelector(30, 62, emg.n_channels, DISABLED_CHANNELS)
        self.slider_rect = pygame.Rect(180, 430, 540, 20)

        self.step = 0
        self._recording = False
        self._progress = 0.0
        self._kind = ""
        self._slider_drag = False
        self._frame = 0

    def reset(self):
        self.step = 0
        self._recording = False
        self._progress = 0.0

    # step 0→1 on first SPACE: auto-advance to rest recording
    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            if not self._recording:
                if self.step == 0:
                    self._start_record('rest')
                elif self.step == 1:
                    self._start_record('rest')
                elif self.step == 2:
                    self._start_record('flex')
                elif self.step == 3:
                    self._start_record('flex')  # redo flex

        elif event.type == pygame.KEYDOWN and event.key in (
                pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                    pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8):
            ch = event.key - pygame.K_1
            if ch < self.emg.n_channels and ch not in DISABLED_CHANNELS:
                self.proc.set_channel(ch)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Channel selector
            ch = self.ch_sel.hit_test(event.pos)
            if ch is not None:
                self.proc.set_channel(ch)
            # Slider (step 3)
            if self.step == 3 and self.slider_rect.collidepoint(event.pos):
                self._slider_drag = True
                self._update_slider(event.pos[0])

        elif event.type == pygame.MOUSEBUTTONUP:
            self._slider_drag = False

        elif event.type == pygame.MOUSEMOTION:
            if self._slider_drag:
                self._update_slider(event.pos[0])

    def _update_slider(self, mx: int):
        ratio = (max(self.slider_rect.x, min(mx, self.slider_rect.right))
                 - self.slider_rect.x) / self.slider_rect.w
        thr = self.SLIDER_MIN + ratio * (self.SLIDER_MAX - self.SLIDER_MIN)
        self.cal.threshold = thr
        self.ctrl.set_threshold(thr)

    def _start_record(self, kind: str):
        self._recording = True
        self._progress = 0.0
        self._kind = kind
        smooth_samples = self.proc.smooth_samples
        ch = self.proc.channel

        def do_record():
            start = time.monotonic()
            activations = []
            while time.monotonic() - start < CAL_DURATION_S:
                disp = self.emg.peek(smooth_samples)
                act = EMGProcessor.compute_activation_from_window(disp, ch, smooth_samples)
                activations.append(act)
                self._progress = min(1.0, (time.monotonic() - start) / CAL_DURATION_S)
                time.sleep(0.04)

            if kind == 'rest':
                self.cal.record_rest(np.array(activations))
                self.step = 2
            else:
                self.cal.record_flex(np.array(activations))
                self.cal.compute_threshold(sensitivity=0.45)
                self.ctrl.set_threshold(self.cal.threshold)
                self.step = 3

            self._recording = False

        threading.Thread(target=do_record, daemon=True).start()

    def run_frame(self, activation: float, active_ch: int) -> str:
        self._frame += 1
        s = self.surf
        s.fill(BG)

        # ── Header ──
        step_label = self.STEPS[min(self.step, len(self.STEPS) - 1)]
        draw_label(s, "CALIBRATION", 30, 18, self.fonts['large'], TEXT)
        step_str = f"Step {min(self.step+1, len(self.STEPS))} / {len(self.STEPS)}  —  {step_label}"
        draw_label(s, step_str, SCREEN_W - 30, 22, self.fonts['small'], DIM, anchor='right')

        # ── Channel selector ──
        self.ch_sel.draw(s, active_ch, self.fonts)

        # ── 4-channel EMG plot ──
        disp = self.emg.peek(DISPLAY_N)
        self.plot.draw(s, disp, self.cal.threshold if self.cal.is_calibrated() else 0.0,
                       active_ch, self.fonts)

        # ── Activation bar ──
        draw_label(s, "ACTIVATION", 30, 314, self.fonts['tiny'], DIM)
        self.act_bar.draw(s, activation, self.ctrl.threshold, self.fonts)

        # ── Step content ──
        self._draw_step_content(activation)

        # ── Bottom hint ──
        hint = "M = menu   G = jump game   R = reaction   1-4 = channel"
        draw_label(s, hint, SCREEN_W // 2, SCREEN_H - 14, self.fonts['tiny'], BORDER, anchor='center')

        return 'calibrating'

    def _draw_step_content(self, activation: float):
        s = self.surf
        cy = 375

        if self.step == 0:
            lines = [
                ("Calibration takes ~10 seconds.", TEXT),
                (f"Step 1: Relax arm  ({CAL_DURATION_S}s of rest signal)", DIM),
                (f"Step 2: Flex arm   ({CAL_DURATION_S}s of flex signal)", DIM),
                ("Step 3: Adjust threshold slider.", DIM),
            ]
            for i, (line, color) in enumerate(lines):
                draw_label(s, line, SCREEN_W // 2, cy + i * 30, self.fonts['small'],
                           color, anchor='center')
            self._draw_big_button("SPACE  →  Start Calibration", cy + 140)

        elif self.step in (1, 2):
            kind = self._kind.upper() if self._kind else ("REST" if self.step == 1 else "FLEX")
            if self._recording:
                color = RED if kind == 'REST' else GREEN
                # Progress bar
                bx, by, bw, bh = 180, cy - 10, 540, 32
                draw_panel(s, pygame.Rect(bx, by, bw, bh), radius=8, color=PANEL2)
                fill_w = int(bw * self._progress)
                if fill_w > 4:
                    pygame.draw.rect(s, color, (bx, by, fill_w, bh), border_radius=8)
                pygame.draw.rect(s, BORDER, (bx, by, bw, bh), 1, border_radius=8)
                pct = int(self._progress * 100)
                secs_left = CAL_DURATION_S * (1 - self._progress)
                draw_label(s, f"Recording {kind}...  {secs_left:.1f}s",
                           SCREEN_W // 2, by + bh // 2, self.fonts['small'], TEXT, anchor='center')
                msg = "Hold still and relax." if kind == 'REST' else "Flex firmly and hold."
                draw_label(s, msg, SCREEN_W // 2, cy + 40, self.fonts['tiny'], DIM, anchor='center')
                self._draw_record_overlay(kind, secs_left)
            else:
                next_kind = "REST" if self.step == 1 else "FLEX"
                msg = "Relax your forearm completely." if next_kind == 'REST' else "Flex your forearm firmly."
                color = RED if next_kind == 'REST' else GREEN
                draw_label(s, msg, SCREEN_W // 2, cy, self.fonts['small'], color, anchor='center')
                self._draw_big_button(f"SPACE  →  Record {next_kind}", cy + 50)

        elif self.step == 3:
            # Threshold slider
            thr = self.cal.threshold
            draw_label(s, f"Threshold:  {thr:.1f} µV", 180, cy - 26,
                       self.fonts['small'], TEXT)
            draw_label(s, "drag to adjust →", SCREEN_W - 180, cy - 26,
                       self.fonts['tiny'], DIM, anchor='right')

            sr = self.slider_rect
            draw_panel(s, sr, radius=sr.h // 2, color=PANEL2)

            # Rest / flex markers
            rest_m = self.cal.rest_mean
            flex_m = self.cal.flex_mean
            max_v = max(self.SLIDER_MAX, flex_m * 1.2)

            def slider_x(v):
                r = (v - self.SLIDER_MIN) / (self.SLIDER_MAX - self.SLIDER_MIN)
                return sr.x + int(max(0, min(1, r)) * sr.w)

            # Colored fill up to threshold
            ratio = (thr - self.SLIDER_MIN) / (self.SLIDER_MAX - self.SLIDER_MIN)
            fill_w = int(sr.w * max(0, min(1, ratio)))
            if fill_w > 2:
                pygame.draw.rect(s, ACCENT, (sr.x, sr.y, fill_w, sr.h), border_radius=sr.h // 2)

            # Rest marker
            rx = slider_x(rest_m)
            pygame.draw.line(s, DIM, (rx, sr.y - 6), (rx, sr.bottom + 6), 2)
            draw_label(s, "REST", rx, sr.y - 18, self.fonts['tiny'], DIM, anchor='center')

            # Flex marker
            fx = slider_x(flex_m)
            pygame.draw.line(s, GREEN, (fx, sr.y - 6), (fx, sr.bottom + 6), 2)
            draw_label(s, "FLEX", fx, sr.y - 18, self.fonts['tiny'], GREEN, anchor='center')

            # Threshold knob
            kx = slider_x(thr)
            pygame.draw.circle(s, YELLOW, (kx, sr.centery), 13)
            pygame.draw.circle(s, TEXT, (kx, sr.centery), 13, 2)

            pygame.draw.rect(s, BORDER, sr, 1, border_radius=sr.h // 2)

            # Live trigger status
            is_triggered = activation >= thr
            tc = GREEN if is_triggered else DIM
            status = "● TRIGGERED" if is_triggered else "○ not triggered"
            draw_label(s, status, SCREEN_W // 2, sr.bottom + 28,
                       self.fonts['med'], tc, anchor='center')

            # Summary
            summary = self.cal.summary()
            draw_label(s, summary, SCREEN_W // 2, sr.bottom + 60,
                       self.fonts['tiny'], DIM, anchor='center')

            gap = self.cal.flex_mean - self.cal.rest_mean
            quality = "Good separation" if gap >= max(20.0, 2.5 * self.cal.rest_std) else "Weak separation"
            qc = GREEN if quality == "Good separation" else YELLOW
            draw_label(s, quality, SCREEN_W // 2, sr.bottom + 78,
                       self.fonts['tiny'], qc, anchor='center')

            self._draw_big_button("SPACE = redo FLEX   G = game   R = reaction",
                                  sr.bottom + 90, color=PANEL2, text_color=DIM)

    def _draw_big_button(self, label: str, y: int, color=ACCENT, text_color=BG):
        w = min(540, len(label) * 11 + 40)
        r = pygame.Rect(SCREEN_W // 2 - w // 2, y, w, 38)
        draw_panel(self.surf, r, radius=8, color=color, border=True)
        draw_label(self.surf, label, SCREEN_W // 2, r.centery,
                   self.fonts['small'], text_color, anchor='center')

    def _draw_record_overlay(self, kind: str, secs_left: float):
        ov = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        ov.fill((5, 8, 18, 145))
        self.surf.blit(ov, (0, 0))

        title_color = RED if kind == "REST" else GREEN
        title = "REST NOW" if kind == "REST" else "FLEX NOW"
        subtitle = "Keep your arm relaxed and still" if kind == "REST" else "Contract and hold steadily"

        draw_label(self.surf, title, SCREEN_W // 2, 160, self.fonts['xl'], title_color, anchor='center')
        draw_label(self.surf, subtitle, SCREEN_W // 2, 208, self.fonts['med'], TEXT, anchor='center')

        countdown = max(0.0, secs_left)
        draw_label(
            self.surf,
            f"{countdown:.1f}s",
            SCREEN_W // 2,
            270,
            self.fonts['large'],
            TEXT,
            anchor='center',
        )


# ── Main menu ──────────────────────────────────────────────────────────────────
class MainMenu:
    def __init__(self, surf, fonts, cal: Calibration, emg: EMGInput,
                 proc: EMGProcessor, ctrl: TriggerController):
        self.surf = surf
        self.fonts = fonts
        self.cal = cal
        self.emg = emg
        self.proc = proc
        self.ctrl = ctrl
        self.plot = MultiChannelPlot(30, 130, SCREEN_W - 60, 220, emg.n_channels)
        self.act_bar = ActivationBar(30, 364, SCREEN_W - 180, 18)
        self.ch_sel = ChannelSelector(30, 100, emg.n_channels, DISABLED_CHANNELS)

        bw, bh = 220, 64
        gap = 24
        total = 3 * bw + 2 * gap
        bx = SCREEN_W // 2 - total // 2
        by = 415
        self.buttons = [
            Button(bx,              by, bw, bh, "CALIBRATE", "Press C", color=(77, 100, 200)),
            Button(bx + bw + gap,   by, bw, bh, "JUMP GAME", "Press G", color=(0, 160, 90)),
            Button(bx + 2*(bw+gap), by, bw, bh, "REACTION",  "Press R", color=(180, 80, 200)),
        ]

    def run_frame(self, activation: float, active_ch: int):
        s = self.surf
        s.fill(BG)
        mx, my = pygame.mouse.get_pos()

        # ── Header ──
        draw_label(s, "EMG BCI WORKSHOP", SCREEN_W // 2, 30, self.fonts['large'],
                   TEXT, anchor='center')
        draw_label(s, "Triton Neurotech  ·  MindRove 4-Channel Armband",
                   SCREEN_W // 2, 68, self.fonts['small'], DIM, anchor='center')

        # ── Channel selector ──
        self.ch_sel.draw(s, active_ch, self.fonts)

        # ── EMG plot ──
        disp = self.emg.peek(DISPLAY_N)
        self.plot.draw(s, disp, self.ctrl.threshold if self.cal.is_calibrated() else 0.0,
                       active_ch, self.fonts)

        # ── Activation bar ──
        draw_label(s, "ACTIVATION", 30, 350, self.fonts['tiny'], DIM)
        self.act_bar.draw(s, activation, self.ctrl.threshold, self.fonts)

        # ── Calibration status ──
        if self.cal.is_calibrated():
            draw_pill(s, SCREEN_W // 2, 398, 380, 22, GREEN,
                      f"Calibrated  ·  {self.cal.summary()}", self.fonts['tiny'])
        else:
            draw_pill(s, SCREEN_W // 2, 398, 320, 22, (100, 40, 40),
                      "Not calibrated  —  press C first", self.fonts['tiny'],
                      text_color=(255, 150, 150))

        # ── Buttons ──
        for btn in self.buttons:
            btn.draw(s, self.fonts, btn.hovered((mx, my)))

        # ── Footer ──
        draw_label(s, "SPACE = manual trigger   1-4 = channel   ESC = quit",
                   SCREEN_W // 2, SCREEN_H - 14, self.fonts['tiny'], BORDER, anchor='center')


# ── App ────────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    pygame.display.set_caption("EMG BCI Workshop — Triton Neurotech")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    fonts = make_fonts()

    # Core components
    emg = EMGInput()
    emg.start()

    default_channel = next((i for i in range(emg.n_channels) if i not in DISABLED_CHANNELS), 0)
    proc = EMGProcessor(sampling_rate=emg.sampling_rate, smooth_ms=80, channel=default_channel)
    cal  = Calibration()
    ctrl = TriggerController(threshold=50.0, refractory_ms=250, release_ratio=0.65)

    # Screens
    menu_scr = MainMenu(screen, fonts, cal, emg, proc, ctrl)
    cal_scr  = CalibrationScreen(screen, fonts, emg, proc, cal, ctrl)
    game_scr = JumpGame(screen, fonts['large'], fonts['med'], fonts['small'])
    rxn_scr  = ReactionChallenge(screen, fonts['large'], fonts['med'], fonts['small'])

    mode = 'menu'
    active_ch = default_channel

    print("=" * 54)
    print("  EMG BCI Workshop  |  Triton Neurotech")
    print(
        f"  Backend: {emg.backend}  |  Source: {emg.source_name}  |  "
        f"Synthetic: {emg.synthetic}"
    )
    print(f"  {emg.n_channels} ch @ {emg.sampling_rate} Hz")
    print("  C=calibrate  G=game  R=reaction  M=menu  1-4=channel")
    if DISABLED_CHANNELS:
        print("  Disabled channels: " + ", ".join(f"CH{i+1}" for i in sorted(DISABLED_CHANNELS)))
    print("=" * 54)

    running = True
    while running:
        manual_trigger = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k == pygame.K_m:
                    mode = 'menu'
                elif k == pygame.K_c:
                    cal_scr.reset()
                    mode = 'calibrate'
                elif k == pygame.K_g:
                    mode = 'game'
                elif k == pygame.K_r:
                    mode = 'reaction'
                elif k in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                    pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8):
                    ch = k - pygame.K_1
                    if ch < emg.n_channels and ch not in DISABLED_CHANNELS:
                        active_ch = ch
                        proc.set_channel(ch)
                elif k == pygame.K_SPACE:
                    if mode == 'calibrate':
                        cal_scr.handle_event(event)
                    else:
                        manual_trigger = True
                elif mode == 'reaction':
                    if k == pygame.K_l:
                        rxn_scr.next_participant()
                    elif k == pygame.K_n:
                        rxn_scr.new_participant()
                    rxn_scr.handle_key(event)
                elif mode == 'calibrate':
                    cal_scr.handle_event(event)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                if mode == 'menu':
                    # Channel selector
                    ch = menu_scr.ch_sel.hit_test(pos)
                    if ch is not None:
                        active_ch = ch
                        proc.set_channel(ch)
                    # Nav buttons
                    if menu_scr.buttons[0].hovered(pos):
                        cal_scr.reset(); mode = 'calibrate'
                    elif menu_scr.buttons[1].hovered(pos):
                        mode = 'game'
                    elif menu_scr.buttons[2].hovered(pos):
                        mode = 'reaction'
                elif mode == 'calibrate':
                    cal_scr.handle_event(event)

            elif event.type in (pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
                if mode == 'calibrate':
                    cal_scr.handle_event(event)

        # ── EMG: pull NEW samples only, push once ──────────────────────────────
        new_samples = emg.pull()
        proc.push(new_samples)
        activation = proc.activation()

        triggered = ctrl.update(activation) or manual_trigger

        # ── Render ────────────────────────────────────────────────────────────
        if mode == 'menu':
            menu_scr.run_frame(activation, active_ch)

        elif mode == 'calibrate':
            # Keep active_ch in sync with proc channel
            active_ch = proc.channel
            cal_scr.run_frame(activation, active_ch)

        elif mode == 'game':
            if triggered:
                game_scr.trigger_jump()
            game_scr.run_frame(activation, ctrl.threshold)

        elif mode == 'reaction':
            if triggered:
                rxn_scr.trigger_flex()
            rxn_scr.run_frame(activation, ctrl.threshold)

        pygame.display.flip()
        clock.tick(FPS)

    emg.stop()
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
