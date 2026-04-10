"""
main.py — EMG BCI Workshop · PyQt6 + pyqtgraph
Triton Neurotech · MindRove 4-Channel Armband / LSL

Keys:
  C         calibrate
  G         jump game
  R         reaction challenge
  M         main menu
  1-8       select active EMG channel
  SPACE     manual trigger (menu/game/reaction)
  ESC       quit
"""

import sys, time, threading
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QFrame, QSlider, QLineEdit,
)
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QPainterPath,
    QKeyEvent,
)
import pyqtgraph as pg

from emg_input import EMGInput
from signal_processing import EMGProcessor
from calibration import Calibration
from controller import TriggerController
from game import JumpGameLogic, GAME_W, GAME_H, GROUND_Y
from reaction import ReactionLogic

# ── pyqtgraph global config ───────────────────────────────────────────────────
pg.setConfigOptions(antialias=True, useOpenGL=False)

# ── Window / timing ───────────────────────────────────────────────────────────
WIN_W, WIN_H    = 1100, 720
TICK_MS         = 16          # ~60 fps update
CAL_DURATION_S  = 10
DISPLAY_SECS_UI = 2.0
DISABLED_CHANNELS: set[int] = set()

# ── Colour palette ────────────────────────────────────────────────────────────
C_BG     = QColor( 11,  15,  26)
C_PANEL  = QColor( 20,  27,  46)
C_PANEL2 = QColor( 28,  38,  62)
C_BORDER = QColor( 42,  58,  94)
C_TEXT   = QColor(232, 236, 248)
C_DIM    = QColor( 90, 106, 148)
C_ACCENT = QColor( 77, 159, 255)
C_GREEN  = QColor(  0, 210, 110)
C_RED    = QColor(255,  72,  72)
C_YELLOW = QColor(255, 210,  50)
C_GOLD   = QColor(255, 210,  50)
C_SILVER = QColor(192, 192, 192)
C_BRONZE = QColor(205, 127,  50)

CH_COLORS = [
    QColor( 77, 159, 255),  # CH1 blue
    QColor(  0, 210, 110),  # CH2 green
    QColor(255, 210,  50),  # CH3 yellow
    QColor(255, 128, 171),  # CH4 pink
    QColor(186, 104, 255),  # CH5 purple
    QColor(255, 165,  50),  # CH6 orange
    QColor( 64, 224, 208),  # CH7 teal
    QColor(200, 200, 200),  # CH8 grey
]
CH_HEX = [c.name() for c in CH_COLORS]

# ── Stylesheet ────────────────────────────────────────────────────────────────
STYLESHEET = """
* {
    font-family: "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
}
QMainWindow, QWidget {
    background-color: #0B0F1A;
    color: #E8ECF8;
}
QLabel { background: transparent; color: #E8ECF8; }
QPushButton {
    background-color: #141B2E;
    border: 1px solid #2A3A5E;
    border-radius: 8px;
    color: #E8ECF8;
    padding: 6px 14px;
    min-height: 28px;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #1C263E;
    border-color: #4D9FFF;
}
QPushButton:checked {
    background-color: #4D9FFF;
    color: #0B0F1A;
    font-weight: bold;
    border-color: #4D9FFF;
}
QPushButton:pressed { background-color: #3A7ACC; }
QLineEdit {
    background-color: #141B2E;
    border: 1px solid #2A3A5E;
    border-radius: 8px;
    color: #E8ECF8;
    padding: 8px 14px;
    font-size: 18px;
    selection-background-color: #4D9FFF;
}
QLineEdit:focus { border-color: #4D9FFF; }
QSlider::groove:horizontal {
    height: 6px;
    background: #1C263E;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 22px; height: 22px;
    margin: -8px 0;
    background: #FFD232;
    border-radius: 11px;
    border: 2px solid #E8ECF8;
}
QSlider::sub-page:horizontal {
    background: #4D9FFF;
    border-radius: 3px;
}
"""

NAV_BTN_CSS = """
QPushButton {{
    background-color: {bg};
    border: none;
    border-radius: 12px;
    color: #E8ECF8;
    font-size: 15px;
    font-weight: bold;
    padding: 14px 10px;
    min-height: 68px;
}}
QPushButton:hover {{ background-color: {hover}; border: 2px solid #E8ECF8; }}
QPushButton:pressed {{ background-color: #0B0F1A; }}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFixedHeight(1)
    f.setStyleSheet("background: #2A3A5E; border: none;")
    return f


def _lbl(text: str, size: int = 13, bold: bool = False,
         color: QColor | None = None) -> QLabel:
    w = QLabel(text)
    f = w.font()
    f.setPointSize(size)
    if bold:
        f.setBold(True)
    w.setFont(f)
    if color:
        pal = w.palette()
        pal.setColor(w.foregroundRole(), color)
        w.setPalette(pal)
    return w


def probe_channel_metrics(emg: EMGInput, settle_s: float = 0.5,
                          window_s: float = 0.6) -> list[tuple[float, float]]:
    """Return (mean_abs, p95_abs) per channel from a short rest window."""
    time.sleep(settle_s)
    n = max(64, int(emg.sampling_rate * window_s))
    disp = emg.peek(n)
    metrics: list[tuple[float, float]] = []
    for ch in range(emg.n_channels):
        if disp.shape[1] == 0:
            metrics.append((0.0, 0.0))
            continue
        sig = np.asarray(disp[ch], dtype=float)
        dc = float(np.mean(sig))
        abs_sig = np.abs(sig - dc)
        metrics.append((float(np.mean(abs_sig)), float(np.percentile(abs_sig, 95))))
    return metrics


def choose_default_channel(metrics: list[tuple[float, float]], n_ch: int) -> int:  # noqa: ARG001
    reasonable = [(i, p) for i, (_, p) in enumerate(metrics) if 1.0 <= p <= 60.0]
    if reasonable:
        return min(reasonable, key=lambda x: x[1])[0]
    nonflat = [(i, p) for i, (_, p) in enumerate(metrics) if p > 1.0]
    if nonflat:
        return min(nonflat, key=lambda x: x[1])[0]
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Shared Widgets
# ─────────────────────────────────────────────────────────────────────────────

class EMGMultiPlot(pg.GraphicsLayoutWidget):
    """
    Stacked multi-channel EMG display via pyqtgraph.
    Each channel has its own PlotItem so Y-scales are independent.
    """

    def __init__(self, n_channels: int = 4, parent=None):
        super().__init__(parent)
        self.n_channels = n_channels
        self.setBackground(C_BG)
        self.ci.layout.setSpacing(1)
        self.ci.layout.setContentsMargins(4, 4, 4, 4)

        self._plots:      list[pg.PlotItem]     = []
        self._curves:     list[pg.PlotDataItem] = []
        self._thr_lines:  list[pg.InfiniteLine] = []
        self._ch_labels:  list[pg.TextItem]     = []
        self._sc_labels:  list[pg.TextItem]     = []

        for i in range(n_channels):
            p = self.addPlot(row=i, col=0)
            p.setMouseEnabled(False, False)
            p.hideAxis('bottom')
            p.setMenuEnabled(False)
            p.setDefaultPadding(0.02)
            ax = p.getAxis('left')
            ax.setStyle(tickLength=0, showValues=False)
            ax.setWidth(4)

            color = CH_HEX[i]
            curve = p.plot(pen=pg.mkPen(color=color, width=1.5))

            ch_lbl = pg.TextItem(
                text=f'CH{i + 1}', color=color, anchor=(0, 0),
                fill=pg.mkBrush(C_BG.red(), C_BG.green(), C_BG.blue(), 180),
            )
            p.addItem(ch_lbl)

            sc_lbl = pg.TextItem(text='', color=C_DIM.name(), anchor=(1, 0))
            p.addItem(sc_lbl)

            thr_line = p.addLine(
                y=0,
                pen=pg.mkPen(color='#FF4848', width=1,
                             style=Qt.PenStyle.DashLine),
            )
            thr_line.setVisible(False)

            self._plots.append(p)
            self._curves.append(curve)
            self._thr_lines.append(thr_line)
            self._ch_labels.append(ch_lbl)
            self._sc_labels.append(sc_lbl)

    def update_data(self, data: np.ndarray, active_ch: int,
                    threshold: float = 0.0):
        n_avail = min(self.n_channels, data.shape[0])
        for i in range(self.n_channels):
            p         = self._plots[i]
            curve     = self._curves[i]
            thr_line  = self._thr_lines[i]
            is_active = (i == active_ch)

            c = CH_COLORS[i]
            if is_active:
                color = c.name()
                width = 2
            else:
                r = max(0, c.red()   - 55)
                g = max(0, c.green() - 55)
                b = max(0, c.blue()  - 55)
                color = QColor(r, g, b).name()
                width = 1
            curve.setPen(pg.mkPen(color=color, width=width))

            if i >= n_avail or data.shape[1] == 0:
                curve.setData([])
                thr_line.setVisible(False)
                continue

            sig = data[i].astype(float)
            dc  = float(np.mean(sig)) if len(sig) > 0 else 0.0
            centered = sig - dc

            nonzero = np.abs(centered[np.abs(centered) > 1e-9])
            scale = max(25.0, float(np.percentile(nonzero, 99)) * 2.2) \
                if len(nonzero) > 10 else 80.0

            curve.setData(centered)
            p.setYRange(-scale, scale, padding=0)

            xmax = len(centered) - 1 if len(centered) > 0 else 1
            self._ch_labels[i].setPos(0, scale * 0.72)
            self._sc_labels[i].setText(f'±{scale:.0f}µV')
            self._sc_labels[i].setPos(xmax, scale * 0.72)

            if is_active and threshold > 0:
                thr_line.setVisible(True)
                thr_line.setValue(threshold)
            else:
                thr_line.setVisible(False)


class ActivationBarWidget(QWidget):
    """Horizontal activation bar with threshold marker and µV readout."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._activation = 0.0
        self._threshold  = 50.0
        self.setFixedHeight(26)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_values(self, activation: float, threshold: float):
        if self._activation != activation or self._threshold != threshold:
            self._activation = activation
            self._threshold  = threshold
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        bar_w = self.width() - 120
        bar_h = self.height()
        r     = QRectF(0, 0, bar_w, bar_h)
        rad   = bar_h / 2

        # Track
        path = QPainterPath()
        path.addRoundedRect(r, rad, rad)
        p.fillPath(path, C_PANEL2)

        # Fill
        a, thr  = self._activation, self._threshold
        max_v   = max(thr * 2.5, a * 1.2, 50.0)
        ratio   = min(1.0, a / max_v)
        fill_w  = ratio * bar_w
        is_act  = a >= thr
        fill_c  = C_GREEN if is_act else C_ACCENT
        if fill_w > 4:
            fp = QPainterPath()
            fp.addRoundedRect(QRectF(0, 0, fill_w, bar_h), rad, rad)
            p.fillPath(fp, fill_c)

        # Threshold marker
        if thr > 0 and max_v > 0:
            tx = bar_w * thr / max_v
            p.setPen(QPen(C_YELLOW, 2))
            p.drawLine(QPointF(tx, -3), QPointF(tx, bar_h + 3))

        # Border
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(C_BORDER, 1))
        p.drawPath(path)

        # Value text
        txt = f"{a:.1f} µV"
        f2  = QFont()
        f2.setPointSize(12)
        p.setFont(f2)
        p.setPen(QPen(C_TEXT if is_act else C_DIM, 1))
        p.drawText(QRectF(bar_w + 10, 0, 110, bar_h),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   txt)
        p.end()


class ChannelSelectorWidget(QWidget):
    """Row of checkable channel buttons."""
    channel_changed = pyqtSignal(int)

    def __init__(self, n_channels: int, disabled: set[int] | None = None,
                 parent=None):
        super().__init__(parent)
        self.n_channels = n_channels
        self._disabled  = set(disabled or set())
        self._buttons: list[QPushButton] = []

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(_lbl("CHANNEL", 10, color=C_DIM))

        for i in range(n_channels):
            c   = CH_COLORS[i]
            btn = QPushButton(f"CH{i + 1}" if i not in self._disabled
                              else f"{i + 1}✕")
            btn.setCheckable(True)
            btn.setFixedSize(52, 26)
            btn.setEnabled(i not in self._disabled)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: #141B2E; border: 1px solid #2A3A5E;
                    border-radius: 5px; color: #5A6A94;
                    font-size: 11px; font-weight: bold;
                }}
                QPushButton:checked {{
                    background: {c.name()}; border-color: {c.name()};
                    color: #0B0F1A;
                }}
                QPushButton:hover:!checked {{
                    border-color: {c.name()}; color: {c.name()};
                }}
            """)
            btn.clicked.connect(lambda _, idx=i: self.channel_changed.emit(idx))
            self._buttons.append(btn)
            row.addWidget(btn)

        row.addStretch()

    def set_active(self, ch: int):
        for i, btn in enumerate(self._buttons):
            btn.setChecked(i == ch)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Main Menu
# ─────────────────────────────────────────────────────────────────────────────

class MenuPage(QWidget):
    nav_calibrate   = pyqtSignal()
    nav_game        = pyqtSignal()
    nav_reaction    = pyqtSignal()
    channel_changed = pyqtSignal(int)

    def __init__(self, emg: EMGInput, ctrl: TriggerController,
                 cal: Calibration, parent=None):
        super().__init__(parent)
        self._emg  = emg
        self._ctrl = ctrl
        self._cal  = cal
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 18, 28, 14)
        root.setSpacing(10)

        # ── Header ─────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title_col.addWidget(_lbl("EMG BCI WORKSHOP", 20, bold=True))
        title_col.addWidget(_lbl("Triton Neurotech  ·  MindRove Armband",
                                  11, color=C_DIM))
        hdr.addLayout(title_col)
        hdr.addStretch()
        self._status_lbl = _lbl("⬤  Connecting…", 11, color=C_YELLOW)
        hdr.addWidget(self._status_lbl)
        root.addLayout(hdr)
        root.addWidget(_sep())

        # ── Channel selector ────────────────────────────────────────────────
        self._ch_sel = ChannelSelectorWidget(
            self._emg.n_channels, DISABLED_CHANNELS
        )
        self._ch_sel.channel_changed.connect(self.channel_changed)
        root.addWidget(self._ch_sel)

        # ── Live EMG plot ───────────────────────────────────────────────────
        self._plot = EMGMultiPlot(self._emg.n_channels)
        self._plot.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        root.addWidget(self._plot, stretch=3)

        # ── Activation row ──────────────────────────────────────────────────
        act_row = QHBoxLayout()
        act_row.addWidget(_lbl("ACTIVATION", 10, color=C_DIM))
        act_row.addSpacing(8)
        self._act_bar = ActivationBarWidget()
        act_row.addWidget(self._act_bar, stretch=1)
        self._pill = _lbl("  rest  ", 11)
        self._pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pill.setFixedSize(76, 24)
        self._pill.setStyleSheet(
            "border-radius:12px; background:#1C263E; "
            "color:#5A6A94; border:1px solid #2A3A5E;"
        )
        act_row.addWidget(self._pill)
        root.addLayout(act_row)

        root.addWidget(_sep())

        # ── Calibration status ──────────────────────────────────────────────
        self._cal_lbl = _lbl(
            "⚠  Not calibrated — press Calibrate before playing",
            12, color=C_DIM
        )
        self._cal_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._cal_lbl)

        # ── Nav buttons ─────────────────────────────────────────────────────
        nav = QHBoxLayout()
        nav.setSpacing(14)

        cal_btn = QPushButton("⚙   CALIBRATE")
        cal_btn.setStyleSheet(
            NAV_BTN_CSS.format(bg='#3A4FA0', hover='#4D64C8')
        )
        cal_btn.clicked.connect(self.nav_calibrate)

        game_btn = QPushButton("▶   JUMP GAME")
        game_btn.setStyleSheet(
            NAV_BTN_CSS.format(bg='#007A40', hover='#00A858')
        )
        game_btn.clicked.connect(self.nav_game)

        rxn_btn = QPushButton("⚡   REACTION")
        rxn_btn.setStyleSheet(
            NAV_BTN_CSS.format(bg='#6E2E90', hover='#9040C0')
        )
        rxn_btn.clicked.connect(self.nav_reaction)

        for btn in (cal_btn, game_btn, rxn_btn):
            nav.addWidget(btn)
        root.addLayout(nav)

        # ── Footer ──────────────────────────────────────────────────────────
        footer = _lbl(
            "C = Calibrate   G = Jump Game   R = Reaction   "
            "SPACE = trigger   1-8 = channel   ESC = quit",
            10, color=C_DIM
        )
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(footer)

    def update_frame(self, activation: float, active_ch: int):
        n    = max(90, int(self._emg.sampling_rate * DISPLAY_SECS_UI))
        disp = self._emg.peek(n)
        self._plot.update_data(disp, active_ch, self._ctrl.threshold)
        self._act_bar.set_values(activation, self._ctrl.threshold)
        self._ch_sel.set_active(active_ch)

        is_act = activation >= self._ctrl.threshold
        if is_act:
            self._pill.setText("  FLEX!  ")
            self._pill.setStyleSheet(
                "border-radius:12px; background:#00D26E; "
                "color:#0B0F1A; font-weight:bold; border:none;"
            )
        else:
            self._pill.setText("  rest  ")
            self._pill.setStyleSheet(
                "border-radius:12px; background:#1C263E; "
                "color:#5A6A94; border:1px solid #2A3A5E;"
            )

        if self._cal.is_calibrated():
            self._cal_lbl.setText(f"✓  Calibrated  ·  {self._cal.summary()}")
            self._cal_lbl.setStyleSheet("color: #00D26E;")
        else:
            self._cal_lbl.setText(
                "⚠  Not calibrated — press Calibrate before playing"
            )
            self._cal_lbl.setStyleSheet("color: #5A6A94;")

        bk = self._emg.backend
        if bk == 'synthetic':
            self._status_lbl.setText("⬤  Synthetic")
            self._status_lbl.setStyleSheet("color: #FFD232;")
        else:
            self._status_lbl.setText(
                f"⬤  {self._emg.source_name[:32]}"
            )
            self._status_lbl.setStyleSheet("color: #00D26E;")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Calibration
# ─────────────────────────────────────────────────────────────────────────────

class CalibrationPage(QWidget):
    nav_back     = pyqtSignal()
    nav_game     = pyqtSignal()
    nav_reaction = pyqtSignal()
    channel_changed = pyqtSignal(int)

    STEPS = ["Instructions", "Record REST", "Record FLEX", "Review Threshold"]
    SLIDER_MIN, SLIDER_MAX = 5.0, 600.0

    def __init__(self, emg: EMGInput, proc: EMGProcessor,
                 cal: Calibration, ctrl: TriggerController, parent=None):
        super().__init__(parent)
        self._emg  = emg
        self._proc = proc
        self._cal  = cal
        self._ctrl = ctrl

        self.step       = 0
        self._recording = False
        self._progress  = 0.0
        self._kind      = ""
        self._frame     = 0

        self._build()

    def reset(self):
        self.step       = 0
        self._recording = False
        self._progress  = 0.0
        self._kind      = ""
        self._update_step_view()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 16, 28, 14)
        root.setSpacing(10)

        # ── Top bar ─────────────────────────────────────────────────────────
        top = QHBoxLayout()
        back_btn = QPushButton("◀  Menu")
        back_btn.setFixedWidth(90)
        back_btn.clicked.connect(self.nav_back)
        top.addWidget(back_btn)
        top.addSpacing(10)
        top.addWidget(_lbl("CALIBRATION", 18, bold=True))
        top.addStretch()
        self._step_lbl = _lbl("Step 1 / 4", 12, color=C_DIM)
        top.addWidget(self._step_lbl)
        root.addLayout(top)

        # Step progress dots
        dots_row = QHBoxLayout()
        dots_row.addStretch()
        self._step_dots: list[QLabel] = []
        for _ in range(4):
            d = QLabel("●")
            d.setFixedSize(22, 22)
            d.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dots_row.addWidget(d)
            self._step_dots.append(d)
        dots_row.addStretch()
        root.addLayout(dots_row)

        root.addWidget(_sep())

        # ── Channel selector ─────────────────────────────────────────────────
        self._ch_sel = ChannelSelectorWidget(
            self._emg.n_channels, DISABLED_CHANNELS
        )
        self._ch_sel.channel_changed.connect(self.channel_changed)
        root.addWidget(self._ch_sel)

        # ── Live EMG plot ────────────────────────────────────────────────────
        self._plot = EMGMultiPlot(self._emg.n_channels)
        self._plot.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        root.addWidget(self._plot, stretch=2)

        # ── Activation bar ───────────────────────────────────────────────────
        act_row = QHBoxLayout()
        act_row.addWidget(_lbl("ACTIVATION", 10, color=C_DIM))
        act_row.addSpacing(8)
        self._act_bar = ActivationBarWidget()
        act_row.addWidget(self._act_bar, stretch=1)
        root.addLayout(act_row)

        root.addWidget(_sep())

        # ── Step content area ────────────────────────────────────────────────
        self._step_stack = QStackedWidget()
        root.addWidget(self._step_stack, stretch=2)

        self._build_step0()
        self._build_step1()
        self._build_step2()
        self._build_step3()

        self._update_step_view()

    # ── Step 0: Instructions ─────────────────────────────────────────────────
    def _build_step0(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.setSpacing(10)

        v.addWidget(_lbl("Calibration Overview", 15, bold=True,
                         color=C_TEXT))
        for line in [
            f"Step 1 — Relax your arm completely  ({CAL_DURATION_S}s rest recording)",
            f"Step 2 — Flex your forearm firmly   ({CAL_DURATION_S}s flex recording)",
            "Step 3 — Review and adjust the activation threshold",
        ]:
            v.addWidget(_lbl(f"   {line}", 12, color=C_DIM))

        v.addSpacing(8)
        start_btn = QPushButton("  START CALIBRATION  (Space)")
        start_btn.setStyleSheet(
            "QPushButton { background:#3A4FA0; border:none; border-radius:10px;"
            " color:#E8ECF8; font-size:15px; font-weight:bold;"
            " padding:14px 24px; min-height:48px; }"
            "QPushButton:hover { background:#4D64C8; }"
        )
        start_btn.clicked.connect(self._space_pressed)
        v.addWidget(start_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self._step_stack.addWidget(w)

    # ── Step 1: REST recording ───────────────────────────────────────────────
    def _build_step1(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.setSpacing(10)

        self._step1_heading = _lbl("RELAX YOUR ARM", 20, bold=True,
                                    color=C_RED)
        self._step1_heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._step1_heading)

        self._step1_sub = _lbl(
            "Keep your forearm completely still and relaxed", 12, color=C_DIM
        )
        self._step1_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._step1_sub)

        v.addSpacing(6)

        # Progress bar (custom drawn)
        self._rest_prog = _ProgressBar()
        self._rest_prog.setFixedHeight(28)
        v.addWidget(self._rest_prog)

        self._step1_status = _lbl("Press SPACE to begin REST recording",
                                   12, color=C_ACCENT)
        self._step1_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._step1_status)

        start_btn = QPushButton("  Record REST  (Space)")
        start_btn.setStyleSheet(
            "QPushButton { background:#8B1A1A; border:none; border-radius:8px;"
            " color:#E8ECF8; font-size:13px; font-weight:bold;"
            " padding:10px 20px; min-height:36px; }"
            "QPushButton:hover { background:#B02020; }"
        )
        start_btn.clicked.connect(self._space_pressed)
        v.addWidget(start_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self._step_stack.addWidget(w)

    # ── Step 2: FLEX recording ───────────────────────────────────────────────
    def _build_step2(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.setSpacing(10)

        self._step2_heading = _lbl("FLEX YOUR ARM", 20, bold=True,
                                    color=C_GREEN)
        self._step2_heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._step2_heading)

        self._step2_sub = _lbl(
            "Contract your forearm muscle and hold it firmly", 12, color=C_DIM
        )
        self._step2_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._step2_sub)

        v.addSpacing(6)

        self._flex_prog = _ProgressBar(color=C_GREEN)
        self._flex_prog.setFixedHeight(28)
        v.addWidget(self._flex_prog)

        self._step2_status = _lbl("Press SPACE to begin FLEX recording",
                                   12, color=C_ACCENT)
        self._step2_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._step2_status)

        start_btn = QPushButton("  Record FLEX  (Space)")
        start_btn.setStyleSheet(
            "QPushButton { background:#005030; border:none; border-radius:8px;"
            " color:#E8ECF8; font-size:13px; font-weight:bold;"
            " padding:10px 20px; min-height:36px; }"
            "QPushButton:hover { background:#007A40; }"
        )
        start_btn.clicked.connect(self._space_pressed)
        v.addWidget(start_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self._step_stack.addWidget(w)

    # ── Step 3: Threshold review ─────────────────────────────────────────────
    def _build_step3(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setAlignment(Qt.AlignmentFlag.AlignTop)
        v.setSpacing(8)
        v.setContentsMargins(20, 10, 20, 10)

        thr_row = QHBoxLayout()
        self._thr_val_lbl = _lbl("Threshold:  50.0 µV", 13, bold=True)
        thr_row.addWidget(self._thr_val_lbl)
        thr_row.addStretch()

        self._trigger_badge = _lbl("  ○ not triggered  ", 12)
        self._trigger_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._trigger_badge.setFixedHeight(26)
        self._trigger_badge.setStyleSheet(
            "border-radius:13px; background:#1C263E; color:#5A6A94;"
            "border:1px solid #2A3A5E;"
        )
        thr_row.addWidget(self._trigger_badge)
        v.addLayout(thr_row)

        self._thr_slider = QSlider(Qt.Orientation.Horizontal)
        self._thr_slider.setMinimum(int(self.SLIDER_MIN * 10))
        self._thr_slider.setMaximum(int(self.SLIDER_MAX * 10))
        self._thr_slider.setValue(500)   # 50.0
        self._thr_slider.valueChanged.connect(self._on_slider)
        v.addWidget(self._thr_slider)

        # REST / FLEX marker labels below slider
        markers = QHBoxLayout()
        self._rest_marker = _lbl("REST: —", 10, color=C_DIM)
        self._flex_marker = _lbl("FLEX: —", 10, color=C_GREEN)
        markers.addWidget(self._rest_marker)
        markers.addStretch()
        markers.addWidget(self._flex_marker)
        v.addLayout(markers)

        self._cal_summary = _lbl("", 11, color=C_DIM)
        self._cal_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._cal_summary)

        self._quality_lbl = _lbl("", 11, color=C_GREEN)
        self._quality_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._quality_lbl)

        btn_row = QHBoxLayout()
        redo_btn = QPushButton("↺  Redo FLEX  (Space)")
        redo_btn.clicked.connect(self._space_pressed)
        game_btn = QPushButton("▶  Jump Game (G)")
        game_btn.setStyleSheet(
            "QPushButton { background:#007A40; border:none; border-radius:8px;"
            " color:#E8ECF8; font-size:13px; font-weight:bold;"
            " padding:8px 18px; } QPushButton:hover { background:#00A858; }"
        )
        game_btn.clicked.connect(self.nav_game)
        rxn_btn = QPushButton("⚡  Reaction (R)")
        rxn_btn.setStyleSheet(
            "QPushButton { background:#6E2E90; border:none; border-radius:8px;"
            " color:#E8ECF8; font-size:13px; font-weight:bold;"
            " padding:8px 18px; } QPushButton:hover { background:#9040C0; }"
        )
        rxn_btn.clicked.connect(self.nav_reaction)
        btn_row.addWidget(redo_btn)
        btn_row.addWidget(game_btn)
        btn_row.addWidget(rxn_btn)
        v.addLayout(btn_row)

        self._step_stack.addWidget(w)

    # ── Internal logic ───────────────────────────────────────────────────────

    def _on_slider(self, val: int):
        thr = val / 10.0
        self._cal.threshold = thr
        self._ctrl.set_threshold(thr)
        self._thr_val_lbl.setText(f"Threshold:  {thr:.1f} µV")

    def handle_space(self):
        self._space_pressed()

    def _space_pressed(self):
        if self._recording:
            return
        if self.step == 0:
            self._start_record('rest')
        elif self.step == 1:
            self._start_record('rest')
        elif self.step == 2:
            self._start_record('flex')
        elif self.step == 3:
            self._start_record('flex')   # redo flex

    def _start_record(self, kind: str):
        if kind == 'rest':
            self.step = 1
            self._cal.reset()
            self._ctrl.reset()
        else:
            self.step = 2

        self._recording = True
        self._progress  = 0.0
        self._kind      = kind
        self._update_step_view()

        smooth_samples = self._proc.smooth_samples
        ch = self._proc.channel

        def do_record():
            start = time.monotonic()
            activations = []
            while time.monotonic() - start < CAL_DURATION_S:
                disp = self._emg.peek(smooth_samples)
                act  = EMGProcessor.compute_activation_from_window(
                    disp, ch, smooth_samples
                )
                activations.append(act)
                self._progress = min(
                    1.0, (time.monotonic() - start) / CAL_DURATION_S
                )
                time.sleep(0.04)

            if kind == 'rest':
                self._cal.record_rest(np.array(activations))
                self.step = 2
            else:
                self._cal.record_flex(np.array(activations))
                self._cal.compute_threshold(sensitivity=0.45)
                self._ctrl.set_threshold(self._cal.threshold)
                self.step = 3

            self._recording = False
            self._progress  = 1.0

        threading.Thread(target=do_record, daemon=True).start()

    def _update_step_view(self):
        # Step dots
        for i, d in enumerate(self._step_dots):
            if i < self.step:
                d.setStyleSheet("color: #00D26E; font-size: 14px;")
            elif i == self.step:
                d.setStyleSheet("color: #4D9FFF; font-size: 18px;")
            else:
                d.setStyleSheet("color: #2A3A5E; font-size: 14px;")
        label = self.STEPS[min(self.step, len(self.STEPS) - 1)]
        self._step_lbl.setText(
            f"Step {min(self.step + 1, 4)} / 4  —  {label}"
        )
        # Show the right step widget
        idx = min(self.step, self._step_stack.count() - 1)
        self._step_stack.setCurrentIndex(idx)

    def update_frame(self, activation: float, active_ch: int):
        self._frame += 1
        n    = max(90, int(self._emg.sampling_rate * DISPLAY_SECS_UI))
        disp = self._emg.peek(n)
        self._plot.update_data(disp, active_ch, 0.0)
        self._act_bar.set_values(activation, self._ctrl.threshold)
        self._ch_sel.set_active(active_ch)
        self._update_step_view()

        # Update recording progress bars
        if self.step == 1:
            if self._recording:
                self._rest_prog.set_progress(self._progress)
                secs = CAL_DURATION_S * (1 - self._progress)
                self._step1_status.setText(
                    f"Recording REST…  {secs:.1f}s remaining"
                )
                self._step1_status.setStyleSheet("color: #FF4848;")
            else:
                self._rest_prog.set_progress(0.0)
                self._step1_status.setText("Press SPACE to begin REST recording")
                self._step1_status.setStyleSheet("color: #4D9FFF;")

        elif self.step == 2:
            if self._recording:
                self._flex_prog.set_progress(self._progress)
                secs = CAL_DURATION_S * (1 - self._progress)
                self._step2_status.setText(
                    f"Recording FLEX…  {secs:.1f}s remaining"
                )
                self._step2_status.setStyleSheet("color: #00D26E;")
            else:
                self._flex_prog.set_progress(0.0)
                self._step2_status.setText("Press SPACE to begin FLEX recording")
                self._step2_status.setStyleSheet("color: #4D9FFF;")

        elif self.step == 3:
            thr = self._cal.threshold
            self._thr_slider.blockSignals(True)
            self._thr_slider.setValue(int(thr * 10))
            self._thr_slider.blockSignals(False)
            self._thr_val_lbl.setText(f"Threshold:  {thr:.1f} µV")

            is_act = activation >= thr
            if is_act:
                self._trigger_badge.setText("  ● TRIGGERED  ")
                self._trigger_badge.setStyleSheet(
                    "border-radius:13px; background:#00D26E; color:#0B0F1A;"
                    " font-weight:bold; border:none;"
                )
            else:
                self._trigger_badge.setText("  ○ not triggered  ")
                self._trigger_badge.setStyleSheet(
                    "border-radius:13px; background:#1C263E; color:#5A6A94;"
                    " border:1px solid #2A3A5E;"
                )

            if self._cal.is_calibrated():
                self._rest_marker.setText(
                    f"REST p99: {self._cal.rest_p99:.1f} µV"
                )
                self._flex_marker.setText(
                    f"FLEX p25: {self._cal.flex_p25:.1f} µV"
                )
                self._cal_summary.setText(self._cal.summary())
                quality = (
                    "Good separation"
                    if self._cal.quality_margin >= 8.0
                    else "Weak separation — try flexing harder"
                )
                qc = "#00D26E" if self._cal.quality_margin >= 8.0 else "#FFD232"
                self._quality_lbl.setText(quality)
                self._quality_lbl.setStyleSheet(f"color: {qc};")


class _ProgressBar(QWidget):
    """Simple horizontal progress bar used in calibration steps."""

    def __init__(self, color: QColor = C_RED, parent=None):
        super().__init__(parent)
        self._color    = color
        self._progress = 0.0
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_progress(self, v: float):
        if self._progress != v:
            self._progress = v
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        r    = QRectF(0, 0, w, h)
        rad  = h / 2
        bg   = QPainterPath()
        bg.addRoundedRect(r, rad, rad)
        p.fillPath(bg, C_PANEL2)
        if self._progress > 0.01:
            fp = QPainterPath()
            fp.addRoundedRect(QRectF(0, 0, w * self._progress, h), rad, rad)
            p.fillPath(fp, self._color)
        p.setPen(QPen(C_BORDER, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(bg)

        # Percentage text
        f2 = QFont()
        f2.setPointSize(10)
        p.setFont(f2)
        p.setPen(QPen(C_TEXT, 1))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter,
                   f"{int(self._progress * 100)}%")
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# Page: Jump Game
# ─────────────────────────────────────────────────────────────────────────────

class GameCanvas(QWidget):
    """Renders the jump game using QPainter, scaled to widget size."""

    def __init__(self, logic: JumpGameLogic, parent=None):
        super().__init__(parent)
        self._logic      = logic
        self._activation = 0.0
        self._threshold  = 50.0
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 300)

    def set_emg(self, activation: float, threshold: float):
        self._activation = activation
        self._threshold  = threshold

    def paintEvent(self, event):
        p   = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cw  = self.width()
        ch  = self.height()
        sx  = cw / GAME_W
        sy  = ch / GAME_H
        lg  = self._logic

        # Scale transform so all game coordinates map to canvas size
        p.save()
        p.scale(sx, sy)

        # Background
        p.fillRect(0, 0, GAME_W, GAME_H, C_BG)

        # Stars
        p.setPen(Qt.PenStyle.NoPen)
        for (stx, sty, br) in lg.stars:
            c = int(60 + br * 100)
            p.setBrush(QBrush(QColor(c, c, min(255, int(c * 1.3)))))
            p.drawEllipse(stx, sty, 2, 2)

        # Ground
        p.fillRect(0, GROUND_Y, GAME_W, 8,
                   QColor(42, 160, 100))
        p.fillRect(0, GROUND_Y + 8, GAME_W, GAME_H - GROUND_Y,
                   QColor(28, 110, 68))

        # Obstacles
        for obs in lg.obstacles:
            ox, oy = int(obs.x), int(obs.y)
            p.setBrush(QBrush(C_RED))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(ox, oy, obs.w, obs.h, 5, 5)
            p.fillRect(ox, oy, obs.w, 8, QColor(255, 140, 140))

        # Player
        pl  = lg.player
        px  = int(pl.x)
        py  = int(pl.y)
        pc  = QColor(140, 210, 255) if pl.flashing else QColor(77, 159, 255)
        p.setBrush(QBrush(pc))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(px, py, pl.W, pl.H, 7, 7)
        # Visor
        p.fillRect(px + 4, py + 6, pl.W - 8, 16, QColor(20, 40, 80))
        p.setBrush(Qt.BrushStyle.NoBrush)
        vpen = QPen(C_ACCENT, 1)
        p.setPen(vpen)
        p.drawRoundedRect(px + 4, py + 6, pl.W - 8, 16, 4, 4)
        # Legs
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(50, 100, 180)))
        p.drawRoundedRect(px + 4,  py + pl.H - 4, 10, 6, 3, 3)
        p.drawRoundedRect(px + 20, py + pl.H - 4, 10, 6, 3, 3)

        # Score HUD
        f_score = QFont()
        f_score.setPointSize(20)
        f_score.setBold(True)
        p.setFont(f_score)
        p.setPen(QPen(C_TEXT, 1))
        fm = p.fontMetrics()
        score_str = str(lg.score)
        sw = fm.horizontalAdvance(score_str)
        p.drawText(GAME_W - sw - 20, 40, score_str)

        f_small = QFont()
        f_small.setPointSize(11)
        p.setFont(f_small)
        p.setPen(QPen(C_DIM, 1))
        best_str = f"BEST  {lg.high_score}"
        bw = p.fontMetrics().horizontalAdvance(best_str)
        p.drawText(GAME_W - bw - 20, 60, best_str)

        # Speed dots
        spd = min(10, int(lg.speed))
        for i in range(10):
            dc = C_ACCENT if i < spd else C_BORDER
            p.setBrush(QBrush(dc))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(20 + i * 14, 18, 8, 8)

        # Overlay (ready / dead)
        if lg.state in ('ready', 'dead'):
            ov = QColor(5, 8, 18, 170)
            p.fillRect(0, 0, GAME_W, GAME_H, ov)

            f_big = QFont()
            f_big.setPointSize(36)
            f_big.setBold(True)
            p.setFont(f_big)
            p.setPen(QPen(C_TEXT, 1))

            if lg.state == 'ready':
                title = "FLEX TO START"
                body  = "Flex your forearm to begin"
            else:
                title = "GAME OVER"
                body  = f"Score  {lg.score}   ·   Best  {lg.high_score}"

            tw = p.fontMetrics().horizontalAdvance(title)
            p.drawText((GAME_W - tw) // 2, GAME_H // 2 - 20, title)

            f_body = QFont()
            f_body.setPointSize(16)
            p.setFont(f_body)
            p.setPen(QPen(C_DIM, 1))
            bw2 = p.fontMetrics().horizontalAdvance(body)
            p.drawText((GAME_W - bw2) // 2, GAME_H // 2 + 30, body)

            if lg.state == 'dead':
                hint = "Flex to restart"
                f_hint = QFont()
                f_hint.setPointSize(13)
                p.setFont(f_hint)
                hw = p.fontMetrics().horizontalAdvance(hint)
                p.drawText((GAME_W - hw) // 2, GAME_H // 2 + 60, hint)

        p.restore()


class _GameSidebar(QWidget):
    """Right-panel EMG display for the game page."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._activation = 0.0
        self._threshold  = 50.0
        self.setFixedWidth(160)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

    def set_values(self, activation: float, threshold: float):
        self._activation = activation
        self._threshold  = threshold
        self.update()

    def paintEvent(self, event):
        p   = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w   = self.width()
        h   = self.height()

        # Background
        p.fillRect(0, 0, w, h, C_PANEL)
        p.setPen(QPen(C_BORDER, 1))
        p.drawLine(0, 0, 0, h)

        # Title
        f_title = QFont()
        f_title.setPointSize(11)
        f_title.setBold(True)
        p.setFont(f_title)
        p.setPen(QPen(C_DIM, 1))
        p.drawText(QRectF(0, 10, w, 24),
                   Qt.AlignmentFlag.AlignCenter, "EMG")

        # Vertical bar
        bx = 30; bw2 = 50; by = 50; bh = 280
        p.setBrush(QBrush(C_PANEL2))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(bx, by, bw2, bh, 6, 6)
        p.setPen(QPen(C_BORDER, 1))
        p.drawRoundedRect(bx, by, bw2, bh, 6, 6)

        a, thr  = self._activation, self._threshold
        max_v   = max(thr * 2.5, a * 1.2, 50.0)
        ratio   = min(1.0, a / max_v)
        fill_h  = int(bh * ratio)
        fill_y  = by + bh - fill_h
        is_act  = a >= thr
        fill_c  = C_GREEN if is_act else C_ACCENT
        if fill_h > 4:
            p.setBrush(QBrush(fill_c))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(bx, fill_y, bw2, fill_h, 6, 6)

        # Threshold marker
        if thr > 0 and max_v > 0:
            thr_ratio = min(1.0, thr / max_v)
            ty = by + bh - int(bh * thr_ratio)
            p.setPen(QPen(C_YELLOW, 2))
            p.drawLine(bx - 6, ty, bx + bw2 + 6, ty)
            f_tiny = QFont()
            f_tiny.setPointSize(9)
            p.setFont(f_tiny)
            p.setPen(QPen(C_YELLOW, 1))
            p.drawText(QRectF(0, ty - 18, w, 14),
                       Qt.AlignmentFlag.AlignCenter, "THR")

        # µV value
        f_val = QFont()
        f_val.setPointSize(13)
        f_val.setBold(True)
        p.setFont(f_val)
        p.setPen(QPen(C_GREEN if is_act else C_TEXT, 1))
        val_y = by + bh + 22
        p.drawText(QRectF(0, val_y, w, 20),
                   Qt.AlignmentFlag.AlignCenter, f"{a:.0f}")
        f_uv = QFont()
        f_uv.setPointSize(9)
        p.setFont(f_uv)
        p.setPen(QPen(C_DIM, 1))
        p.drawText(QRectF(0, val_y + 22, w, 14),
                   Qt.AlignmentFlag.AlignCenter, "µV")

        # FLEX / rest pill
        pill_y = val_y + 46
        pill_h = 26
        pill_c = C_GREEN if is_act else C_PANEL2
        p.setBrush(QBrush(pill_c))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(10, pill_y, w - 20, pill_h, 13, 13)
        p.setPen(QPen(C_BORDER, 1))
        p.drawRoundedRect(10, pill_y, w - 20, pill_h, 13, 13)
        f_pill = QFont()
        f_pill.setPointSize(10)
        f_pill.setBold(is_act)
        p.setFont(f_pill)
        p.setPen(QPen(QColor(11, 15, 26) if is_act else C_DIM, 1))
        p.drawText(QRectF(10, pill_y, w - 20, pill_h),
                   Qt.AlignmentFlag.AlignCenter,
                   "FLEX!" if is_act else "rest")

        # Hints
        f_hint = QFont()
        f_hint.setPointSize(9)
        p.setFont(f_hint)
        p.setPen(QPen(C_BORDER, 1))
        p.drawText(QRectF(0, h - 42, w, 14),
                   Qt.AlignmentFlag.AlignCenter, "C = calibrate")
        p.drawText(QRectF(0, h - 24, w, 14),
                   Qt.AlignmentFlag.AlignCenter, "M = menu")
        p.end()


class GamePage(QWidget):
    nav_back = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logic   = JumpGameLogic()
        self._build()

    def _build(self):
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        self._canvas  = GameCanvas(self.logic)
        self._sidebar = _GameSidebar()
        row.addWidget(self._canvas, stretch=1)
        row.addWidget(self._sidebar)

    def trigger_jump(self):
        self.logic.trigger_jump()

    def update_frame(self, activation: float, threshold: float):
        self.logic.update()
        self._sidebar.set_values(activation, threshold)
        self._canvas.update()


# ─────────────────────────────────────────────────────────────────────────────
# Page: Reaction Challenge  (QPainter canvas)
# ─────────────────────────────────────────────────────────────────────────────

def _reaction_tier(ms: float) -> tuple[str, QColor]:
    """Return (label, color) performance tier for a reaction time in ms."""
    if ms < 180:  return ("⚡ LIGHTNING",  QColor(255, 220,  50))
    if ms < 250:  return ("🔥 EXCELLENT",  QColor(  0, 210, 110))
    if ms < 350:  return ("✓  GOOD",        QColor( 77, 159, 255))
    if ms < 500:  return ("○  AVERAGE",     QColor(180, 180, 180))
    return            ("△  SLOW",          QColor(160, 100, 100))


class _ReactionCanvas(QWidget):
    """Draws all reaction-challenge states via QPainter."""

    def __init__(self, logic: ReactionLogic, parent=None):
        super().__init__(parent)
        self.logic       = logic
        self._activation = 0.0
        self._threshold  = 50.0
        self._frame      = 0
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    def set_emg(self, activation: float, threshold: float):
        self._activation = activation
        self._threshold  = threshold

    def paintEvent(self, _):
        self._frame += 1
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        w, h = self.width(), self.height()
        state = self.logic.state

        if state == 'go':
            # Pulsing green background
            pulse = 0.5 + 0.5 * abs((self._frame % 20) / 10.0 - 1.0)
            g = int(20 + pulse * 30)
            p.fillRect(0, 0, w, h, QColor(0, g, 0))
        else:
            p.fillRect(0, 0, w, h, C_BG)

        cx, cy = w // 2, h // 2

        if state == 'waiting':
            self._draw_waiting(p, cx, cy, w, h)
        elif state == 'go':
            self._draw_go(p, cx, cy, w, h)
        elif state == 'result':
            self._draw_result(p, cx, cy, w, h)
        elif state == 'leaderboard':
            self._draw_leaderboard(p, cx, cy, w, h)

        self._draw_emg_strip(p, w, h)
        p.end()

    # ── State drawers ─────────────────────────────────────────────────────────

    def _draw_waiting(self, p: QPainter, cx, cy, w, h):
        # Countdown ring showing fraction of wait time remaining
        t_left  = self.logic.time_until_go
        t_total = self.logic.wait_total
        frac    = 1.0 - (t_left / t_total if t_total > 0 else 0)

        ring_r = min(w, h) // 5
        ring_x = cx - ring_r
        ring_y = cy - ring_r - h // 8
        ring_sz = ring_r * 2

        # Track ring
        p.setPen(QPen(C_PANEL2, 10, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(ring_x, ring_y, ring_sz, ring_sz)

        # Progress arc (red → yellow as GO approaches)
        if frac > 0:
            r = int(255)
            g = int(min(210, frac * 210))
            arc_color = QColor(r, g, 0)
            pen = QPen(arc_color, 10, Qt.PenStyle.SolidLine,
                       Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            span = int(frac * 360 * 16)
            p.drawArc(ring_x, ring_y, ring_sz, ring_sz, 90 * 16, -span)

        # Seconds inside ring
        self._text(p, f"{t_left:.1f}s", cx, cy - h // 8,
                   QColor(232, 236, 248), 22, bold=True)

        # Heading
        self._text(p, "GET READY", cx, cy + ring_r + 20 - h // 8,
                   C_RED, 34, bold=True)

        # Animated dots
        dots = "·" * (1 + (self._frame // 18) % 3)
        self._text(p, f"Wait for it {dots}", cx, cy + ring_r + 60 - h // 8,
                   C_DIM, 16)

        # Participant name
        self._text(p, f"Participant: {self.logic.current_name}",
                   cx, cy + ring_r + 90 - h // 8, C_DIM, 13)

    def _draw_go(self, p: QPainter, cx, cy, w, h):
        # Pulsing scale for GO text
        pulse = 0.92 + 0.08 * abs((self._frame % 14) / 7.0 - 1.0)
        size  = int(90 * pulse)

        # Outer glow rings
        for r_off, alpha in [(60, 30), (40, 60), (20, 100)]:
            glow = QColor(0, 210, 110, alpha)
            p.setPen(QPen(glow, 4))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(cx - r_off, cy - r_off - 40, r_off * 2, r_off * 2)

        self._text(p, "GO!", cx, cy - 40, C_GREEN, size, bold=True)
        self._text(p, "FLEX NOW!", cx, cy + 55, QColor(180, 255, 180), 26)
        self._text(p, self.logic.current_name, cx, cy + 95, C_DIM, 14)

    def _draw_result(self, p: QPainter, cx, cy, w, h):
        if self.logic.false_start:
            self._text(p, "FALSE START!", cx, cy - 60, C_RED, 42, bold=True)
            self._text(p, "You flexed before GO!", cx, cy, C_DIM, 18)
            self._text(p, "Press N to try again", cx, cy + 40, C_DIM, 14)
            return

        ms = self.logic.result_ms
        tier_label, tier_color = _reaction_tier(ms)

        # Tier badge background
        badge_w, badge_h = 260, 44
        bx, by = cx - badge_w // 2, cy - 130
        path = QPainterPath()
        path.addRoundedRect(QRectF(bx, by, badge_w, badge_h), 22, 22)
        badge_bg = QColor(tier_color)
        badge_bg.setAlpha(40)
        p.fillPath(path, badge_bg)
        p.setPen(QPen(tier_color, 2))
        p.drawPath(path)
        self._text(p, tier_label, cx, by + badge_h // 2, tier_color, 16, bold=True)

        # Name
        self._text(p, self.logic.current_name, cx, cy - 60, C_TEXT, 20, bold=True)

        # Big time
        self._text(p, f"{ms:.0f}", cx, cy + 20, tier_color, 72, bold=True)
        self._text(p, "ms", cx, cy + 70, C_DIM, 20)

        # Rank
        rank = self.logic.get_rank_label()
        rank_color = C_GOLD if rank == "NEW RECORD!" else C_DIM
        self._text(p, rank, cx, cy + 100, rank_color, 18, bold=(rank == "NEW RECORD!"))

        # Action hint
        self._text(p, "L = Leaderboard   ·   N = Next participant",
                   cx, h - 60, C_DIM, 12)

    def _draw_leaderboard(self, p: QPainter, cx, cy, w, h):
        # Header
        self._text(p, "LEADERBOARD", cx, 50, C_GOLD, 28, bold=True)

        lb = self.logic.leaderboard
        if not lb:
            self._text(p, "No scores yet", cx, cy, C_DIM, 16)
            self._text(p, "N = Add participant", cx, h - 60, C_DIM, 12)
            return

        # Time bars — scale to best time
        best_ms = lb[0][1] if lb else 1.0
        bar_max_w = min(w * 0.45, 400)
        row_h = min(48, (h - 160) // max(len(lb[:8]), 1))
        start_y = 90

        medals = ["🥇", "🥈", "🥉"]
        colors = [C_GOLD, C_SILVER, C_BRONZE]

        for i, (name, ms) in enumerate(lb[:8]):
            ry = start_y + i * row_h
            color = colors[i] if i < 3 else C_TEXT
            prefix = medals[i] if i < 3 else f"#{i+1}"

            # Bar (relative to best)
            ratio = best_ms / ms if ms > 0 else 1.0
            bar_w = int(bar_max_w * ratio)
            bar_x = cx - int(bar_max_w) - 10
            bar_y = ry + row_h // 2 - 10

            bar_bg = QColor(color)
            bar_bg.setAlpha(20)
            p.fillRect(bar_x, bar_y, int(bar_max_w), 20, bar_bg)
            bar_fill = QColor(color)
            bar_fill.setAlpha(120)
            p.fillRect(bar_x, bar_y, bar_w, 20, bar_fill)

            # Rank + name
            p.setPen(QPen(color))
            f = QFont("Arial", 14 if i < 3 else 13)
            if i < 3:
                f.setBold(True)
            p.setFont(f)
            p.drawText(QRectF(cx - int(bar_max_w) - 10, ry, bar_max_w, row_h),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       f"  {prefix}  {name}")

            # Time
            f2 = QFont("Arial", 13, QFont.Weight.Bold if i == 0 else QFont.Weight.Normal)
            p.setFont(f2)
            p.drawText(QRectF(cx + 14, ry, 120, row_h),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       f"{ms:.0f} ms")

        self._text(p, "N = Add next participant", cx, h - 40, C_DIM, 12)

    def _draw_emg_strip(self, p: QPainter, w, h):
        a, thr = self._activation, self._threshold
        bx, by, bw, bh = 16.0, h - 30.0, 220.0, 14.0

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(C_PANEL2))
        p.drawRoundedRect(QRectF(bx, by, bw, bh), bh / 2, bh / 2)

        max_v = max(thr * 2.5, a * 1.1, 50.0)
        fw    = bw * min(1.0, a / max_v) if max_v > 0 else 0
        color = C_GREEN if a >= thr else C_ACCENT
        if fw > 2:
            p.setBrush(QBrush(color))
            p.drawRoundedRect(QRectF(bx, by, fw, bh), bh / 2, bh / 2)

        if thr > 0 and max_v > 0:
            tx = bx + bw * min(1.0, thr / max_v)
            p.setPen(QPen(C_YELLOW, 2))
            p.drawLine(QPointF(tx, by - 4), QPointF(tx, by + bh + 4))

        p.setPen(QPen(C_BORDER, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(bx, by, bw, bh), bh / 2, bh / 2)

        p.setPen(QPen(C_DIM))
        f = QFont("Arial", 9)
        p.setFont(f)
        p.drawText(QRectF(bx, by - 18, 180, 14),
                   Qt.AlignmentFlag.AlignLeft, f"EMG  {a:.0f} µV")

    @staticmethod
    def _text(p: QPainter, text: str, cx, cy, color: QColor,
              size: int, bold=False):
        p.setPen(QPen(color))
        f = QFont("Arial", size)
        if bold:
            f.setBold(True)
        p.setFont(f)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(text)
        th = fm.height()
        p.drawText(cx - tw // 2, cy + th // 3, text)


class ReactionPage(QWidget):
    nav_back = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logic  = ReactionLogic()
        self._frame = 0
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top bar ──────────────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setContentsMargins(16, 10, 16, 0)
        back_btn = QPushButton("◀  Menu")
        back_btn.setFixedWidth(90)
        back_btn.clicked.connect(self.nav_back)
        top.addWidget(back_btn)
        top.addStretch()
        hint = _lbl("F11 = fullscreen   M = menu", 10, color=C_DIM)
        top.addWidget(hint)
        root.addLayout(top)

        # ── Name-entry form (only shown in enter_name state) ─────────────────
        self._name_card = QWidget()
        nc = QVBoxLayout(self._name_card)
        nc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nc.setSpacing(14)
        nc.setContentsMargins(0, 0, 0, 0)

        nc.addWidget(_lbl("⚡  Reaction Challenge", 28, bold=True, color=C_TEXT),
                     alignment=Qt.AlignmentFlag.AlignCenter)
        nc.addWidget(_lbl("Flex your forearm as fast as possible when GO appears!",
                           13, color=C_DIM),
                     alignment=Qt.AlignmentFlag.AlignCenter)
        nc.addSpacing(12)

        name_row = QHBoxLayout()
        name_row.setSpacing(8)
        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("Enter your name…")
        self._name_input.setMaxLength(14)
        self._name_input.setFixedSize(300, 48)
        self._name_input.returnPressed.connect(self._submit_name)
        go_btn = QPushButton("Go →")
        go_btn.setFixedSize(72, 48)
        go_btn.setStyleSheet(
            "QPushButton { background:#4D9FFF; border:none; border-radius:8px;"
            " color:#0B0F1A; font-size:15px; font-weight:bold; }"
            "QPushButton:hover { background:#77BFFF; }"
        )
        go_btn.clicked.connect(self._submit_name)
        name_row.addStretch()
        name_row.addWidget(self._name_input)
        name_row.addWidget(go_btn)
        name_row.addStretch()
        nc.addLayout(name_row)

        nc.addWidget(_lbl("Tip: false-starting before GO resets your attempt!",
                           11, color=C_DIM),
                     alignment=Qt.AlignmentFlag.AlignCenter)

        root.addWidget(self._name_card, stretch=1)

        # ── Canvas (waiting / go / result / leaderboard) ─────────────────────
        self._canvas = _ReactionCanvas(self.logic)
        root.addWidget(self._canvas, stretch=1)

        self._name_card.hide()  # canvas visible by default; toggled in update

    def _submit_name(self):
        text = self._name_input.text().strip()
        if text:
            self.logic.name_input = text
            self.logic.submit_name()
            self._name_input.clear()

    def handle_key(self, key: int, text: str):
        state = self.logic.state
        if state == 'result':
            if key == Qt.Key.Key_L:
                self.logic.next_participant()
            elif key == Qt.Key.Key_N:
                self.logic.new_participant()
        elif state == 'leaderboard':
            if key == Qt.Key.Key_N:
                self.logic.new_participant()

    def update_frame(self, activation: float, threshold: float,
                     triggered: bool):
        self._frame += 1
        self.logic.update()
        if triggered:
            self.logic.trigger_flex()

        self._canvas.set_emg(activation, threshold)

        # Show name-entry card OR the canvas
        in_name = (self.logic.state == 'enter_name')
        self._name_card.setVisible(in_name)
        self._canvas.setVisible(not in_name)

        if not in_name:
            self._canvas.update()


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class EMGWorkshopWindow(QMainWindow):

    def __init__(self, emg: EMGInput, proc: EMGProcessor,
                 cal: Calibration, ctrl: TriggerController,
                 default_ch: int):
        super().__init__()
        self._emg        = emg
        self._proc       = proc
        self._cal        = cal
        self._ctrl       = ctrl
        self._active_ch  = default_ch
        self._manual_trig = False

        self.setWindowTitle("EMG BCI Workshop — Triton Neurotech")
        self.setMinimumSize(800, 560)
        self.resize(WIN_W, WIN_H)
        self.setStyleSheet(STYLESHEET)

        # Pages
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._menu_page = MenuPage(emg, ctrl, cal)
        self._cal_page  = CalibrationPage(emg, proc, cal, ctrl)
        self._game_page = GamePage()
        self._rxn_page  = ReactionPage()

        self._stack.addWidget(self._menu_page)   # 0
        self._stack.addWidget(self._cal_page)    # 1
        self._stack.addWidget(self._game_page)   # 2
        self._stack.addWidget(self._rxn_page)    # 3

        # Wire navigation signals
        self._menu_page.nav_calibrate.connect(self._goto_cal)
        self._menu_page.nav_game.connect(self._goto_game)
        self._menu_page.nav_reaction.connect(self._goto_rxn)
        self._menu_page.channel_changed.connect(self._set_channel)

        self._cal_page.nav_back.connect(self._goto_menu)
        self._cal_page.nav_game.connect(self._goto_game)
        self._cal_page.nav_reaction.connect(self._goto_rxn)
        self._cal_page.channel_changed.connect(self._set_channel)

        self._game_page.nav_back.connect(self._goto_menu)
        self._rxn_page.nav_back.connect(self._goto_menu)

        # Update timer
        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        # Set initial channel selection
        self._menu_page._ch_sel.set_active(default_ch)
        self._cal_page._ch_sel.set_active(default_ch)

    # ── Navigation ─────────────────────────────────────────────────────────────

    def _goto_menu(self):
        self._stack.setCurrentIndex(0)

    def _goto_cal(self):
        self._cal_page.reset()
        self._stack.setCurrentIndex(1)

    def _goto_game(self):
        self._stack.setCurrentIndex(2)

    def _goto_rxn(self):
        self._stack.setCurrentIndex(3)

    def _set_channel(self, ch: int):
        self._active_ch = ch
        self._proc.set_channel(ch)
        self._menu_page._ch_sel.set_active(ch)
        self._cal_page._ch_sel.set_active(ch)

    # ── Per-frame update ───────────────────────────────────────────────────────

    def _tick(self):
        # Compute activation (always from peek — never from pull)
        _peek_n = max(self._proc.smooth_samples * 4, 64)
        _disp   = self._emg.peek(_peek_n)
        activation = EMGProcessor.compute_activation_from_window(
            _disp, self._active_ch, self._proc.smooth_samples
        )

        triggered         = self._ctrl.update(activation) or self._manual_trig
        self._manual_trig = False

        idx = self._stack.currentIndex()

        if idx == 0:
            self._menu_page.update_frame(activation, self._active_ch)
        elif idx == 1:
            self._active_ch = self._proc.channel
            self._cal_page.update_frame(activation, self._active_ch)
        elif idx == 2:
            if triggered:
                self._game_page.trigger_jump()
            self._game_page.update_frame(activation, self._ctrl.threshold)
        elif idx == 3:
            self._rxn_page.update_frame(activation, self._ctrl.threshold,
                                         triggered)

    # ── Keyboard ───────────────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent):
        k = event.key()
        idx = self._stack.currentIndex()

        if k == Qt.Key.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.close()
        elif k == Qt.Key.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif k == Qt.Key.Key_M:
            self._goto_menu()
        elif k == Qt.Key.Key_C:
            self._goto_cal()
        elif k == Qt.Key.Key_G:
            self._goto_game()
        elif k == Qt.Key.Key_R:
            self._goto_rxn()
        elif Qt.Key.Key_1 <= k <= Qt.Key.Key_8:
            ch = k - Qt.Key.Key_1
            if ch < self._emg.n_channels and ch not in DISABLED_CHANNELS:
                self._set_channel(ch)
        elif k == Qt.Key.Key_Space:
            if idx == 1:
                self._cal_page.handle_space()
            else:
                self._manual_trig = True
        elif idx == 3:
            self._rxn_page.handle_key(k, event.text())

        super().keyPressEvent(event)

    def closeEvent(self, event):
        self._timer.stop()
        self._emg.stop()
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("EMG BCI Workshop")

    emg = EMGInput()
    emg.start()

    channel_metrics = probe_channel_metrics(emg)
    default_ch      = choose_default_channel(channel_metrics, emg.n_channels)

    proc = EMGProcessor(
        sampling_rate=emg.sampling_rate, smooth_ms=90, channel=default_ch
    )
    cal  = Calibration()
    ctrl = TriggerController(
        threshold=50.0, refractory_ms=250, release_ratio=0.65, hold_ms=70
    )

    print("=" * 54)
    print("  EMG BCI Workshop  |  Triton Neurotech")
    print(f"  Backend: {emg.backend}  |  {emg.n_channels} ch @ {emg.sampling_rate} Hz")
    metrics_str = "  ".join(
        f"CH{i+1}:{m:.1f}/{p:.1f}"
        for i, (m, p) in enumerate(channel_metrics)
    )
    print(f"  Rest probe mean/p95 µV: {metrics_str}")
    print(f"  Default channel: CH{default_ch + 1}")
    print("=" * 54)

    win = EMGWorkshopWindow(emg, proc, cal, ctrl, default_ch)
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
