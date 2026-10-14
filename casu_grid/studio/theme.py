"""Colours, fonts and the application stylesheet (light and dark), plus the semantic status vocabulary."""
from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtGui

# Okabe-Ito colour-blind safe palette for data series (identity is also carried by legend text and line dashes)
SERIES_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]


@dataclass(frozen=True)
class Palette:
    name: str
    window: str
    panel: str
    panel_alt: str
    border: str
    text: str
    muted: str
    accent: str
    accent_text: str
    chart_bg: str
    chart_grid: str
    chart_axis: str
    good: str
    warn: str
    bad: str
    info: str


LIGHT = Palette("light", window="#f3f5f8", panel="#ffffff", panel_alt="#eef1f6", border="#d5dbe5", text="#1c2430", muted="#5d6b7e",
                accent="#1f5fbf", accent_text="#ffffff", chart_bg="#ffffff", chart_grid="#e3e8ef", chart_axis="#4a566a",
                good="#1a7f4b", warn="#a86400", bad="#b3261e", info="#1f5fbf")
DARK = Palette("dark", window="#14181f", panel="#1c222c", panel_alt="#232a36", border="#323b4a", text="#e6eaf0", muted="#9aa6b8",
               accent="#5b9bff", accent_text="#0b1220", chart_bg="#1c222c", chart_grid="#2c3442", chart_axis="#b5c0d0",
               good="#4cc38a", warn="#e0a43a", bad="#ff7b72", info="#5b9bff")

# status -> (glyph, label, palette attribute); the glyph and the word carry the meaning, the colour only reinforces it
STATUS = {
    "ready": ("○", "Ready", "muted"),
    "starting": ("◐", "Starting", "info"),
    "running": ("●", "Running", "good"),
    "stopping": ("◑", "Stopping", "warn"),
    "completed": ("✔", "Completed", "good"),
    "failed": ("✖", "Failed", "bad"),
    "interrupted": ("■", "Interrupted", "warn"),
    "unknown": ("?", "Unknown", "muted"),
}


def status_text(status: str) -> str:
    glyph, label, _ = STATUS.get(status, STATUS["unknown"])
    return f"{glyph} {label}"


def status_color(palette: Palette, status: str) -> str:
    return getattr(palette, STATUS.get(status, STATUS["unknown"])[2])


def system_prefers_dark(app) -> bool:
    try:
        from PySide6 import QtCore
        return app.styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark
    except Exception:
        return False


def stylesheet(p: Palette) -> str:
    return f"""
    QWidget {{ color: {p.text}; font-size: 13px; }}
    QMainWindow, QStackedWidget, #page {{ background: {p.window}; }}
    #sidebar {{ background: {p.panel}; border-right: 1px solid {p.border}; }}
    #sidebar QPushButton {{ text-align: left; padding: 9px 14px; border: none; border-radius: 6px; background: transparent; color: {p.text}; font-size: 14px; }}
    #sidebar QPushButton:hover {{ background: {p.panel_alt}; }}
    #sidebar QPushButton:checked {{ background: {p.accent}; }}
    #sidebar QLabel#navLabel {{ background: transparent; font-size: 14px; }}
    #brand {{ font-size: 16px; font-weight: 700; padding: 14px 14px 2px 14px; }}
    #brandSub {{ color: {p.muted}; padding: 0 14px 10px 14px; font-size: 11px; }}
    #pageTitle {{ font-size: 22px; font-weight: 700; }}
    #pageSubtitle, .muted {{ color: {p.muted}; }}
    #card {{ background: {p.panel}; border: 1px solid {p.border}; border-radius: 10px; }}
    #cardTitle {{ font-size: 14px; font-weight: 650; }}
    #heroCard {{ background: {p.panel}; border: 1px solid {p.accent}; border-radius: 12px; }}
    #statTile {{ background: {p.panel_alt}; border-radius: 8px; }}
    #statValue {{ font-size: 20px; font-weight: 700; }}
    #statLabel {{ color: {p.muted}; font-size: 11px; }}
    #hint {{ color: {p.muted}; font-size: 12px; }}
    #banner_bad {{ background: {p.panel}; border: 1px solid {p.bad}; border-left: 5px solid {p.bad}; border-radius: 6px; }}
    #banner_warn {{ background: {p.panel}; border: 1px solid {p.warn}; border-left: 5px solid {p.warn}; border-radius: 6px; }}
    #banner_info {{ background: {p.panel}; border: 1px solid {p.info}; border-left: 5px solid {p.info}; border-radius: 6px; }}
    #banner_good {{ background: {p.panel}; border: 1px solid {p.good}; border-left: 5px solid {p.good}; border-radius: 6px; }}
    QPushButton {{ background: {p.panel}; border: 1px solid {p.border}; border-radius: 6px; padding: 6px 14px; }}
    QPushButton:hover {{ border-color: {p.accent}; }}
    QPushButton:disabled {{ color: {p.muted}; background: {p.panel_alt}; }}
    QPushButton#primary {{ background: {p.accent}; color: {p.accent_text}; border: 1px solid {p.accent}; font-weight: 600; padding: 8px 20px; }}
    QPushButton#primary:disabled {{ background: {p.panel_alt}; color: {p.muted}; border-color: {p.border}; }}
    QPushButton#danger {{ border-color: {p.bad}; color: {p.bad}; }}
    QPushButton#flat {{ border: none; background: transparent; color: {p.accent}; padding: 2px 6px; }}
    QToolButton#info {{ border: 1px solid {p.border}; border-radius: 9px; background: {p.panel_alt}; color: {p.muted}; min-width: 18px; max-width: 18px; min-height: 18px; max-height: 18px; font-size: 11px; font-weight: 700; }}
    QToolButton#info:hover {{ color: {p.accent_text}; background: {p.accent}; }}
    QLineEdit, QPlainTextEdit, QTextEdit {{ background: {p.panel}; border: 1px solid {p.border}; border-radius: 6px; padding: 4px 6px; selection-background-color: {p.accent}; selection-color: {p.accent_text}; }}
    QLineEdit:focus {{ border-color: {p.accent}; }}
    QLineEdit[readOnly="true"] {{ background: {p.panel_alt}; color: {p.muted}; }}
    QTableWidget, QTreeWidget, QListWidget {{ background: {p.panel}; border: 1px solid {p.border}; border-radius: 6px; gridline-color: {p.border}; alternate-background-color: {p.panel_alt}; }}
    QTableWidget::item:selected, QTreeWidget::item:selected, QListWidget::item:selected {{ background: {p.accent}; color: {p.accent_text}; }}
    QHeaderView::section {{ background: {p.panel_alt}; border: none; border-bottom: 1px solid {p.border}; padding: 5px 8px; font-weight: 600; }}
    QTabWidget::pane {{ border: 1px solid {p.border}; border-radius: 6px; background: {p.panel}; top: -1px; }}
    QTabBar::tab {{ padding: 7px 16px; border: 1px solid transparent; border-top-left-radius: 6px; border-top-right-radius: 6px; color: {p.muted}; }}
    QTabBar::tab:selected {{ background: {p.panel}; border-color: {p.border}; border-bottom-color: {p.panel}; color: {p.text}; font-weight: 600; }}
    QScrollArea {{ border: none; background: transparent; }}
    QScrollBar:vertical {{ width: 12px; background: transparent; }}
    QScrollBar::handle:vertical {{ background: {p.border}; border-radius: 5px; min-height: 30px; }}
    QProgressBar {{ border: 1px solid {p.border}; border-radius: 6px; background: {p.panel_alt}; text-align: center; height: 16px; }}
    QProgressBar::chunk {{ background: {p.accent}; border-radius: 5px; }}
    QToolTip {{ background: {p.panel}; color: {p.text}; border: 1px solid {p.border}; padding: 6px; }}
    QGroupBox {{ border: 1px solid {p.border}; border-radius: 8px; margin-top: 12px; padding-top: 10px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {p.muted}; font-weight: 600; }}
    QSplitter::handle {{ background: {p.border}; }}
    """


def qcolor(value: str, alpha: int | None = None) -> QtGui.QColor:
    c = QtGui.QColor(value)
    if alpha is not None:
        c.setAlpha(alpha)
    return c


def apply_application_style(app, p: Palette) -> None:
    """Fusion + a matching QPalette: native-looking spin boxes, combo boxes and arrows in both themes; the stylesheet adds the cards."""
    from PySide6 import QtWidgets
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
    pal = QtGui.QPalette()
    roles = {QtGui.QPalette.Window: p.window, QtGui.QPalette.WindowText: p.text, QtGui.QPalette.Base: p.panel, QtGui.QPalette.AlternateBase: p.panel_alt,
             QtGui.QPalette.Text: p.text, QtGui.QPalette.Button: p.panel, QtGui.QPalette.ButtonText: p.text, QtGui.QPalette.Highlight: p.accent,
             QtGui.QPalette.HighlightedText: p.accent_text, QtGui.QPalette.ToolTipBase: p.panel, QtGui.QPalette.ToolTipText: p.text,
             QtGui.QPalette.PlaceholderText: p.muted, QtGui.QPalette.Mid: p.border, QtGui.QPalette.Dark: p.border, QtGui.QPalette.Light: p.panel}
    for role, color in roles.items():
        pal.setColor(role, QtGui.QColor(color))
    for role in (QtGui.QPalette.Text, QtGui.QPalette.ButtonText, QtGui.QPalette.WindowText):
        pal.setColor(QtGui.QPalette.Disabled, role, QtGui.QColor(p.muted))
    app.setPalette(pal)
    app.setStyleSheet(stylesheet(p))
