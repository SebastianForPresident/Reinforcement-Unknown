"""Interactive 2-D line charts drawn with QPainter (no third-party plotting library).

Design rules
* Every chart is its own widget with a title, axis labels and units; nothing is rendered into a single large image.
* Missing values are gaps. A line is never interpolated across a missing sample, and a lone sample is drawn as a dot.
* The cost of a repaint depends on the width of the plot, not on the length of the run: long series are reduced to a
  min / max envelope per pixel column, the underlying values stay intact for tooltips, zoom and export.
* Hover shows the exact value of every visible series at the nearest sample, with the update and decision counts.
"""
from __future__ import annotations

import bisect
import csv
import math
from dataclasses import dataclass, field
from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from . import theme
from .widgets import FlowLayout

NAN = float("nan")


@dataclass
class Series:
    key: str
    label: str
    xs: list[float]
    ys: list[float | None]
    color: str = "#0072B2"
    run: str = ""
    dashed: bool = False
    visible: bool = True
    smoothable: bool = True
    info: Callable[[int], str] | None = None       # text for the tooltip of sample i (e.g. "update 12 · 49,152 decisions")
    smoothed: list[float | None] | None = None

    def finite_count(self) -> int:
        return sum(1 for y in self.ys if y is not None and y == y and math.isfinite(y))


def smooth_values(ys: list[float | None], window: int) -> list[float | None]:
    """Bias-corrected exponential moving average over the samples that exist; gaps stay gaps."""
    if window <= 1:
        return list(ys)
    alpha = 2.0 / (window + 1.0)
    out: list[float | None] = []
    avg, weight = 0.0, 0.0
    for y in ys:
        if y is None or y != y or not math.isfinite(y):
            out.append(None)
            continue
        avg = (1 - alpha) * avg + alpha * y
        weight = (1 - alpha) * weight + alpha
        out.append(avg / weight)
    return out


def nice_step(span: float, target: int) -> float:
    if span <= 0 or not math.isfinite(span):
        return 1.0
    raw = span / max(1, target)
    magnitude = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 2.5, 5, 10):
        if raw <= factor * magnitude:
            return factor * magnitude
    return 10 * magnitude


def ticks(lo: float, hi: float, target: int) -> list[float]:
    step = nice_step(hi - lo, target)
    first = math.ceil(lo / step - 1e-9) * step
    values, v, guard = [], first, 0
    while v <= hi + step * 1e-9 and guard < 200:
        values.append(round(v, 12))
        v += step
        guard += 1
    return values


def format_number(v: float, step: float | None = None) -> str:
    if v is None or v != v:
        return "–"
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:.3g} G"
    if a >= 1e6:
        return f"{v / 1e6:.3g} M"
    if a >= 1e4:
        return f"{v / 1e3:.3g} k"
    if a == 0:
        return "0"
    if step is not None and step >= 1:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:,.1f}"
    if a >= 1:
        return f"{v:.3g}"
    if a >= 0.001:
        return f"{v:.3g}"
    return f"{v:.2e}"


def format_exact(v: float | None) -> str:
    if v is None or v != v:
        return "no sample"
    a = abs(v)
    if a >= 1e5 or (a != 0 and a < 1e-3):
        return f"{v:.4g}"
    if a >= 100:
        return f"{v:,.2f}"
    return f"{v:.5g}"


@dataclass
class ChartSpec:
    key: str
    title: str
    ylabel: str
    help_key: str = ""
    caption: str = ""                                    # one-line interpretation shown under the chart
    empty: str = "No samples have been recorded yet."
    y_min: float | None = None                           # hard lower bound of the axis (e.g. 0 for rates)
    y_max: float | None = None
    include_zero: bool = False
    references: list[tuple[str, float]] = field(default_factory=list)      # (label, y) horizontal guides, e.g. the KL target
    percent: bool = False


class ChartView(QtWidgets.QWidget):
    """The plot area. Owns the view transform, the hover state and all mouse interaction."""
    viewChanged = QtCore.Signal(object)               # (xmin, xmax) or None after a reset

    def __init__(self, spec: ChartSpec, palette: theme.Palette, compact: bool = False, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.palette_ = palette
        self.compact = compact
        self.series: list[Series] = []
        self.xlabel = "Update"
        self.x_is_int = True
        self.show_raw = True
        self.show_smooth = True
        self.smooth_window = 10
        self._xrange: tuple[float, float] | None = None
        self._yrange: tuple[float, float] | None = None
        self._hover: QtCore.QPointF | None = None
        self._drag_origin: QtCore.QPointF | None = None
        self._drag_view: tuple[float, float, float, float] | None = None
        self._box_origin: QtCore.QPointF | None = None
        self._box_now: QtCore.QPointF | None = None
        self.setMouseTracking(not compact)
        self.setMinimumHeight(90 if compact else 220)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._cache_key = None
        self._cache_paths = None

    # ------------------------------------------------------------------------------------------------ data
    def set_palette(self, palette: theme.Palette):
        self.palette_ = palette
        self.update()

    def set_series(self, series: list[Series], xlabel: str, x_is_int: bool):
        self.series = series
        self.xlabel, self.x_is_int = xlabel, x_is_int
        self._apply_smoothing()
        self._cache_key = None
        self.update()

    def _apply_smoothing(self):
        for s in self.series:
            s.smoothed = smooth_values(s.ys, self.smooth_window) if s.smoothable else None

    def set_smoothing(self, window: int, show_raw: bool, show_smooth: bool):
        self.smooth_window, self.show_raw, self.show_smooth = max(1, window), show_raw, show_smooth
        self._apply_smoothing()
        self._cache_key = None
        self.update()

    def has_data(self) -> bool:
        return any(s.visible and s.finite_count() for s in self.series)

    # ------------------------------------------------------------------------------------------------ view
    def data_extent(self) -> tuple[float, float, float, float] | None:
        xs_lo, xs_hi, ys_lo, ys_hi = math.inf, -math.inf, math.inf, -math.inf
        for s in self.series:
            if not s.visible:
                continue
            for x, y in zip(s.xs, s.ys):
                if y is not None and y == y and math.isfinite(y):
                    xs_lo, xs_hi = min(xs_lo, x), max(xs_hi, x)
                    ys_lo, ys_hi = min(ys_lo, y), max(ys_hi, y)
        if xs_lo > xs_hi:
            return None
        return xs_lo, xs_hi, ys_lo, ys_hi

    def x_view(self) -> tuple[float, float] | None:
        if self._xrange is not None:
            return self._xrange
        ext = self.data_extent()
        if ext is None:
            return None
        lo, hi = ext[0], ext[1]
        if hi - lo < 1e-12:
            lo, hi = lo - 0.5, hi + 0.5
        pad = (hi - lo) * 0.015
        return lo - pad, hi + pad

    def _visible_values(self, x0: float, x1: float):
        for s in self.series:
            if not s.visible:
                continue
            lo = bisect.bisect_left(s.xs, x0)
            hi = bisect.bisect_right(s.xs, x1)
            if self.show_raw or s.smoothed is None:
                for y in s.ys[lo:hi]:
                    if y is not None and y == y and math.isfinite(y):
                        yield y
            if self.show_smooth and s.smoothed is not None:
                for y in s.smoothed[lo:hi]:
                    if y is not None:
                        yield y

    def y_view(self, x0: float, x1: float) -> tuple[float, float]:
        if self._yrange is not None:
            return self._yrange
        values = list(self._visible_values(x0, x1))
        values.extend(v for _, v in self.spec.references)
        if not values:
            return 0.0, 1.0
        lo, hi = min(values), max(values)
        if self.spec.include_zero:
            lo = min(lo, 0.0)
        if hi - lo < 1e-12:
            pad = abs(hi) * 0.1 or 0.5
            lo, hi = lo - pad, hi + pad
        else:
            pad = (hi - lo) * 0.07
            lo, hi = lo - pad, hi + pad
        if self.spec.y_min is not None:
            lo = max(lo, self.spec.y_min)
        if self.spec.y_max is not None:
            hi = min(hi, self.spec.y_max)
        if hi <= lo:
            hi = lo + 1.0
        # a series that sits exactly on a hard bound (e.g. 0 % of agents lost) must stay visible inside the frame, not hide on its border
        margin = 0.04 * (hi - lo)
        if self.spec.y_min is not None and lo <= self.spec.y_min:
            lo -= margin
        if self.spec.y_max is not None and hi >= self.spec.y_max:
            hi += margin
        return lo, hi

    def reset_view(self):
        self._xrange = None
        self._yrange = None
        self._cache_key = None
        self.update()
        self.viewChanged.emit(None)

    def set_x_range(self, rng: tuple[float, float] | None, emit: bool = False):
        self._xrange = rng
        self._yrange = None
        self._cache_key = None
        self.update()
        if emit:
            self.viewChanged.emit(rng)

    def x_range_user(self) -> tuple[float, float] | None:
        return self._xrange

    # ------------------------------------------------------------------------------------------------ geometry
    def plot_rect(self) -> QtCore.QRectF:
        if self.compact:
            return QtCore.QRectF(34, 6, max(10, self.width() - 42), max(10, self.height() - 22))
        return QtCore.QRectF(68, 10, max(10, self.width() - 84), max(10, self.height() - 52))

    def _to_px(self, x, y, rect, xv, yv):
        return QtCore.QPointF(rect.left() + (x - xv[0]) / (xv[1] - xv[0]) * rect.width(),
                              rect.bottom() - (y - yv[0]) / (yv[1] - yv[0]) * rect.height())

    def _from_px(self, px, py, rect, xv, yv):
        return (xv[0] + (px - rect.left()) / rect.width() * (xv[1] - xv[0]),
                yv[0] + (rect.bottom() - py) / rect.height() * (yv[1] - yv[0]))

    # ------------------------------------------------------------------------------------------------ painting
    def _paths_for(self, values: list[float | None], xs: list[float], rect, xv, yv, width_px: int):
        """Polylines for one value list: gap-separated runs, each reduced to a min/max envelope per pixel column when long."""
        lo = max(0, bisect.bisect_left(xs, xv[0]) - 1)
        hi = min(len(xs), bisect.bisect_right(xs, xv[1]) + 1)
        runs: list[list[QtCore.QPointF]] = []
        current: list[tuple[float, float]] = []
        for i in range(lo, hi):
            y = values[i]
            if y is None or y != y or not math.isfinite(y):
                if current:
                    runs.append(self._reduce(current, rect, xv, yv, width_px))
                    current = []
                continue
            current.append((xs[i], y))
        if current:
            runs.append(self._reduce(current, rect, xv, yv, width_px))
        return runs

    def _reduce(self, pts: list[tuple[float, float]], rect, xv, yv, width_px: int) -> list[QtCore.QPointF]:
        if len(pts) <= 3 * width_px:
            return [self._to_px(x, y, rect, xv, yv) for x, y in pts]
        buckets: dict[int, list[tuple[float, float]]] = {}
        span = xv[1] - xv[0]
        for x, y in pts:
            buckets.setdefault(int((x - xv[0]) / span * width_px), []).append((x, y))
        out: list[QtCore.QPointF] = []
        for b in sorted(buckets):
            group = buckets[b]
            first, last = group[0], group[-1]
            ymin = min(group, key=lambda p: p[1])
            ymax = max(group, key=lambda p: p[1])
            for p in sorted({first, ymin, ymax, last}, key=lambda p: p[0]):
                out.append(self._to_px(p[0], p[1], rect, xv, yv))
        return out

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        pal = self.palette_
        p.fillRect(self.rect(), QtGui.QColor(pal.chart_bg))
        rect = self.plot_rect()
        xv = self.x_view()
        small = QtGui.QFont(self.font())
        small.setPointSizeF(max(7.0, self.font().pointSizeF() - (3.0 if self.compact else 1.5)))
        p.setFont(small)
        fm = QtGui.QFontMetrics(small)
        if xv is None or not self.has_data():
            p.setPen(QtGui.QColor(pal.muted))
            p.drawRect(rect)
            text = self.spec.empty if not self.compact else "no data"
            p.drawText(rect.adjusted(14, 8, -14, -8), QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap, text)
            p.end()
            return
        yv = self.y_view(*xv)
        # grid and axes
        grid = QtGui.QPen(QtGui.QColor(pal.chart_grid)); grid.setWidthF(1)
        axis_pen = QtGui.QPen(QtGui.QColor(pal.chart_axis))
        ystep = nice_step(yv[1] - yv[0], 3 if self.compact else 5)
        for v in ticks(yv[0], yv[1], 3 if self.compact else 5):
            y = self._to_px(0, v, rect, (0, 1), yv).y()
            p.setPen(grid)
            p.drawLine(QtCore.QPointF(rect.left(), y), QtCore.QPointF(rect.right(), y))
            p.setPen(axis_pen)
            label = f"{v * 100:g}%" if self.spec.percent else format_number(v, ystep)
            p.drawText(QtCore.QRectF(0, y - 8, rect.left() - 4, 16), QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, label)
        xstep = nice_step(xv[1] - xv[0], 3 if self.compact else 7)
        for v in ticks(xv[0], xv[1], 3 if self.compact else 7):
            x = self._to_px(v, 0, rect, xv, (0, 1)).x()
            p.setPen(grid)
            p.drawLine(QtCore.QPointF(x, rect.top()), QtCore.QPointF(x, rect.bottom()))
            p.setPen(axis_pen)
            p.drawText(QtCore.QRectF(x - 40, rect.bottom() + 3, 80, 16), QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop, self._format_x(v, xstep))
        p.setPen(axis_pen)
        p.drawRect(rect)
        if not self.compact:
            p.drawText(QtCore.QRectF(rect.left(), self.height() - 18, rect.width(), 16), QtCore.Qt.AlignHCenter, self.xlabel)
            p.save()
            p.translate(13, rect.center().y())
            p.rotate(-90)
            p.drawText(QtCore.QRectF(-rect.height() / 2, -9, rect.height(), 18), QtCore.Qt.AlignCenter, self.spec.ylabel)
            p.restore()
        # references (e.g. KL target)
        p.save()
        p.setClipRect(rect)
        for label, value in self.spec.references:
            y = self._to_px(0, value, rect, (0, 1), yv).y()
            pen = QtGui.QPen(QtGui.QColor(pal.muted)); pen.setStyle(QtCore.Qt.DashLine)
            p.setPen(pen)
            p.drawLine(QtCore.QPointF(rect.left(), y), QtCore.QPointF(rect.right(), y))
            if not self.compact:
                p.drawText(QtCore.QPointF(rect.left() + 6, y - 4), label)
        # series
        width_px = max(1, int(rect.width()))
        for s in self.series:
            if not s.visible:
                continue
            color = QtGui.QColor(s.color)
            layers = []
            if s.smoothed is not None and self.show_smooth:
                if self.show_raw:
                    layers.append((s.ys, 0.35, 1.0))
                layers.append((s.smoothed, 1.0, 2.2))
            else:
                layers.append((s.ys, 1.0, 1.8))
            for values, alpha, width in layers:
                c = QtGui.QColor(color); c.setAlphaF(alpha)
                pen = QtGui.QPen(c); pen.setWidthF(width); pen.setJoinStyle(QtCore.Qt.RoundJoin); pen.setCapStyle(QtCore.Qt.RoundCap)
                if s.dashed:
                    pen.setStyle(QtCore.Qt.DashLine)
                p.setPen(pen)
                for run in self._paths_for(values, s.xs, rect, xv, yv, width_px):
                    if len(run) == 1:
                        p.setBrush(c)
                        p.drawEllipse(run[0], 2.6, 2.6)
                        p.setBrush(QtCore.Qt.NoBrush)
                    else:
                        path = QtGui.QPainterPath(run[0])
                        for pt in run[1:]:
                            path.lineTo(pt)
                        p.drawPath(path)
        p.restore()
        # box zoom
        if self._box_origin is not None and self._box_now is not None:
            box = QtCore.QRectF(self._box_origin, self._box_now).normalized()
            fill = QtGui.QColor(pal.accent); fill.setAlpha(40)
            p.fillRect(box, fill)
            p.setPen(QtGui.QPen(QtGui.QColor(pal.accent)))
            p.drawRect(box)
        if self._hover is not None and not self.compact and rect.contains(self._hover):
            self._paint_hover(p, rect, xv, yv, fm)
        p.end()

    def _format_x(self, v: float, step: float) -> str:
        if self.xlabel.startswith("Training time"):
            if v >= 3600:
                return f"{v / 3600:.2g} h"
            return f"{v / 60:.3g} min"
        if self.xlabel.startswith("Decisions"):
            return f"{v / 1e6:.3g} M" if abs(v) >= 1e5 else format_number(v, step)
        return f"{v:,.0f}" if step >= 1 else f"{v:.3g}"

    def _nearest(self, s: Series, x: float) -> int | None:
        if not s.xs:
            return None
        i = bisect.bisect_left(s.xs, x)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(s.xs):
                if best is None or abs(s.xs[j] - x) < abs(s.xs[best] - x):
                    best = j
        return best

    def _paint_hover(self, p, rect, xv, yv, fm):
        pal = self.palette_
        x, _ = self._from_px(self._hover.x(), self._hover.y(), rect, xv, yv)
        lines: list[tuple[str, str | None]] = []
        header_done = False
        anchor_x = None
        dots = []
        for s in self.series:
            if not s.visible:
                continue
            i = self._nearest(s, x)
            if i is None:
                continue
            raw = s.ys[i]
            if not header_done:
                anchor_x = s.xs[i]
                header_done = True
            text = format_exact(raw)
            if s.smoothed is not None and self.show_smooth and s.smoothed[i] is not None and raw is not None:
                text += f"   (smoothed {format_exact(s.smoothed[i])})"
            name = f"{s.run + ' · ' if s.run else ''}{s.label}"
            lines.append((f"{name}: {text}", s.color))
            if s.info is not None:
                lines.append(("    " + s.info(i), None))
            if raw is not None:
                dots.append((self._to_px(s.xs[i], raw, rect, xv, yv), s.color))
        if not lines:
            return
        pen = QtGui.QPen(QtGui.QColor(pal.chart_axis)); pen.setStyle(QtCore.Qt.DotLine)
        p.setPen(pen)
        cx = self._to_px(anchor_x if anchor_x is not None else x, 0, rect, xv, (0, 1)).x()
        p.drawLine(QtCore.QPointF(cx, rect.top()), QtCore.QPointF(cx, rect.bottom()))
        for pt, color in dots:
            p.setBrush(QtGui.QColor(color)); p.setPen(QtGui.QPen(QtGui.QColor(pal.chart_bg), 1.5))
            p.drawEllipse(pt, 4, 4)
        width = max(fm.horizontalAdvance(t) for t, _ in lines) + 24
        height = len(lines) * (fm.height() + 2) + 10
        left = self._hover.x() + 14
        if left + width > self.width() - 4:
            left = self._hover.x() - 14 - width
        top = min(max(rect.top(), self._hover.y() - height / 2), self.height() - height - 4)
        box = QtCore.QRectF(left, top, width, height)
        bg = QtGui.QColor(pal.panel); bg.setAlpha(240)
        p.setBrush(bg); p.setPen(QtGui.QPen(QtGui.QColor(pal.border)))
        p.drawRoundedRect(box, 6, 6)
        y = box.top() + 6
        for text, color in lines:
            if color:
                p.setBrush(QtGui.QColor(color)); p.setPen(QtCore.Qt.NoPen)
                p.drawRect(QtCore.QRectF(box.left() + 7, y + 4, 8, 8))
                p.setPen(QtGui.QColor(pal.text))
                p.drawText(QtCore.QPointF(box.left() + 19, y + fm.ascent()), text)
            else:
                p.setPen(QtGui.QColor(pal.muted))
                p.drawText(QtCore.QPointF(box.left() + 7, y + fm.ascent()), text)
            y += fm.height() + 2

    # ------------------------------------------------------------------------------------------------ interaction
    def mouseMoveEvent(self, e):
        if self.compact:
            return
        pos = e.position()
        if self._drag_origin is not None and self._drag_view is not None:
            rect = self.plot_rect()
            x0, x1, y0, y1 = self._drag_view
            dx = (pos.x() - self._drag_origin.x()) / rect.width() * (x1 - x0)
            self.set_x_range((x0 - dx, x1 - dx), emit=True)
            if self._yrange is not None:
                dy = (pos.y() - self._drag_origin.y()) / rect.height() * (y1 - y0)
                self._yrange = (y0 + dy, y1 + dy)
        elif self._box_origin is not None:
            self._box_now = pos
        self._hover = pos
        self.update()

    def mousePressEvent(self, e):
        if self.compact:
            return
        if e.button() == QtCore.Qt.LeftButton:
            xv = self.x_view()
            if xv is None:
                return
            if e.modifiers() & QtCore.Qt.ShiftModifier:
                self._box_origin, self._box_now = e.position(), e.position()
            else:
                yv = self.y_view(*xv)
                self._drag_origin = e.position()
                self._drag_view = (xv[0], xv[1], yv[0], yv[1])
                self.setCursor(QtCore.Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, e):
        if self._box_origin is not None and self._box_now is not None:
            rect, xv = self.plot_rect(), self.x_view()
            box = QtCore.QRectF(self._box_origin, self._box_now).normalized()
            if xv is not None and box.width() > 6 and box.height() > 6:
                yv = self.y_view(*xv)
                a = self._from_px(box.left(), box.bottom(), rect, xv, yv)
                b = self._from_px(box.right(), box.top(), rect, xv, yv)
                self._yrange = (a[1], b[1])
                self.set_x_range((a[0], b[0]), emit=True)
        self._box_origin = self._box_now = None
        self._drag_origin = self._drag_view = None
        self.setCursor(QtCore.Qt.ArrowCursor)
        self.update()

    def mouseDoubleClickEvent(self, e):
        if not self.compact:
            self.reset_view()

    def leaveEvent(self, e):
        self._hover = None
        self.update()

    def wheelEvent(self, e):
        if self.compact:
            e.ignore()
            return
        xv = self.x_view()
        if xv is None:
            return
        rect = self.plot_rect()
        factor = 0.8 if e.angleDelta().y() > 0 else 1.25
        yv = self.y_view(*xv)
        cx, cy = self._from_px(e.position().x(), e.position().y(), rect, xv, yv)
        if e.modifiers() & QtCore.Qt.ShiftModifier:
            self._yrange = (cy - (cy - yv[0]) * factor, cy + (yv[1] - cy) * factor)
            self.update()
        else:
            self.set_x_range((cx - (cx - xv[0]) * factor, cx + (xv[1] - cx) * factor), emit=True)
        e.accept()

    def zoom(self, factor: float):
        xv = self.x_view()
        if xv is None:
            return
        mid = (xv[0] + xv[1]) / 2
        half = (xv[1] - xv[0]) / 2 * factor
        self.set_x_range((mid - half, mid + half), emit=True)


def export_csv(path: str, series: list[Series], xlabel: str):
    """All samples of all series (raw and smoothed) so any plotted number can be re-analysed elsewhere."""
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", "run", xlabel, "raw", "smoothed"])
        for s in series:
            for i, (x, y) in enumerate(zip(s.xs, s.ys)):
                sm = s.smoothed[i] if s.smoothed is not None else None
                writer.writerow([s.label, s.run, x, "" if y is None else y, "" if sm is None else sm])


class ChartCard(QtWidgets.QFrame):
    """Title + help button + legend + chart + one-line interpretation. Used in the grid (compact toolbar) and in the focus view."""
    expandRequested = QtCore.Signal(str)
    viewChanged = QtCore.Signal(object)

    def __init__(self, spec: ChartSpec, palette: theme.Palette, help_callback: Callable[[str], None] | None = None, parent=None, large: bool = False):
        super().__init__(parent)
        self.setObjectName("card")
        self.spec, self.large = spec, large
        self.view = ChartView(spec, palette, parent=self)
        self.view.viewChanged.connect(self.viewChanged)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        head = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(spec.title)
        title.setObjectName("cardTitle")
        title.setWordWrap(True)
        head.addWidget(title, 1)
        if spec.help_key and help_callback:
            info = QtWidgets.QToolButton()
            info.setObjectName("info")
            info.setText("i")
            info.setToolTip("What does this show?")
            info.clicked.connect(lambda: help_callback(spec.help_key))
            head.addWidget(info)
        self.reset_button = QtWidgets.QToolButton()
        self.reset_button.setText("⟲")
        self.reset_button.setToolTip("Reset zoom (double-click the plot)")
        self.reset_button.clicked.connect(self.view.reset_view)
        head.addWidget(self.reset_button)
        save = QtWidgets.QToolButton()
        save.setText("⇩")
        save.setToolTip("Save this chart as an image, or its data as CSV")
        save.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(save)
        menu.addAction("Save image (PNG)…", self._save_png)
        menu.addAction("Save data (CSV)…", self._save_csv)
        save.setMenu(menu)
        head.addWidget(save)
        if not large:
            expand = QtWidgets.QToolButton()
            expand.setText("⤢")
            expand.setToolTip("Inspect this chart in detail")
            expand.clicked.connect(lambda: self.expandRequested.emit(spec.key))
            head.addWidget(expand)
        layout.addLayout(head)
        self.legend = QtWidgets.QWidget()
        self.legend_layout = FlowLayout(self.legend)
        layout.addWidget(self.legend)
        layout.addWidget(self.view, 1)
        self.caption = QtWidgets.QLabel(spec.caption)
        self.caption.setObjectName("hint")
        self.caption.setWordWrap(True)
        self.caption.setVisible(bool(spec.caption))
        layout.addWidget(self.caption)
        self.view.setMinimumHeight(300 if large else 190)

    def set_series(self, series: list[Series], xlabel: str, x_is_int: bool):
        self.view.set_series(series, xlabel, x_is_int)
        self._rebuild_legend(series)

    def _rebuild_legend(self, series):
        while self.legend_layout.count():
            item = self.legend_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        multi = len(series) > 1
        self.legend.setVisible(multi)
        for s in series:
            full = f"{s.run + ' · ' if s.run else ''}{s.label}"
            button = QtWidgets.QPushButton("■ " + (full if len(full) <= 46 else full[:43] + "…").replace("&", "&&"))
            button.setCheckable(True)
            button.setChecked(s.visible)
            button.setObjectName("flat")
            button.setStyleSheet(f"color:{s.color}; text-align:left; padding:1px 6px; font-weight:600;")
            button.setToolTip(full + "\nClick to show or hide this series")
            button.toggled.connect(lambda on, ss=s: self._toggle(ss, on))
            self.legend_layout.addWidget(button)

    def _toggle(self, s: Series, on: bool):
        s.visible = on
        self.view.update()

    def _save_png(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save chart image", f"{self.spec.key}.png", "PNG image (*.png)")
        if path:
            self.export_png(path)

    def _save_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save chart data", f"{self.spec.key}.csv", "CSV (*.csv)")
        if path:
            self.export_data(path)

    def export_png(self, path: str) -> bool:
        return self.grab().save(path, "PNG")

    def export_data(self, path: str) -> None:
        export_csv(path, self.view.series, self.view.xlabel)
