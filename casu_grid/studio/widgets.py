"""Small reusable building blocks so that every page has the same look and vocabulary."""
from __future__ import annotations

from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from . import help as helptext
from . import theme


class FlowLayout(QtWidgets.QLayout):
    """Left-to-right layout that wraps to new lines (chart legends with many entries must never force horizontal scrolling)."""

    def __init__(self, parent=None, spacing: int = 6):
        super().__init__(parent)
        self._items: list[QtWidgets.QLayoutItem] = []
        self._space = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._layout(QtCore.QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size + QtCore.QSize(0, 0)

    def _layout(self, rect, test_only):
        x, y, line_height = rect.x(), rect.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            if x + hint.width() > rect.right() and line_height > 0:
                x = rect.x()
                y += line_height + self._space
                line_height = 0
            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x += hint.width() + self._space
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y()


def make_label(text: str = "", object_name: str = "", wrap: bool = True, selectable: bool = False) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    if object_name:
        label.setObjectName(object_name)
    label.setWordWrap(wrap)
    label.setTextFormat(QtCore.Qt.AutoText)
    if selectable:
        label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    return label


def clear_layout(layout: QtWidgets.QLayout) -> None:
    """Remove every item and hide its widget immediately (deleteLater alone leaves it painted at a stale position until the event loop runs)."""
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.hide()
            widget.setParent(None)
            widget.deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())


class Card(QtWidgets.QFrame):
    """A titled panel. ``body`` is the layout to add content to."""

    def __init__(self, title: str = "", subtitle: str = "", object_name: str = "card", help_key: str = "", help_callback=None, parent=None):
        super().__init__(parent)
        self.setObjectName(object_name)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(8)
        self.header = QtWidgets.QHBoxLayout()
        self.title_label = make_label(title, "cardTitle", wrap=False)
        self.header.addWidget(self.title_label)
        if help_key and help_callback:
            self.header.addWidget(InfoButton(help_key, help_callback))
        self.header.addStretch(1)
        if title:
            outer.addLayout(self.header)
        if subtitle:
            outer.addWidget(make_label(subtitle, "hint"))
        self.body = QtWidgets.QVBoxLayout()
        self.body.setSpacing(8)
        outer.addLayout(self.body)

    def add(self, widget_or_layout):
        if isinstance(widget_or_layout, QtWidgets.QLayout):
            self.body.addLayout(widget_or_layout)
        else:
            self.body.addWidget(widget_or_layout)
        return widget_or_layout


class InfoButton(QtWidgets.QToolButton):
    """A small 'i' that explains a concept in place; the same wording appears in the Help page."""

    def __init__(self, key: str, callback: Callable[[str], None], parent=None):
        super().__init__(parent)
        self.setObjectName("info")
        self.setText("i")
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setToolTip(helptext.short(key) or "Explain")
        self.clicked.connect(lambda: callback(key))


class StatTile(QtWidgets.QFrame):
    def __init__(self, label: str, value: str = "–", help_key: str = "", help_callback=None, parent=None):
        super().__init__(parent)
        self.setObjectName("statTile")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(0)
        row = QtWidgets.QHBoxLayout()
        self.label = make_label(label, "statLabel", wrap=False)
        row.addWidget(self.label)
        if help_key and help_callback:
            row.addWidget(InfoButton(help_key, help_callback))
        row.addStretch(1)
        layout.addLayout(row)
        self.value = make_label(value, "statValue", wrap=False)
        layout.addWidget(self.value)
        self.sub = make_label("", "hint", wrap=True)
        layout.addWidget(self.sub)
        self.sub.setVisible(False)

    def set(self, value: str, sub: str = "", color: str | None = None):
        self.value.setText(value)
        self.value.setStyleSheet(f"color:{color};" if color else "")
        self.sub.setText(sub)
        self.sub.setVisible(bool(sub))


class Banner(QtWidgets.QFrame):
    """Inline message with a clear level; used instead of modal pop-ups for everything that is not a confirmation."""

    def __init__(self, level: str = "info", title: str = "", text: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName(f"banner_{level}")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)
        self.title = make_label(f"<b>{title}</b>" if title else "", wrap=True)
        self.text = make_label(text, wrap=True)
        layout.addWidget(self.title)
        layout.addWidget(self.text)
        self.actions = QtWidgets.QHBoxLayout()
        layout.addLayout(self.actions)
        self.title.setVisible(bool(title))
        self.text.setVisible(bool(text))

    def set(self, level: str, title: str, text: str = ""):
        self.setObjectName(f"banner_{level}")
        self.style().unpolish(self)
        self.style().polish(self)
        self.title.setText(f"<b>{title}</b>")
        self.title.setVisible(bool(title))
        self.text.setText(text)
        self.text.setVisible(bool(text))

    def add_button(self, text: str, callback, primary: bool = False) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        if primary:
            button.setObjectName("primary")
        button.clicked.connect(callback)
        self.actions.addWidget(button)
        return button


class PageHeader(QtWidgets.QWidget):
    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        text = QtWidgets.QVBoxLayout()
        text.setSpacing(2)
        self.title = make_label(title, "pageTitle", wrap=False)
        self.subtitle = make_label(subtitle, "pageSubtitle")
        text.addWidget(self.title)
        text.addWidget(self.subtitle)
        layout.addLayout(text, 1)
        self.actions = QtWidgets.QHBoxLayout()
        layout.addLayout(self.actions)


class StatusBadge(QtWidgets.QLabel):
    """Status as glyph + word + colour, so it is readable without colour vision."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.palette_ = theme.LIGHT
        self.set_status("unknown")

    def set_palette(self, palette: theme.Palette):
        self.palette_ = palette
        self.set_status(self.status)

    def set_status(self, status: str):
        self.status = status
        color = theme.status_color(self.palette_, status)
        self.setText(theme.status_text(status))
        self.setStyleSheet(f"color:{color}; border:1px solid {color}; border-radius:11px; padding:3px 12px; font-weight:700;")


def scroll_page(widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QtWidgets.QFrame.NoFrame)
    area.setWidget(widget)
    return area


def page_container(spacing: int = 14) -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
    w = QtWidgets.QWidget()
    w.setObjectName("page")
    layout = QtWidgets.QVBoxLayout(w)
    layout.setContentsMargins(28, 22, 28, 22)
    layout.setSpacing(spacing)
    return w, layout


class FlowRow(QtWidgets.QWidget):
    """Tiles that wrap to the next row when the window is narrow."""

    def __init__(self, min_width: int = 190, parent=None):
        super().__init__(parent)
        self._grid = QtWidgets.QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(10)
        self._items: list[QtWidgets.QWidget] = []
        self._min = min_width
        self._cols = 0

    def add(self, widget: QtWidgets.QWidget):
        self._items.append(widget)
        self._relayout(force=True)

    def resizeEvent(self, e):
        self._relayout()
        super().resizeEvent(e)

    def _relayout(self, force: bool = False):
        cols = max(1, min(len(self._items) or 1, self.width() // self._min or 1))
        if cols == self._cols and not force:
            return
        self._cols = cols
        for i, w in enumerate(self._items):
            self._grid.addWidget(w, i // cols, i % cols)
        for c in range(cols):
            self._grid.setColumnStretch(c, 1)


def open_path(path) -> None:
    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))


class HelpDialog(QtWidgets.QDialog):
    def __init__(self, key: str, parent=None):
        super().__init__(parent)
        topic = helptext.TOPICS.get(key)
        self.setWindowTitle(topic.title if topic else "Help")
        self.setMinimumWidth(460)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(make_label(f"<h3>{topic.title if topic else key}</h3>"))
        layout.addWidget(make_label(f"<p><b>{topic.short}</b></p><p>{topic.body}</p>" if topic else ""))
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
