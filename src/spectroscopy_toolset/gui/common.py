"""Shared GUI helpers: Qt/pyqtgraph setup, palette, colours, dialogs and export."""

from __future__ import annotations

import sys
from pathlib import Path

try:  # import PyQt5 before pyqtgraph so pyqtgraph binds to it
    from PyQt5 import QtCore, QtGui, QtWidgets, uic
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        "The graphical interfaces need PyQt5 and pyqtgraph: pip install 'spectroscopy-toolset[gui]'"
    ) from exc
import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter, SVGExporter

from ..plotting import AXIS, CATEGORICAL, DIVERGING, INK_SECONDARY, MUTED

__all__ = ["QtCore", "QtGui", "QtWidgets", "pg"]

UI_DIR = Path(__file__).with_name("ui")

pg.setConfigOptions(
    imageAxisOrder="row-major", antialias=True, background="w", foreground=INK_SECONDARY
)

LABEL_STYLE = {"color": INK_SECONDARY, "font-size": "11pt"}
OTHER_COLOR = MUTED  # series beyond the eight categorical colours

#: Marker symbols offered in the curve-style menu (label -> pyqtgraph symbol).
SYMBOLS = {
    "None": None,
    "● circle": "o",
    "+ plus": "+",
    "◆ diamond": "d",
    "▲ triangle": "t",
    "■ square": "s",
    "⬟ pentagon": "p",
}
LINE_STYLES = {
    "Solid": QtCore.Qt.SolidLine,
    "Dash": QtCore.Qt.DashLine,
    "Dot": QtCore.Qt.DotLine,
    "Dash-dot": QtCore.Qt.DashDotLine,
}


def series_color(index: int) -> str:
    """Categorical colour for series ``index`` (gray beyond the eighth - never cycled)."""
    return CATEGORICAL[index] if index < len(CATEGORICAL) else OTHER_COLOR


def diverging_colormap() -> pg.ColorMap:
    """The blue - gray - red map of :mod:`spectroscopy_toolset.plotting` as a pyqtgraph ColorMap."""
    stops = np.linspace(0.0, 1.0, 7)
    colors = [tuple(int(255 * c) for c in DIVERGING(s)[:3]) for s in stops]
    return pg.ColorMap(stops, colors)


def add_legend(plot: pg.PlotItem) -> pg.LegendItem:
    """Top-right legend on a translucent white background (drag it to move)."""
    return plot.addLegend(
        offset=(-10, 10), brush=pg.mkBrush(255, 255, 255, 210), labelTextColor=INK_SECONDARY
    )


def style_plot(
    plot: pg.PlotItem, left: str, bottom: str, left_units=None, bottom_units=None
) -> None:
    plot.setLabel("left", left, units=left_units, **LABEL_STYLE)
    plot.setLabel("bottom", bottom, units=bottom_units, **LABEL_STYLE)
    plot.showGrid(x=True, y=True, alpha=0.25)
    for name in ("left", "bottom"):
        plot.getAxis(name).setPen(pg.mkPen(AXIS))
        plot.getAxis(name).setTextPen(pg.mkPen(INK_SECONDARY))


def dark_palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    roles = {
        QtGui.QPalette.Window: QtGui.QColor(53, 53, 53),
        QtGui.QPalette.WindowText: QtCore.Qt.white,
        QtGui.QPalette.Base: QtGui.QColor(25, 25, 25),
        QtGui.QPalette.AlternateBase: QtGui.QColor(53, 53, 53),
        QtGui.QPalette.ToolTipBase: QtGui.QColor(25, 25, 25),
        QtGui.QPalette.ToolTipText: QtCore.Qt.white,
        QtGui.QPalette.Text: QtCore.Qt.white,
        QtGui.QPalette.Button: QtGui.QColor(53, 53, 53),
        QtGui.QPalette.ButtonText: QtCore.Qt.white,
        QtGui.QPalette.BrightText: QtCore.Qt.red,
        QtGui.QPalette.Link: QtGui.QColor(42, 130, 218),
        QtGui.QPalette.Highlight: QtGui.QColor(42, 130, 218),
        QtGui.QPalette.HighlightedText: QtCore.Qt.black,
    }
    for role, color in roles.items():
        palette.setColor(role, QtGui.QColor(color))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor(127, 127, 127))
    palette.setColor(
        QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor(127, 127, 127)
    )
    return palette


def make_app(argv=None) -> QtWidgets.QApplication:
    """The running QApplication, created (Fusion style, dark palette) if needed."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(list(argv) if argv is not None else sys.argv)
        app.setStyle("Fusion")
        app.setPalette(dark_palette())
    return app


def load_ui(name: str, base: QtWidgets.QWidget | None = None) -> QtWidgets.QWidget:
    """Load a Qt Designer file from the package's ``ui`` folder (into ``base`` if given)."""
    return (
        uic.loadUi(str(UI_DIR / name), base) if base is not None else uic.loadUi(str(UI_DIR / name))
    )


def export_plot(item: pg.GraphicsItem, path: str | Path, width: int = 2400) -> None:
    """Save a plot item as PNG (``width`` px) or SVG, chosen by the file suffix."""
    path = str(path)
    if path.lower().endswith(".svg"):
        SVGExporter(item).export(path)
    else:
        exporter = ImageExporter(item)
        exporter.parameters()["width"] = width
        exporter.export(path)


def save_dialog(parent, title: str, default: str | Path, filters: str) -> str | None:
    path, _ = QtWidgets.QFileDialog.getSaveFileName(parent, title, str(default), filters)
    return path or None


def warn(parent, title: str, text: str) -> None:
    QtWidgets.QMessageBox.warning(parent, title, text)


def parse_float(line_edit: QtWidgets.QLineEdit, default: float | None = None) -> float | None:
    """Float from a line edit (accepts a decimal comma); ``default`` if empty/invalid."""
    text = line_edit.text().strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return default


class BusyCursor:
    """Context manager showing a wait cursor during long computations."""

    def __enter__(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()
        return self

    def __exit__(self, *exc):
        QtWidgets.QApplication.restoreOverrideCursor()
        return False
