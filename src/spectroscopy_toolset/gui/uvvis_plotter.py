"""Interactive plotter for steady-state (UV-vis) spectra.

Run with ``spectro-uvvis [files...]`` or ``python -m spectroscopy_toolset.gui.uvvis_plotter``.
"""

from __future__ import annotations

import csv
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import models, uvvis
from ..fitting import fit
from ..io import read_spectra
from ..plotting import CATEGORICAL, INK_SECONDARY, series_colors
from ..processing import average_curves, crop_mask, normalize
from ..spectrum import Spectrum
from .common import (
    LABEL_STYLE,
    LINE_STYLES,
    SYMBOLS,
    QtCore,
    QtGui,
    QtWidgets,
    add_legend,
    export_plot,
    load_ui,
    make_app,
    parse_float,
    pg,
    save_dialog,
    series_color,
    style_plot,
    warn,
)

FILE_FILTER = "Spectra (*.csv *.txt *.dat *.asc *.prn *.tsv);;All files (*)"
FLOAT_VALIDATOR_RE = r"[-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?"


@dataclass
class Curve:
    spectrum: Spectrum
    item: pg.PlotDataItem
    color: str
    width: float = 2.0
    symbol: str | None = None
    style: int = QtCore.Qt.SolidLine
    visible: bool = True

    @property
    def name(self) -> str:
        return self.spectrum.name


@dataclass
class FitModel:
    label: str
    names: list[str]
    func: Callable
    guess: Callable[[np.ndarray, np.ndarray], list[float]]
    shift_x: bool = False  # fit on x - x_start so exponential amplitudes stay well scaled


def _exp_guess(n: int):
    def guess(x, y):
        span = float(x[-1] - x[0]) or 1.0
        taus = np.geomspace(span / 20.0, span, n) if n > 1 else [span / 3.0]
        params = []
        for tau in taus:
            params += [(y[0] - y[-1]) / n, tau]
        return params + [float(y[-1])]

    return guess


def _peak_guess(width: str):
    def guess(x, y):
        base = float(np.min(y))
        k = int(np.argmax(y))
        amp = float(y[k] - base)
        above = x[y - base >= amp / 2.0]
        fwhm = float(above[-1] - above[0]) if above.size > 1 else float(x[-1] - x[0]) / 10.0
        w = fwhm / models.FWHM_PER_SIGMA if width == "sigma" else fwhm / 2.0
        return [amp, float(x[k]), max(w, 1e-9), base]

    return guess


def _power_guess(x, y):
    ok = (x > 0) & (y > 0)
    if ok.sum() >= 2:
        k, log_a = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
        return [float(np.exp(log_a)), float(k), 0.0]
    return [1.0, -1.0, 0.0]


#: Models offered by the radio buttons of the "Fit model" box (keyed by button name).
FIT_MODELS = {
    "exp_button": FitModel(
        "A1 exp(-(x-x0)/t1) + y0",
        ["A1", "t1", "y0"],
        lambda x, *p: models.multi_exp(x, *p),
        _exp_guess(1),
        True,
    ),
    "biexp_button": FitModel(
        "bi-exponential + y0",
        ["A1", "t1", "A2", "t2", "y0"],
        lambda x, *p: models.multi_exp(x, *p),
        _exp_guess(2),
        True,
    ),
    "triexp_button": FitModel(
        "tri-exponential + y0",
        ["A1", "t1", "A2", "t2", "A3", "t3", "y0"],
        lambda x, *p: models.multi_exp(x, *p),
        _exp_guess(3),
        True,
    ),
    "power_button": FitModel("A x^k + y0", ["A", "k", "y0"], models.power_law, _power_guess),
    "gaussian_button": FitModel(
        "Gaussian + y0",
        ["A", "x0", "sigma", "y0"],
        lambda x, a, c, s, y0: models.gaussian(x, a, c, s) + y0,
        _peak_guess("sigma"),
    ),
    "lorentzian_button": FitModel(
        "Lorentzian + y0",
        ["A", "x0", "gamma", "y0"],
        lambda x, a, c, g, y0: models.lorentzian(x, a, c, g) + y0,
        _peak_guess("gamma"),
    ),
}
_POSITIVE = {"t1", "t2", "t3", "sigma", "gamma"}


def float_validator(parent=None) -> QtGui.QRegExpValidator:
    """Validator accepting '.' or ',' as decimal separator regardless of the system locale."""
    return QtGui.QRegExpValidator(QtCore.QRegExp(FLOAT_VALIDATOR_RE), parent)


class UVVisPlotter(QtWidgets.QMainWindow):
    """Main window: plot area on the left, formatting/analysis panel on the right."""

    windows: list[UVVisPlotter] = []  # keeps top-level windows alive

    def __init__(self):
        super().__init__()
        self.setWindowTitle("UV-vis Plotter")
        self.resize(1400, 850)
        self.curves: list[Curve] = []
        self.overlays: list[pg.GraphicsObject] = []
        self.directory = Path.home()
        self.confirm_close = True
        self._build_ui()
        self._connect()
        UVVisPlotter.windows.append(self)

    # ------------------------------------------------------------------ setup
    def _action(self, text, slot, shortcut=None, tip=""):
        action = QtWidgets.QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.setStatusTip(tip)
        action.triggered.connect(slot)
        return action

    def _build_ui(self):
        self.plot_widget = pg.PlotWidget()
        self.area = self.plot_widget.getPlotItem()
        style_plot(self.area, "Absorbance", "Wavelength (nm)")
        self.legend = add_legend(self.area)

        self.panel = load_ui("uvvis_formatting.ui")
        self.panel.setSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding
        )
        p = self.panel
        for edit in (p.xaxis_min, p.xaxis_max, p.yaxis_min, p.yaxis_max):
            edit.setValidator(float_validator(edit))
        p.curve_marker.addItems(list(SYMBOLS))
        p.curve_linestyle.addItems(list(LINE_STYLES))
        p.xaxis_title.setText("Wavelength (nm)")
        p.yaxis_title.setText("Absorbance")
        p.fit_table.verticalHeader().setVisible(False)
        p.fit_table.horizontalHeader().setStretchLastSection(True)
        p.fitting_button.setChecked(True)
        p.gaussian_button.setChecked(True)
        p.hide_button.setToolTip("Hide or show the selected curve")
        p.recolor_button.setToolTip("Choose the colour of the selected curve")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.plot_widget)
        splitter.addWidget(self.panel)
        splitter.setStretchFactor(0, 1)
        self.setCentralWidget(splitter)
        self.status = self.statusBar()

        actions = [
            self._action("Load", self.on_load, "Ctrl+O", "Load spectra"),
            self._action("Clear", self.clear, "Ctrl+K", "Remove all curves"),
            self._action(
                "Save figure", self.save_figure, "Ctrl+S", "Export the plot as PNG or SVG"
            ),
            self._action(
                "Export data", self.export_data, "Ctrl+E", "Write the displayed curves to CSV"
            ),
            self._action(
                "Avg. Data", self.average, None, "Average the visible curves on a common grid"
            ),
            self._action(
                "Ramp colours", self.ramp_colors, None, "Colour curves along a ramp (series)"
            ),
        ]
        toolbar = self.addToolBar("main")
        menu = self.menuBar().addMenu("&File")
        for action in actions:
            toolbar.addAction(action)
            menu.addAction(action)
        menu.addSeparator()
        menu.addAction(
            self._action("New window", self.new_window, "Ctrl+N", "Open another plotter")
        )
        menu.addAction(self._action("Close", self.close, "Ctrl+W", "Close this window"))

    def _connect(self):
        p = self.panel
        p.applyOpts_button.clicked.connect(self.apply_graph_options)
        p.autoscale_button.clicked.connect(self.autoscale)
        p.normalize_checkbox.toggled.connect(lambda *_: self.redraw())
        p.energy_checkbox.toggled.connect(self.on_energy_toggled)
        p.legend_list.currentIndexChanged.connect(self.on_curve_selected)
        p.curve_title.editingFinished.connect(self.rename_curve)
        p.curveOpts_button.clicked.connect(self.apply_curve_options)
        p.recolor_button.clicked.connect(self.recolor)
        p.hide_button.clicked.connect(self.toggle_visibility)
        for name in FIT_MODELS:
            getattr(p, name).toggled.connect(self.on_model_changed)
        p.calcAnalysis_button.clicked.connect(self.run_analysis)
        p.clearAnalysis_button.clicked.connect(self.clear_analysis)
        p.saveNotes_button.clicked.connect(self.save_notes)

    # ------------------------------------------------------------------- data
    def on_load(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Load spectra", str(self.directory), FILE_FILTER
        )
        if paths:
            self.load_files(paths)

    def load_files(self, paths) -> int:
        loaded, failed = 0, []
        for path in paths:
            try:
                spectra = read_spectra(path)
            except Exception as exc:  # report every unreadable file, keep going
                failed.append(f"{Path(path).name}: {exc}")
                continue
            for spec in spectra:
                self.add_spectrum(spec)
                loaded += 1
        if paths:
            self.directory = Path(paths[0]).parent
        if loaded:
            if not self.panel.graph_title.text():
                self.panel.graph_title.setText(self.directory.name)
            self.autoscale()
        message = f"Loaded {loaded} spectra"
        if failed:
            message += f"; {len(failed)} file(s) could not be read"
            warn(self, "Some files could not be read", "\n".join(failed))
        self.status.showMessage(message)
        return loaded

    def add_spectrum(self, spec: Spectrum) -> Curve:
        index = len(self.curves)
        color = series_color(index)
        item = self.area.plot([], [], pen=pg.mkPen(color, width=2.0), name=spec.name)
        curve = Curve(spec, item, color)
        self.curves.append(curve)
        item.setData(*self.displayed(curve))
        self.panel.legend_list.addItem(spec.name)
        if index == len(CATEGORICAL):
            self.status.showMessage(
                "More than 8 curves: extra curves are gray - use Recolor or Ramp colours", 8000
            )
        return curve

    def displayed(self, curve: Curve) -> tuple[np.ndarray, np.ndarray]:
        """x/y of a curve as currently shown (energy axis and normalisation applied)."""
        spec = curve.spectrum
        if self.panel.energy_checkbox.isChecked() and np.all(spec.x > 0):
            spec = spec.to_energy()
        y = spec.y
        if self.panel.normalize_checkbox.isChecked():
            try:
                y = normalize(y, "max")
            except ValueError:
                pass
        return spec.x, y

    def clear(self):
        self.clear_analysis()
        for curve in self.curves:
            self.area.removeItem(curve.item)
        self.curves.clear()
        self.legend.clear()
        self.panel.legend_list.clear()
        self.status.showMessage("Cleared")

    def average(self):
        visible = [c for c in self.curves if c.visible]
        if len(visible) < 2:
            self.status.showMessage("Load (or show) at least two spectra to average")
            return
        try:
            grid, mean, std = average_curves(
                [c.spectrum.x for c in visible], [c.spectrum.y for c in visible]
            )
        except ValueError as exc:
            warn(self, "Cannot average", str(exc))
            return
        first = visible[0].spectrum
        self.add_spectrum(
            Spectrum(
                grid,
                mean,
                name=f"average of {len(visible)}",
                x_label=first.x_label,
                y_label=first.y_label,
                meta={"std": std},
            )
        )
        self.status.showMessage(
            f"Averaged {len(visible)} spectra; largest standard deviation {np.nanmax(std):.3g}"
        )

    # ---------------------------------------------------------- graph options
    def data_bounds(self):
        shown = [self.displayed(c) for c in self.curves if c.visible]
        if not shown:
            return None
        xs = np.concatenate([x for x, _ in shown])
        ys = np.concatenate([y for _, y in shown])
        return np.nanmin(xs), np.nanmax(xs), np.nanmin(ys), np.nanmax(ys)

    def autoscale(self):
        self.area.enableAutoRange()
        bounds = self.data_bounds()
        if bounds is None:
            return
        xmin, xmax, ymin, ymax = bounds
        headroom = 0.05 * (ymax - ymin)  # room for peak labels above the tallest band
        p = self.panel
        for edit, value in zip(
            (p.xaxis_min, p.xaxis_max, p.yaxis_min, p.yaxis_max),
            (xmin, xmax, ymin, ymax + headroom),
        ):
            edit.setText(f"{value:.6g}")

    def apply_graph_options(self):
        p = self.panel
        self.area.setTitle(p.graph_title.text(), color=INK_SECONDARY)
        self.area.setLabel("bottom", p.xaxis_title.text(), **LABEL_STYLE)
        ylabel = p.yaxis_title.text()
        if p.normalize_checkbox.isChecked() and "norm" not in ylabel:
            ylabel += " (norm.)"
        self.area.setLabel("left", ylabel, **LABEL_STYLE)
        xmin, xmax = parse_float(p.xaxis_min), parse_float(p.xaxis_max)
        ymin, ymax = parse_float(p.yaxis_min), parse_float(p.yaxis_max)
        if None not in (xmin, xmax) and xmin < xmax:
            self.area.setXRange(xmin, xmax, padding=0)
        if None not in (ymin, ymax) and ymin < ymax:
            self.area.setYRange(ymin, ymax, padding=0)

    def on_energy_toggled(self, checked):
        p = self.panel
        if checked and "wavelength" in p.xaxis_title.text().lower():
            p.xaxis_title.setText("Energy (eV)")
        elif not checked and p.xaxis_title.text() == "Energy (eV)":
            p.xaxis_title.setText("Wavelength (nm)")
        self.area.setLabel("bottom", p.xaxis_title.text(), **LABEL_STYLE)
        self.redraw()
        self.autoscale()

    def redraw(self):
        self.clear_analysis()
        for curve in self.curves:
            curve.item.setData(*self.displayed(curve))
        self.apply_graph_options()

    # ---------------------------------------------------------- curve options
    def current_curve(self) -> Curve | None:
        i = self.panel.legend_list.currentIndex()
        return self.curves[i] if 0 <= i < len(self.curves) else None

    def on_curve_selected(self, *_):
        curve = self.current_curve()
        if curve is None:
            return
        p = self.panel
        p.curve_title.setText(curve.name)
        p.curve_thickness.setValue(curve.width)
        p.curve_marker.setCurrentIndex(list(SYMBOLS.values()).index(curve.symbol))
        p.curve_linestyle.setCurrentIndex(list(LINE_STYLES.values()).index(curve.style))
        p.hide_button.setText("Hide" if curve.visible else "Show")
        if p.fitting_button.isChecked():
            self.populate_fit_table()

    def rename_curve(self):
        curve = self.current_curve()
        name = self.panel.curve_title.text().strip()
        if curve is None or not name or name == curve.name:
            return
        curve.spectrum = curve.spectrum.copy(name=name)
        curve.item.opts["name"] = name
        self.panel.legend_list.setItemText(self.panel.legend_list.currentIndex(), name)
        self.rebuild_legend()

    def _apply_pen(self, curve: Curve):
        curve.item.setPen(pg.mkPen(curve.color, width=curve.width, style=curve.style))
        curve.item.setSymbol(curve.symbol)
        if curve.symbol:
            curve.item.setSymbolBrush(curve.color)
            curve.item.setSymbolPen(pg.mkPen("w"))
            curve.item.setSymbolSize(7)

    def apply_curve_options(self):
        curve = self.current_curve()
        if curve is None:
            return
        p = self.panel
        curve.width = p.curve_thickness.value()
        curve.symbol = SYMBOLS[p.curve_marker.currentText()]
        curve.style = LINE_STYLES[p.curve_linestyle.currentText()]
        self._apply_pen(curve)
        self.rename_curve()

    def recolor(self):
        curve = self.current_curve()
        if curve is None:
            return
        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(curve.color), self, f"Colour of {curve.name}"
        )
        if color.isValid():
            curve.color = color.name()
            self._apply_pen(curve)
            self.rebuild_legend()

    def toggle_visibility(self):
        curve = self.current_curve()
        if curve is None:
            return
        curve.visible = not curve.visible
        curve.item.setVisible(curve.visible)
        self.panel.hide_button.setText("Hide" if curve.visible else "Show")
        self.rebuild_legend()

    def ramp_colors(self):
        for curve, color in zip(self.curves, series_colors(len(self.curves), ordered=True)):
            curve.color = color
            self._apply_pen(curve)
        self.rebuild_legend()

    def rebuild_legend(self):
        self.legend.clear()
        for curve in self.curves:
            if curve.visible:
                self.legend.addItem(curve.item, curve.name)

    # --------------------------------------------------------------- analysis
    def selected_model(self) -> FitModel:
        for name, model in FIT_MODELS.items():
            if getattr(self.panel, name).isChecked():
                return model
        return FIT_MODELS["gaussian_button"]

    def analysis_window(self, curve: Curve):
        """Displayed data of ``curve`` inside the x-axis limits of the panel."""
        x, y = self.displayed(curve)
        lo, hi = parse_float(self.panel.xaxis_min), parse_float(self.panel.xaxis_max)
        mask = crop_mask(x, lo, hi) & np.isfinite(y)
        return x[mask], y[mask]

    def on_model_changed(self, checked=True):
        if checked:
            self.populate_fit_table()

    def populate_fit_table(self):
        model = self.selected_model()
        guesses = [1.0] * len(model.names)
        curve = self.current_curve()
        if curve is not None:
            x, y = self.analysis_window(curve)
            if x.size > len(model.names):
                guesses = model.guess(x - x[0] if model.shift_x else x, y)
        table = self.panel.fit_table
        table.setRowCount(len(model.names))
        locked = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        for row, (name, guess) in enumerate(zip(model.names, guesses)):
            label = QtWidgets.QTableWidgetItem(name)
            label.setFlags(locked)
            fitted = QtWidgets.QTableWidgetItem("")
            fitted.setFlags(locked)
            table.setItem(row, 0, label)
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{guess:.6g}"))
            table.setItem(row, 2, fitted)

    def run_analysis(self):
        curve = self.current_curve()
        if curve is None:
            self.status.showMessage("Load and select a curve first")
            return
        if self.panel.findpeaks_button.isChecked():
            self.find_peaks(curve)
        else:
            self.fit_curve(curve)

    def fit_curve(self, curve: Curve):
        model = self.selected_model()
        x, y = self.analysis_window(curve)
        if x.size <= len(model.names):
            warn(self, "Not enough data", "Too few points inside the x-axis limits for this model.")
            return
        table = self.panel.fit_table
        if table.rowCount() != len(model.names) or table.item(0, 0).text() != model.names[0]:
            self.populate_fit_table()
        x_start = float(x[0]) if model.shift_x else 0.0
        defaults = model.guess(x - x_start, y)
        p0 = []
        for row, default in enumerate(defaults):
            try:
                p0.append(float(table.item(row, 1).text().replace(",", ".")))
            except (AttributeError, ValueError):
                p0.append(default)
        bounds = {n: (1e-12, None) for n in model.names if n in _POSITIVE}
        try:
            result = fit(
                model.func, x - x_start, y, p0, names=model.names, bounds=bounds, model=model.label
            )
        except Exception as exc:
            warn(self, "Fit failed", str(exc))
            return
        for row, (value, err) in enumerate(zip(result.values, result.stderr)):
            table.item(row, 2).setText(f"{value:.5g} ± {err:.2g}")
        xx = np.linspace(x.min(), x.max(), 600)
        item = self.area.plot(
            xx,
            result.eval(xx - x_start),
            pen=pg.mkPen(
                CATEGORICAL[1] if curve.color != CATEGORICAL[1] else CATEGORICAL[0],
                width=2.5,
                style=QtCore.Qt.DashLine,
            ),
        )
        self.overlays.append(item)
        shift = f" (x0 = {x_start:.6g})" if model.shift_x else ""
        self.append_note(
            f"Fit of '{curve.name}' over {x.min():.5g}-{x.max():.5g}{shift}\n{result.summary()}"
        )
        self.status.showMessage(f"{model.label}: R² = {result.r_squared:.5f}")

    def find_peaks(self, curve: Curve):
        x, y = self.analysis_window(curve)
        if x.size < 3:
            return
        peaks = uvvis.find_peaks(x, y)
        if peaks.empty:
            self.status.showMessage("No peaks found")
            return
        scatter = pg.ScatterPlotItem(
            peaks["position"].to_numpy(),
            peaks["height"].to_numpy(),
            size=10,
            brush=pg.mkBrush(curve.color),
            pen=pg.mkPen("w", width=1.5),
        )
        self.area.addItem(scatter)
        self.overlays.append(scatter)
        for pos, height in zip(peaks["position"], peaks["height"]):
            label = pg.TextItem(f"{pos:.4g}", color=INK_SECONDARY, anchor=(0.5, 1.3))
            label.setPos(pos, height)
            self.area.addItem(label)
            self.overlays.append(label)
        lines = [f"Peaks of '{curve.name}' (position, height, FWHM):"]
        lines += [f"  {r.position:.5g}\t{r.height:.4g}\t{r.fwhm:.4g}" for r in peaks.itertuples()]
        self.append_note("\n".join(lines))
        self.status.showMessage(f"{len(peaks)} peak(s) found")

    def clear_analysis(self):
        for item in self.overlays:
            self.area.removeItem(item)
        self.overlays.clear()
        table = self.panel.fit_table
        for row in range(table.rowCount()):
            if table.item(row, 2) is not None:
                table.item(row, 2).setText("")

    # ------------------------------------------------------------ notes & I/O
    def append_note(self, text: str):
        self.panel.notes_textbox.append(text + "\n")

    def save_notes(self):
        name = self.panel.notes_filename.text().strip()
        if name:
            path = Path(name) if Path(name).is_absolute() else self.directory / name
            if not path.suffix:
                path = path.with_suffix(".txt")
        else:
            chosen = save_dialog(self, "Save notes", self.directory / "notes.txt", "Text (*.txt)")
            if not chosen:
                return
            path = Path(chosen)
        path.write_text(self.panel.notes_textbox.toPlainText(), encoding="utf-8")
        self.status.showMessage(f"Notes saved to {path}")

    def save_figure(self, path: str | None = None):
        if not self.curves:
            return
        if path is None:
            title = self.panel.graph_title.text().strip() or "spectra"
            path = save_dialog(
                self,
                "Save figure",
                self.directory / f"{title}.png",
                "PNG image (*.png);;SVG vector graphics (*.svg)",
            )
            if not path:
                return
        export_plot(self.area, path)
        self.status.showMessage(f"Figure saved to {path}")

    def export_data(self, path: str | None = None):
        shown = [(c.name, *self.displayed(c)) for c in self.curves if c.visible]
        if not shown:
            return
        if path is None:
            path = save_dialog(self, "Export data", self.directory / "spectra.csv", "CSV (*.csv)")
            if not path:
                return
        xlabel = self.panel.xaxis_title.text() or "x"
        length = max(len(x) for _, x, _ in shown)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([h for name, _, _ in shown for h in (f"{name} {xlabel}", name)])
            for i in range(length):
                row = []
                for _, x, y in shown:
                    row += [f"{x[i]:.8g}", f"{y[i]:.8g}"] if i < len(x) else ["", ""]
                writer.writerow(row)
        self.status.showMessage(f"Data exported to {path}")

    # ---------------------------------------------------------------- windows
    def new_window(self):
        window = UVVisPlotter()
        window.show()
        return window

    def closeEvent(self, event):
        if self.confirm_close and self.curves:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Close plotter",
                "Close this window? Unsaved figures and notes will be lost.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
        if self in UVVisPlotter.windows:
            UVVisPlotter.windows.remove(self)
        event.accept()


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = make_app(argv)
    window = UVVisPlotter()
    window.showMaximized()
    files = [a for a in argv[1:] if Path(a).is_file()]
    if files:
        window.load_files(files)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
