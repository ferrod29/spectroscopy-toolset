"""Interactive inspector for transient-absorption data.

Run with ``spectro-ta [files...]`` or ``python -m spectroscopy_toolset.gui.ta_inspector``.

The suite window collects normalised traces exported from any number of
inspector windows; each inspector shows one data set as a map, with the
kinetic trace and transient spectrum at the clicked point.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

from ..fitting import GlobalFitResult, fit_kinetics
from ..io import read_ta, write_ta_matrix
from ..plotting import CATEGORICAL, INK, INK_SECONDARY, MUTED, cell_edges, robust_limit
from ..processing import normalize
from ..transient import TAData, estimate_chirp, fit_chirp, oscillation_spectrum
from .common import (
    BusyCursor,
    QtCore,
    QtWidgets,
    add_legend,
    diverging_colormap,
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
from .uvvis_plotter import float_validator

FILE_FILTER = "TA data (*.dat *.txt *.csv *.scan);;All files (*)"
MAX_GLOBAL_WAVELENGTHS = 256


def raster_image(data: TAData, max_cols: int = 2000, max_rows: int = 1500):
    """Resample ``dA`` (mOD) onto a uniform raster for display.

    Each raster cell shows the measured value whose (non-uniform) cell it falls
    in, so the delay axis is not distorted; cells inside wavelength gaps (e.g.
    removed pump scatter) are NaN (transparent).

    Returns ``(image, (x, y, width, height))``.
    """
    t, wl = data.delays, data.wavelengths
    te, we = cell_edges(t), cell_edges(wl)
    t_span, wl_span = te[-1] - te[0], we[-1] - we[0]
    ncols = int(min(max_cols, max(t.size, np.ceil(t_span / np.min(np.diff(te))))))
    dwl = float(np.median(np.diff(wl))) if wl.size > 1 else 1.0
    nrows = int(min(max_rows, max(wl.size, np.ceil(wl_span / dwl))))
    tc = te[0] + (np.arange(ncols) + 0.5) * t_span / ncols
    wc = we[0] + (np.arange(nrows) + 0.5) * wl_span / nrows
    cols = np.clip(np.searchsorted(te, tc) - 1, 0, t.size - 1)
    rows = np.clip(np.searchsorted(we, wc) - 1, 0, wl.size - 1)
    image = 1e3 * data.dA[np.ix_(rows, cols)]
    image[np.abs(wc - wl[rows]) > dwl] = np.nan
    return image, (te[0], we[0], t_span, wl_span)


def parse_ranges(text: str) -> list[tuple[float, float]]:
    """Parse wavelength bands such as ``"510-522; 1020-1040"``."""
    ranges = []
    for part in re.split(r"[;\n]", text):
        numbers = re.findall(r"\d+(?:[.,]\d+)?", part)
        if len(numbers) >= 2:
            a, b = (float(n.replace(",", ".")) for n in numbers[:2])
            ranges.append((min(a, b), max(a, b)))
    return ranges


def _with_gaps(x, y):
    """Insert NaN where ``x`` jumps by more than 3 median steps (so lines break at gaps)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3:
        return x, y
    gaps = np.flatnonzero(np.diff(x) > 3 * np.median(np.diff(x))) + 1
    return np.insert(x, gaps, np.nan), np.insert(y, gaps, np.nan)


class TracesWidget(pg.GraphicsLayoutWidget):
    """A kinetic-trace plot and a transient-spectrum plot."""

    def __init__(self, parent=None, side_by_side: bool = False, title: str | None = None):
        super().__init__(parent)
        if title:
            self.setWindowTitle(title)
        self.kinetics = self.addPlot(row=0, col=0)
        self.spectral = self.addPlot(row=0, col=1) if side_by_side else self.addPlot(row=1, col=0)
        style_plot(self.kinetics, "ΔA (mOD)", "Delay (ps)")
        style_plot(self.spectral, "ΔA (mOD)", "Wavelength (nm)")
        for plot in (self.kinetics, self.spectral):
            add_legend(plot)
            plot.addLine(y=0, pen=pg.mkPen(MUTED, width=1))
        self.count = 0

    def next_color(self) -> str:
        color = series_color(self.count)
        self.count += 1
        return color

    def clear_all(self):
        for plot in (self.kinetics, self.spectral):
            plot.clear()
            plot.addLine(y=0, pen=pg.mkPen(MUTED, width=1))
        self.count = 0


class MapWidget(pg.GraphicsLayoutWidget):
    """False-colour dA map with a histogram/colour-bar and a crosshair."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot = self.addPlot(row=0, col=0)
        style_plot(self.plot, "Wavelength (nm)", "Delay (ps)")
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        self.hist = pg.HistogramLUTItem(fillHistogram=True)
        self.hist.setImageItem(self.image)
        self.hist.gradient.setColorMap(diverging_colormap())
        self.hist.axis.setLabel("ΔA (mOD)")
        self.addItem(self.hist, row=0, col=1)
        pen = pg.mkPen(INK, width=1, style=QtCore.Qt.DashLine)
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        self.plot.addItem(self.vline, ignoreBounds=True)
        self.plot.addItem(self.hline, ignoreBounds=True)
        self.marker = pg.ScatterPlotItem(
            size=14, symbol="+", pen=pg.mkPen(INK, width=2), brush=None
        )
        self.plot.addItem(self.marker)

    def show_data(self, data: TAData, levels: tuple[float, float]):
        image, rect = raster_image(data)
        self.image.setImage(image, autoLevels=False, levels=levels)
        self.image.setRect(QtCore.QRectF(*rect))
        self.hist.setLevels(*levels)
        self.hist.setHistogramRange(*levels, padding=0.05)
        self.plot.autoRange(padding=0)


class ControlPanel(QtWidgets.QWidget):
    """The Qt Designer control panel (``ta_control_panel.ui``)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        load_ui("ta_control_panel.ui", self)
        for edit in (
            self.W0_value,
            self.delta_wl,
            self.T0_value,
            self.time_min,
            self.time_max,
            self.wlen_min,
            self.wlen_max,
            self.inty_min,
            self.inty_max,
        ):
            edit.setValidator(float_validator(edit))


class GlobalResultWindow(pg.GraphicsLayoutWidget):
    """Decay- and evolution-associated spectra of a global fit."""

    def __init__(self, result: GlobalFitResult, title: str):
        super().__init__()
        self.setWindowTitle(f"Global analysis - {title}")
        self.resize(900, 750)
        self.result = result
        lifetimes = ", ".join(
            f"τ{i + 1} = {t:.3g} ± {e:.2g} ps"
            for i, (t, e) in enumerate(zip(result.taus, result.taus_stderr))
        )
        self.addLabel(
            lifetimes + ("  + long-lived" if result.step else ""), row=0, col=0, color=INK_SECONDARY
        )
        panels = [("DAS (mOD)", result.das, result.labels)]
        try:
            names = [
                f"{chr(65 + i)} ({lab.split('= ')[-1]})" for i, lab in enumerate(result.labels)
            ]
            panels.append(("EADS (mOD)", result.eads(), names))
        except (ValueError, np.linalg.LinAlgError):
            pass  # degenerate lifetimes: no sequential interpretation
        self.plots = []
        for row, (label, spectra, names) in enumerate(panels, start=1):
            plot = self.addPlot(row=row, col=0)
            style_plot(plot, label, "Wavelength (nm)")
            add_legend(plot)
            plot.addLine(y=0, pen=pg.mkPen(MUTED, width=1))
            for i, (spec, name) in enumerate(zip(spectra, names)):
                x, y = _with_gaps(result.wavelengths, 1e3 * spec)
                plot.plot(x, y, pen=pg.mkPen(series_color(i), width=2), name=name, connect="finite")
            self.plots.append(plot)


class TAInspector(QtWidgets.QMainWindow):
    """One data set: map + control panel on the left, traces at the clicked point on the right."""

    def __init__(self, suite: TASuite | None = None):
        super().__init__()
        self.suite = suite
        self.raw: TAData | None = None
        self.data: TAData | None = None
        self.point: tuple[float, float] | None = None
        self.current: dict | None = None
        self.chirp = None
        self.last_fit = None
        self.directory = Path.home()
        self.confirm_close = True
        self.extra_windows: list[QtWidgets.QWidget] = []
        self.setWindowTitle("TA Inspector")
        self.resize(1600, 1000)
        self._build_ui()
        self._connect()

    def _action(self, text, slot, shortcut=None, tip=""):
        action = QtWidgets.QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.setStatusTip(tip)
        action.triggered.connect(slot)
        return action

    def _build_ui(self):
        self.map = MapWidget()
        self.panel = ControlPanel()
        self.traces = TracesWidget()
        self.compare = TracesWidget(side_by_side=True, title="Traces comparison")
        self.compare.resize(1300, 500)
        left = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left.addWidget(self.map)
        left.addWidget(self.panel)
        left.setStretchFactor(0, 3)
        main = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main.addWidget(left)
        main.addWidget(self.traces)
        main.setStretchFactor(0, 3)
        main.setStretchFactor(1, 2)
        self.setCentralWidget(main)
        self.status = self.statusBar()
        actions = [
            self._action("Open", self.on_load, "Ctrl+O", "Load one or several scans (averaged)"),
            self._action("Save data", self.save_data, "Ctrl+S", "Write the processed matrix"),
            self._action("Export map", self.export_map, "Ctrl+E", "Save the map as PNG or SVG"),
            self._action(
                "Chirp", self.show_chirp, None, "Show the time-zero estimates and dispersion fit"
            ),
        ]
        toolbar = self.addToolBar("main")
        menu = self.menuBar().addMenu("&File")
        for action in actions:
            toolbar.addAction(action)
            menu.addAction(action)
        menu.addSeparator()
        menu.addAction(self._action("Close", self.close, "Ctrl+W"))

    def _connect(self):
        p = self.panel
        p.loadData_button.clicked.connect(self.on_load)
        p.correctData_button.clicked.connect(self.apply_corrections)
        p.plotData_button.clicked.connect(self.plot_map)
        p.exportplot_button.clicked.connect(self.export_map)
        p.gettraces_button.clicked.connect(self.get_traces)
        p.plottraces_button.clicked.connect(self.toggle_compare)
        p.exporttraces_button.clicked.connect(self.export_traces)
        p.cleartraces_button.clicked.connect(self.clear_traces)
        p.globalA_button.clicked.connect(self.global_analysis)
        p.savenotes_button.clicked.connect(self.save_notes)
        for box in (p.fit_checkbox, p.std_checkbox, p.fft_checkbox, p.artifact_checkbox):
            box.toggled.connect(lambda *_: self.refresh_traces())
        p.nexp_spin.valueChanged.connect(lambda *_: self.refresh_traces())
        self.map.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.map.scene().sigMouseClicked.connect(self.on_mouse_clicked)

    # ------------------------------------------------------------------ data
    def on_load(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Load TA data", str(self.directory), FILE_FILTER
        )
        if paths:
            self.load_files(paths)

    def load_files(self, paths) -> bool:
        try:
            with BusyCursor():
                data = read_ta(paths)
        except Exception as exc:
            warn(self, "Could not load data", str(exc))
            self.status.showMessage("Failed to load data")
            return False
        # Delays are counted from the first recorded delay; the pre-time-zero window
        # (dT0) and background subtraction refer to this origin.
        self.raw = data.shift_time(data.delays[0])
        self.directory = Path(paths[0]).parent
        p = self.panel
        p.filename_loaded.setText(
            str(paths[0]) if len(paths) == 1 else f"{len(paths)} scans in {self.directory}"
        )
        p.dataname_input.setText(data.name)
        p.W0_value.setText(f"{data.wavelengths[0]:.1f}")
        p.plotOpt_box.setEnabled(True)
        p.dataOpt_box.setEnabled(True)
        p.loadData_button.setText("New Data")
        self.setWindowTitle(f"TA Inspector - {data.name}")
        self.point = self.current = self.chirp = None
        self.apply_corrections()
        self.status.showMessage(
            f"Loaded {data.name}: {data.shape[0]} wavelengths x {data.shape[1]} delays"
        )
        return True

    def apply_corrections(self):
        if self.raw is None:
            return
        p = self.panel
        data = self.raw
        steps = []
        t_pre = parse_float(p.T0_value, 0.5)
        try:
            if p.w0_checkbox.isChecked():
                data = data.crop(wl=(parse_float(p.W0_value), None))
                steps.append("crop")
            if p.exclude_checkbox.isChecked():
                ranges = parse_ranges(p.exclude_range.text())
                if ranges:
                    data = data.exclude_wavelengths(*ranges)
                    steps.append("exclusion")
            if p.t0_checkbox.isChecked():
                data = data.subtract_background(before=t_pre)
                steps.append("background")
            try:
                noise = data.noise(before=t_pre)
            except ValueError:
                noise = data.noise()
            p.I0_value.setText(f"{1e3 * float(np.nanmedian(noise)):.3g}")
            self.chirp = None
            if p.chirp_checkbox.isChecked():
                wl, t0 = estimate_chirp(data)
                model = fit_chirp(wl, t0)
                data = data.correct_chirp(model)
                self.chirp = (wl, t0, model)
                steps.append("chirp")
            if p.dlambda_checkbox.isChecked():
                data = data.bin_wavelengths(width=parse_float(p.delta_wl, 1.0))
                steps.append("binning")
        except ValueError as exc:
            warn(self, "Correction failed", str(exc))
            return
        self.data = data
        self.update_fields()
        self.plot_map()
        self.status.showMessage("Corrections: " + (", ".join(steps) or "none"))

    def update_fields(self):
        d, p = self.data, self.panel
        p.time_min.setText(f"{d.delays[0]:.4g}")
        p.time_max.setText(f"{d.delays[-1]:.4g}")
        p.wlen_min.setText(f"{d.wavelengths[0]:.4g}")
        p.wlen_max.setText(f"{d.wavelengths[-1]:.4g}")
        vmax = robust_limit(1e3 * d.dA)
        p.inty_min.setText(f"{-vmax:.3g}")
        p.inty_max.setText(f"{vmax:.3g}")

    def view_data(self) -> TAData:
        """The processed data inside the axis ranges of the control panel."""
        p = self.panel
        t = (parse_float(p.time_min), parse_float(p.time_max))
        wl = (parse_float(p.wlen_min), parse_float(p.wlen_max))
        try:
            return self.data.crop(wl=wl, t=t)
        except ValueError:
            self.status.showMessage("The axis ranges contain no data - showing everything")
            return self.data

    def plot_map(self):
        if self.data is None:
            return
        p = self.panel
        lo, hi = parse_float(p.inty_min), parse_float(p.inty_max)
        if lo is None or hi is None or lo >= hi:
            vmax = robust_limit(1e3 * self.data.dA)
            lo, hi = -vmax, vmax
        self.map.show_data(self.view_data(), (lo, hi))
        self.refresh_traces()

    # --------------------------------------------------------------- mouse
    def on_mouse_moved(self, pos):
        plot = self.map.plot
        if not plot.sceneBoundingRect().contains(pos):
            return
        point = plot.vb.mapSceneToView(pos)
        self.map.vline.setPos(point.x())
        self.map.hline.setPos(point.y())
        if self.data is not None:
            d = self.data
            i = int(np.argmin(np.abs(d.wavelengths - point.y())))
            j = int(np.argmin(np.abs(d.delays - point.x())))
            self.status.showMessage(
                f"t = {point.x():.4g} ps   λ = {point.y():.1f} nm   ΔA = {1e3 * d.dA[i, j]:.3g} mOD"
            )

    def on_mouse_clicked(self, event):
        if self.data is None or event.button() != QtCore.Qt.LeftButton:
            return
        plot = self.map.plot
        if plot.sceneBoundingRect().contains(event.scenePos()):
            point = plot.vb.mapSceneToView(event.scenePos())
            self.select_point(point.x(), point.y())

    def select_point(self, delay: float, wavelength: float):
        self.point = (delay, wavelength)
        self.map.marker.setData([delay], [wavelength])
        self.refresh_traces()

    # -------------------------------------------------------------- traces
    def refresh_traces(self):
        if self.data is None or self.point is None:
            return
        p = self.panel
        view = self.view_data()
        t_sel, wl_sel = self.point
        delays, kinetic = view.kinetic(wl_sel)
        wls, spectrum = view.spectrum(t_sel)
        wl = float(view.wavelengths[np.argmin(np.abs(view.wavelengths - wl_sel))])
        t = float(view.delays[np.argmin(np.abs(view.delays - t_sel))])
        self.current = {
            "delays": delays,
            "kinetic": kinetic,
            "wavelengths": wls,
            "spectrum": spectrum,
            "wl": wl,
            "t": t,
        }

        k, s = self.traces.kinetics, self.traces.spectral
        self.traces.clear_all()
        x, y = _with_gaps(wls, 1e3 * spectrum)
        s.plot(
            x, y, pen=pg.mkPen(CATEGORICAL[0], width=2), name=f"Δt = {t:.3g} ps", connect="finite"
        )

        do_fft = p.fft_checkbox.isChecked()
        result = None
        if p.fit_checkbox.isChecked() or do_fft:
            try:
                result = fit_kinetics(
                    delays,
                    kinetic,
                    n_exp=p.nexp_spin.value(),
                    step=True,
                    artifact=3 if p.artifact_checkbox.isChecked() else 0,
                    sigma=view.kinetic_error(wl_sel),
                )
                self.last_fit = result
            except Exception as exc:
                self.status.showMessage(f"Kinetic fit failed: {exc}")

        if do_fft and result is not None:
            late = result.x > result["t0"] + 2.0 * result["fwhm"]
            wavenumbers, amplitude = oscillation_spectrum(result.x[late], result.residuals[late])
            style_plot(k, "Amplitude (mOD)", "Wavenumber (cm⁻¹)")
            k.plot(
                wavenumbers,
                1e3 * amplitude,
                pen=pg.mkPen(CATEGORICAL[0], width=2),
                name=f"λ = {wl:.1f} nm, residuals",
            )
            return

        style_plot(k, "ΔA (mOD)", "Delay (ps)")
        k.plot(
            delays, 1e3 * kinetic, pen=pg.mkPen(CATEGORICAL[0], width=1.5), name=f"λ = {wl:.1f} nm"
        )
        if p.std_checkbox.isChecked():
            err = view.kinetic_error(wl_sel)
            if err is None:
                self.status.showMessage(
                    "No standard error available (load several scans to average)"
                )
            else:
                k.addItem(
                    pg.ErrorBarItem(
                        x=delays, y=1e3 * kinetic, height=2e3 * err, beam=0, pen=pg.mkPen(MUTED)
                    )
                )
        if result is not None:
            n = p.nexp_spin.value()
            taus = ", ".join(
                f"τ{i + 1} = {result[f'tau{i + 1}']:.3g}±{result.errors[f'tau{i + 1}']:.2g}"
                for i in range(n)
            )
            k.plot(
                result.x,
                1e3 * result.best_fit,
                pen=pg.mkPen(CATEGORICAL[1], width=2.5),
                name=f"fit: {taus} ps",
            )

    def get_traces(self):
        if not self.current:
            self.status.showMessage("Click on the map to select a point first")
            return
        c, cur = self.compare, self.current
        color = c.next_color()
        name = self.panel.dataname_input.text()
        c.kinetics.plot(
            cur["delays"],
            1e3 * cur["kinetic"],
            pen=pg.mkPen(color, width=2),
            name=f"{name}: {cur['wl']:.1f} nm",
        )
        x, y = _with_gaps(cur["wavelengths"], 1e3 * cur["spectrum"])
        c.spectral.plot(
            x, y, pen=pg.mkPen(color, width=2), name=f"{name}: {cur['t']:.3g} ps", connect="finite"
        )
        self.status.showMessage("Traces added - press 'Plot Traces' to show the comparison")

    def toggle_compare(self):
        self.compare.setVisible(not self.compare.isVisible())

    def export_traces(self):
        """Send normalised traces to the suite window to compare different data sets."""
        if not self.current or self.suite is None:
            self.status.showMessage("Click on the map to select a point first")
            return
        cur, target = self.current, self.suite.comparison
        color = target.next_color()
        name = self.panel.dataname_input.text()
        try:
            kinetic = normalize(cur["kinetic"], "absmax")
            spectrum = normalize(cur["spectrum"], "absmax")
        except ValueError:
            return
        target.kinetics.plot(
            cur["delays"],
            kinetic,
            pen=pg.mkPen(color, width=2),
            name=f"{name} ({cur['wl']:.1f} nm)",
        )
        x, y = _with_gaps(cur["wavelengths"], spectrum)
        target.spectral.plot(
            x, y, pen=pg.mkPen(color, width=2), name=f"{name} ({cur['t']:.3g} ps)", connect="finite"
        )
        for plot in (target.kinetics, target.spectral):
            plot.setLabel("left", "ΔA (normalised)")
        self.suite.show()
        self.suite.raise_()

    def clear_traces(self):
        self.compare.clear_all()
        if self.suite is not None:
            self.suite.comparison.clear_all()

    # ------------------------------------------------------------ analysis
    def global_analysis(self):
        if self.data is None:
            return
        view = self.view_data()
        if view.shape[0] > MAX_GLOBAL_WAVELENGTHS:
            view = view.bin_wavelengths(n=int(np.ceil(view.shape[0] / MAX_GLOBAL_WAVELENGTHS)))
        n = self.panel.nexp_spin.value()
        artifact = 3 if self.panel.artifact_checkbox.isChecked() else 0
        try:
            with BusyCursor():
                result = view.global_fit(n_exp=n, step=True, artifact=artifact)
        except Exception as exc:
            warn(self, "Global analysis failed", str(exc))
            return
        name = self.panel.dataname_input.text()
        window = GlobalResultWindow(result, name)
        window.show()
        self.extra_windows.append(window)
        t0, t1 = view.delays[0], view.delays[-1]
        self.panel.annotations.append(
            f"Global analysis of {name} ({t0:.3g}-{t1:.4g} ps):\n{result.summary()}\n"
        )
        self.status.showMessage(result.summary().splitlines()[0])
        return result

    def show_chirp(self):
        if self.chirp is None:
            self.status.showMessage("Tick 'Unchirp' and apply corrections to estimate the chirp")
            return
        wl, t0, model = self.chirp
        widget = pg.PlotWidget(title="Time zero versus wavelength")
        plot = widget.getPlotItem()
        style_plot(plot, "Time zero (ps)", "Wavelength (nm)")
        add_legend(plot)
        ok = np.isfinite(t0)
        plot.plot(
            wl[ok],
            t0[ok],
            pen=None,
            symbol="o",
            symbolSize=7,
            symbolBrush=CATEGORICAL[0],
            symbolPen="w",
            name="estimates",
        )
        grid = np.linspace(wl.min(), wl.max(), 200)
        plot.plot(grid, model(grid), pen=pg.mkPen(CATEGORICAL[1], width=2), name="polynomial fit")
        widget.resize(700, 450)
        widget.show()
        self.extra_windows.append(widget)

    # ----------------------------------------------------------------- I/O
    def export_map(self, path: str | None = None):
        if self.data is None:
            return
        if path is None:
            name = self.panel.dataname_input.text() or "ta_map"
            path = save_dialog(
                self,
                "Export map",
                self.directory / f"{name}_map.png",
                "PNG image (*.png);;SVG vector graphics (*.svg)",
            )
            if not path:
                return
        export_plot(self.map.ci, path)
        self.status.showMessage(f"Map saved to {path}")

    def save_data(self, path: str | None = None):
        if self.data is None:
            return
        if path is None:
            name = self.panel.dataname_input.text() or "ta_data"
            path = save_dialog(
                self,
                "Save processed data",
                self.directory / f"{name}_processed.dat",
                "TA matrix (*.dat *.txt)",
            )
            if not path:
                return
        write_ta_matrix(path, self.data)
        self.status.showMessage(f"Processed data saved to {path}")

    def save_notes(self, path: str | None = None):
        if path is None:
            name = self.panel.dataname_input.text() or "ta"
            path = save_dialog(
                self, "Save notes", self.directory / f"{name}_notes.txt", "Text (*.txt)"
            )
            if not path:
                return
        Path(path).write_text(self.panel.annotations.toPlainText(), encoding="utf-8")
        self.status.showMessage(f"Notes saved to {path}")

    def closeEvent(self, event):
        if self.confirm_close and self.raw is not None:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Close inspector",
                "Close this data set? Unsaved results and notes will be lost.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
        for window in [self.compare, *self.extra_windows]:
            window.close()
        if self.suite is not None and self in self.suite.inspectors:
            self.suite.inspectors.remove(self)
        event.accept()


class TASuite(QtWidgets.QMainWindow):
    """Launcher window holding the cross-data-set comparison of normalised traces."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TA Analyser Suite")
        self.resize(1300, 600)
        self.inspectors: list[TAInspector] = []
        self.comparison = TracesWidget(side_by_side=True)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        hint = QtWidgets.QLabel(
            "Normalised traces sent with 'Export Traces' from any inspector appear here."
        )
        layout.addWidget(hint)
        layout.addWidget(self.comparison)
        self.setCentralWidget(central)
        self.statusBar()
        new = QtWidgets.QAction("New inspector", self)
        new.setShortcut("Ctrl+N")
        new.triggered.connect(self.new_inspector)
        clear = QtWidgets.QAction("Clear comparison", self)
        clear.triggered.connect(self.comparison.clear_all)
        quit_ = QtWidgets.QAction("Quit", self)
        quit_.setShortcut("Ctrl+Q")
        quit_.triggered.connect(self.close)
        toolbar = self.addToolBar("main")
        menu = self.menuBar().addMenu("&Menu")
        for action in (new, clear, quit_):
            toolbar.addAction(action)
            menu.addAction(action)

    def new_inspector(self) -> TAInspector:
        window = TAInspector(self)
        window.show()
        self.inspectors.append(window)
        return window

    def closeEvent(self, event):
        for window in list(self.inspectors):
            if not window.close():
                event.ignore()
                return
        event.accept()


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = make_app(argv)
    suite = TASuite()
    suite.show()
    inspector = suite.new_inspector()
    inspector.showMaximized()
    files = [a for a in argv[1:] if Path(a).is_file()]
    if files:
        inspector.load_files(files)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
