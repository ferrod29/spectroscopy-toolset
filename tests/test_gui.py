"""Offscreen smoke tests of the Qt interfaces (skipped without PyQt5/pyqtgraph)."""

import os

import numpy as np
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spectroscopy_toolset import models  # noqa: E402
from spectroscopy_toolset.gui.common import make_app  # noqa: E402
from spectroscopy_toolset.gui.ta_inspector import TASuite, parse_ranges, raster_image  # noqa: E402
from spectroscopy_toolset.gui.uvvis_plotter import UVVisPlotter  # noqa: E402
from spectroscopy_toolset.io import write_ta_matrix  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return make_app(["test"])


@pytest.fixture
def spectra_files(tmp_path):
    x = np.linspace(300, 800, 501)
    paths = []
    for i, scale in enumerate((0.5, 1.0)):
        y = scale * (models.gaussian(x, 1.0, 420, 25) + 0.6 * models.gaussian(x, 1.0, 520, 35))
        path = tmp_path / f"s{i}.csv"
        np.savetxt(path, np.c_[x, y], delimiter=",", header="Wavelength (nm),Abs", comments="")
        paths.append(str(path))
    return paths


def test_uvvis_plotter_workflow(app, tmp_path, spectra_files):
    w = UVVisPlotter()
    w.confirm_close = False
    assert w.load_files(spectra_files) == 2
    w.average()
    assert len(w.curves) == 3
    p = w.panel
    p.legend_list.setCurrentIndex(1)
    p.xaxis_min.setText("470")
    p.xaxis_max.setText("600")
    p.gaussian_button.setChecked(True)
    w.run_analysis()
    fitted = p.fit_table.item(1, 2).text()
    assert fitted and abs(float(fitted.split()[0]) - 520) < 3
    p.findpeaks_button.setChecked(True)
    p.xaxis_min.setText("300")
    p.xaxis_max.setText("800")
    w.run_analysis()
    assert "Peaks of" in p.notes_textbox.toPlainText()
    p.energy_checkbox.setChecked(True)
    assert w.displayed(w.curves[0])[0].max() < 5  # eV
    p.energy_checkbox.setChecked(False)
    w.toggle_visibility()
    w.ramp_colors()
    w.save_figure(str(tmp_path / "fig.png"))
    w.export_data(str(tmp_path / "data.csv"))
    assert (tmp_path / "fig.png").exists() and (tmp_path / "data.csv").exists()
    w.clear()
    assert not w.curves
    w.close()


def test_ta_inspector_workflow(app, tmp_path, synthetic_ta):
    path = tmp_path / "ta.dat"
    write_ta_matrix(path, synthetic_ta)
    suite = TASuite()
    ins = suite.new_inspector()
    ins.confirm_close = False
    assert ins.load_files([str(path)])
    p = ins.panel
    p.t0_checkbox.setChecked(True)
    p.T0_value.setText("0.5")
    p.dlambda_checkbox.setChecked(True)
    p.delta_wl.setText("10")
    p.exclude_range.setText("600-610")
    p.exclude_checkbox.setChecked(True)
    ins.apply_corrections()
    assert ins.data.shape[0] < synthetic_ta.shape[0]
    p.fit_checkbox.setChecked(True)
    ins.select_point(5.0, 520.0)
    assert ins.last_fit is not None and ins.current["wl"] == pytest.approx(520, abs=10)
    ins.get_traces()
    ins.export_traces()
    p.fft_checkbox.setChecked(True)
    p.fft_checkbox.setChecked(False)
    p.nexp_spin.setValue(3)
    result = ins.global_analysis()
    np.testing.assert_allclose(result.taus, [0.5, 8.0, 120.0], rtol=0.05)
    ins.export_map(str(tmp_path / "map.png"))
    ins.save_data(str(tmp_path / "processed.dat"))
    ins.save_notes(str(tmp_path / "notes.txt"))
    for name in ("map.png", "processed.dat", "notes.txt"):
        assert (tmp_path / name).exists()
    suite.close()


def test_raster_and_range_helpers(synthetic_ta):
    gapped = synthetic_ta.exclude_wavelengths((550, 600))
    image, (x, y, w, h) = raster_image(gapped)
    assert np.isnan(image).any() and np.isfinite(image).any()
    assert x < gapped.delays[0] and x + w > gapped.delays[-1]
    assert parse_ranges("510-522; 1020,5 - 1040") == [(510.0, 522.0), (1020.5, 1040.0)]
