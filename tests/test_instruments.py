"""Readers for the HELIOS, HARPIA and HARBOR TA setups and the HARBOR wavelength calibration."""

import numpy as np
import pytest

import spectroscopy_toolset as st
from spectroscopy_toolset import io as stio
from spectroscopy_toolset.calibration import PrismCalibration, fused_silica_index
from spectroscopy_toolset.cli import main

TRUE_CAL = PrismCalibration(focal_length=166.0, offset=-325.0)
BANDS = [(445, 20, 1.1), (475, 8, 0.9), (525, 7, 2.5), (583, 10, 3.5), (680, 6, 0.6)]


def filter_absorbance(wl):
    """A BG36-like absorbance spectrum with several sharp bands."""
    return 0.05 + sum(a * np.exp(-0.5 * ((wl - c) / w) ** 2) for c, w, a in BANDS)


# --------------------------------------------------------------------------
# Synthetic HARBOR files
# --------------------------------------------------------------------------
def write_harbor(path, delays_fs, dA_mod, n_pix=600, scan=1, open_beam=None, jitter=None):
    """One HARBOR scan: forward then backward sweep, three channels plus zero padding."""
    rng = np.random.default_rng(scan)
    pixels = np.arange(1, n_pix + 1)
    header = np.r_[scan, np.tile(pixels, 3), np.zeros(1860 - 3 * n_pix)]
    rows = []
    for order in (slice(None), slice(None, None, -1)):
        t = delays_fs[order]
        noise = rng.normal(0, 2.0, t.size) if jitter is None else jitter[order]
        for k, tk in zip(range(t.size), t + noise):
            idx = np.arange(delays_fs.size)[order][k]
            ch0 = dA_mod[:n_pix, idx]
            ch1 = np.ones(n_pix) if open_beam is None else open_beam[:n_pix]
            rows.append(np.r_[tk, ch0, ch1, ch1, np.zeros(1860 - 3 * n_pix)])
    np.savetxt(path, np.vstack([header, *rows]), delimiter="\t", fmt="%.6f")
    return path


@pytest.fixture
def harbor_signal():
    delays = np.r_[-2000.0, -1000.0, np.arange(-500.0, 1000.0, 100.0), 5000.0, 20000.0]
    px = np.arange(1, 601)
    dA = (
        10.0
        * np.exp(-0.5 * ((px[:, None] - 300) / 40) ** 2)
        * (delays > 0)
        * np.exp(-np.clip(delays, 0, None) / 3000.0)
    )
    dA[560:] = np.nan  # unused detector pixels
    return delays, dA


def test_detect_formats(tmp_path, harbor_signal, synthetic_ta):
    delays, dA = harbor_signal
    harbor = write_harbor(tmp_path / "s_0001.dat", delays, dA)
    matrix = tmp_path / "m.dat"
    stio.write_ta_matrix(matrix, synthetic_ta)
    harpia = tmp_path / "h_matrix.dat"
    harpia.write_text("XAxisTitle Wavelength (nm)\nYAxisTitle Delay (ps)\n0 400 401\n1 2 3\n")
    raw = tmp_path / "raw.dat"
    raw.write_text("Pump-probe, Not referenced, 500 shots per spectrum,\n")
    assert stio.detect_ta_format(harbor) == "harbor"
    assert stio.detect_ta_format(matrix) == "matrix"
    assert stio.detect_ta_format(harpia) == "harpia"
    with pytest.raises(ValueError, match="_matrix.dat"):
        stio.detect_ta_format(raw)


def test_read_harbor_sweeps_units_and_errors(tmp_path, harbor_signal):
    delays, dA = harbor_signal
    files = [write_harbor(tmp_path / f"s_{k:04d}.dat", delays, dA, scan=k) for k in (1, 2)]
    data = st.read_ta(files)
    assert data.meta["n_scans"] == 4  # two files x (forward + backward)
    assert data.meta["wavelength_axis"] == "pixel"
    np.testing.assert_allclose(data.delays, delays * 1e-3, atol=0.01)  # fs -> ps, mean of jitter
    assert 0 < data.meta["delay_spread_ps"] < 0.02
    assert data.wavelengths[-1] == 560  # all-NaN pixels dropped
    i300 = np.searchsorted(data.wavelengths, 300)
    np.testing.assert_allclose(data.dA[i300], 1e-3 * dA[299], atol=1e-9)  # mOD -> OD
    assert data.std is not None and np.all(data.std[i300] < 1e-9)
    fwd = st.read_harbor(files[0], sweeps="forward", pixel_range=(250, 350))
    assert fwd.meta["n_scans"] == 1 and fwd.wavelengths[[0, -1]].tolist() == [250, 350]


def test_read_harbor_mixed_channel_widths(tmp_path, harbor_signal):
    delays, dA = harbor_signal
    a = write_harbor(tmp_path / "a.dat", delays, dA, n_pix=600)
    b = write_harbor(tmp_path / "b.dat", delays, dA, n_pix=580, scan=2)
    data = st.read_harbor([a, b])
    assert data.wavelengths.max() == 560
    i = np.searchsorted(data.wavelengths, 300)
    np.testing.assert_allclose(data.dA[i], 1e-3 * dA[299], atol=1e-9)


def test_read_harbor_with_calibration(tmp_path, harbor_signal):
    delays, dA = harbor_signal
    f = write_harbor(tmp_path / "s.dat", delays, dA)
    json_path = tmp_path / "cal.json"
    TRUE_CAL.save(json_path)
    data = st.read_harbor(f, calibration=json_path, pixel_range=(220, 340))
    np.testing.assert_allclose(data.wavelengths, np.sort(TRUE_CAL.wavelengths(np.arange(220, 341))))
    table = tmp_path / "wl.txt"
    np.savetxt(table, TRUE_CAL.wavelengths(np.arange(220, 341)), header="nm")
    same = st.read_harbor(f, calibration=table, pixel_range=(220, 340))
    np.testing.assert_allclose(same.wavelengths, data.wavelengths)
    with pytest.raises(ValueError, match="wavelengths for"):
        st.read_harbor(f, calibration=np.arange(10.0))


# --------------------------------------------------------------------------
# Wavelength calibration
# --------------------------------------------------------------------------
def test_fused_silica_index_and_round_trip(tmp_path):
    assert fused_silica_index(587.6) == pytest.approx(1.4585, abs=2e-4)  # n_d of fused silica
    px = np.arange(200.0, 400.0)
    np.testing.assert_allclose(TRUE_CAL.pixels(TRUE_CAL.wavelengths(px)), px, atol=1e-3)
    assert np.all(np.diff(TRUE_CAL.wavelengths(px)) > 0)
    assert np.isnan(TRUE_CAL.wavelengths([-5000.0])).all()
    TRUE_CAL.save(tmp_path / "c.json")
    assert PrismCalibration.load(tmp_path / "c.json") == TRUE_CAL


def synthetic_transmission(px, sigma=4.0):
    from scipy.ndimage import gaussian_filter1d

    grid = np.arange(380.0, 800.0, 0.2)
    t = gaussian_filter1d(10 ** -filter_absorbance(grid), sigma / 0.2)
    return np.interp(TRUE_CAL.wavelengths(px), grid, t)


def test_fit_prism_calibration_recovers_parameters():
    px = np.arange(224.0, 337.0)
    ref_wl = np.arange(390.0, 800.0, 0.2)
    fit = st.fit_prism_calibration(
        px, synthetic_transmission(px), ref_wl, filter_absorbance(ref_wl)
    )
    np.testing.assert_allclose(fit.wavelengths, TRUE_CAL.wavelengths(px), atol=0.3)
    assert fit.broadening == pytest.approx(4.0, abs=0.2)
    assert fit.scale == pytest.approx(1.0, abs=0.02) and abs(fit.baseline) < 0.01
    assert fit.r_squared > 0.999
    assert fit.calibration.valid_pixels == (224.0, 336.0)
    assert "focal length" in str(fit)


def test_fit_prism_calibration_swapped_ratio_warns():
    px = np.arange(224.0, 337.0)
    ref_wl = np.arange(390.0, 800.0, 0.2)
    with pytest.warns(UserWarning, match="swapped"):
        fit = st.fit_prism_calibration(
            px, 1.0 / synthetic_transmission(px), ref_wl, filter_absorbance(ref_wl)
        )
    np.testing.assert_allclose(fit.wavelengths, TRUE_CAL.wavelengths(px), atol=0.3)


@pytest.fixture
def calibration_files(tmp_path):
    px = np.arange(1, 601, dtype=float)
    lamp = 0.5 * np.exp(-0.5 * ((px - 280) / 30) ** 2)  # probe light only near the centre
    t = np.ones_like(px)
    lit = (px >= 190) & (px <= 370)
    t[lit] = synthetic_transmission(px[lit])
    delays = np.array([-9000.0, -9000.0])
    zeros = np.zeros((600, 2))
    with_filter = write_harbor(
        tmp_path / "cal_b.dat", delays, zeros, open_beam=lamp * t, jitter=np.zeros(2)
    )
    without = write_harbor(
        tmp_path / "cal_a.dat", delays, zeros, open_beam=lamp, jitter=np.zeros(2)
    )
    # the intensity sits in channel 1 in these synthetic files; put it in channel 0 as on the setup
    for f in (with_filter, without):
        table = np.loadtxt(f)
        table[1:, 1:601] = table[1:, 601:1201]
        np.savetxt(f, table, delimiter="\t", fmt="%.6g")
    ref = tmp_path / "RefBG36.txt"
    wl = np.arange(400.0, 1000.0, 0.2)
    np.savetxt(
        ref, np.c_[wl, filter_absorbance(wl)], header="Wavelength\tAbsorbance\nnm\tOD", comments=""
    )
    return with_filter, without, ref


def test_calibrate_harbor_and_cli(tmp_path, calibration_files, capsys):
    with_filter, without, ref = calibration_files
    fit = st.calibrate_harbor(with_filter, without, ref)
    lo, hi = fit.calibration.valid_pixels
    assert lo > 200 and hi < 360  # dim pixels left out
    np.testing.assert_allclose(fit.wavelengths, TRUE_CAL.wavelengths(fit.pixels), atol=0.5)

    cal_json = tmp_path / "cal.json"
    argv = ["wl-calibrate", "--with-filter", str(with_filter), "--without-filter", str(without)]
    argv += ["--reference", str(ref), "--save", str(cal_json), "--out", str(tmp_path / "wl")]
    assert main(argv) == 0
    assert "focal length" in capsys.readouterr().out
    assert (tmp_path / "wl.png").exists()
    assert PrismCalibration.load(cal_json).focal_length == pytest.approx(166.0, abs=2.0)


def test_cli_reads_harbor_with_calibration(tmp_path, harbor_signal, capsys):
    delays, dA = harbor_signal
    f = write_harbor(tmp_path / "s.dat", delays, dA)
    TRUE_CAL.save(tmp_path / "cal.json")
    argv = ["ta-export", str(f), "--calibration", str(tmp_path / "cal.json")]
    argv += ["--pixel-range", "220", "340", "--matrix", str(tmp_path / "out.dat")]
    assert main(argv) == 0
    out = st.read_ta(tmp_path / "out.dat")
    assert out.wavelengths.min() == pytest.approx(TRUE_CAL.wavelengths(220), abs=1e-4)
    assert main(["ta-export", str(f), "--format", "matrix", "--calibration", "x.json"]) == 1
    assert "HARBOR files only" in capsys.readouterr().err


# --------------------------------------------------------------------------
# HELIOS and HARPIA
# --------------------------------------------------------------------------
def write_helios(path, wl, delays, dA):
    table = np.zeros((wl.size + 1, delays.size + 1))
    table[0, 1:], table[1:, 0], table[1:, 1:] = delays, wl, dA
    np.savetxt(path, table, delimiter="\t", fmt="%.6f")
    return path


def test_read_helios_nan_pixels_and_delay_jitter(tmp_path, rng):
    wl = np.linspace(300, 800, 60)
    delays = np.r_[np.arange(-2.0, 2.0, 0.1), np.arange(2.0, 50.0, 2.0)]
    dA = 1e-3 * np.exp(-0.5 * ((wl[:, None] - 550) / 50) ** 2) * (delays > 0)
    scans = []
    for k in range(3):
        d = dA.copy()
        d[: 3 + k] = np.nan  # the flagged pixels differ between scans
        jitter = rng.uniform(-0.02, 0.02, delays.size) if k else 0.0
        scans.append(write_helios(tmp_path / f"scan_{k:02d}.dat", wl, delays + jitter, d))
    one = st.read_helios(scans[2])
    assert one.shape == (55, delays.size)
    with pytest.raises(ValueError, match="delay_tolerance"):
        st.read_ta(scans)
    data = st.read_ta(scans, format="helios", delay_tolerance=0.05)
    assert data.shape == (57, delays.size)  # rows 3-4 are finite in some scans only
    assert data.meta["n_scans"] == 3 and data.meta["delay_spread_ps"] <= 0.02
    np.testing.assert_allclose(data.dA, dA[3:], atol=1e-6)


def test_read_helios_raw(tmp_path):
    wl = np.linspace(400, 700, 20)
    delays = np.linspace(-1, 10, 15)
    off = 0.5 + 0.1 * np.sin(wl / 30)[None, :] * np.ones((delays.size, 1))
    dA = 2e-3 * np.cos(delays / 5)[:, None] * np.ones(wl.size)
    on = off * 10**-dA

    def write(path, intensity):  # delays down the first column, blank lines between rows
        with open(path, "w") as fh:
            fh.write("0\t" + "\t".join(f"{w:.4f}" for w in wl) + "\n\n")
            for t, row in zip(delays, intensity):
                fh.write(f"{t:.4f}\t" + "\t".join(f"{v:.8f}" for v in row) + "\n\n")
        return path

    data = st.read_helios_raw(
        write(tmp_path / "s_RAW_Signal Pump on.dat", on),
        write(tmp_path / "s_RAW_Signal Pump off.dat", off),
    )
    np.testing.assert_allclose(data.dA, dA.T, atol=1e-6)
    assert data.name == "s"


def test_read_harpia(tmp_path):
    wl = np.linspace(350, 550, 12)
    delays = np.array([-5.0, 0.0, 1.0, 10.0, 6000.0])
    dA_mod = 20.0 * np.outer(np.ones(delays.size), np.exp(-0.5 * ((wl - 450) / 30) ** 2))
    table = np.zeros((delays.size + 1, wl.size + 1))
    table[0, 1:], table[1:, 0], table[1:, 1:] = wl, delays, dA_mod
    path = tmp_path / "Run1_matrix.dat"
    with open(path, "w") as fh:
        fh.write("XAxisTitle Wavelength (nm)\nYAxisTitle Delay (ps)\n")
        np.savetxt(fh, table, fmt="%20.8E")
    data = st.read_ta(path)
    np.testing.assert_allclose(data.delays, delays)
    np.testing.assert_allclose(data.dA, 1e-3 * dA_mod.T)
    fs = tmp_path / "fs_matrix.dat"
    fs.write_text(path.read_text().replace("Delay (ps)", "Delay (fs)"))
    np.testing.assert_allclose(st.read_harpia(fs).delays, delays * 1e-3)
