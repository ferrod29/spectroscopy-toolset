import numpy as np
import pytest

import spectroscopy_toolset as st
from conftest import EXAMPLE_DATA
from spectroscopy_toolset import models
from spectroscopy_toolset.processing import C_CM_PER_PS
from spectroscopy_toolset.transient import ChirpModel, average_scans


def test_container_validation_and_sorting():
    d = st.TAData([510, 500], [1.0, 0.0], [[1, 2], [3, 4]])
    np.testing.assert_allclose(d.wavelengths, [500, 510])
    np.testing.assert_allclose(d.dA, [[4, 3], [2, 1]])
    with pytest.raises(ValueError):
        st.TAData([1, 2, 3], [0, 1], np.zeros((2, 3)))


def test_crop_exclude_shift(synthetic_ta):
    c = synthetic_ta.crop(wl=(500, 600), t=(0, 10))
    assert c.wavelengths.min() >= 500 and c.wavelengths.max() <= 600
    assert c.delays.min() >= 0 and c.delays.max() <= 10
    e = synthetic_ta.exclude_wavelengths((510, 530), (700, 710))
    assert not np.any((e.wavelengths >= 510) & (e.wavelengths <= 530))
    np.testing.assert_allclose(synthetic_ta.shift_time(0.2).delays, synthetic_ta.delays - 0.2)
    with pytest.raises(ValueError):
        synthetic_ta.crop(wl=(1000, 1100))


def test_background_and_noise(synthetic_ta):
    shifted = synthetic_ta.copy(dA=synthetic_ta.dA + 3e-4)
    corrected = shifted.subtract_background(before=-0.5)
    assert np.abs(corrected.dA[:, synthetic_ta.delays < -0.5]).mean() < 1e-5
    assert np.median(corrected.noise(before=-0.5)) == pytest.approx(5e-6, rel=0.2)
    assert np.median(corrected.noise()) == pytest.approx(5e-6, rel=0.3)
    with pytest.raises(ValueError):
        synthetic_ta.background(before=-5.0)


def test_binning_and_slices(synthetic_ta):
    std = np.full(synthetic_ta.shape, 1e-5)
    data = synthetic_ta.copy(std=std)
    b = data.bin_wavelengths(n=3)
    assert b.shape == (30, synthetic_ta.shape[1])
    np.testing.assert_allclose(b.std, 1e-5 / np.sqrt(3))
    assert data.bin_wavelengths(width=10.0).shape[0] == 30
    with pytest.raises(ValueError):
        data.bin_wavelengths()
    t, k = synthetic_ta.kinetic(520.0, width=10.0)
    rows = np.abs(synthetic_ta.wavelengths - 520) <= 5
    np.testing.assert_allclose(k, synthetic_ta.dA[rows].mean(axis=0))
    assert data.kinetic_error(520.0, width=10.0) == pytest.approx(1e-5 / np.sqrt(rows.sum()))
    wl, s = synthetic_ta.spectrum(5.0)
    assert s.shape == wl.shape


def test_svd_rank(synthetic_ta):
    _, s, _ = synthetic_ta.svd()
    assert s[2] / s[3] > 20  # three components above the noise floor
    assert synthetic_ta.svd(n=4)[1].size == 4


def _chirped(model, method):
    wl = np.linspace(450, 750, 150)
    t = np.arange(-2.0, 5.0, 0.02)
    t0 = model(wl)
    if method == "onset":
        d = np.vstack([-1e-3 * models.exp_irf(t, 50.0, ti, 0.1) for ti in t0])
    else:
        d = np.vstack([1e-3 * models.gaussian_irf(t, ti, 0.1) for ti in t0])
    d = d + np.random.default_rng(3).normal(0, 1e-5, d.shape)
    return st.TAData(wl, t, d)


@pytest.mark.parametrize("method", ["onset", "max"])
def test_chirp_estimation_and_correction(method):
    true = ChirpModel(np.array([0.3, 0.25, -0.05]), 600.0)
    data = _chirped(true, method)
    wl, t0 = data.estimate_chirp(window=(-1.0, 1.5), method=method)
    model = st.fit_chirp(wl, t0, order=2, center=600.0)
    grid = np.linspace(460, 740, 50)
    np.testing.assert_allclose(model(grid), true(grid), atol=0.01)
    corrected = data.correct_chirp(model)
    assert np.all(np.isfinite(corrected.dA))
    _, t0_after = corrected.estimate_chirp(window=(-1.0, 1.0), method=method)
    assert np.nanstd(t0_after) < 0.01
    assert abs(np.nanmedian(t0_after)) < 0.02
    with pytest.raises(ValueError):
        data.correct_chirp(np.zeros(3))


def test_oscillation_spectrum_units():
    t = np.arange(0.0, 10.0, 0.01)
    wavenumber = 150.0
    y = 2e-4 * np.cos(2 * np.pi * C_CM_PER_PS * wavenumber * t)
    wn, amp = st.oscillation_spectrum(t, y, max_wavenumber=500, n=1000)
    assert wn[np.argmax(amp)] == pytest.approx(wavenumber, abs=3.0)
    assert amp.max() == pytest.approx(2e-4, rel=0.1)


def test_average_scans(synthetic_ta):
    avg = average_scans([synthetic_ta, synthetic_ta.copy(dA=synthetic_ta.dA * 1.0)])
    np.testing.assert_allclose(avg.dA, synthetic_ta.dA)
    with pytest.raises(ValueError):
        average_scans([synthetic_ta, synthetic_ta.crop(wl=(500, 600))])


@pytest.mark.skipif(
    not (EXAMPLE_DATA / "TestData_1.dat").exists(), reason="example data not present"
)
def test_example_data_chirp_is_normal_dispersion():
    data = st.read_ta(EXAMPLE_DATA / "TestData_1.dat")
    data = data.shift_time(data.delays[0]).exclude_wavelengths((509, 523)).subtract_background(0.45)
    wl, t0 = data.estimate_chirp(window=(0.3, 1.5))
    model = st.fit_chirp(wl, t0)
    assert model(700.0) > model(550.0)  # red light arrives later (normal dispersion)
