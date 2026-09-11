import numpy as np
import pytest

import spectroscopy_toolset as st
from spectroscopy_toolset import models, uvvis
from spectroscopy_toolset.processing import ev_to_nm


@pytest.fixture
def two_bands():
    x = np.linspace(300, 700, 801)
    y = models.gaussian(x, 1.0, 420, 20) + models.gaussian(x, 0.6, 540, 30)
    return x, y


def test_find_peaks_positions_and_widths(two_bands):
    x, y = two_bands
    peaks = uvvis.find_peaks(x, y)
    np.testing.assert_allclose(peaks["position"], [420, 540], atol=0.5)
    # well-separated bands on a flat baseline: width at half prominence = FWHM
    separated = models.gaussian(x, 1.0, 380, 12) + models.gaussian(x, 0.6, 600, 18) + 0.2
    widths = uvvis.find_peaks(x, separated)["fwhm"]
    np.testing.assert_allclose(
        widths, [12 * models.FWHM_PER_SIGMA, 18 * models.FWHM_PER_SIGMA], rtol=0.01
    )
    assert len(uvvis.find_peaks(x, y, max_peaks=1)) == 1
    valleys = uvvis.find_peaks(x, -y, valleys=True)
    np.testing.assert_allclose(valleys["position"], [420, 540], atol=0.5)
    assert uvvis.find_peaks(x, np.zeros_like(x) + 1.0).empty


def test_fit_peaks_deconvolves_overlapping_bands(rng):
    x = np.linspace(350, 650, 301)
    y = models.gaussian(x, 0.8, 470, 25) + models.gaussian(x, 0.5, 520, 20) + 0.05
    y = y + rng.normal(0, 0.003, x.size)
    res = uvvis.fit_peaks(x, y, centers=[460, 530], baseline="constant")
    assert res["center1"] == pytest.approx(470, abs=1.0)
    assert res["center2"] == pytest.approx(520, abs=1.0)
    assert res["sigma2"] == pytest.approx(20, rel=0.05)
    assert res["c0"] == pytest.approx(0.05, abs=0.01)
    assert set(res.components) == {"peak1", "peak2", "baseline"}
    lor = uvvis.fit_peaks(x, models.lorentzian(x, 1.0, 500, 10), n_peaks=1, shape="lorentzian")
    assert lor["gamma1"] == pytest.approx(10, rel=1e-3)
    voigt = uvvis.fit_peaks(x, models.pseudo_voigt(x, 1.0, 500, 30, 0.3), n_peaks=1, shape="voigt")
    assert voigt["eta1"] == pytest.approx(0.3, abs=0.01)
    with pytest.raises(ValueError):
        uvvis.fit_peaks(x, y, shape="triangle")


def test_beer_lambert_and_conversions():
    assert uvvis.concentration_from_absorbance(0.5, 1e4, 1.0) == pytest.approx(5e-5)
    assert uvvis.molar_absorptivity(0.5, 5e-5) == pytest.approx(1e4)
    a = np.array([0.0, 0.5, 2.0])
    np.testing.assert_allclose(
        uvvis.transmittance_to_absorbance(uvvis.absorbance_to_transmittance(a)), a
    )
    assert uvvis.absorbance_to_transmittance(1.0, percent=True) == pytest.approx(10.0)
    assert uvvis.transmittance_to_absorbance(10.0, percent=True) == pytest.approx(1.0)


def test_calibration_curve(rng):
    c = np.array([0.0, 10e-6, 20e-6, 40e-6, 60e-6, 80e-6])
    eps, blank = 1.2e4, 0.01
    a = eps * c + blank + rng.normal(0, 0.002, c.size)
    cal = uvvis.calibration_curve(c, a)
    assert cal.molar_absorptivity == pytest.approx(eps, rel=0.03)
    assert cal.intercept == pytest.approx(blank, abs=0.005)
    assert cal.r_squared > 0.999
    assert cal.lod == pytest.approx(3.3 * cal.residual_std / cal.slope)
    assert cal.loq > cal.lod
    assert cal.concentration(eps * 30e-6 + blank) == pytest.approx(30e-6, rel=0.03)
    origin = uvvis.calibration_curve(c, eps * c, path_length=0.5, through_origin=True)
    assert origin.molar_absorptivity == pytest.approx(2 * eps)
    assert "LOD" in cal.summary()
    with pytest.raises(ValueError):
        uvvis.calibration_curve([1, 2], [1, 2])


def _semiconductor(eg, kind, urbach=0.05):
    wl = np.linspace(300, 800, 1001)
    e = 1239.841984 / wl
    above = np.clip(e - eg, 0, None)
    if kind == "direct":
        alpha = np.sqrt(above) / e
    else:
        alpha = above**2 / e
    tail = 0.02 * np.exp((e - eg) / urbach) * (e < eg)
    return wl, alpha + tail


@pytest.mark.parametrize("kind", ["direct", "indirect"])
def test_tauc_band_gap(kind):
    wl, a = _semiconductor(2.5, kind)
    auto = uvvis.tauc_bandgap(wl, a, transition=kind)
    assert auto.band_gap == pytest.approx(2.5, abs=0.02)
    manual = uvvis.tauc_bandgap(wl, a, transition=kind, fit_range=(2.7, 3.5))
    assert manual.band_gap == pytest.approx(2.5, abs=0.01)
    assert manual.r_squared > 0.99
    assert "Eg" in manual.summary()
    assert ev_to_nm(auto.band_gap) == pytest.approx(ev_to_nm(2.5), abs=5)


def test_spectrum_container(tmp_path, two_bands):
    x, y = two_bands
    spec = st.Spectrum(x[::-1], (y + 0.1 + 1e-4 * x)[::-1], name="s")
    assert spec.x[0] < spec.x[-1]
    cropped = spec.crop(400, 600)
    assert cropped.x.min() >= 400 and cropped.x.max() <= 600
    assert len(spec) == x.size
    flat = spec.subtract_baseline("poly", degree=1, regions=[(300, 340), (660, 700)])
    assert abs(flat.y[0]) < 1e-3
    assert spec.subtract_baseline("offset", at=700).y[-1] == pytest.approx(0.0)
    assert spec.normalize().y.max() == pytest.approx(1.0)
    energy = spec.to_energy()
    assert energy.x_label == "Energy (eV)" and energy.x[0] < energy.x[-1]
    assert spec.derivative(order=2).y.shape == spec.y.shape
    assert len(spec.find_peaks()) == 2
    assert spec.smooth(7).y.shape == spec.y.shape
    np.testing.assert_allclose(spec.resample([400, 500]).x, [400, 500])
    with pytest.raises(ValueError):
        st.Spectrum([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        spec.baseline("magic")
