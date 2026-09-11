import matplotlib.pyplot as plt
import numpy as np
import pytest

import spectroscopy_toolset as st
from spectroscopy_toolset import models, plotting, uvvis


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_series_colors_follow_the_rules():
    assert plotting.series_colors(3) == plotting.CATEGORICAL[:3]
    assert plotting.series_colors(8) == plotting.CATEGORICAL
    many = plotting.series_colors(12)
    assert len(set(many)) == 12 and not set(many) & set(plotting.CATEGORICAL[1:])
    assert len(plotting.series_colors(4, ordered=True)) == 4


def test_robust_limit_ignores_saturated_rows():
    z = np.random.default_rng(0).normal(0, 0.1, (100, 50))
    z[:50] += 1.0
    z[3] = 300.0  # e.g. pump scatter
    assert 1.0 < plotting.robust_limit(z) < 2.0
    assert plotting.robust_limit(np.full((3, 3), np.nan)) == 1.0


def test_steady_state_plots():
    x = np.linspace(300, 700, 401)
    spectra = [
        st.Spectrum(x, models.gaussian(x, a, 450, 30), name=f"s{a}") for a in (0.5, 1.0, 1.5)
    ]
    ax = plotting.plot_spectra(spectra, normalize="max", energy=True)
    assert len(ax.get_lines()) == 3 and ax.get_legend() is not None
    peaks = spectra[1].find_peaks()
    assert plotting.plot_peaks(spectra[1], peaks) is not None
    fit = spectra[1].fit_peaks(n_peaks=1)
    main, resid = plotting.plot_fit(fit, xlabel="nm")
    assert main.get_legend() is not None and resid.get_ylabel() == "residual"
    cal = uvvis.calibration_curve([0, 1, 2, 3], [0.01, 0.2, 0.41, 0.6])
    assert plotting.plot_calibration(cal) is not None


def test_tauc_plot():
    wl = np.linspace(300, 800, 501)
    e = 1239.84 / wl
    a = np.sqrt(np.clip(e - 2.5, 0, None)) / e
    result = uvvis.tauc_bandgap(wl, a)
    ax = plotting.plot_tauc(result)
    assert ax.get_xlabel() == "Photon energy (eV)"


def test_ta_plots(synthetic_ta):
    gapped = synthetic_ta.exclude_wavelengths((550, 580))
    ax = plotting.plot_ta_map(gapped, linthresh=1.0)
    assert len(ax.collections) == 2  # one mesh per contiguous wavelength block
    fits = [synthetic_ta.fit_kinetics(520, n_exp=2)]
    assert plotting.plot_kinetics(synthetic_ta, [520], fits=fits, linthresh=1.0) is not None
    assert plotting.plot_ta_spectra(gapped, [0.5, 5, 50]) is not None
    result = synthetic_ta.global_fit(n_exp=3)
    assert len(plotting.plot_das(result).get_lines()) >= 3
    assert plotting.plot_das(result, eads=True).get_ylabel().startswith("EADS")
    wl, t0 = np.linspace(450, 750, 20), np.linspace(0.1, 0.4, 20)
    assert plotting.plot_chirp(wl, t0, st.fit_chirp(wl, t0)) is not None
    assert plotting.plot_singular_values(synthetic_ta) is not None
