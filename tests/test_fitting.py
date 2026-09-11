import numpy as np
import pytest

import spectroscopy_toolset as st
from conftest import FWHM, T0, TAUS, make_das
from spectroscopy_toolset import fitting, models


def test_fit_gaussian_with_uncertainties(rng):
    x = np.linspace(-5, 5, 201)
    y = models.gaussian(x, 2.0, 0.5, 1.2) + rng.normal(0, 0.02, x.size)
    res = st.fit(models.gaussian, x, y, [1.0, 0.0, 1.0], names=["amplitude", "center", "sigma"])
    assert res.success
    for name, true in zip(res.names, (2.0, 0.5, 1.2)):
        assert abs(res[name] - true) < 4 * res.errors[name]
    assert np.all(res.stderr > 0)
    assert res.r_squared > 0.99
    assert "amplitude" in res.summary()
    assert list(res.to_frame().index) == ["amplitude", "center", "sigma"]
    np.testing.assert_allclose(res.eval(x), res.best_fit)


def test_fit_fixed_bounds_and_weights(rng):
    x = np.linspace(0, 10, 101)
    noise = rng.normal(0, 0.05, x.size)
    y = models.exp_decay(x, 3.0, 2.0, 0.5) + noise
    res = st.fit(
        models.exp_decay, x, y, [1.0, 1.0, 0.5], names=["a", "tau", "c"], fixed=["c"], sigma=0.05
    )
    assert res["c"] == 0.5 and res.errors["c"] == 0.0
    # with correct weights the reduced chi-square equals the (sample) noise variance / sigma^2
    assert res.reduced_chi2 == pytest.approx((noise.std(ddof=1) / 0.05) ** 2, rel=0.1)
    bounded = st.fit(
        models.exp_decay, x, y, [1.0, 1.0, 0.0], names=["a", "tau", "c"], bounds={"tau": (0.1, 1.0)}
    )
    assert "tau" in bounded.at_bound
    with pytest.raises(ValueError):
        st.fit(models.exp_decay, x, y, [1, 1, 1], names=["a", "tau", "c"], fixed=["a", "tau", "c"])


def test_fit_kinetics_recovers_irf_and_lifetimes(delays, rng):
    y = models.multi_exp_irf(delays, T0, FWHM, -1.0, 0.8, 0.6, 35.0) + rng.normal(
        0, 0.01, delays.size
    )
    res = st.fit_kinetics(delays, y, n_exp=2)
    assert res["tau1"] == pytest.approx(0.8, rel=0.05)
    assert res["tau2"] == pytest.approx(35.0, rel=0.05)
    assert res["t0"] == pytest.approx(T0, abs=0.01)
    assert res["fwhm"] == pytest.approx(FWHM, rel=0.1)
    assert res["tau1"] < res["tau2"]
    assert len(res.components) == 2


def test_fit_kinetics_sorts_components_whatever_the_start(delays, rng):
    y = models.multi_exp_irf(delays, T0, FWHM, 1.0, 50.0, -0.5, 1.0) + rng.normal(
        0, 0.005, delays.size
    )
    res = st.fit_kinetics(delays, y, n_exp=2, taus=[60.0, 0.5])
    assert res["tau1"] == pytest.approx(1.0, rel=0.05)
    assert res["a1"] == pytest.approx(-0.5, rel=0.05)
    assert res["tau2"] == pytest.approx(50.0, rel=0.05)


def test_fit_kinetics_step_artifact_and_no_irf(delays, rng):
    shapes = fitting._artifact_basis(delays, T0, FWHM, 2)
    y = (
        models.multi_exp_irf(delays, T0, FWHM, -1.0, 3.0)
        + 0.3 * models.exp_irf(delays, np.inf, T0, FWHM)
        + 0.8 * shapes[0]
        - 0.5 * shapes[1]
        + rng.normal(0, 0.005, delays.size)
    )
    res = st.fit_kinetics(delays, y, n_exp=1, step=True, artifact=2)
    assert res["tau1"] == pytest.approx(3.0, rel=0.05)
    assert res["a_inf"] == pytest.approx(0.3, abs=0.02)
    assert res["ca0"] == pytest.approx(0.8, abs=0.05)
    assert "coherent artefact" in res.components

    t = np.linspace(0, 20, 200)
    plain = st.fit_kinetics(t, models.exp_decay(t, 2.0, 4.0), n_exp=1, irf=False, t0=0.0)
    assert plain["tau1"] == pytest.approx(4.0, rel=1e-4)
    with pytest.raises(ValueError):
        st.fit_kinetics(t, t, irf=False, artifact=1)


def test_global_fit_recovers_lifetimes_and_das(synthetic_ta):
    res = synthetic_ta.global_fit(n_exp=3)
    np.testing.assert_allclose(res.taus, TAUS, rtol=0.01)
    assert res.t0 == pytest.approx(T0, abs=0.005)
    assert res.fwhm == pytest.approx(FWHM, rel=0.03)
    np.testing.assert_allclose(res.das, synthetic_ta.meta["das"], atol=2e-5)
    assert res.rmse == pytest.approx(5e-6, rel=0.1)
    assert not res.at_bound
    assert res.concentrations().shape == (3, synthetic_ta.delays.size)
    assert "tau1" in res.summary()


def test_global_fit_many_wavelengths_and_artifact(delays, rng):
    wl = np.linspace(450, 750, 400)  # more wavelengths than delays: exercises SVD compression
    t = delays[delays < 60]
    das = make_das(wl)[:2]
    c = np.column_stack([models.exp_irf(t, tau, T0, FWHM) for tau in (0.5, 8.0)])
    art = np.outer(2e-4 * np.cos(wl / 20), fitting._artifact_basis(t, T0, FWHM, 1)[0])
    d = (c @ das).T + art + rng.normal(0, 5e-6, (wl.size, t.size))
    res = st.TAData(wl, t, d).global_fit(n_exp=2, artifact=1)
    np.testing.assert_allclose(res.taus, [0.5, 8.0], rtol=0.01)
    assert res.artifact_spectra.shape == (1, wl.size)
    np.testing.assert_allclose(res.artifact_spectra[0], 2e-4 * np.cos(wl / 20), atol=2e-5)


def test_sequential_scheme_and_eads(delays, rng):
    taus = np.array([0.5, 8.0, 120.0])
    wl = np.linspace(450, 750, 60)
    eads = make_das(wl)
    b = fitting.sequential_matrix(taus)
    basis = np.column_stack([models.exp_irf(delays, tau, T0, FWHM) for tau in taus])
    conc = basis @ b.T  # species populations
    np.testing.assert_allclose(b.sum(axis=1)[0], 1.0)  # species A starts with all population
    d = (conc @ eads).T + rng.normal(0, 2e-6, (wl.size, delays.size))
    res = st.TAData(wl, delays, d).global_fit(n_exp=3)
    np.testing.assert_allclose(res.eads(), eads, atol=3e-5)
    np.testing.assert_allclose(st.das_to_eads(b.T @ eads, taus), eads, atol=1e-12)
    with pytest.raises(ValueError):
        fitting.sequential_matrix([1.0, 1.0])


def test_sequential_with_long_lived_species():
    b = fitting.sequential_matrix([2.0, np.inf])
    t = np.array([0.0, 2.0, 1e6])
    pops = b @ np.vstack([np.exp(-t / 2.0), np.ones_like(t)])
    np.testing.assert_allclose(pops.sum(axis=0), 1.0)  # population is conserved
