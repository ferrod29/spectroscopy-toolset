import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erfc

from spectroscopy_toolset import models


def test_gaussian_fwhm():
    sigma = 3.0
    half = models.FWHM_PER_SIGMA * sigma / 2
    assert models.gaussian(10.0 + half, 2.0, 10.0, sigma) == pytest.approx(1.0)


def test_lorentzian_hwhm():
    assert models.lorentzian(7.0, 4.0, 5.0, 2.0) == pytest.approx(2.0)


def test_pseudo_voigt_limits():
    x = np.linspace(-5, 5, 101)
    fwhm = 2.0
    g = models.gaussian(x, 1.0, 0.0, fwhm / models.FWHM_PER_SIGMA)
    lor = models.lorentzian(x, 1.0, 0.0, fwhm / 2)
    np.testing.assert_allclose(models.pseudo_voigt(x, 1.0, 0.0, fwhm, 0.0), g)
    np.testing.assert_allclose(models.pseudo_voigt(x, 1.0, 0.0, fwhm, 1.0), lor)


def test_voigt_peak_height():
    assert models.voigt(3.0, 2.5, 3.0, 0.7, 0.4) == pytest.approx(2.5)


def test_multi_exp_offset():
    t = np.array([0.0, 1.0])
    np.testing.assert_allclose(models.multi_exp(t, 2.0, 1.0, 0.5), [2.5, 2 * np.exp(-1) + 0.5])
    np.testing.assert_allclose(models.multi_exp(t, 2.0, 1.0), [2.0, 2 * np.exp(-1)])


@pytest.mark.parametrize("tau", [0.05, 0.4, 5.0])
def test_exp_irf_matches_numerical_convolution(tau):
    t0, fwhm = 0.3, 0.2
    sigma = fwhm / models.FWHM_PER_SIGMA

    def convolved(t):
        # the Gaussian is negligible beyond 12 sigma, so integrate over that window only
        lo, hi = max(t0, t - 12 * sigma), t + 12 * sigma
        if hi <= lo:
            return 0.0
        integrand = lambda s: np.exp(-(s - t0) / tau) * np.exp(-0.5 * ((t - s) / sigma) ** 2)  # noqa: E731
        val, _ = quad(integrand, lo, hi, epsabs=1e-15, epsrel=1e-11, limit=200)
        return val / (sigma * np.sqrt(2 * np.pi))

    t = np.array([-0.2, 0.2, 0.3, 0.45, 1.0, 3.0])
    expected = [convolved(ti) for ti in t]
    np.testing.assert_allclose(models.exp_irf(t, tau, t0, fwhm), expected, rtol=1e-8, atol=1e-15)


def test_exp_irf_limits_and_stability():
    t = np.linspace(-1, 5, 61)
    np.testing.assert_allclose(
        models.exp_irf(t, 1.0, 0.0, 0.0), np.where(t >= 0, np.exp(-np.clip(t, 0, None)), 0)
    )
    sigma = 0.3 / models.FWHM_PER_SIGMA
    np.testing.assert_allclose(
        models.exp_irf(t, np.inf, 0.0, 0.3), 0.5 * erfc(-t / (sigma * np.sqrt(2)))
    )
    extreme = models.exp_irf(np.array([-1e4, -10.0, 0.0, 10.0, 1e6]), 1e-3, 0.0, 1e-5)
    assert np.all(np.isfinite(extreme))
    assert models.exp_irf(0.0, 2.0, 0.0, 0.1).shape == ()


def test_multi_exp_irf_long_time_limit():
    y = models.multi_exp_irf(np.array([100.0]), 0.0, 0.1, 1.0, 1.0, 2.0, 1e9)
    assert y[0] == pytest.approx(2.0, rel=1e-6)
