import numpy as np
import pytest
from scipy.integrate import trapezoid

from spectroscopy_toolset import models
from spectroscopy_toolset import processing as pr


def test_unit_conversions():
    assert pr.nm_to_ev(500.0) == pytest.approx(2.4797, abs=1e-4)
    assert pr.ev_to_nm(pr.nm_to_ev(633.0)) == pytest.approx(633.0)
    assert pr.nm_to_wavenumber(500.0) == pytest.approx(20000.0)
    assert pr.wavenumber_to_nm(20000.0) == pytest.approx(500.0)


def test_crop_mask_inclusive_and_open():
    x = np.arange(10.0)
    assert pr.crop_mask(x, 2, 4).sum() == 3
    assert pr.crop_mask(x, 4, 2).sum() == 3
    assert pr.crop_mask(x, None, 4).sum() == 5
    assert pr.crop_mask(x).all()


def test_bin_mean():
    a = np.arange(10.0)
    np.testing.assert_allclose(pr.bin_mean(a, 3), [1.0, 4.0, 7.0])
    m = np.arange(12.0).reshape(3, 4)
    np.testing.assert_allclose(pr.bin_mean(m, 2, axis=1), [[0.5, 2.5], [4.5, 6.5], [8.5, 10.5]])
    with pytest.raises(ValueError):
        pr.bin_mean(a, 0)
    with pytest.raises(ValueError):
        pr.bin_mean(a, 20)


def test_smooth_preserves_polynomial_and_derivative():
    x = np.linspace(0, 1, 101)
    y = 3 * x**2 - x
    np.testing.assert_allclose(pr.smooth(y, 11, 3), y, atol=1e-10)
    xs = np.linspace(0, 2 * np.pi, 400)
    np.testing.assert_allclose(pr.derivative(xs, np.sin(xs))[20:-20], np.cos(xs)[20:-20], atol=1e-4)
    xn = 2 * np.pi * np.linspace(0, 1, 500) ** 1.5  # smoothly non-uniform, like an energy axis
    np.testing.assert_allclose(pr.derivative(xn, np.sin(xn))[30:-30], np.cos(xn)[30:-30], atol=5e-3)


def test_baseline_als_recovers_sloping_baseline():
    x = np.linspace(300, 800, 501)
    baseline = 0.1 + 2e-4 * (x - 300)
    y = baseline + models.gaussian(x, 1.0, 450, 15) + models.gaussian(x, 0.5, 600, 20)
    est = pr.baseline_als(y, lam=1e6, p=0.001)
    far = (np.abs(x - 450) > 60) & (np.abs(x - 600) > 70)
    assert np.max(np.abs(est - baseline)[far]) < 0.01


def test_baseline_polynomial_and_offset():
    x = np.linspace(0, 10, 101)
    y = 1 + 0.5 * x + models.gaussian(x, 3.0, 5.0, 0.5)
    base = pr.baseline_polynomial(x, y, degree=1, regions=[(0, 2), (8, 10)])
    np.testing.assert_allclose(base, 1 + 0.5 * x, atol=1e-6)
    np.testing.assert_allclose(pr.baseline_offset(x, y, at=10.0), y[-1])
    assert pr.baseline_offset(x, y, at=(0, 1))[0] == pytest.approx(y[x <= 1].mean())
    with pytest.raises(ValueError):
        pr.baseline_polynomial(x, y, degree=3, regions=[(0, 0.15)])


def test_normalize_methods():
    x = np.linspace(0, 1, 11)
    y = np.array([0, 1, 2, 4, 2, 1, 0, -1, -2, -1, 0], dtype=float)
    assert pr.normalize(y, "max").max() == pytest.approx(1.0)
    assert pr.normalize(-y * 3, "absmax")[3] == pytest.approx(1.0)
    n = pr.normalize(y, "minmax")
    assert (n.min(), n.max()) == (0.0, 1.0)
    assert trapezoid(np.abs(pr.normalize(y, "area", x=x)), x) == pytest.approx(1.0)
    assert pr.normalize(y, "at", x=x, at=0.3)[3] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        pr.normalize(np.zeros(3))
    with pytest.raises(ValueError):
        pr.normalize(y, "bogus")


def test_average_curves_on_different_grids():
    x1 = np.linspace(400, 700, 301)
    x2 = np.linspace(390, 710, 97)
    f = lambda x: models.gaussian(x, 1.0, 550, 40)  # noqa: E731
    grid, mean, std = pr.average_curves([x1, x2], [f(x1), f(x2)])
    assert grid[0] >= 400 and grid[-1] <= 700
    np.testing.assert_allclose(mean, f(grid), atol=2e-3)
    assert np.nanmax(std) < 3e-3
    with pytest.raises(ValueError):
        pr.common_grid([np.array([0.0, 1.0]), np.array([2.0, 3.0])])


def test_robust_std_ignores_slow_signal():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 10, 5000)
    y = 5 * np.exp(-t) + rng.normal(0, 0.1, t.size)
    assert pr.robust_std(y) == pytest.approx(0.1, rel=0.1)
