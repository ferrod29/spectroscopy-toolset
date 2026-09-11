"""Generic 1-D/2-D processing: unit conversion, binning, smoothing, baselines,
normalisation and resampling."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

#: Planck constant times speed of light, in eV nm.
HC_EV_NM = 1239.841984
#: Speed of light in cm / ps (converts wavenumbers to angular frequencies for delays in ps).
C_CM_PER_PS = 0.0299792458


# --------------------------------------------------------------------------
# Unit conversion
# --------------------------------------------------------------------------
def nm_to_ev(wavelength_nm):
    """Photon energy (eV) for a wavelength in nm."""
    return HC_EV_NM / np.asarray(wavelength_nm, dtype=float)


def ev_to_nm(energy_ev):
    """Wavelength (nm) for a photon energy in eV."""
    return HC_EV_NM / np.asarray(energy_ev, dtype=float)


def nm_to_wavenumber(wavelength_nm):
    """Wavenumber (cm^-1) for a wavelength in nm."""
    return 1e7 / np.asarray(wavelength_nm, dtype=float)


def wavenumber_to_nm(wavenumber_cm):
    """Wavelength (nm) for a wavenumber in cm^-1."""
    return 1e7 / np.asarray(wavenumber_cm, dtype=float)


# --------------------------------------------------------------------------
# Selection, binning, smoothing
# --------------------------------------------------------------------------
def crop_mask(x, lo=None, hi=None):
    """Boolean mask of ``lo <= x <= hi``; ``None`` leaves that side open."""
    x = np.asarray(x, dtype=float)
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    mask = np.ones(x.shape, dtype=bool)
    if lo is not None:
        mask &= x >= lo
    if hi is not None:
        mask &= x <= hi
    return mask


def bin_mean(array, n, axis=0):
    """Average consecutive groups of ``n`` samples along ``axis``.

    A trailing remainder shorter than ``n`` is dropped.
    """
    a = np.asarray(array, dtype=float)
    n = int(n)
    if n < 1:
        raise ValueError("bin size must be >= 1")
    if n == 1:
        return a.copy()
    a = np.moveaxis(a, axis, 0)
    m = a.shape[0] // n
    if m == 0:
        raise ValueError(f"cannot bin {a.shape[0]} samples in groups of {n}")
    out = a[: m * n].reshape((m, n) + a.shape[1:]).mean(axis=1)
    return np.moveaxis(out, 0, axis)


def _odd_window(window, n):
    window = int(window)
    if window % 2 == 0:
        window += 1
    max_window = n if n % 2 else n - 1
    return max(1, min(window, max_window))


def smooth(y, window=11, polyorder=3, deriv=0, delta=1.0, axis=-1):
    """Savitzky-Golay smoothing (or differentiation with ``deriv > 0``)."""
    y = np.asarray(y, dtype=float)
    window = _odd_window(window, y.shape[axis])
    polyorder = min(polyorder, window - 1)
    return savgol_filter(y, window, polyorder, deriv=deriv, delta=delta, axis=axis)


def derivative(x, y, order=1, window=11, polyorder=3):
    """Savitzky-Golay derivative ``d^n y / dx^n``.

    A non-uniform ``x`` grid is resampled to a uniform one, differentiated and
    interpolated back.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    polyorder = max(polyorder, order)
    dx = np.diff(x)
    if np.allclose(dx, dx.mean(), rtol=1e-3):
        return smooth(y, window, polyorder, deriv=order, delta=dx.mean())
    xu = np.linspace(x[0], x[-1], x.size)
    du = smooth(np.interp(xu, x, y), window, polyorder, deriv=order, delta=xu[1] - xu[0])
    return np.interp(x, xu, du)


# --------------------------------------------------------------------------
# Baselines and normalisation
# --------------------------------------------------------------------------
def baseline_als(y, lam=1e5, p=0.01, niter=10):
    """Asymmetric least-squares baseline (Eilers & Boelens, 2005).

    Parameters
    ----------
    lam : float
        Smoothness penalty (larger = stiffer baseline; typically 1e2 - 1e9).
    p : float
        Asymmetry: weight given to points above the baseline (0.001 - 0.1 for
        positive bands on top of the baseline).
    niter : int
        Number of re-weighting iterations.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    d = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(n, n - 2))
    penalty = lam * (d @ d.T)
    w = np.ones(n)
    z = y
    for _ in range(niter):
        z = spsolve((sparse.diags(w) + penalty).tocsc(), w * y)
        w = p * (y > z) + (1.0 - p) * (y <= z)
    return z


def baseline_polynomial(x, y, degree=1, regions=None):
    """Polynomial baseline fitted to the points inside ``regions``.

    ``regions`` is a list of ``(x0, x1)`` intervals known to contain no bands
    (default: all points).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if regions is None:
        mask = np.ones(x.shape, dtype=bool)
    else:
        mask = np.zeros(x.shape, dtype=bool)
        for lo, hi in regions:
            mask |= crop_mask(x, lo, hi)
    if mask.sum() <= degree:
        raise ValueError("not enough points in the baseline regions for this degree")
    poly = np.polynomial.Polynomial.fit(x[mask], y[mask], degree)
    return poly(x)


def baseline_offset(x, y, at):
    """Constant baseline: ``y`` at ``at`` (a position) or its mean over ``at = (x0, x1)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.ndim(at) == 0:
        value = y[np.argmin(np.abs(x - at))]
    else:
        mask = crop_mask(x, *at)
        if not mask.any():
            raise ValueError(f"no points inside {at}")
        value = y[mask].mean()
    return np.full_like(y, value)


def normalize(y, method="max", x=None, at=None):
    """Normalise ``y``.

    ``method`` is one of ``'max'`` (maximum = 1), ``'absmax'`` (largest
    magnitude = +/-1), ``'minmax'`` (scaled to 0..1), ``'area'`` (integral of
    ``|y|`` = 1, needs ``x``) or ``'at'`` (``y(at)`` = 1, needs ``x``).
    """
    y = np.asarray(y, dtype=float)
    if method == "max":
        scale = np.nanmax(y)
    elif method == "absmax":
        scale = y[np.nanargmax(np.abs(y))]
    elif method == "minmax":
        lo, hi = np.nanmin(y), np.nanmax(y)
        return (y - lo) / (hi - lo)
    elif method == "area":
        if x is None:
            raise ValueError("'area' normalisation needs x")
        order = np.argsort(x)
        scale = trapezoid(np.abs(y)[order], np.asarray(x, dtype=float)[order])
    elif method == "at":
        if x is None or at is None:
            raise ValueError("'at' normalisation needs x and at")
        scale = np.interp(at, x, y)
    else:
        raise ValueError(f"unknown normalisation method {method!r}")
    if scale == 0 or not np.isfinite(scale):
        raise ValueError("normalisation factor is zero or not finite")
    return y / scale


# --------------------------------------------------------------------------
# Resampling and averaging
# --------------------------------------------------------------------------
def resample(x_new, x, y, fill=np.nan):
    """Linear interpolation of ``y(x)`` onto ``x_new`` (``fill`` outside the data range)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    return np.interp(np.asarray(x_new, dtype=float), x[order], y[order], left=fill, right=fill)


def common_grid(xs: Sequence[np.ndarray], step=None):
    """Uniform grid spanning the range shared by all ``xs``.

    The default step is the finest median sample spacing among the inputs.
    """
    lo = max(float(np.min(x)) for x in xs)
    hi = min(float(np.max(x)) for x in xs)
    if lo >= hi:
        raise ValueError("the curves do not share a common x range")
    if step is None:
        step = min(float(np.median(np.abs(np.diff(np.sort(x))))) for x in xs)
    n = int(np.floor((hi - lo) / step + 1e-9)) + 1
    return lo + step * np.arange(n)


def average_curves(xs, ys, x_grid=None):
    """Mean and sample standard deviation of several curves on a common grid.

    Curves are linearly interpolated onto ``x_grid`` (default:
    :func:`common_grid`), so curves recorded on different grids are averaged
    correctly.

    Returns
    -------
    grid, mean, std : ndarray
    """
    xs = [np.asarray(x, dtype=float) for x in xs]
    ys = [np.asarray(y, dtype=float) for y in ys]
    if not xs:
        raise ValueError("no curves to average")
    grid = common_grid(xs) if x_grid is None else np.asarray(x_grid, dtype=float)
    stack = np.vstack([resample(grid, x, y) for x, y in zip(xs, ys)])
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0, ddof=1) if len(xs) > 1 else np.zeros_like(mean)
    return grid, mean, std


def robust_std(y):
    """Noise estimate insensitive to slow signals: scaled MAD of first differences."""
    d = np.diff(np.asarray(y, dtype=float))
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)
