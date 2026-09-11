"""Model functions for spectral band shapes and kinetic decays.

Every function follows the ``f(x, *params)`` convention used by
:func:`scipy.optimize.curve_fit` and :func:`spectroscopy_toolset.fitting.fit`
and is vectorised over ``x``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc, erfcx, voigt_profile

#: Ratio FWHM / sigma of a Gaussian, 2*sqrt(2*ln 2).
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))

_SQRT2 = np.sqrt(2.0)


# --------------------------------------------------------------------------
# Band shapes
# --------------------------------------------------------------------------
def gaussian(x, amplitude, center, sigma):
    """Gaussian band of peak height ``amplitude`` and standard deviation ``sigma``."""
    x = np.asarray(x, dtype=float)
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def lorentzian(x, amplitude, center, gamma):
    """Lorentzian (Cauchy) band of peak height ``amplitude`` and half width at half maximum ``gamma``."""
    x = np.asarray(x, dtype=float)
    return amplitude / (1.0 + ((x - center) / gamma) ** 2)


def pseudo_voigt(x, amplitude, center, fwhm, eta):
    """Pseudo-Voigt band: ``eta * Lorentzian + (1 - eta) * Gaussian`` sharing one FWHM."""
    gauss = gaussian(x, 1.0, center, fwhm / FWHM_PER_SIGMA)
    lorentz = lorentzian(x, 1.0, center, fwhm / 2.0)
    return amplitude * ((1.0 - eta) * gauss + eta * lorentz)


def voigt(x, amplitude, center, sigma, gamma):
    """Voigt profile (Gaussian ``sigma`` convolved with Lorentzian HWHM ``gamma``) of peak height ``amplitude``."""
    x = np.asarray(x, dtype=float)
    return amplitude * voigt_profile(x - center, sigma, gamma) / voigt_profile(0.0, sigma, gamma)


def power_law(x, amplitude, exponent, offset=0.0):
    """Power law ``amplitude * x**exponent + offset``."""
    return amplitude * np.power(np.asarray(x, dtype=float), exponent) + offset


# --------------------------------------------------------------------------
# Kinetics
# --------------------------------------------------------------------------
def exp_decay(t, amplitude, tau, offset=0.0):
    """Single exponential ``amplitude * exp(-t / tau) + offset``."""
    return amplitude * np.exp(-np.asarray(t, dtype=float) / tau) + offset


def multi_exp(t, *params):
    """Sum of exponentials ``sum_i a_i exp(-t / tau_i)``.

    ``params`` is ``a1, tau1, a2, tau2, ...``; an odd number of parameters
    adds the last one as a constant offset.
    """
    t = np.asarray(t, dtype=float)
    n = len(params) // 2
    y = np.full_like(t, params[-1] if len(params) % 2 else 0.0)
    for a, tau in zip(params[0 : 2 * n : 2], params[1 : 2 * n : 2]):
        y = y + a * np.exp(-t / tau)
    return y


def gaussian_irf(t, t0=0.0, fwhm=0.1):
    """Gaussian instrument response of unit peak height (a model for the coherent artefact)."""
    return gaussian(t, 1.0, t0, fwhm / FWHM_PER_SIGMA)


def exp_irf(t, tau, t0=0.0, fwhm=0.1):
    """Unit-amplitude exponential decay convolved with a Gaussian instrument response.

    Computes ``(H(t') exp(-t'/tau)) * G(t'; fwhm)`` evaluated at ``t' = t - t0``,
    where ``H`` is the Heaviside step and ``G`` a unit-area Gaussian.
    ``tau = inf`` gives a (long-lived) step, ``fwhm <= 0`` the unconvolved decay.

    The closed form ``0.5 exp(s^2/2tau^2 - u/tau) erfc((s/tau - u/s)/sqrt 2)`` overflows
    for large ``u / s``; it is evaluated with ``erfcx`` on the rising edge and
    with ``erfc`` on the decaying side so that no intermediate over/underflows.
    """
    t = np.asarray(t, dtype=float)
    u = np.atleast_1d(t - t0)
    if fwhm <= 0:
        if np.isinf(tau):
            out = (u >= 0).astype(float)
        else:
            out = np.where(u >= 0, np.exp(-np.clip(u, 0.0, None) / tau), 0.0)
        return out.reshape(t.shape)

    sigma = fwhm / FWHM_PER_SIGMA
    if np.isinf(tau):
        return (0.5 * erfc(-u / (sigma * _SQRT2))).reshape(t.shape)

    z = (sigma / tau - u / sigma) / _SQRT2
    out = np.empty_like(u)
    rising = z >= 0
    out[rising] = 0.5 * np.exp(-0.5 * (u[rising] / sigma) ** 2) * erfcx(z[rising])
    decaying = ~rising
    out[decaying] = 0.5 * np.exp(0.5 * (sigma / tau) ** 2 - u[decaying] / tau) * erfc(z[decaying])
    return out.reshape(t.shape)


def multi_exp_irf(t, t0, fwhm, *params):
    """Sum of IRF-convolved exponentials.

    ``params`` is ``a1, tau1, a2, tau2, ...``; an odd number of parameters adds
    the last one as a constant offset (present at all times, e.g. a baseline).
    """
    t = np.asarray(t, dtype=float)
    n = len(params) // 2
    y = np.full_like(t, params[-1] if len(params) % 2 else 0.0)
    for a, tau in zip(params[0 : 2 * n : 2], params[1 : 2 * n : 2]):
        y = y + a * exp_irf(t, tau, t0, fwhm)
    return y
