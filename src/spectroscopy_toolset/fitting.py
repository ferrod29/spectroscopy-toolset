"""Least-squares fitting with parameter uncertainties.

* :func:`fit` - generic non-linear fit of any ``f(x, *params)`` model.
* :func:`fit_kinetics` - multi-exponential decay (optionally convolved with a
  Gaussian instrument response) of a single kinetic trace.
* :func:`global_fit` - global analysis of a transient-absorption matrix with
  shared lifetimes, yielding decay-associated spectra (DAS);
  :func:`das_to_eads` converts them to evolution-associated spectra (EADS) of a
  sequential scheme.

Kinetic fits use variable projection: amplitudes enter the model linearly and
are solved exactly for every trial set of lifetimes, so only ``t0``, the IRF
width and the lifetimes are optimised non-linearly. This makes the fits far
less sensitive to starting values than optimising all parameters at once.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import models

if TYPE_CHECKING:
    from .transient import TAData


# --------------------------------------------------------------------------
# Generic fitting
# --------------------------------------------------------------------------
@dataclass
class FitResult:
    """Outcome of a least-squares fit.

    Parameter values are available as ``result.params`` (dict),
    ``result["name"]`` or ``result.values``; one-sigma standard errors (from
    the covariance matrix scaled by the reduced chi-square) as
    ``result.errors`` / ``result.stderr``.
    """

    model: str
    names: list[str]
    values: np.ndarray
    stderr: np.ndarray
    covariance: np.ndarray
    x: np.ndarray
    y: np.ndarray
    best_fit: np.ndarray
    func: Callable = field(repr=False)
    sigma: np.ndarray | None = field(default=None, repr=False)
    fixed: tuple[str, ...] = ()
    at_bound: tuple[str, ...] = ()
    success: bool = True
    message: str = ""
    components: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def params(self) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.values)}

    @property
    def errors(self) -> dict[str, float]:
        return {n: float(e) for n, e in zip(self.names, self.stderr)}

    def __getitem__(self, name: str) -> float:
        return self.params[name]

    @property
    def residuals(self) -> np.ndarray:
        return self.y - self.best_fit

    @property
    def n_free(self) -> int:
        return len(self.names) - len(self.fixed)

    @property
    def dof(self) -> int:
        return max(self.y.size - self.n_free, 0)

    @property
    def chi2(self) -> float:
        r = self.residuals if self.sigma is None else self.residuals / self.sigma
        return float(r @ r)

    @property
    def reduced_chi2(self) -> float:
        return self.chi2 / self.dof if self.dof else float("nan")

    @property
    def r_squared(self) -> float:
        ss_tot = float(np.sum((self.y - self.y.mean()) ** 2))
        return 1.0 - float(np.sum(self.residuals**2)) / ss_tot if ss_tot > 0 else float("nan")

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.residuals**2)))

    def eval(self, x) -> np.ndarray:
        """Evaluate the fitted model at ``x``."""
        return self.func(np.asarray(x, dtype=float), *self.values)

    def to_frame(self) -> pd.DataFrame:
        """Parameters as a DataFrame with ``value``, ``stderr`` and ``fixed`` columns."""
        return pd.DataFrame(
            {
                "value": self.values,
                "stderr": self.stderr,
                "fixed": [n in self.fixed for n in self.names],
            },
            index=pd.Index(self.names, name="parameter"),
        )

    def summary(self) -> str:
        lines = [
            f"Model: {self.model}",
            f"  points: {self.y.size}   free parameters: {self.n_free}   "
            f"R^2: {self.r_squared:.5f}   reduced chi^2: {self.reduced_chi2:.4g}",
        ]
        for name, value, err in zip(self.names, self.values, self.stderr):
            if name in self.fixed:
                note, err_txt = " (fixed)", ""
            else:
                note = " (at bound)" if name in self.at_bound else ""
                err_txt = f" +/- {err:.3g}" if np.isfinite(err) else ""
            lines.append(f"  {name:>10s} = {value:.6g}{err_txt}{note}")
        if not self.success:
            lines.append(f"  WARNING: {self.message}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _bounds_arrays(bounds, names: Sequence[str]):
    n = len(names)
    lower = np.full(n, -np.inf)
    upper = np.full(n, np.inf)
    if bounds is None:
        return lower, upper
    if isinstance(bounds, dict):
        for key, (lo, hi) in bounds.items():
            i = names.index(key)
            lower[i] = -np.inf if lo is None else lo
            upper[i] = np.inf if hi is None else hi
    else:
        lo, hi = bounds
        lower = np.broadcast_to(np.asarray(lo, dtype=float), (n,)).copy()
        upper = np.broadcast_to(np.asarray(hi, dtype=float), (n,)).copy()
    return lower, upper


def _interior_start(start, lo, hi):
    """Clip start values into the bounds and move values on a bound slightly inside."""
    start = np.clip(start, lo, hi)
    span = np.where(np.isfinite(hi - lo), hi - lo, np.abs(start) + 1.0)
    start = np.where(start <= lo, lo + 1e-6 * span, start)
    return np.where(start >= hi, hi - 1e-6 * span, start)


def _covariance(jac, resid, dof):
    """Covariance ``s^2 (J^T J)^-1`` via SVD (pseudo-inverse for rank-deficient J)."""
    if dof <= 0:
        return None
    _, s, vt = np.linalg.svd(np.asarray(jac), full_matrices=False)
    if s.size == 0 or s[0] == 0:
        return None
    keep = s > np.finfo(float).eps * max(jac.shape) * s[0]
    s, vt = s[keep], vt[keep]
    return (vt.T / s**2) @ vt * (float(resid @ resid) / dof)


def fit(
    func: Callable,
    x,
    y,
    p0,
    names: Sequence[str] | None = None,
    bounds=None,
    sigma=None,
    fixed: Iterable[str] | dict[str, float] | None = None,
    loss: str = "linear",
    f_scale: float = 1.0,
    model: str | None = None,
    max_nfev: int | None = None,
) -> FitResult:
    """Non-linear least-squares fit of ``y = func(x, *params)``.

    Parameters
    ----------
    func : callable
        Model ``func(x, *params)``.
    x, y : array_like
        Data; non-finite points are ignored.
    p0 : sequence of float
        Starting values.
    names : sequence of str, optional
        Parameter names (default ``p0, p1, ...``).
    bounds : dict or (lower, upper), optional
        ``{name: (lo, hi)}`` (``None`` = unbounded) or two sequences/scalars.
    sigma : array_like, optional
        One-sigma uncertainties of ``y``; residuals are weighted by ``1/sigma``.
    fixed : iterable of str or dict, optional
        Parameters held constant (at their ``p0`` value, or at the dict value).
    loss : str
        ``'linear'`` (ordinary least squares) or a robust loss accepted by
        :func:`scipy.optimize.least_squares` (``'soft_l1'``, ``'huber'``,
        ``'cauchy'``, ``'arctan'``) to down-weight outliers.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p0 = np.atleast_1d(np.asarray(p0, dtype=float)).copy()
    names = list(names) if names is not None else [f"p{i}" for i in range(p0.size)]
    if len(names) != p0.size:
        raise ValueError(f"{len(names)} names given for {p0.size} parameters")
    lower, upper = _bounds_arrays(bounds, names)

    if isinstance(fixed, dict):
        for key, value in fixed.items():
            p0[names.index(key)] = value
        fixed_names = tuple(fixed)
    else:
        fixed_names = tuple(fixed or ())
    unknown = set(fixed_names) - set(names)
    if unknown:
        raise ValueError(f"unknown fixed parameters: {sorted(unknown)}")
    free = np.array([n not in fixed_names for n in names])
    if not free.any():
        raise ValueError("all parameters are fixed")

    mask = np.isfinite(x) & np.isfinite(y)
    weights = None
    if sigma is not None:
        sigma = np.broadcast_to(np.asarray(sigma, dtype=float), y.shape)
        mask &= np.isfinite(sigma) & (sigma > 0)
        sigma = sigma[mask]
        weights = 1.0 / sigma
    x, y = x[mask], y[mask]
    if y.size < free.sum():
        raise ValueError(f"{y.size} data points are not enough for {free.sum()} free parameters")

    def expand(p_free):
        p = p0.copy()
        p[free] = p_free
        return p

    def residual(p_free):
        r = func(x, *expand(p_free)) - y
        return r * weights if weights is not None else r

    lo, hi = lower[free], upper[free]
    res = least_squares(
        residual,
        _interior_start(p0[free], lo, hi),
        bounds=(lo, hi),
        method="trf",
        loss=loss,
        f_scale=f_scale,
        x_scale="jac",
        max_nfev=max_nfev,
    )

    values = expand(res.x)
    n = len(names)
    cov = np.full((n, n), np.nan)
    stderr = np.where(free, np.nan, 0.0)
    cov_free = _covariance(res.jac, res.fun, y.size - int(free.sum()))
    if cov_free is not None:
        idx = np.flatnonzero(free)
        cov[np.ix_(idx, idx)] = cov_free
        stderr[idx] = np.sqrt(np.clip(np.diag(cov_free), 0.0, None))
    at_bound = tuple(names[i] for i, a in zip(np.flatnonzero(free), res.active_mask) if a != 0)

    return FitResult(
        model=model or getattr(func, "__name__", "model"),
        names=names,
        values=values,
        stderr=stderr,
        covariance=cov,
        x=x,
        y=y,
        best_fit=func(x, *values),
        func=func,
        sigma=sigma,
        fixed=fixed_names,
        at_bound=at_bound,
        success=bool(res.success),
        message=str(res.message),
    )


# --------------------------------------------------------------------------
# Kinetics: shared helpers
# --------------------------------------------------------------------------
def _artifact_basis(t, t0, fwhm, order):
    """Coherent-artefact shapes: the Gaussian IRF and its first two derivatives (Hermite form)."""
    u = (np.asarray(t, dtype=float) - t0) / (fwhm / models.FWHM_PER_SIGMA)
    g = np.exp(-0.5 * u * u)
    return [g, u * g, (u * u - 1.0) * g][:order]


def _basis(t, taus, t0, fwhm, step, offset, artifact=0):
    """Design matrix of unit-amplitude components: exponentials, step, artefact, offset."""
    cols = [models.exp_irf(t, tau, t0, fwhm) for tau in taus]
    if step:
        cols.append(models.exp_irf(t, np.inf, t0, fwhm))
    if artifact:
        cols.extend(_artifact_basis(t, t0, fwhm, artifact))
    if offset:
        cols.append(np.ones_like(t))
    return np.column_stack(cols)


def _check_artifact(artifact, irf):
    if artifact not in (0, 1, 2, 3):
        raise ValueError(
            "artifact must be 0 (none) or the number of Gaussian-derivative terms, 1-3"
        )
    if artifact and not irf:
        raise ValueError("the coherent-artefact model needs irf=True")


def _guess_t0(t, y):
    """Time at which the smoothed |signal| first reaches half its maximum."""
    ys = np.convolve(np.abs(y), np.ones(3) / 3.0, mode="same")
    return float(t[np.argmax(ys >= 0.5 * ys.max())])


def _min_step(t):
    dt = np.diff(t)
    dt = dt[dt > 0]
    return float(dt.min()) if dt.size else 1.0


def _default_taus(t, t0, fwhm, n):
    lo = max(2.0 * fwhm, 3.0 * _min_step(t))
    hi = max((t[-1] - t0) / 3.0, 2.0 * lo)
    if n == 1:
        return np.array([np.sqrt(lo * hi)])
    return np.geomspace(lo, hi, n)


class _NonlinearParams:
    """Maps the free non-linear kinetic parameters (t0, log fwhm, log taus) to a vector."""

    def __init__(self, t, t0, fwhm, taus, fit_t0, fit_fwhm):
        self.t0, self.fwhm, self.taus = float(t0), float(fwhm), np.asarray(taus, dtype=float)
        self.fit_t0, self.fit_fwhm = fit_t0, fit_fwhm
        step, span = _min_step(t), float(t[-1] - t[0])
        lo, hi, start = [], [], []
        if fit_t0:
            lo.append(t[0])
            hi.append(t[-1])
            start.append(self.t0)
        if fit_fwhm:
            lo.append(np.log(step / 10.0))
            hi.append(np.log(max(span, step)))
            start.append(np.log(self.fwhm))
        lo += [np.log(step / 10.0)] * self.taus.size
        hi += [np.log(100.0 * max(span, step))] * self.taus.size
        start += list(np.log(self.taus))
        self.lower, self.upper = np.array(lo), np.array(hi)
        self.start = _interior_start(np.array(start), self.lower, self.upper)

    def unpack(self, theta):
        i = 0
        t0, fwhm = self.t0, self.fwhm
        if self.fit_t0:
            t0 = theta[i]
            i += 1
        if self.fit_fwhm:
            fwhm = np.exp(theta[i])
            i += 1
        return t0, fwhm, np.exp(theta[i:])


# --------------------------------------------------------------------------
# Single-trace kinetics
# --------------------------------------------------------------------------
def _kinetic_model(n_exp: int, step: bool, offset: bool, artifact: int = 0):
    def model(t, t0, fwhm, *p):
        t = np.asarray(t, dtype=float)
        y = np.zeros_like(t)
        for i in range(n_exp):
            y = y + p[2 * i] * models.exp_irf(t, p[2 * i + 1], t0, fwhm)
        k = 2 * n_exp
        if step:
            y = y + p[k] * models.exp_irf(t, np.inf, t0, fwhm)
            k += 1
        if artifact:
            for j, shape in enumerate(_artifact_basis(t, t0, fwhm, artifact)):
                y = y + p[k + j] * shape
            k += artifact
        if offset:
            y = y + p[k]
        return y

    return model


def fit_kinetics(
    t,
    y,
    n_exp: int = 1,
    irf: bool = True,
    t0: float | None = None,
    fwhm: float | None = None,
    taus: Sequence[float] | None = None,
    step: bool = False,
    offset: bool = False,
    artifact: int = 0,
    fix_t0: bool = False,
    fix_fwhm: bool = False,
    sigma=None,
    loss: str = "linear",
) -> FitResult:
    """Fit a kinetic trace with a sum of exponential decays.

    With ``irf=True`` the model is ``sum_i a_i exp(-(t-t0)/tau_i)`` convolved
    with a Gaussian instrument response of width ``fwhm`` (both fitted unless
    fixed), which describes the rise of the signal around time zero. With
    ``irf=False`` the unconvolved model is fitted to ``t >= t0`` only.

    ``artifact = 1..3`` adds the coherent artefact around time zero as the
    Gaussian IRF and its first and second derivatives (amplitudes ``ca0..ca2``),
    so no exponential is wasted on it.

    Parameters
    ----------
    t, y : array_like
        Delays and signal.
    n_exp : int
        Number of exponential components.
    t0, fwhm, taus : optional
        Starting values (``t0`` defaults to the half-rise point, ``fwhm`` to
        four times the smallest delay step, ``taus`` to a geometric series
        spanning the time window).
    step : bool
        Add a component with infinite lifetime (signal that does not decay
        within the window).
    offset : bool
        Add a constant offset present at all delays (e.g. residual background).
    fix_t0, fix_fwhm : bool
        Keep ``t0`` / ``fwhm`` at their starting values.
    sigma : array_like, optional
        Uncertainties of ``y`` used as weights.

    Returns
    -------
    FitResult
        Parameters ``t0, fwhm, a1, tau1, ..., [a_inf], [offset]`` with the
        components sorted by increasing lifetime; individual contributions are
        in ``result.components``.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    order = np.argsort(t)
    t, y = t[order], y[order]
    if sigma is not None:
        sigma = np.broadcast_to(np.asarray(sigma, dtype=float), ok.shape)[ok][order]
    n_exp = int(n_exp)
    if n_exp < 1:
        raise ValueError("n_exp must be >= 1")
    _check_artifact(artifact, irf)

    if irf:
        t0 = _guess_t0(t, y) if t0 is None else float(t0)
        fwhm = 4.0 * _min_step(t) if fwhm is None else float(fwhm)
    else:
        t0 = float(t[np.argmax(np.abs(y))]) if t0 is None else float(t0)
        keep = t >= t0
        t, y = t[keep], y[keep]
        if sigma is not None:
            sigma = sigma[keep]
        fwhm = 0.0
        fix_t0 = fix_fwhm = True
    taus = (
        _default_taus(t, t0, max(fwhm, _min_step(t)), n_exp)
        if taus is None
        else np.asarray(taus, float)
    )
    if taus.size != n_exp:
        raise ValueError(f"{taus.size} starting lifetimes given for n_exp={n_exp}")

    weights = None if sigma is None else 1.0 / sigma
    yw = y if weights is None else y * weights
    nl = _NonlinearParams(t, t0, max(fwhm, 1e-12), taus, not fix_t0, irf and not fix_fwhm)

    def design(theta):
        t0_, fwhm_, taus_ = nl.unpack(theta)
        c = _basis(t, taus_, t0_, fwhm_ if irf else 0.0, step, offset, artifact)
        return c if weights is None else c * weights[:, None]

    def projected_residual(theta):
        c = design(theta)
        coef = np.linalg.lstsq(c, yw, rcond=None)[0]
        return c @ coef - yw

    # Stage 1: variable projection over the non-linear parameters only.
    if nl.start.size:
        theta = least_squares(
            projected_residual, nl.start, bounds=(nl.lower, nl.upper), method="trf"
        ).x
    else:
        theta = nl.start
    t0_, fwhm_, taus_ = nl.unpack(theta)
    fwhm_ = fwhm_ if irf else 0.0
    order = np.argsort(taus_)
    taus_ = taus_[order]
    amps = np.linalg.lstsq(design(_repack(nl, t0_, fwhm_, taus_)), yw, rcond=None)[0]

    # Stage 2: refine all parameters together to obtain the full covariance.
    names = ["t0", "fwhm"]
    p0 = [t0_, fwhm_]
    step_min, span = _min_step(t), float(t[-1] - t[0])
    bounds = {"t0": (t[0], t[-1]), "fwhm": (step_min / 10.0, max(span, step_min))}
    for i in range(n_exp):
        names += [f"a{i + 1}", f"tau{i + 1}"]
        p0 += [amps[i], taus_[i]]
        bounds[f"tau{i + 1}"] = (step_min / 10.0, 100.0 * max(span, step_min))
    k = n_exp
    if step:
        names.append("a_inf")
        p0.append(amps[k])
        k += 1
    for j in range(artifact):
        names.append(f"ca{j}")
        p0.append(amps[k])
        k += 1
    if offset:
        names.append("offset")
        p0.append(amps[k])
    fixed = [n for n, f in (("t0", fix_t0), ("fwhm", fix_fwhm)) if f]
    if not irf:
        bounds.pop("fwhm")
    model = _kinetic_model(n_exp, step, offset, artifact)
    label = f"{n_exp}-exponential" + (" * Gaussian IRF" if irf else f" (t >= {t0:.4g})")
    if artifact:
        label += " + coherent artefact"
    result = fit(
        model,
        t,
        y,
        p0,
        names=names,
        bounds=bounds,
        sigma=sigma,
        fixed=fixed,
        loss=loss,
        model=label,
    )
    return _sort_kinetic_result(result, n_exp, step, offset, artifact)


def _repack(nl: _NonlinearParams, t0, fwhm, taus):
    theta = []
    if nl.fit_t0:
        theta.append(t0)
    if nl.fit_fwhm:
        theta.append(np.log(fwhm))
    return np.array(theta + list(np.log(taus)))


def _sort_kinetic_result(
    result: FitResult, n_exp: int, step: bool, offset: bool, artifact: int = 0
) -> FitResult:
    """Order the exponential components by lifetime and attach component curves."""
    p = result.params
    order = np.argsort([p[f"tau{i + 1}"] for i in range(n_exp)])
    perm = [0, 1]
    for i in order:
        perm += [2 + 2 * i, 3 + 2 * i]
    perm += list(range(2 + 2 * n_exp, len(result.names)))
    perm = np.array(perm)
    result.values = result.values[perm]
    result.stderr = result.stderr[perm]
    result.covariance = result.covariance[np.ix_(perm, perm)]
    result.at_bound = tuple(_renamed(result.names, perm, name) for name in result.at_bound)

    p = result.params
    t0, fwhm = p["t0"], p["fwhm"]
    for i in range(n_exp):
        a, tau = p[f"a{i + 1}"], p[f"tau{i + 1}"]
        result.components[f"tau{i + 1} = {tau:.3g}"] = a * models.exp_irf(result.x, tau, t0, fwhm)
    if step:
        result.components["long-lived"] = p["a_inf"] * models.exp_irf(result.x, np.inf, t0, fwhm)
    if artifact:
        shapes = _artifact_basis(result.x, t0, fwhm, artifact)
        result.components["coherent artefact"] = sum(p[f"ca{j}"] * s for j, s in enumerate(shapes))
    if offset:
        result.components["offset"] = np.full_like(result.x, p["offset"])
    return result


def _renamed(names, perm, name):
    """Name that parameter ``name`` carries after permuting values by ``perm``."""
    return names[int(np.flatnonzero(perm == names.index(name))[0])]


# --------------------------------------------------------------------------
# Global analysis
# --------------------------------------------------------------------------
@dataclass
class GlobalFitResult:
    """Outcome of :func:`global_fit`.

    ``das[i]`` is the decay-associated spectrum of lifetime ``taus[i]``
    (plus a final long-lived component when ``step`` is True).
    """

    taus: np.ndarray
    taus_stderr: np.ndarray
    t0: float
    t0_stderr: float
    fwhm: float
    fwhm_stderr: float
    step: bool
    das: np.ndarray
    wavelengths: np.ndarray
    delays: np.ndarray
    data: np.ndarray
    fit: np.ndarray
    artifact_spectra: np.ndarray | None = None
    at_bound: tuple[str, ...] = ()
    success: bool = True
    message: str = ""

    @property
    def residuals(self) -> np.ndarray:
        return self.data - self.fit

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.residuals**2)))

    @property
    def lifetimes(self) -> np.ndarray:
        """Lifetimes of all components (``inf`` for the long-lived one)."""
        return np.append(self.taus, np.inf) if self.step else self.taus.copy()

    @property
    def labels(self) -> list[str]:
        labels = [f"tau{i + 1} = {tau:.3g} ps" for i, tau in enumerate(self.taus)]
        return labels + (["long-lived"] if self.step else [])

    def eads(self) -> np.ndarray:
        """Evolution-associated spectra of the sequential scheme A -> B -> ... (see :func:`das_to_eads`)."""
        return das_to_eads(self.das, self.lifetimes)

    def concentrations(self, sequential: bool = False) -> np.ndarray:
        """Time profiles (n_components, n_delays) of the components (or sequential species)."""
        c = _basis(self.delays, self.taus, self.t0, self.fwhm, self.step, False).T
        return sequential_matrix(self.lifetimes) @ c if sequential else c

    def summary(self) -> str:
        lines = [
            f"Global fit: {len(self.taus)} exponential(s)"
            + (" + long-lived component" if self.step else "")
            + (" + coherent artefact" if self.artifact_spectra is not None else ""),
            f"  wavelengths: {self.wavelengths.size}   delays: {self.delays.size}   RMSE: {self.rmse:.4g}",
            f"  t0   = {self.t0:.5g} +/- {self.t0_stderr:.2g}",
            f"  fwhm = {self.fwhm:.5g} +/- {self.fwhm_stderr:.2g}",
        ]
        for i, (tau, err) in enumerate(zip(self.taus, self.taus_stderr)):
            lines.append(f"  tau{i + 1} = {tau:.5g} +/- {err:.2g}")
        if self.at_bound:
            lines.append(
                f"  WARNING: {', '.join(self.at_bound)} ended at a bound - the model may have "
                "too many components or the time window misses that process"
            )
        if not self.success:
            lines.append(f"  WARNING: {self.message}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def global_fit(
    data: TAData,
    n_exp: int = 2,
    irf: bool = True,
    t0: float | None = None,
    fwhm: float | None = None,
    taus: Sequence[float] | None = None,
    step: bool = False,
    artifact: int = 0,
    fix_t0: bool = False,
    fix_fwhm: bool = False,
    max_nfev: int | None = None,
) -> GlobalFitResult:
    """Global analysis of a TA data set with lifetimes shared by all wavelengths.

    Fits ``dA(lambda, t) = sum_i DAS_i(lambda) * [exp(-(t - t0)/tau_i) (x) IRF](t)``
    by variable projection: the decay-associated spectra are solved by linear
    least squares for every trial set of ``(t0, fwhm, taus)``.

    The data should be chirp-corrected (:meth:`TAData.correct_chirp`) so a
    single ``t0`` applies to every wavelength. Rows containing NaN are dropped.
    Binning wavelengths first (:meth:`TAData.bin_wavelengths`) speeds up the fit
    without changing the lifetimes appreciably.

    ``artifact = 1..3`` models the coherent artefact with the Gaussian IRF and
    its derivatives; their spectra are returned in ``artifact_spectra``.
    """
    _check_artifact(artifact, irf)
    d = np.asarray(data.dA, dtype=float)
    t = np.asarray(data.delays, dtype=float)
    wl = np.asarray(data.wavelengths, dtype=float)
    good = np.all(np.isfinite(d), axis=1)
    if not good.all():
        warnings.warn(f"dropping {np.sum(~good)} wavelength rows containing NaN", stacklevel=2)
        d, wl = d[good], wl[good]
    if d.shape[0] == 0:
        raise ValueError("no finite data to fit")

    # The time profile of the dominant singular vector gives robust starting values.
    u, s, vt = np.linalg.svd(d, full_matrices=False)
    trace = s[0] * vt[0]
    if irf:
        t0 = _guess_t0(t, trace) if t0 is None else float(t0)
        fwhm = 4.0 * _min_step(t) if fwhm is None else float(fwhm)
    else:
        t0 = float(t[np.argmax(np.abs(trace))]) if t0 is None else float(t0)
        keep = t >= t0
        t, d, vt = t[keep], d[:, keep], vt[:, keep]
        u, s, vt = np.linalg.svd(d, full_matrices=False)
        fwhm, fix_t0, fix_fwhm = 0.0, True, True
    taus = (
        _default_taus(t, t0, max(fwhm, _min_step(t)), n_exp)
        if taus is None
        else np.asarray(taus, float)
    )
    if taus.size != n_exp:
        raise ValueError(f"{taus.size} starting lifetimes given for n_exp={n_exp}")

    # d.T = V S U^T with orthonormal U: fitting V S instead of d.T gives identical
    # residual norms and Jacobian products but is much smaller when n_wl > n_t.
    target = vt.T * s
    nl = _NonlinearParams(t, t0, max(fwhm, 1e-12), taus, not fix_t0, irf and not fix_fwhm)

    def design(theta):
        t0_, fwhm_, taus_ = nl.unpack(theta)
        return _basis(t, taus_, t0_, fwhm_ if irf else 0.0, step, False, artifact)

    def projected_residual(theta):
        c = design(theta)
        coef = np.linalg.lstsq(c, target, rcond=None)[0]
        return (c @ coef - target).ravel()

    res = least_squares(
        projected_residual, nl.start, bounds=(nl.lower, nl.upper), method="trf", max_nfev=max_nfev
    )
    t0_, fwhm_, taus_ = nl.unpack(res.x)
    fwhm_ = fwhm_ if irf else 0.0

    n_components = n_exp + int(step)
    n_linear = (n_components + artifact) * d.shape[0]
    cov = _covariance(res.jac, res.fun, d.size - n_linear - res.x.size)
    err = (
        np.sqrt(np.clip(np.diag(cov), 0.0, None))
        if cov is not None
        else np.full(res.x.size, np.nan)
    )
    i = 0
    t0_err = fwhm_err = 0.0
    if nl.fit_t0:
        t0_err = float(err[i])
        i += 1
    if nl.fit_fwhm:
        fwhm_err = float(fwhm_ * err[i])  # d(fwhm) = fwhm * d(log fwhm)
        i += 1
    taus_err = taus_ * err[i:]

    order = np.argsort(taus_)
    taus_, taus_err = taus_[order], taus_err[order]
    theta_names = ["t0"] * nl.fit_t0 + ["fwhm"] * nl.fit_fwhm
    theta_names += [f"tau{int(np.flatnonzero(order == j)[0]) + 1}" for j in range(n_exp)]
    at_bound = tuple(name for name, active in zip(theta_names, res.active_mask) if active != 0)
    c = _basis(t, taus_, t0_, fwhm_, step, False, artifact)
    coef = np.linalg.lstsq(c, d.T, rcond=None)[0]
    return GlobalFitResult(
        taus=taus_,
        taus_stderr=taus_err,
        t0=float(t0_),
        t0_stderr=t0_err,
        fwhm=float(fwhm_),
        fwhm_stderr=fwhm_err,
        step=step,
        das=coef[:n_components],
        wavelengths=wl,
        delays=t,
        data=d,
        fit=(c @ coef).T,
        artifact_spectra=coef[n_components:] if artifact else None,
        at_bound=at_bound,
        success=bool(res.success),
        message=str(res.message),
    )


def sequential_matrix(lifetimes: Sequence[float]) -> np.ndarray:
    """Coefficients of the sequential scheme 1 -> 2 -> ... -> n.

    Returns ``B`` with ``c_j(t) = sum_i B[j, i] exp(-t / tau_i)`` for unit initial
    population of species 1. ``lifetimes`` must be distinct and sorted in the
    order the species appear (an infinite value makes the last species
    long-lived).
    """
    k = 1.0 / np.asarray(lifetimes, dtype=float)
    n = k.size
    if np.unique(k).size != n:
        raise ValueError("the lifetimes of a sequential scheme must be distinct")
    b = np.zeros((n, n))
    for j in range(n):
        numerator = np.prod(k[:j])
        for i in range(j + 1):
            b[j, i] = numerator / np.prod([k[m] - k[i] for m in range(j + 1) if m != i])
    return b


def das_to_eads(das, lifetimes: Sequence[float]) -> np.ndarray:
    """Convert decay-associated spectra to evolution-associated spectra.

    For a sequential scheme with the given (ascending) lifetimes,
    ``DAS = B^T @ EADS`` with ``B`` from :func:`sequential_matrix`.
    """
    return np.linalg.solve(sequential_matrix(lifetimes).T, np.asarray(das, dtype=float))
