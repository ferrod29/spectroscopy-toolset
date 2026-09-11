"""Transient-absorption (pump-probe) data: container, pre-processing, chirp
correction and oscillation analysis.

Conventions: wavelengths in nm, delays in ps and signals ``dA`` in optical
density (OD), stored as an ``(n_wavelengths, n_delays)`` array.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lombscargle

from . import fitting
from .processing import C_CM_PER_PS, bin_mean, crop_mask, robust_std


@dataclass
class TAData:
    """A transient-absorption data matrix ``dA[wavelength, delay]``.

    Axes are sorted in ascending order on construction. ``std`` optionally
    holds the standard error of ``dA`` (e.g. from averaging several scans).
    Every processing method returns a new object, so steps can be chained::

        ta = (TAData.from_file("scan.dat")
              .exclude_wavelengths((510, 522))       # pump scatter
              .subtract_background(before=-1.0)
              .bin_wavelengths(width=2.0))
    """

    wavelengths: np.ndarray
    delays: np.ndarray
    dA: np.ndarray
    std: np.ndarray | None = None
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        wl = np.asarray(self.wavelengths, dtype=float).ravel()
        t = np.asarray(self.delays, dtype=float).ravel()
        d = np.asarray(self.dA, dtype=float)
        if d.shape != (wl.size, t.size):
            raise ValueError(
                f"dA has shape {d.shape}, expected (n_wavelengths, n_delays) = ({wl.size}, {t.size})"
            )
        iw, it = np.argsort(wl, kind="stable"), np.argsort(t, kind="stable")
        self.wavelengths, self.delays = wl[iw], t[it]
        self.dA = d[np.ix_(iw, it)]
        if self.std is not None:
            s = np.asarray(self.std, dtype=float)
            if s.shape != d.shape:
                raise ValueError("std must have the same shape as dA")
            self.std = s[np.ix_(iw, it)]
        self.meta = dict(self.meta)

    # ------------------------------------------------------------ basics
    @classmethod
    def from_file(cls, paths, **kwargs) -> TAData:
        """Read (and average) scans with :func:`spectroscopy_toolset.io.read_ta`."""
        from .io import read_ta

        return read_ta(paths, **kwargs)

    @property
    def shape(self) -> tuple[int, int]:
        return self.dA.shape

    def copy(self, **changes) -> TAData:
        return replace(self, **changes)

    def _take(self, rows=None, cols=None, **changes) -> TAData:
        rows = slice(None) if rows is None else rows
        cols = slice(None) if cols is None else cols
        return self.copy(
            wavelengths=self.wavelengths[rows],
            delays=self.delays[cols],
            dA=self.dA[rows][:, cols],
            std=None if self.std is None else self.std[rows][:, cols],
            **changes,
        )

    def crop(self, wl: tuple | None = None, t: tuple | None = None) -> TAData:
        """Keep wavelengths within ``wl = (min, max)`` and delays within ``t`` (inclusive, ``None`` = open)."""
        rows = crop_mask(self.wavelengths, *(wl or (None, None)))
        cols = crop_mask(self.delays, *(t or (None, None)))
        if not rows.any() or not cols.any():
            raise ValueError("the crop range contains no data")
        return self._take(rows, cols)

    def exclude_wavelengths(self, *ranges: tuple[float, float]) -> TAData:
        """Remove wavelength bands, e.g. pump scatter: ``exclude_wavelengths((510, 522))``."""
        drop = np.zeros(self.wavelengths.shape, dtype=bool)
        for lo, hi in ranges:
            drop |= crop_mask(self.wavelengths, lo, hi)
        return self._take(~drop)

    def shift_time(self, t0: float) -> TAData:
        """Redefine time zero: delays become ``delays - t0``."""
        return self.copy(delays=self.delays - t0)

    # ------------------------------------------------ background & noise
    def _pre(self, before: float) -> np.ndarray:
        mask = self.delays < before
        if mask.sum() < 2:
            raise ValueError(f"fewer than 2 delays before t = {before}")
        return mask

    def background(self, before: float) -> np.ndarray:
        """Per-wavelength mean signal at delays ``< before`` (pump-induced background/scatter)."""
        return self.dA[:, self._pre(before)].mean(axis=1)

    def subtract_background(self, before: float) -> TAData:
        """Subtract the pre-time-zero signal of every wavelength."""
        return self.copy(dA=self.dA - self.background(before)[:, None])

    def noise(self, before: float | None = None) -> np.ndarray:
        """Per-wavelength noise level (OD).

        The standard deviation at delays ``< before`` if given, otherwise a
        robust estimate from point-to-point fluctuations along each trace.
        """
        if before is not None:
            return self.dA[:, self._pre(before)].std(axis=1, ddof=1)
        return np.array([robust_std(row) for row in self.dA])

    # --------------------------------------------------------- binning
    def bin_wavelengths(self, n: int | None = None, width: float | None = None) -> TAData:
        """Average groups of ``n`` adjacent wavelengths (or groups spanning ``width`` nm)."""
        if (n is None) == (width is None):
            raise ValueError("give exactly one of n or width")
        if width is not None:
            n = max(1, int(round(width / float(np.median(np.diff(self.wavelengths))))))
        if n == 1:
            return self.copy()
        std = None
        if self.std is not None:
            std = np.sqrt(bin_mean(self.std**2, n, axis=0) / n)
        return self.copy(
            wavelengths=bin_mean(self.wavelengths, n),
            dA=bin_mean(self.dA, n, axis=0),
            std=std,
        )

    # -------------------------------------------------------- slicing
    def _rows(self, wavelength: float, width: float) -> np.ndarray:
        if width > 0:
            rows = np.flatnonzero(np.abs(self.wavelengths - wavelength) <= width / 2.0)
            if rows.size:
                return rows
        return np.array([np.argmin(np.abs(self.wavelengths - wavelength))])

    def _cols(self, delay: float, width: float) -> np.ndarray:
        if width > 0:
            cols = np.flatnonzero(np.abs(self.delays - delay) <= width / 2.0)
            if cols.size:
                return cols
        return np.array([np.argmin(np.abs(self.delays - delay))])

    def kinetic(self, wavelength: float, width: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Kinetic trace at ``wavelength`` (averaged over ``+/- width/2`` nm); returns ``(delays, dA)``."""
        return self.delays, self.dA[self._rows(wavelength, width)].mean(axis=0)

    def kinetic_error(self, wavelength: float, width: float = 0.0) -> np.ndarray | None:
        """Standard error of :meth:`kinetic` (``None`` without ``std``)."""
        if self.std is None:
            return None
        rows = self._rows(wavelength, width)
        return np.sqrt((self.std[rows] ** 2).sum(axis=0)) / rows.size

    def spectrum(self, delay: float, width: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Transient spectrum at ``delay`` (averaged over ``+/- width/2``); returns ``(wavelengths, dA)``."""
        return self.wavelengths, self.dA[:, self._cols(delay, width)].mean(axis=1)

    def svd(self, n: int | None = None):
        """Singular value decomposition ``dA = U diag(s) Vt`` (truncated to ``n`` components).

        The number of singular values clearly above the noise floor estimates
        how many kinetic components the data support.
        """
        if not np.all(np.isfinite(self.dA)):
            raise ValueError("dA contains NaN/inf; crop or correct the data first")
        u, s, vt = np.linalg.svd(self.dA, full_matrices=False)
        if n is not None:
            u, s, vt = u[:, :n], s[:n], vt[:n]
        return u, s, vt

    # ----------------------------------------------------------- chirp
    def estimate_chirp(self, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Time zero versus wavelength (see :func:`estimate_chirp`)."""
        return estimate_chirp(self, **kwargs)

    def correct_chirp(self, chirp: Callable | np.ndarray, crop: bool = True) -> TAData:
        """Remove the group-velocity dispersion (chirp) of the probe.

        Parameters
        ----------
        chirp : callable or array
            Time zero for every wavelength, e.g. a :class:`ChirpModel` from
            :func:`fit_chirp`, or an array of length ``n_wavelengths``.
        crop : bool
            Drop delays at which not every wavelength has data after the shift.

        Every kinetic trace is linearly interpolated onto ``t + t0(lambda)``; the
        corrected data have time zero at ``t = 0`` for all wavelengths.
        """
        t0 = np.asarray(chirp(self.wavelengths) if callable(chirp) else chirp, dtype=float)
        if t0.shape != self.wavelengths.shape:
            raise ValueError("chirp must give one time zero per wavelength")
        new_delays = self.delays - float(np.median(t0))
        dA = np.empty_like(self.dA)
        std = None if self.std is None else np.empty_like(self.std)
        for i, shift in enumerate(t0):
            dA[i] = np.interp(
                new_delays + shift, self.delays, self.dA[i], left=np.nan, right=np.nan
            )
            if std is not None:
                std[i] = np.interp(
                    new_delays + shift, self.delays, self.std[i], left=np.nan, right=np.nan
                )
        meta = {**self.meta, "chirp_t0": t0}
        if isinstance(chirp, ChirpModel):
            meta["chirp_model"] = chirp
        out = self.copy(delays=new_delays, dA=dA, std=std, meta=meta)
        if crop:
            valid = np.all(np.isfinite(dA), axis=0)
            out = out._take(cols=valid)
        return out

    # -------------------------------------------------------- analysis
    def fit_kinetics(self, wavelength: float, width: float = 0.0, **kwargs) -> fitting.FitResult:
        """Fit the kinetic trace at ``wavelength`` (see :func:`spectroscopy_toolset.fitting.fit_kinetics`)."""
        t, y = self.kinetic(wavelength, width)
        kwargs.setdefault("sigma", self.kinetic_error(wavelength, width))
        return fitting.fit_kinetics(t, y, **kwargs)

    def global_fit(self, n_exp: int = 2, **kwargs) -> fitting.GlobalFitResult:
        """Global analysis (see :func:`spectroscopy_toolset.fitting.global_fit`)."""
        return fitting.global_fit(self, n_exp, **kwargs)

    # ------------------------------------------------------------ I/O
    def save(self, path: str | Path, delimiter: str = "\t") -> None:
        """Write the matrix in the same layout the readers accept."""
        from .io import write_ta_matrix

        write_ta_matrix(path, self, delimiter=delimiter)

    def save_xyz(self, path: str | Path, scale: float = 1e3) -> None:
        """Write ``wavelength delay value`` triplets (gnuplot/Origin 3-D format)."""
        from .io import write_ta_xyz

        write_ta_xyz(path, self, scale=scale)


def average_scans(scans: Sequence[TAData]) -> TAData:
    """Average repeated scans measured on identical axes.

    The returned ``std`` is the standard error of the mean.
    """
    scans = list(scans)
    first = scans[0]
    for other in scans[1:]:
        if other.shape != first.shape or not np.allclose(other.wavelengths, first.wavelengths):
            raise ValueError(
                f"scan {other.name!r} has different wavelengths or shape than {first.name!r}"
            )
        if not np.allclose(other.delays, first.delays):
            offset = other.delays - first.delays
            detail = (
                f"its delays are offset by {offset[0]:+.4g} ps"
                if np.allclose(offset, offset[0])
                else "its delay grid differs"
            )
            raise ValueError(
                f"cannot average {other.name!r} with {first.name!r}: {detail}. If these are repeated "
                "scans with a re-zeroed delay stage, align them with TAData.shift_time() first."
            )
    if len(scans) == 1:
        return first.copy()
    stack = np.stack([s.dA for s in scans])
    sources = [src for s in scans for src in s.meta.get("source", [s.name])]
    return first.copy(
        dA=stack.mean(axis=0),
        std=stack.std(axis=0, ddof=1) / np.sqrt(len(scans)),
        name=f"{first.name} (mean of {len(scans)})",
        meta={**first.meta, "source": sources, "n_scans": len(scans)},
    )


# --------------------------------------------------------------------------
# Chirp (group-velocity dispersion) correction
# --------------------------------------------------------------------------
@dataclass
class ChirpModel:
    """Time zero as a polynomial in ``x = (wavelength - center) / 100``."""

    coefficients: np.ndarray
    center: float

    def __call__(self, wavelengths) -> np.ndarray:
        x = (np.asarray(wavelengths, dtype=float) - self.center) / 100.0
        return np.polynomial.polynomial.polyval(x, self.coefficients)

    def __str__(self) -> str:
        terms = " + ".join(f"{c:.4g} x^{i}" for i, c in enumerate(self.coefficients))
        return f"t0(lambda) = {terms},  x = (lambda - {self.center:.1f} nm) / 100 nm"


def _vertex(t, y, k):
    """Sub-sample position of the extremum at ``k`` from a parabola through 3 points."""
    if k == 0 or k == t.size - 1:
        return t[k]
    tt, yy = t[k - 1 : k + 2], y[k - 1 : k + 2]
    a, b, _ = np.polyfit(tt - tt[1], yy, 2)
    if a == 0:
        return t[k]
    return float(np.clip(tt[1] - b / (2.0 * a), tt[0], tt[2]))


def _onset(t, y, fraction):
    """First time ``y`` reaches ``fraction`` of its maximum, linearly interpolated.

    NaN if ``y`` is already above that level at the start of the window.
    """
    level = fraction * y.max()
    k = int(np.argmax(y >= level))
    if k == 0:
        return np.nan
    return t[k - 1] + (level - y[k - 1]) / (y[k] - y[k - 1]) * (t[k] - t[k - 1])


def estimate_chirp(
    data: TAData,
    window: tuple[float, float] | None = None,
    bin_width: float | None = 5.0,
    method: str = "onset",
    fraction: float = 0.5,
    threshold: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate time zero as a function of wavelength.

    Time zero is located, for every wavelength bin, from the first strong
    signal inside ``window`` - normally the coherent artefact (cross-phase
    modulation) and/or the rise of the pump-induced signal.

    Parameters
    ----------
    window : (float, float), optional
        Delay range searched (default: +/- 1 ps around the delay of the largest
        mean |dA|).
    bin_width : float or None
        Average wavelengths in bins of this width (nm) to reduce noise.
    method : {'onset', 'max', 'derivative'}
        ``'onset'`` (default): first delay at which ``|dA|`` reaches
        ``fraction`` of its maximum in the window - robust when the artefact
        has two lobes or overlaps a strong population signal.
        ``'max'``: delay of the largest ``|dA|`` (refined with a parabola).
        ``'derivative'``: delay of the steepest change of ``dA``.
    threshold : float
        Bins whose peak is below ``threshold`` times their noise level are
        discarded (returned as NaN).

    Returns
    -------
    wavelengths, t0 : ndarray
    """
    if method not in ("onset", "max", "derivative"):
        raise ValueError("method must be 'onset', 'max' or 'derivative'")
    d = data.bin_wavelengths(width=bin_width) if bin_width else data
    t = d.delays
    if window is None:
        envelope = np.nanmean(
            np.abs(d.dA - np.nanmedian(d.dA[:, :3], axis=1, keepdims=True)), axis=0
        )
        center = t[int(np.nanargmax(envelope))]
        window = (center - 1.0, center + 1.0)
    in_win = crop_mask(t, *window)
    if in_win.sum() < 3:
        raise ValueError("the chirp search window contains fewer than 3 delays")
    pre = t < window[0]
    tw = t[in_win]
    t0 = np.full(d.wavelengths.shape, np.nan)
    for i, row in enumerate(d.dA):
        base = np.median(row[pre]) if pre.sum() >= 3 else 0.0
        seg = np.abs(row[in_win] - base)
        k = int(np.argmax(seg))
        if seg[k] < threshold * robust_std(row):
            continue
        if method == "onset":
            t0[i] = _onset(tw, seg, fraction)
            continue
        if method == "derivative":
            k = int(np.argmax(np.abs(np.gradient(row[in_win], tw))))
        # an extremum on the edge of the window is not a measurement of time zero
        if 0 < k < tw.size - 1:
            t0[i] = _vertex(tw, seg, k) if method == "max" else tw[k]
    return d.wavelengths, t0


def fit_chirp(
    wavelengths,
    t0,
    order: int = 2,
    center: float | None = None,
    loss: str = "soft_l1",
) -> ChirpModel:
    """Robust polynomial fit of time zero versus wavelength.

    A robust loss (default ``'soft_l1'``) keeps outlying ``t0`` estimates from
    pulling the curve; NaN estimates are ignored.
    """
    wl = np.asarray(wavelengths, dtype=float)
    t0 = np.asarray(t0, dtype=float)
    ok = np.isfinite(wl) & np.isfinite(t0)
    wl, t0 = wl[ok], t0[ok]
    if wl.size <= order:
        raise ValueError(f"need more than {order} valid time-zero estimates")
    center = float(np.mean(wl)) if center is None else float(center)
    x = (wl - center) / 100.0
    c0 = np.polynomial.polynomial.polyfit(x, t0, order)
    r0 = np.polynomial.polynomial.polyval(x, c0) - t0
    scale = max(1.4826 * float(np.median(np.abs(r0 - np.median(r0)))), 1e-6)
    res = least_squares(
        lambda c: np.polynomial.polynomial.polyval(x, c) - t0, c0, loss=loss, f_scale=scale
    )
    return ChirpModel(coefficients=res.x, center=center)


# --------------------------------------------------------------------------
# Coherent oscillations
# --------------------------------------------------------------------------
def oscillation_spectrum(
    t,
    y,
    max_wavenumber: float = 1000.0,
    n: int = 1000,
    window: str | None = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude spectrum of oscillations in a (residual) kinetic trace.

    Uses the Lomb-Scargle periodogram, which handles the non-uniform delay
    grids of TA experiments. Subtract the population kinetics first (e.g. the
    residuals of :func:`~spectroscopy_toolset.fitting.fit_kinetics`) and
    restrict ``t`` to delays after the coherent artefact.

    Parameters
    ----------
    t : array_like
        Delays in ps.
    y : array_like
        Signal (e.g. fit residuals).
    max_wavenumber : float
        Upper limit of the spectrum in cm^-1.
    n : int
        Number of frequencies.
    window : {'hann', None}
        Taper applied in time to suppress spectral leakage.

    Returns
    -------
    wavenumbers, amplitude : ndarray
        Wavenumbers in cm^-1 and the estimated oscillation amplitude (same
        units as ``y``).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    y = y - y.mean()
    gain = 1.0
    if window == "hann":
        w = 0.5 * (1.0 - np.cos(2.0 * np.pi * (t - t[0]) / (t[-1] - t[0])))
        y = y * w
        gain = float(w.mean())
    elif window is not None:
        raise ValueError("window must be 'hann' or None")
    wavenumbers = np.linspace(max_wavenumber / n, max_wavenumber, n)
    omega = 2.0 * np.pi * C_CM_PER_PS * wavenumbers  # rad/ps
    power = lombscargle(t, y, omega)
    return wavenumbers, np.sqrt(4.0 * power / t.size) / gain
