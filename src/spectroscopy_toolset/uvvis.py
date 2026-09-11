"""Steady-state UV-vis absorption analysis.

* peak detection with widths (:func:`find_peaks`) and band deconvolution
  (:func:`fit_peaks`);
* absorbance/transmittance conversion and Beer-Lambert quantification,
  including calibration curves with LOD/LOQ (:func:`calibration_curve`);
* optical band gaps from Tauc plots (:func:`tauc_bandgap`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal

from . import models
from .fitting import FitResult, fit
from .processing import crop_mask, nm_to_ev

LN10 = np.log(10.0)


# --------------------------------------------------------------------------
# Peaks
# --------------------------------------------------------------------------
def _sorted_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    return x[order], y[order]


def find_peaks(
    x,
    y,
    prominence: float | None = None,
    height: float | None = None,
    distance: float | None = None,
    min_width: float | None = None,
    max_peaks: int | None = None,
    valleys: bool = False,
) -> pd.DataFrame:
    """Locate bands and measure their widths.

    Parameters
    ----------
    prominence : float, optional
        Minimum prominence (default: 5 % of the signal range).
    height : float, optional
        Minimum peak height.
    distance, min_width : float, optional
        Minimum peak separation and width, in ``x`` units.
    max_peaks : int, optional
        Keep only the most prominent peaks.
    valleys : bool
        Find minima instead of maxima (e.g. bleach bands).

    Returns
    -------
    DataFrame
        Columns ``position, height, prominence, fwhm, left, right, index``. The
        width is measured at half prominence (equal to the FWHM for an isolated
        band on a flat baseline).
    """
    x, y = _sorted_xy(x, y)
    ys = -y if valleys else y
    if prominence is None:
        prominence = 0.05 * float(np.nanmax(ys) - np.nanmin(ys))
    dx = float(np.median(np.diff(x)))
    idx, props = signal.find_peaks(
        ys,
        prominence=prominence,
        height=height,
        distance=max(1, int(round(distance / dx))) if distance else None,
        width=min_width / dx if min_width else None,
    )
    columns = ["position", "height", "prominence", "fwhm", "left", "right", "index"]
    if idx.size == 0:
        return pd.DataFrame(columns=columns)
    _, _, left_ips, right_ips = signal.peak_widths(
        ys,
        idx,
        rel_height=0.5,
        prominence_data=(props["prominences"], props["left_bases"], props["right_bases"]),
    )
    samples = np.arange(x.size)
    left, right = np.interp(left_ips, samples, x), np.interp(right_ips, samples, x)
    table = pd.DataFrame(
        {
            "position": x[idx],
            "height": y[idx],
            "prominence": props["prominences"],
            "fwhm": right - left,
            "left": left,
            "right": right,
            "index": idx,
        },
        columns=columns,
    )
    if max_peaks is not None and len(table) > max_peaks:
        table = table.nlargest(max_peaks, "prominence").sort_values("position")
    return table.reset_index(drop=True)


_PEAK_SHAPES = {
    "gaussian": (models.gaussian, ("amplitude", "center", "sigma")),
    "lorentzian": (models.lorentzian, ("amplitude", "center", "gamma")),
    "voigt": (models.pseudo_voigt, ("amplitude", "center", "fwhm", "eta")),
}


def fit_peaks(
    x,
    y,
    centers: Sequence[float] | None = None,
    n_peaks: int | None = None,
    shape: str = "gaussian",
    baseline: str = "none",
    x_range: tuple[float, float] | None = None,
    fix_centers: bool = False,
    loss: str = "linear",
) -> FitResult:
    """Deconvolve overlapping bands into a sum of peak functions.

    Parameters
    ----------
    centers : sequence of float, optional
        Starting band positions. By default the ``n_peaks`` most prominent
        peaks found by :func:`find_peaks` are used.
    shape : {'gaussian', 'lorentzian', 'voigt'}
        Band shape (``'voigt'`` is a pseudo-Voigt with a free mixing ratio).
        Gaussian bands fitted on an energy axis are the usual choice for
        electronic absorption bands.
    baseline : {'none', 'constant', 'linear'}
        Background fitted together with the bands.
    x_range : (float, float), optional
        Restrict the fit to this ``x`` window.

    Returns
    -------
    FitResult
        Parameters ``amplitude1, center1, sigma1, ...``; individual band curves
        are in ``result.components``.
    """
    if shape not in _PEAK_SHAPES:
        raise ValueError(f"shape must be one of {sorted(_PEAK_SHAPES)}")
    func, pnames = _PEAK_SHAPES[shape]
    x, y = _sorted_xy(x, y)
    if x_range is not None:
        mask = crop_mask(x, *x_range)
        x, y = x[mask], y[mask]
    span = float(x[-1] - x[0])
    dx = float(np.median(np.diff(x)))
    sign = 1.0 if np.nanmax(y) >= -np.nanmin(y) else -1.0

    if centers is None:
        peaks = find_peaks(x, sign * y, max_peaks=n_peaks)
        if peaks.empty:
            raise ValueError("no peaks found; pass `centers` explicitly")
        centers = peaks["position"].to_numpy()
        fwhms = np.clip(peaks["fwhm"].to_numpy(), 2 * dx, span)
    else:
        centers = np.atleast_1d(np.asarray(centers, dtype=float))
        fwhms = np.full(centers.size, span / (4.0 * centers.size))
    heights = np.interp(centers, x, y)

    names, p0, lower, upper = [], [], [], []
    for i, (c, w, h) in enumerate(zip(centers, fwhms, heights), start=1):
        width = {"sigma": w / models.FWHM_PER_SIGMA, "gamma": w / 2.0, "fwhm": w}[pnames[2]]
        names += [f"amplitude{i}", f"center{i}", f"{pnames[2]}{i}"]
        p0 += [h, c, width]
        lower += [0.0 if sign > 0 else -np.inf, x[0], dx / 4.0]
        upper += [np.inf if sign > 0 else 0.0, x[-1], span]
        if shape == "voigt":
            names.append(f"eta{i}")
            p0.append(0.5)
            lower.append(0.0)
            upper.append(1.0)
    n_band = len(pnames)
    n_peak = len(centers)
    if baseline in ("constant", "linear"):
        names.append("c0")
        p0.append(0.0)
        lower.append(-np.inf)
        upper.append(np.inf)
    if baseline == "linear":
        names.append("c1")
        p0.append(0.0)
        lower.append(-np.inf)
        upper.append(np.inf)
    elif baseline not in ("none", "constant"):
        raise ValueError("baseline must be 'none', 'constant' or 'linear'")
    x_mid = float(np.mean(x))

    def model(xx, *p):
        xx = np.asarray(xx, dtype=float)
        out = np.zeros_like(xx)
        for i in range(n_peak):
            out = out + func(xx, *p[i * n_band : (i + 1) * n_band])
        k = n_peak * n_band
        if baseline in ("constant", "linear"):
            out = out + p[k]
        if baseline == "linear":
            out = out + p[k + 1] * (xx - x_mid)
        return out

    fixed = [f"center{i}" for i in range(1, n_peak + 1)] if fix_centers else None
    result = fit(
        model,
        x,
        y,
        p0,
        names=names,
        bounds=(lower, upper),
        fixed=fixed,
        loss=loss,
        model=f"{n_peak} x {shape}" + (f" + {baseline} baseline" if baseline != "none" else ""),
    )
    for i in range(n_peak):
        result.components[f"peak{i + 1}"] = func(x, *result.values[i * n_band : (i + 1) * n_band])
    if baseline != "none":
        k = n_peak * n_band
        base = np.full_like(x, result.values[k])
        if baseline == "linear":
            base = base + result.values[k + 1] * (x - x_mid)
        result.components["baseline"] = base
    return result


# --------------------------------------------------------------------------
# Beer-Lambert
# --------------------------------------------------------------------------
def absorbance_to_transmittance(absorbance, percent: bool = False):
    """``T = 10**(-A)`` (in % if ``percent``)."""
    t = np.power(10.0, -np.asarray(absorbance, dtype=float))
    return 100.0 * t if percent else t


def transmittance_to_absorbance(transmittance, percent: bool = False):
    """``A = -log10(T)``; ``percent`` if ``T`` is given in %."""
    t = np.asarray(transmittance, dtype=float)
    if percent:
        t = t / 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log10(t)


def concentration_from_absorbance(absorbance, epsilon, path_length=1.0):
    """Beer-Lambert concentration ``c = A / (epsilon * l)``.

    With ``epsilon`` in M^-1 cm^-1 and ``path_length`` in cm, ``c`` is in M.
    """
    return np.asarray(absorbance, dtype=float) / (epsilon * path_length)


def molar_absorptivity(absorbance, concentration, path_length=1.0):
    """Beer-Lambert molar absorptivity ``epsilon = A / (c * l)``."""
    return np.asarray(absorbance, dtype=float) / (
        np.asarray(concentration, dtype=float) * path_length
    )


@dataclass
class CalibrationResult:
    """Linear calibration ``A = slope * c + intercept`` (see :func:`calibration_curve`)."""

    slope: float
    intercept: float
    slope_stderr: float
    intercept_stderr: float
    r_squared: float
    residual_std: float
    n: int
    path_length: float
    concentrations: np.ndarray
    absorbances: np.ndarray

    @property
    def molar_absorptivity(self) -> float:
        """``slope / path_length`` (M^-1 cm^-1 for concentrations in M and l in cm)."""
        return self.slope / self.path_length

    @property
    def lod(self) -> float:
        """Limit of detection, ``3.3 s_res / slope`` (ICH Q2)."""
        return 3.3 * self.residual_std / self.slope

    @property
    def loq(self) -> float:
        """Limit of quantification, ``10 s_res / slope`` (ICH Q2)."""
        return 10.0 * self.residual_std / self.slope

    def concentration(self, absorbance):
        """Concentration of an unknown sample from its absorbance."""
        return (np.asarray(absorbance, dtype=float) - self.intercept) / self.slope

    def summary(self) -> str:
        return "\n".join(
            [
                f"Calibration ({self.n} standards): A = ({self.slope:.5g} +/- {self.slope_stderr:.2g}) c"
                f" + ({self.intercept:.4g} +/- {self.intercept_stderr:.2g})",
                f"  R^2 = {self.r_squared:.5f}   s_res = {self.residual_std:.3g}",
                f"  epsilon = {self.molar_absorptivity:.5g} (per unit concentration and path length)",
                f"  LOD = {self.lod:.3g}   LOQ = {self.loq:.3g}",
            ]
        )

    def __str__(self) -> str:
        return self.summary()


def calibration_curve(
    concentrations, absorbances, path_length: float = 1.0, through_origin: bool = False
) -> CalibrationResult:
    """Linear Beer-Lambert calibration from standards of known concentration."""
    c = np.asarray(concentrations, dtype=float)
    a = np.asarray(absorbances, dtype=float)
    n = c.size
    if through_origin:
        if n < 2:
            raise ValueError("need at least 2 standards")
        slope = float(c @ a / (c @ c))
        intercept = 0.0
        resid = a - slope * c
        s2 = float(resid @ resid) / (n - 1)
        slope_err, intercept_err = float(np.sqrt(s2 / (c @ c))), 0.0
    else:
        if n < 3:
            raise ValueError("need at least 3 standards")
        cm = c.mean()
        sxx = float(np.sum((c - cm) ** 2))
        slope = float(np.sum((c - cm) * (a - a.mean())) / sxx)
        intercept = float(a.mean() - slope * cm)
        resid = a - (slope * c + intercept)
        s2 = float(resid @ resid) / (n - 2)
        slope_err = float(np.sqrt(s2 / sxx))
        intercept_err = float(np.sqrt(s2 * (1.0 / n + cm**2 / sxx)))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return CalibrationResult(
        slope=slope,
        intercept=intercept,
        slope_stderr=slope_err,
        intercept_stderr=intercept_err,
        r_squared=1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan"),
        residual_std=float(np.sqrt(s2)),
        n=n,
        path_length=path_length,
        concentrations=c,
        absorbances=a,
    )


# --------------------------------------------------------------------------
# Tauc analysis
# --------------------------------------------------------------------------
#: Tauc exponents n in (alpha h nu)^n for each transition type.
TAUC_EXPONENTS = {
    "direct": 2.0,
    "indirect": 0.5,
    "direct-forbidden": 2.0 / 3.0,
    "indirect-forbidden": 1.0 / 3.0,
}


def tauc_transform(
    wavelength_nm, absorbance, transition: str = "direct", thickness_cm: float | None = None
):
    """Photon energy (eV) and Tauc ordinate ``(alpha h nu)^n``, sorted by energy.

    ``alpha = ln(10) A / d`` if a film thickness ``d`` is given, otherwise the
    absorbance itself is used (the band gap does not depend on this scale).
    """
    if transition not in TAUC_EXPONENTS:
        raise ValueError(f"transition must be one of {sorted(TAUC_EXPONENTS)}")
    energy = nm_to_ev(wavelength_nm)
    a = np.clip(np.asarray(absorbance, dtype=float), 0.0, None)
    alpha = LN10 * a / thickness_cm if thickness_cm else a
    tauc = (alpha * energy) ** TAUC_EXPONENTS[transition]
    order = np.argsort(energy)
    return energy[order], tauc[order]


@dataclass
class TaucResult:
    band_gap: float
    slope: float
    intercept: float
    fit_range: tuple[float, float]
    r_squared: float
    transition: str
    energy: np.ndarray
    tauc: np.ndarray

    def line(self, energy=None):
        """Fitted straight line (evaluated on ``energy``, default the Tauc axis)."""
        e = self.energy if energy is None else np.asarray(energy, dtype=float)
        return self.slope * e + self.intercept

    def summary(self) -> str:
        lo, hi = self.fit_range
        return (
            f"Tauc ({self.transition}): Eg = {self.band_gap:.4f} eV "
            f"(linear fit {lo:.3f}-{hi:.3f} eV, R^2 = {self.r_squared:.4f})"
        )

    def __str__(self) -> str:
        return self.summary()


def _windowed_linear_fits(x, y, w):
    """Slope, intercept and R^2 of least-squares lines through every window of ``w`` points."""
    kernel = np.ones(w)

    def s(v):
        return np.convolve(v, kernel, mode="valid")

    sx, sy, sxx, sxy, syy = s(x), s(y), s(x * x), s(x * y), s(y * y)
    vx = w * sxx - sx**2
    vy = w * syy - sy**2
    cxy = w * sxy - sx * sy
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = cxy / vx
        r2 = cxy**2 / (vx * vy)
    intercept = (sy - slope * sx) / w
    return slope, intercept, r2


def tauc_bandgap(
    wavelength_nm,
    absorbance,
    transition: str = "direct",
    fit_range: tuple[float, float] | None = None,
    thickness_cm: float | None = None,
    window: int | None = None,
    min_r2: float = 0.99,
) -> TaucResult:
    """Optical band gap from the linear region of a Tauc plot.

    The band gap is the energy axis intercept of a straight line fitted to
    ``(alpha h nu)^n`` versus ``h nu``. Give the linear region as ``fit_range``
    (eV) whenever possible. Otherwise the steepest window of ``window`` points
    (default 10 % of the data) whose linear fit reaches ``min_r2`` is used,
    which picks the absorption edge above the Urbach tail.
    """
    energy, tauc = tauc_transform(wavelength_nm, absorbance, transition, thickness_cm)
    if fit_range is not None:
        mask = crop_mask(energy, *fit_range)
        if mask.sum() < 3:
            raise ValueError("fewer than 3 points inside fit_range")
        slope, intercept = np.polyfit(energy[mask], tauc[mask], 1)
        pred = slope * energy[mask] + intercept
        ss_tot = np.sum((tauc[mask] - tauc[mask].mean()) ** 2)
        r2 = 1.0 - np.sum((tauc[mask] - pred) ** 2) / ss_tot
        lo, hi = float(energy[mask][0]), float(energy[mask][-1])
    else:
        w = window or max(5, energy.size // 10)
        if energy.size < w:
            raise ValueError("not enough points for automatic Tauc analysis")
        slopes, intercepts, r2s = _windowed_linear_fits(energy, tauc, w)
        valid = np.isfinite(slopes) & (slopes > 0)
        candidates = valid & (r2s >= min_r2)
        if not candidates.any():
            candidates = valid
        if not candidates.any():
            raise ValueError("no rising absorption edge found")
        best = int(np.argmax(np.where(candidates, slopes, -np.inf)))
        slope, intercept, r2 = slopes[best], intercepts[best], r2s[best]
        lo, hi = float(energy[best]), float(energy[best + w - 1])
    return TaucResult(
        band_gap=float(-intercept / slope),
        slope=float(slope),
        intercept=float(intercept),
        fit_range=(lo, hi),
        r_squared=float(r2),
        transition=transition,
        energy=energy,
        tauc=tauc,
    )
