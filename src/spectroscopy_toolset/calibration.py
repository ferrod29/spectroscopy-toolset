"""Wavelength calibration of prism spectrometers.

Some TA detectors (e.g. the HARBOR setup) disperse the probe with a fused-silica
prism and record pixel numbers instead of wavelengths. :class:`PrismCalibration`
maps pixels to wavelengths with the prism geometry and the Sellmeier dispersion
of fused silica::

    pixel = n_pixels + offset - focal_length * sin(theta(lambda) - theta(lambda_0)) / pixel_size

``theta`` is the exit angle of the prism for a ray entering at
``incidence_angle``. Only ``focal_length / pixel_size`` and
``n_pixels + offset`` affect the result, so two parameters are fitted.

:func:`fit_prism_calibration` finds them from the transmission of a filter with
sharp bands (a BG36 didymium glass), measured on the detector as
``I(with filter) / I(without filter)``. It compares that ratio with
``10**-A_ref`` (the reference absorbance measured on a spectrophotometer)
broadened to the detector resolution. Intensities add linearly, so the
comparison is done in transmission, not absorbance. A coarse grid search
comes first because the band pattern repeats and local fits often lock onto
the wrong band.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

__all__ = [
    "PrismCalibration",
    "PrismCalibrationFit",
    "calibrate_harbor",
    "fit_prism_calibration",
    "fused_silica_index",
]

# Wavelength range (nm) of the pixel -> wavelength lookup; the Sellmeier fit of
# Malitson (1965) is valid from 210 nm to 3.7 um.
_GRID_NM = np.geomspace(210.0, 2500.0, 20000)


def fused_silica_index(wavelength_nm):
    """Refractive index of fused silica (Sellmeier fit of I. H. Malitson, JOSA 55, 1205 (1965))."""
    l2 = (np.asarray(wavelength_nm, dtype=float) * 1e-3) ** 2
    return np.sqrt(
        1.0
        + 0.6961663 * l2 / (l2 - 0.0684043**2)
        + 0.4079426 * l2 / (l2 - 0.1162414**2)
        + 0.8974794 * l2 / (l2 - 9.896161**2)
    )


@dataclass
class PrismCalibration:
    """Pixel -> wavelength mapping of a prism spectrometer.

    The defaults describe the HARBOR TA detector (600 pixels of 25 um, a 60 deg
    fused-silica prism at 50 deg incidence). ``valid_pixels`` records the pixel
    range the calibration was fitted on; outside it the wavelengths are
    extrapolated with the dispersion model.
    """

    focal_length: float  # mm
    offset: float  # px
    n_pixels: int = 600
    pixel_size: float = 0.025  # mm
    apex_angle: float = 60.0  # deg
    incidence_angle: float = 50.0  # deg
    reference_wavelength: float = 500.0  # nm
    valid_pixels: tuple[float, float] | None = None
    meta: dict = field(default_factory=dict)

    def _exit_angle(self, wavelength_nm):
        n = fused_silica_index(wavelength_nm)
        apex, incidence = np.radians(self.apex_angle), np.radians(self.incidence_angle)
        return np.arcsin(n * np.sin(apex - np.arcsin(np.sin(incidence) / n)))

    def _lookup(self) -> tuple[np.ndarray, np.ndarray]:
        """Exit-angle deviation ``sin(theta - theta_0)`` on the wavelength grid (decreasing)."""
        theta0 = self._exit_angle(self.reference_wavelength)
        return np.sin(self._exit_angle(_GRID_NM) - theta0), _GRID_NM

    def pixels(self, wavelengths) -> np.ndarray:
        """Pixel positions of the given wavelengths (nm)."""
        theta0 = self._exit_angle(self.reference_wavelength)
        s = np.sin(self._exit_angle(wavelengths) - theta0)
        return self.n_pixels + self.offset - self.focal_length * s / self.pixel_size

    def wavelengths(self, pixels) -> np.ndarray:
        """Wavelengths (nm) of the given pixel positions; NaN outside 210-2500 nm."""
        s_grid, wl_grid = self._lookup()
        s = (self.n_pixels + self.offset - np.asarray(pixels, dtype=float)) * (
            self.pixel_size / self.focal_length
        )
        # s_grid decreases with wavelength; np.interp needs increasing x.
        return np.interp(s, s_grid[::-1], wl_grid[::-1], left=np.nan, right=np.nan)

    __call__ = wavelengths

    def to_dict(self) -> dict:
        d = asdict(self)
        d["valid_pixels"] = None if self.valid_pixels is None else list(self.valid_pixels)
        return d

    def save(self, path) -> None:
        """Write the calibration as JSON (read it back with :meth:`load`)."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> PrismCalibration:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("valid_pixels") is not None:
            d["valid_pixels"] = tuple(d["valid_pixels"])
        return cls(**d)


@dataclass
class PrismCalibrationFit:
    """Result of :func:`fit_prism_calibration`.

    ``model = scale * T_ref(lambda(pixel)) + baseline``, where ``T_ref`` is the
    reference transmission broadened by a Gaussian of standard deviation
    ``broadening`` (nm). A good calibration has ``scale`` close to 1 and
    ``baseline`` close to 0.
    """

    calibration: PrismCalibration
    pixels: np.ndarray
    transmission: np.ndarray
    model: np.ndarray
    scale: float
    baseline: float
    broadening: float
    stderr: dict
    r_squared: float

    @property
    def wavelengths(self) -> np.ndarray:
        return self.calibration.wavelengths(self.pixels)

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean((self.transmission - self.model) ** 2)))

    def __str__(self) -> str:
        c, e = self.calibration, self.stderr
        wl = self.wavelengths
        return (
            f"focal length  {c.focal_length:.3f} +/- {e['focal_length']:.2g} mm\n"
            f"pixel offset  {c.offset:.3f} +/- {e['offset']:.2g} px\n"
            f"broadening    {self.broadening:.2f} +/- {e['broadening']:.2g} nm (Gaussian sigma)\n"
            f"scale         {self.scale:.3f}, baseline {self.baseline:+.3f}\n"
            f"R^2           {self.r_squared:.4f} (rms {self.rms:.3g})\n"
            f"pixels {self.pixels[0]:g}-{self.pixels[-1]:g} -> {wl[0]:.1f}-{wl[-1]:.1f} nm"
        )


def _reference_transmission(ref_wavelengths, ref_absorbance, step: float):
    wl = np.asarray(ref_wavelengths, dtype=float)
    a = np.asarray(ref_absorbance, dtype=float)
    ok = np.isfinite(wl) & np.isfinite(a)
    order = np.argsort(wl[ok])
    wl, a = wl[ok][order], a[ok][order]
    grid = np.arange(wl[0], wl[-1] + step / 2, step)
    return grid, 10.0 ** -np.interp(grid, wl, a)


def _broaden(t_ref, sigma_nm: float, step: float):
    return gaussian_filter1d(t_ref, sigma_nm / step, mode="nearest") if sigma_nm > 0 else t_ref


def fit_prism_calibration(
    pixels,
    transmission,
    ref_wavelengths,
    ref_absorbance,
    *,
    pixel_range: tuple[float, float] | None = None,
    focal_lengths=None,
    offsets=None,
    broadenings=(0.0, 1.0, 2.0, 4.0, 6.0, 8.0),
    **constants,
) -> PrismCalibrationFit:
    """Fit a :class:`PrismCalibration` to a filter transmission measured on the detector.

    Parameters
    ----------
    pixels, transmission:
        Pixel numbers and ``I(with filter) / I(without filter)``. Use only
        pixels with enough probe light, via ``pixel_range`` or by passing NaN.
    ref_wavelengths, ref_absorbance:
        Reference absorbance (OD) of the same filter, e.g. from a UV-vis
        spectrophotometer.
    focal_lengths, offsets:
        Grids (mm, px) for the initial search. The defaults cover focal lengths
        of 60-500 mm and offsets of -600 to +200 px.
    broadenings:
        Detector resolutions (Gaussian sigma, nm) tried in the grid search. The
        best one is refined together with the other parameters.
    constants:
        Geometry passed to :class:`PrismCalibration` (``n_pixels``,
        ``pixel_size``, ``apex_angle``, ...).
    """
    px = np.asarray(pixels, dtype=float)
    y = np.asarray(transmission, dtype=float)
    mask = np.isfinite(px) & np.isfinite(y)
    if pixel_range is not None:
        mask &= (px >= min(pixel_range)) & (px <= max(pixel_range))
    px, y = px[mask], y[mask]
    order = np.argsort(px)
    px, y = px[order], y[order]
    if px.size < 8:
        raise ValueError(f"only {px.size} usable pixels for the calibration fit")
    if np.median(y) > 1.0:
        warnings.warn(
            "the transmission is mostly > 1; assuming the filter and open-beam spectra "
            "were swapped and using the inverse ratio",
            stacklevel=2,
        )
        y = 1.0 / y

    step = 0.2
    ref_grid, t_ref = _reference_transmission(ref_wavelengths, ref_absorbance, step)
    template = PrismCalibration(1.0, 0.0, **constants)
    s_grid, wl_grid = template._lookup()
    s_inc, wl_inc = s_grid[::-1], wl_grid[::-1]

    def wavelengths(f, c):
        s = (template.n_pixels + c - px) * (template.pixel_size / f)
        return np.interp(s, s_inc, wl_inc, left=np.nan, right=np.nan)

    def predict(wl, t_b):
        return np.interp(wl, ref_grid, t_b, left=np.nan, right=np.nan)

    # ---- grid search: for every (f, c, sigma) the best scale and baseline by linear least squares
    fs = np.arange(60.0, 500.0, 2.0) if focal_lengths is None else np.asarray(focal_lengths, float)
    cs = np.arange(-600.0, 200.0, 2.0) if offsets is None else np.asarray(offsets, float)
    n, sy = px.size, y.sum()
    broadened = [(sigma, _broaden(t_ref, sigma, step)) for sigma in broadenings]
    best = (np.inf, None)
    for f in fs:
        wl = np.interp(
            (template.n_pixels + cs[:, None] - px) * (template.pixel_size / f),
            s_inc,
            wl_inc,
            left=np.nan,
            right=np.nan,
        )
        for sigma, t_b in broadened:
            m = predict(wl, t_b)
            complete = np.isfinite(m).all(axis=-1)
            m = np.where(np.isfinite(m), m, 0.0)
            sm, smm, smy = m.sum(-1), (m * m).sum(-1), (m * y).sum(-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                a = (n * smy - sm * sy) / (n * smm - sm**2)
                b = (sy - a * sm) / n
                ssr = ((a[:, None] * m + b[:, None] - y) ** 2).sum(-1)
            ssr = np.where(complete & (a > 0) & np.isfinite(ssr), ssr, np.inf)
            j = int(np.argmin(ssr))
            if ssr[j] < best[0]:
                best = (ssr[j], (f, cs[j], sigma, a[j], b[j]))
    if best[1] is None:
        raise ValueError("no calibration on the search grid maps the pixels onto the reference")

    # ---- refinement of all five parameters
    def residuals(p):
        f, c, sigma, a, b = p
        m = predict(wavelengths(f, c), _broaden(t_ref, abs(sigma), step))
        return np.where(np.isfinite(m), a * m + b - y, 1.0)

    x0 = np.array(best[1], dtype=float)
    x0[2] = max(x0[2], 0.3)  # a zero width has no gradient
    sol = least_squares(residuals, x0, x_scale=[10.0, 1.0, 1.0, 0.1, 0.01], diff_step=1e-4)
    f, c, sigma, a, b = sol.x
    sigma = abs(sigma)
    r = sol.fun
    dof = max(n - 5, 1)
    try:
        cov = np.linalg.pinv(sol.jac.T @ sol.jac) * (r @ r / dof)
        err = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        err = np.full(5, np.nan)
    model = a * predict(wavelengths(f, c), _broaden(t_ref, sigma, step)) + b
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    calibration = PrismCalibration(
        float(f),
        float(c),
        valid_pixels=(float(px[0]), float(px[-1])),
        meta={"broadening_nm": float(sigma), "scale": float(a), "baseline": float(b)},
        **constants,
    )
    return PrismCalibrationFit(
        calibration=calibration,
        pixels=px,
        transmission=y,
        model=model,
        scale=float(a),
        baseline=float(b),
        broadening=float(sigma),
        stderr=dict(zip(["focal_length", "offset", "broadening", "scale", "baseline"], err)),
        r_squared=1.0 - float(r @ r) / ss_tot if ss_tot > 0 else float("nan"),
    )


def calibrate_harbor(
    with_filter,
    without_filter,
    reference,
    *,
    pixel_range: tuple[float, float] | None = None,
    min_signal: float = 0.05,
    **kwargs,
) -> PrismCalibrationFit:
    """Calibrate the HARBOR detector from BG36 filter measurements.

    ``with_filter`` and ``without_filter`` are one or several HARBOR files
    recorded with and without the filter in the probe beam. ``reference`` is
    a two-column file with the filter absorbance (nm, OD). Pixels whose
    open-beam intensity is below ``min_signal`` times its maximum are left out
    unless ``pixel_range`` is given. Other ``kwargs`` go to
    :func:`fit_prism_calibration`.
    """
    from .io import _common_pixels, read_harbor_spectrum, read_spectrum

    def spectra(paths):
        return [
            read_harbor_spectrum(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)
        ]

    filt, open_ = spectra(with_filter), spectra(without_filter)
    px = _common_pixels([s[0] for s in filt + open_])

    def mean_intensity(group):
        return np.mean([s[1][np.searchsorted(s[0], px)] for s in group], axis=0)

    i_filter, i_open = mean_intensity(filt), mean_intensity(open_)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = i_filter / i_open
    if pixel_range is None:
        t = np.where(i_open > min_signal * np.nanmax(i_open), t, np.nan)
    ref = read_spectrum(reference)
    fit = fit_prism_calibration(px, t, ref.x, ref.y, pixel_range=pixel_range, **kwargs)
    fit.calibration.meta.update(
        {
            "with_filter": [str(p) for p in np.atleast_1d(with_filter)],
            "without_filter": [str(p) for p in np.atleast_1d(without_filter)],
            "reference": str(reference),
        }
    )
    return fit
