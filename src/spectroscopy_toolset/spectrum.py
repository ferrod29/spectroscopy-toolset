"""Container for a single one-dimensional spectrum (e.g. a UV-vis absorbance spectrum)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import processing, uvvis


@dataclass
class Spectrum:
    """A spectrum ``y(x)`` with its axis labels and free-form metadata.

    ``x`` is sorted in ascending order on construction. All processing methods
    return a new :class:`Spectrum` and leave the original untouched, so steps
    can be chained::

        spec = Spectrum.from_file("sample.csv").crop(300, 800).subtract_baseline("als")
    """

    x: np.ndarray
    y: np.ndarray
    name: str = ""
    x_label: str = "Wavelength (nm)"
    y_label: str = "Absorbance"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        x = np.asarray(self.x, dtype=float).ravel()
        y = np.asarray(self.y, dtype=float).ravel()
        if x.shape != y.shape:
            raise ValueError(f"x and y differ in length ({x.size} vs {y.size})")
        order = np.argsort(x, kind="stable")
        self.x, self.y = x[order], y[order]
        self.meta = dict(self.meta)

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> Spectrum:
        """Read a spectrum with :func:`spectroscopy_toolset.io.read_spectrum`."""
        from .io import read_spectrum

        return read_spectrum(path, **kwargs)

    def __len__(self) -> int:
        return self.x.size

    def copy(self, **changes) -> Spectrum:
        return replace(self, **changes)

    # ------------------------------------------------------------------ edits
    def crop(self, xmin=None, xmax=None) -> Spectrum:
        """Keep ``xmin <= x <= xmax`` (``None`` leaves a side open)."""
        mask = processing.crop_mask(self.x, xmin, xmax)
        return self.copy(x=self.x[mask], y=self.y[mask])

    def baseline(self, method: str = "als", **kwargs) -> np.ndarray:
        """Baseline estimate.

        ``method``: ``'als'`` (asymmetric least squares, kwargs ``lam``, ``p``),
        ``'poly'`` (polynomial through band-free ``regions``, kwargs ``degree``,
        ``regions``) or ``'offset'`` (constant equal to ``y`` at ``at`` or its
        mean over ``at = (x0, x1)``, e.g. a non-absorbing region to remove
        scattering offsets).
        """
        if method == "als":
            return processing.baseline_als(self.y, **kwargs)
        if method in ("poly", "polynomial"):
            return processing.baseline_polynomial(self.x, self.y, **kwargs)
        if method == "offset":
            return processing.baseline_offset(self.x, self.y, **kwargs)
        raise ValueError(f"unknown baseline method {method!r}")

    def subtract_baseline(self, method: str = "als", **kwargs) -> Spectrum:
        return self.copy(y=self.y - self.baseline(method, **kwargs))

    def normalize(self, method: str = "max", at=None) -> Spectrum:
        """Normalise ``y`` (see :func:`spectroscopy_toolset.processing.normalize`)."""
        y = processing.normalize(self.y, method, x=self.x, at=at)
        return self.copy(y=y, y_label=f"{self.y_label} (norm.)")

    def smooth(self, window: int = 11, polyorder: int = 3) -> Spectrum:
        """Savitzky-Golay smoothing (``window`` in points)."""
        return self.copy(y=processing.smooth(self.y, window, polyorder))

    def derivative(self, order: int = 1, window: int = 11, polyorder: int = 3) -> Spectrum:
        """Savitzky-Golay derivative (second derivatives resolve overlapping bands)."""
        y = processing.derivative(self.x, self.y, order, window, polyorder)
        return self.copy(y=y, y_label=f"d{order}({self.y_label})/dx{order}")

    def resample(self, x_new) -> Spectrum:
        return self.copy(x=np.asarray(x_new, float), y=processing.resample(x_new, self.x, self.y))

    def to_energy(self) -> Spectrum:
        """Spectrum on a photon-energy axis (eV), assuming ``x`` is a wavelength in nm."""
        return self.copy(x=processing.nm_to_ev(self.x), x_label="Energy (eV)")

    # --------------------------------------------------------------- analysis
    def find_peaks(self, **kwargs):
        """Peak table (see :func:`spectroscopy_toolset.uvvis.find_peaks`)."""
        return uvvis.find_peaks(self.x, self.y, **kwargs)

    def fit_peaks(self, **kwargs):
        """Band deconvolution (see :func:`spectroscopy_toolset.uvvis.fit_peaks`)."""
        return uvvis.fit_peaks(self.x, self.y, **kwargs)

    def save(self, path: str | Path, delimiter: str = ",") -> None:
        from .io import write_spectrum

        write_spectrum(path, self, delimiter=delimiter)
