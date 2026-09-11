"""Spectroscopy Toolset - analysis of UV-vis absorption and transient-absorption data.

Quick start::

    import spectroscopy_toolset as st

    spec = st.read_spectrum("sample.csv").subtract_baseline("als")
    print(spec.find_peaks())

    ta = st.read_ta("scan.dat").exclude_wavelengths((510, 522))
    print(ta.fit_kinetics(550, n_exp=2))
"""

from . import models, processing, uvvis
from .fitting import FitResult, GlobalFitResult, das_to_eads, fit, fit_kinetics, global_fit
from .io import (
    load_table,
    read_spectra,
    read_spectrum,
    read_ta,
    read_ta_matrix,
    read_ta_scan,
    write_spectrum,
    write_ta_matrix,
    write_ta_xyz,
)
from .spectrum import Spectrum
from .transient import (
    ChirpModel,
    TAData,
    average_scans,
    estimate_chirp,
    fit_chirp,
    oscillation_spectrum,
)

__version__ = "0.2.0"

__all__ = [
    "ChirpModel",
    "FitResult",
    "GlobalFitResult",
    "Spectrum",
    "TAData",
    "average_scans",
    "das_to_eads",
    "estimate_chirp",
    "fit",
    "fit_chirp",
    "fit_kinetics",
    "global_fit",
    "load_table",
    "models",
    "oscillation_spectrum",
    "processing",
    "read_spectra",
    "read_spectrum",
    "read_ta",
    "read_ta_matrix",
    "read_ta_scan",
    "uvvis",
    "write_spectrum",
    "write_ta_matrix",
    "write_ta_xyz",
]
