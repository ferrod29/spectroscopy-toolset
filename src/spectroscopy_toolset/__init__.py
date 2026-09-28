"""Spectroscopy Toolset - analysis of UV-vis absorption and transient-absorption data.

Quick start::

    import spectroscopy_toolset as st

    spec = st.read_spectrum("sample.csv").subtract_baseline("als")
    print(spec.find_peaks())

    ta = st.read_ta("scan.dat").exclude_wavelengths((510, 522))
    print(ta.fit_kinetics(550, n_exp=2))
"""

from . import calibration, models, processing, uvvis
from .calibration import PrismCalibration, calibrate_harbor, fit_prism_calibration
from .fitting import FitResult, GlobalFitResult, das_to_eads, fit, fit_kinetics, global_fit
from .io import (
    detect_ta_format,
    load_table,
    read_harbor,
    read_harpia,
    read_helios,
    read_helios_raw,
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
    "PrismCalibration",
    "Spectrum",
    "TAData",
    "average_scans",
    "calibrate_harbor",
    "calibration",
    "das_to_eads",
    "detect_ta_format",
    "estimate_chirp",
    "fit",
    "fit_chirp",
    "fit_kinetics",
    "fit_prism_calibration",
    "global_fit",
    "load_table",
    "models",
    "oscillation_spectrum",
    "processing",
    "read_harbor",
    "read_harpia",
    "read_helios",
    "read_helios_raw",
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
