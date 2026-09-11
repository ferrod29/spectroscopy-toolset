import os
from pathlib import Path

import matplotlib
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
matplotlib.use("Agg")

from spectroscopy_toolset import TAData, models  # noqa: E402

EXAMPLE_DATA = Path(__file__).resolve().parents[1] / "examples" / "data"
TAUS = np.array([0.5, 8.0, 120.0])
T0, FWHM = 0.2, 0.15


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def delays():
    """A typical non-uniform TA delay grid (ps)."""
    return np.r_[np.arange(-1.0, 2.0, 0.02), np.arange(2.0, 20.0, 0.2), np.arange(20.0, 500.0, 5.0)]


def make_das(wl):
    return np.vstack(
        [
            1e-3 * np.exp(-0.5 * ((wl - 520) / 30) ** 2),
            -0.7e-3 * np.exp(-0.5 * ((wl - 620) / 40) ** 2),
            0.3e-3 * np.exp(-0.5 * ((wl - 680) / 25) ** 2),
        ]
    )


@pytest.fixture
def synthetic_ta(delays, rng):
    """Three-component TA data (lifetimes 0.5, 8 and 120 ps; t0 = 0.2 ps, IRF FWHM 0.15 ps)."""
    wl = np.linspace(450, 750, 90)
    das = make_das(wl)
    c = np.column_stack([models.exp_irf(delays, tau, T0, FWHM) for tau in TAUS])
    d = (c @ das).T + rng.normal(0.0, 5e-6, (wl.size, delays.size))
    return TAData(wl, delays, d, name="synthetic", meta={"das": das})
