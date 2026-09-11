"""Transient-absorption workflow on the example data set ``data/TestData_1.dat``:
background subtraction, chirp correction, kinetic fits and global analysis.

Run:  python examples/ta_analysis.py [output_dir]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

import spectroscopy_toolset as st  # noqa: E402
from spectroscopy_toolset import plotting  # noqa: E402

HERE = Path(__file__).parent
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "output")
OUT.mkdir(parents=True, exist_ok=True)


def save(ax, name):
    ax.figure.savefig(OUT / name, dpi=150)


data = st.read_ta(HERE / "data" / "TestData_1.dat")
print(f"loaded {data.shape[0]} wavelengths x {data.shape[1]} delays")

# Delays are raw stage positions; count them from the first recorded delay.
data = data.shift_time(data.delays[0])
# The 515 nm pump scatters into 512-519 nm (saturated values): remove that band.
data = data.exclude_wavelengths((509, 523))
# The first ~0.45 ps precede time zero: subtract that background, report the noise.
data = data.subtract_background(before=0.45)
print(f"noise level: {1e3 * np.median(data.noise(before=0.45)):.3f} mOD")

# Chirp: time zero per wavelength from the coherent artefact, robust polynomial fit.
wl, t0 = data.estimate_chirp(window=(0.3, 1.5))
chirp = st.fit_chirp(wl, t0)
print(chirp)
save(plotting.plot_chirp(wl, t0, chirp), "ta_chirp.png")
data = data.correct_chirp(chirp).bin_wavelengths(width=2.0)

save(plotting.plot_ta_map(data, linthresh=2.0), "ta_map.png")
save(plotting.plot_singular_values(data), "ta_svd.png")
save(plotting.plot_ta_spectra(data, [0.5, 2, 10, 50, 150], width=0.3), "ta_spectra.png")

# Kinetics: 2 exponentials + long-lived component + coherent artefact, Gaussian IRF.
fits = []
for wavelength in (550, 650):
    result = data.fit_kinetics(wavelength, width=5.0, n_exp=2, step=True, artifact=3)
    print(f"\n--- {wavelength} nm ---\n{result}")
    fits.append(result)
save(
    plotting.plot_kinetics(data, [550, 650], width=5.0, fits=fits, linthresh=2.0), "ta_kinetics.png"
)

# Global analysis: shared lifetimes -> decay- and evolution-associated spectra.
result = data.global_fit(n_exp=2, step=True, artifact=3)
print("\n" + result.summary())
save(plotting.plot_das(result), "ta_das.png")
save(plotting.plot_das(result, eads=True), "ta_eads.png")
residuals = data.copy(dA=result.residuals, name="residuals of the global fit")
save(plotting.plot_ta_map(residuals, linthresh=2.0), "ta_residuals.png")
print(f"\nfigures written to {OUT}")
