"""Steady-state UV-vis workflow: baseline, peaks, band deconvolution,
Beer-Lambert calibration and a Tauc band gap.

The spectra are *synthetic* (generated below) so that the script runs
anywhere and the right answers are known. For your own data replace
``dilution_series()`` by ``st.read_spectra("my_file.csv")``.

Run:  python examples/uvvis_analysis.py [output_dir]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

import spectroscopy_toolset as st  # noqa: E402
from spectroscopy_toolset import models, plotting, uvvis  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("output"))
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)
wl = np.linspace(300, 800, 501)

# Dye with bands at 430 nm (eps = 12000 M^-1 cm^-1) and 520 nm (7000 M^-1 cm^-1),
# Gaussian on the energy axis, measured in a 1 cm cuvette on a scattering background.
energy_axis = st.processing.nm_to_ev(wl)
epsilon = 1.2e4 * models.gaussian(
    energy_axis, 1, st.processing.nm_to_ev(430), 0.13
) + 0.7e4 * models.gaussian(energy_axis, 1, st.processing.nm_to_ev(520), 0.12)
concentrations = np.array([5, 10, 20, 40, 60]) * 1e-6  # M


def dilution_series():
    spectra = []
    for c in concentrations:
        scatter = 0.02 * (400.0 / wl) ** 4
        a = epsilon * c + scatter + 0.01 + rng.normal(0, 0.002, wl.size)
        spectra.append(st.Spectrum(wl, a, name=f"{c * 1e6:.0f} uM"))
    return spectra


raw = dilution_series()
ax = plotting.plot_spectra(raw, ordered=True)
ax.figure.savefig(OUT / "uvvis_raw.png", dpi=150)

# 1) Baseline: a cubic through band-free regions removes the scattering background.
corrected = [s.subtract_baseline("poly", degree=3, regions=[(300, 340), (650, 800)]) for s in raw]
ax = plotting.plot_spectra(corrected, ordered=True)
ax.figure.savefig(OUT / "uvvis_corrected.png", dpi=150)

# 2) Peak table (positions, heights, widths at half prominence).
peaks = corrected[-1].find_peaks()
print("Peaks of the most concentrated sample:\n", peaks.round(3).to_string(index=False))
plotting.plot_peaks(corrected[-1], peaks).figure.savefig(OUT / "uvvis_peaks.png", dpi=150)

# 3) Band deconvolution on the energy axis (Gaussian bands are appropriate in eV).
energy = corrected[-1].to_energy().crop(1.8, 3.6)
deconv = energy.fit_peaks(n_peaks=2, baseline="constant")
print("\n" + deconv.summary())
main, _ = plotting.plot_fit(deconv, xlabel="Energy (eV)", ylabel="Absorbance")
main.figure.savefig(OUT / "uvvis_deconvolution.png", dpi=150)

# 4) Beer-Lambert calibration at the 430 nm maximum, then quantify an "unknown".
a430 = [np.interp(430, s.x, s.y) for s in corrected]
cal = uvvis.calibration_curve(concentrations, a430, path_length=1.0)
print("\n" + cal.summary())
unknown = 0.30
print(f"unknown with A430 = {unknown}: c = {cal.concentration(unknown) * 1e6:.2f} uM")
plotting.plot_calibration(cal).figure.savefig(OUT / "uvvis_calibration.png", dpi=150)

# 5) Tauc analysis of a synthetic direct-gap semiconductor film (Eg = 2.40 eV).
e = st.processing.nm_to_ev(wl)
film = np.sqrt(np.clip(e - 2.40, 0, None)) / e + 0.02 * np.exp((e - 2.40) / 0.05) * (e < 2.40)
tauc = uvvis.tauc_bandgap(wl, film, transition="direct")
print("\n" + tauc.summary())
plotting.plot_tauc(tauc).figure.savefig(OUT / "uvvis_tauc.png", dpi=150)
print(f"\nfigures written to {OUT}")
