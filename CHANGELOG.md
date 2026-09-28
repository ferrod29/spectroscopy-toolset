# Changelog

## Unreleased - instrument formats and HARBOR wavelength calibration

Ported and reworked from the retinal TA notebooks.

### New

* Readers for three TA setups, detected automatically by `read_ta`, the CLI
  and the TA Inspector: HELIOS ΔA matrices and raw pump-on/off intensities
  (`read_helios`, `read_helios_raw`), HARPIA `*_matrix.dat` exports
  (`read_harpia`, mOD → OD) and HARBOR scans (`read_harbor`). HARBOR scans
  have forward and backward sweeps, delays in fs, mOD, and 580- or 600-pixel
  channels.
* `calibration` module: `PrismCalibration` (fused-silica prism, Sellmeier
  dispersion, JSON save/load), `fit_prism_calibration` and `calibrate_harbor`,
  plus `spectro wl-calibrate` and `plot_wavelength_calibration`.
* `average_scans(..., delay_tolerance=)` / `--delay-tolerance` averages scans
  whose measured delay grids jitter slightly, and records the spread.
  NaN points are averaged over the scans that have them.
* TA matrices: `dA_scale`, and dead pixels (all-NaN rows) are dropped.

### Problems found in the notebook code

* **BG36 calibration can lock onto the wrong bands.** The notebook fit scales
  the reference *absorbance* linearly onto the normalised intensity ratio, and
  the result depends on the starting values. From `WavelengthCalibrator`'s
  starting values it converges to a wrong band assignment: the 585 nm band
  (OD 3.7, the strongest) does not appear in the calibrated data at all. That
  run wrote `raw/20260327/calib_wavelengths.txt` (411-728 nm, residual 21×
  larger than the new fit). The earlier fit in
  `processed/20260327/calib_wavelengths.txt`, which the retinal notebooks use,
  is good: it differs from the new calibration by 0.3-2.4 nm over the lit
  pixels (221-336). The new fit works in transmission, includes the detector
  resolution (σ ≈ 5.8 nm) and starts from a grid search, so it does not depend
  on starting values (R² = 0.957, scale 0.93).
* `WavelengthCalibrator` used a 24 µm pixel while `GetLambda2024` and
  `TfsGet36_2024` used 25 µm. Only `f / pixel size` matters, so the
  wavelengths agree, but the reported focal lengths do not. Its `__init__`
  also discarded the result. The saved file header says µm for values in nm.
* HARBOR files were sliced as `columns 1:601`. For files with 580-pixel
  channels this mixed 20 pixels of the second channel into the spectrum.
* `snippets.chirp_correction` ignored its fitted polynomial and used a
  hard-coded one. It modified the input array in place, and it returned the
  corrected time axis of the last pixel only. Use `TAData.estimate_chirp` /
  `correct_chirp`.
* The HELIOS notebook stored each scan as both "forward" and "backward".
  The standard error was therefore divided by √(2N) instead of √N, which
  understated it by √2.

## 0.2.0 - restructure into a package

The loose scripts became an installable package (`src/spectroscopy_toolset`) with
a tested analysis library, a command-line interface (`spectro`), rebuilt Qt
interfaces (`spectro-uvvis`, `spectro-ta`), examples, a pytest suite and CI.

### New analysis methods

* UV-vis: baseline correction (asymmetric least squares, polynomial, offset),
  peak tables with widths, band deconvolution with parameter errors,
  Beer-Lambert calibration with LOD/LOQ, Tauc band gaps, derivative spectra,
  energy axis, and averaging on a common grid.
* TA: chirp estimation and correction; kinetic fits with a Gaussian IRF,
  long-lived component and coherent-artefact model; global analysis with
  DAS/EADS; SVD; noise estimates; averaging of repeated scans with standard
  errors; and oscillation spectra with a correct cm⁻¹ axis.
* The GUI "Global A." button, which was disabled before, now runs the global
  analysis. The TA Inspector also gained a pump-scatter exclusion field, an
  "Artefact" option and an "n exp" selector.

### Bugs fixed in behaviour carried over from the legacy scripts

Results:

* **Tri-exponential kinetic fit**: the bounds and the legend treated
  parameters 0-2 as the three lifetimes, but the model used parameters 1, 3
  and 5. The displayed "τ" values mixed amplitudes and lifetimes, and the
  bounds forced amplitudes to be positive, so bleach signals could not be fitted.
* **Map time axis**: the image assumed uniformly spaced delays (the data use
  steps from 0.05 to 2 ps), which distorted the time axis. The translation and
  scaling also accumulated on every redraw.
* **"Unchirp"** deleted samples from each trace (shifting later points in time)
  and ignored the fitted dispersion curve. It also modified the data in place
  and blocked the GUI with `plt.show()`.
* **Averaging several files**: the per-scan array stored running sums, so the
  standard deviation was wrong. Delay axes that differed were averaged
  silently (`TestData_2.dat` is offset by 1.9 ps). Differing axes now raise a
  clear error.
* `std` was set to the data itself for `.dat` files.
* Kinetic traces were shifted towards zero by the noise level.
* **FFT**: the frequency axis was in rad/ps but labelled cm⁻¹ (wrong by a
  factor of 2πc), and a sample-index Hann window was applied to non-uniform data.

Crashes:

* UV-vis plotter: every file load crashed (`self.status_bar`, and
  `error_bad_lines` was removed in pandas 2). "Avg. Data", "Recolor", find
  peaks and fitting could not run: undefined names, widget names missing from
  the .ui file, and bounds of the wrong size. The fitted curve was never
  drawn. More than 13 curves raised an `IndexError`. The close confirmation
  never ran.
* TA GUI: axis limits had to match a grid value exactly. "Clear Traces" always
  crashed and "Export Traces" crashed unless "Get Traces" had been used first,
  because of typos and wrong parents. The data handler was created only after
  clicking "connect", once more per click. Notes were written to the current
  working directory.
* `data3dtransform.py` could not run: undefined names, and a mesh grid of
  flattened arrays that would have needed about 10¹² elements.
* `.ui` files were loaded relative to the working directory.

### Housekeeping

* Removed tracked `__pycache__` files and Eclipse project files, and added a `.gitignore`.
* The Qt Designer files now set the font once on the root widget instead of
  per widget, which halves their size with the same appearance. Labels were
  corrected ("Wavelenght", "Analisys", "Experiement", and the swapped X/Y
  axis-limit headers).
* Example data moved to `examples/data/`.
