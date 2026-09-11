# Changelog

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
