"""Publication-style matplotlib figures for spectra, fits and TA data.

Every function draws into ``ax`` when given (or a new figure) and returns the
axes. Colours follow fixed rules: unordered series take categorical colours in
a fixed order (at most eight), ordered series (e.g. spectra at increasing
delays) take a one-hue ramp, and signed TA maps use a blue/red diverging map
with a neutral midpoint at dA = 0 (bleach/stimulated emission blue, induced
absorption red).
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_hex

from .fitting import FitResult, GlobalFitResult
from .spectrum import Spectrum
from .transient import TAData
from .uvvis import TAUC_EXPONENTS, CalibrationResult, TaucResult

#: Categorical colours, used in this fixed order (colour-vision-deficiency safe for adjacent series).
CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

#: One-hue ramp for ordered series (light to dark blue).
ORDINAL = LinearSegmentedColormap.from_list(
    "ordinal_blue", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
)
#: Diverging map for signed data: blue (negative) - neutral gray (zero) - red (positive).
DIVERGING = LinearSegmentedColormap.from_list(
    "ta_diverging", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f4b0ae", "#e34948", "#8c1d1f"]
)


# --------------------------------------------------------------------------
# Styling helpers
# --------------------------------------------------------------------------
def style_axes(ax) -> None:
    """Recessive chrome: hairline solid grid, light axis lines, no top/right spines."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK)


def _axes(ax=None, figsize=(6.4, 4.2)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, layout="constrained")
    style_axes(ax)
    return ax


def series_colors(n: int, ordered: bool = False) -> list[str]:
    """Colours for ``n`` series: fixed-order categorical for up to 8 unordered
    series, otherwise (or when ``ordered``) steps of a one-hue ramp."""
    if n <= 1:
        return CATEGORICAL[:1]
    if not ordered and n <= len(CATEGORICAL):
        return CATEGORICAL[:n]
    return [to_hex(ORDINAL(v)) for v in np.linspace(0.0, 1.0, n)]


def robust_limit(values) -> float:
    """Symmetric colour limit for a signed map that ignores a few extreme rows.

    Takes the 90th percentile of the per-row 99.5th-percentile magnitudes, so a
    handful of saturated rows (e.g. pump scatter) do not wash out the signal.
    """
    z = np.abs(np.asarray(values, dtype=float))
    if z.ndim == 1:
        z = z[None, :]
    finite_rows = np.isfinite(z).any(axis=1)
    if not finite_rows.any():
        return 1.0
    row_max = np.nanpercentile(z[finite_rows], 99.5, axis=1)
    limit = float(np.percentile(row_max, 90))
    return limit if np.isfinite(limit) and limit > 0 else 1.0


def _legend(ax, n: int) -> None:
    if n > 1:
        ax.legend(
            frameon=False, fontsize="small", ncol=1 if n <= 8 else 2, labelcolor=INK_SECONDARY
        )


def _zero_line(ax) -> None:
    ax.axhline(0.0, color=AXIS, linewidth=0.8, zorder=1)


# --------------------------------------------------------------------------
# Steady-state spectra
# --------------------------------------------------------------------------
def plot_spectra(
    spectra: Sequence[Spectrum] | Spectrum,
    ax=None,
    normalize: str | None = None,
    energy: bool = False,
    ordered: bool = False,
    legend: bool = True,
):
    """Overlay spectra. ``ordered=True`` colours them along a ramp (titrations, series)."""
    spectra = [spectra] if isinstance(spectra, Spectrum) else list(spectra)
    ax = _axes(ax)
    for spec, color in zip(spectra, series_colors(len(spectra), ordered)):
        shown = spec.to_energy() if energy else spec
        if normalize:
            shown = shown.normalize(normalize)
        ax.plot(shown.x, shown.y, color=color, linewidth=1.5, label=spec.name)
    ax.set_xlabel("Energy (eV)" if energy else spectra[0].x_label)
    ylabel = spectra[0].y_label
    ax.set_ylabel(f"{ylabel} (norm.)" if normalize and "norm" not in ylabel else ylabel)
    if legend:
        _legend(ax, len(spectra))
    return ax


def plot_peaks(spectrum: Spectrum, peaks, ax=None, annotate: bool = True):
    """Spectrum with detected peaks marked and their half-prominence widths drawn."""
    ax = plot_spectra(spectrum, ax=ax, legend=False)
    color = CATEGORICAL[1]
    ax.plot(
        peaks["position"],
        peaks["height"],
        "o",
        color=color,
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=3,
        label="peaks",
    )
    half = peaks["height"] - peaks["prominence"] / 2.0
    ax.hlines(half, peaks["left"], peaks["right"], color=color, linewidth=1.0, zorder=2)
    if annotate:
        for pos, height in zip(peaks["position"], peaks["height"]):
            ax.annotate(
                f"{pos:.4g}",
                (pos, height),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize="small",
                color=INK_SECONDARY,
            )
    return ax


def plot_fit(
    result: FitResult,
    ax=None,
    residuals: bool = True,
    components: bool = True,
    xlabel: str = "",
    ylabel: str = "",
    scale: float = 1.0,
):
    """Data, fitted curve, optional components and a residual panel.

    Returns the main axes, or ``(main, residual)`` axes when a residual panel is drawn.
    """
    axr = None
    if ax is None and residuals:
        fig, (ax, axr) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(6.4, 5.0),
            layout="constrained",
            gridspec_kw={"height_ratios": [3, 1]},
        )
        style_axes(axr)
    ax = _axes(ax)
    order = np.argsort(result.x)
    x = result.x[order]
    ax.plot(
        x,
        scale * result.y[order],
        "o",
        color=MUTED,
        markersize=3,
        alpha=0.7,
        label="data",
        zorder=2,
    )
    ax.plot(
        x,
        scale * result.best_fit[order],
        color=CATEGORICAL[0],
        linewidth=2.0,
        label="fit",
        zorder=3,
    )
    if components and len(result.components) > 1:
        for (name, curve), color in zip(result.components.items(), CATEGORICAL[1:]):
            ax.plot(
                x,
                scale * np.asarray(curve)[order],
                color=color,
                linewidth=1.0,
                label=name,
                zorder=2,
            )
    ax.set_ylabel(ylabel)
    _legend(ax, 2)
    if axr is not None:
        axr.plot(x, scale * result.residuals[order], color=MUTED, linewidth=1.0)
        _zero_line(axr)
        axr.set_ylabel("residual")
        axr.set_xlabel(xlabel)
        return ax, axr
    ax.set_xlabel(xlabel)
    return ax


def plot_tauc(result: TaucResult, ax=None):
    """Tauc plot with the fitted linear region and its extrapolation to the band gap."""
    ax = _axes(ax)
    n = TAUC_EXPONENTS[result.transition]
    exponent = {2.0: "2", 0.5: "1/2"}.get(n, f"{n:.3g}")
    ax.plot(result.energy, result.tauc, color=CATEGORICAL[0], linewidth=1.5, label="data")
    e = np.linspace(result.band_gap, result.fit_range[1], 50)
    ax.plot(e, result.line(e), color=CATEGORICAL[1], linewidth=1.5, label="linear fit")
    ax.plot(
        [result.band_gap], [0.0], "o", color=CATEGORICAL[1], markersize=6, markeredgecolor="white"
    )
    ax.annotate(
        f"$E_g$ = {result.band_gap:.3f} eV",
        (result.band_gap, 0.0),
        xytext=(8, 10),
        textcoords="offset points",
        color=INK_SECONDARY,
    )
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel(rf"$(\alpha h\nu)^{{{exponent}}}$")
    _legend(ax, 2)
    return ax


def plot_calibration(result: CalibrationResult, ax=None, unit: str = "M"):
    """Calibration standards with the regression line."""
    ax = _axes(ax)
    c = result.concentrations
    ax.plot(
        c,
        result.absorbances,
        "o",
        color=CATEGORICAL[0],
        markersize=6,
        markeredgecolor="white",
        label="standards",
        zorder=3,
    )
    cc = np.linspace(0.0, c.max() * 1.05, 50)
    ax.plot(
        cc,
        result.slope * cc + result.intercept,
        color=CATEGORICAL[1],
        linewidth=1.5,
        label=f"A = {result.slope:.4g} c + {result.intercept:.3g}  (R$^2$ = {result.r_squared:.4f})",
    )
    ax.set_xlabel(f"Concentration ({unit})")
    ax.set_ylabel("Absorbance")
    _legend(ax, 2)
    return ax


# --------------------------------------------------------------------------
# Transient absorption
# --------------------------------------------------------------------------
def cell_edges(centers):
    """Cell boundaries for pcolormesh-style plotting of (possibly non-uniform) cell centres."""
    c = np.asarray(centers, dtype=float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[1:] + c[:-1])
    return np.r_[2 * c[0] - mid[0], mid, 2 * c[-1] - mid[-1]]


def _contiguous_blocks(x, factor=3.0):
    """Index ranges of ``x`` without gaps larger than ``factor`` times the median step."""
    if x.size < 2:
        return [slice(0, x.size)]
    breaks = np.flatnonzero(np.diff(x) > factor * np.median(np.diff(x))) + 1
    bounds = np.r_[0, breaks, x.size]
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:])]


def plot_ta_map(
    data: TAData,
    ax=None,
    scale: float = 1e3,
    vmax: float | None = None,
    vmin: float | None = None,
    linthresh: float | None = None,
    colorbar: bool = True,
    units: str = "mOD",
):
    """False-colour map of dA(wavelength, delay) on the true (non-uniform) delay grid.

    Parameters
    ----------
    scale : float
        Multiplier for display (``1e3`` shows mOD).
    vmax, vmin : float, optional
        Colour limits (default: symmetric, from :func:`robust_limit`).
    linthresh : float, optional
        Use a symmetric-log delay axis, linear within ``+/- linthresh`` ps -
        the usual way to show femtosecond and nanosecond dynamics together.
    """
    ax = _axes(ax, figsize=(7.0, 4.6))
    z = scale * data.dA
    if vmax is None:
        vmax = robust_limit(z)
    vmin = -vmax if vmin is None else vmin
    norm = TwoSlopeNorm(vcenter=0.0, vmin=min(vmin, -1e-12), vmax=max(vmax, 1e-12))
    t_edges = cell_edges(data.delays)
    mesh = None
    for rows in _contiguous_blocks(data.wavelengths):
        mesh = ax.pcolormesh(
            t_edges,
            cell_edges(data.wavelengths[rows]),
            z[rows],
            cmap=DIVERGING,
            norm=norm,
            shading="flat",
            rasterized=True,
        )
    ax.grid(False)
    if linthresh:
        ax.set_xscale("symlog", linthresh=linthresh, linscale=1.0)
    ax.set_xlabel("Delay (ps)")
    ax.set_ylabel("Wavelength (nm)")
    if data.name:
        ax.set_title(data.name, fontsize="medium", loc="left")
    if colorbar and mesh is not None:
        cbar = ax.figure.colorbar(mesh, ax=ax, label=rf"$\Delta A$ ({units})", extend="both")
        cbar.outline.set_edgecolor(AXIS)
        cbar.ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY)
        cbar.ax.yaxis.label.set_color(INK_SECONDARY)
    return ax


def plot_kinetics(
    data: TAData,
    wavelengths: Sequence[float],
    width: float = 0.0,
    ax=None,
    fits: Sequence[FitResult] | None = None,
    scale: float = 1e3,
    linthresh: float | None = None,
    units: str = "mOD",
):
    """Kinetic traces at the given wavelengths, with fitted curves if ``fits`` is given."""
    ax = _axes(ax)
    wavelengths = list(wavelengths)
    for i, (wl, color) in enumerate(zip(wavelengths, series_colors(len(wavelengths)))):
        t, y = data.kinetic(wl, width)
        label = f"{wl:g} nm"
        if fits is not None:
            ax.plot(t, scale * y, "o", color=color, markersize=2.5, alpha=0.6, zorder=2)
            order = np.argsort(fits[i].x)
            ax.plot(
                fits[i].x[order],
                scale * fits[i].best_fit[order],
                color=color,
                linewidth=2.0,
                label=label,
                zorder=3,
            )
        else:
            ax.plot(t, scale * y, color=color, linewidth=1.5, label=label)
    _zero_line(ax)
    if linthresh:
        ax.set_xscale("symlog", linthresh=linthresh, linscale=1.0)
    ax.set_xlim(data.delays[0], data.delays[-1])
    ax.set_xlabel("Delay (ps)")
    ax.set_ylabel(rf"$\Delta A$ ({units})")
    _legend(ax, len(wavelengths))
    return ax


def plot_ta_spectra(
    data: TAData,
    delays: Sequence[float],
    width: float = 0.0,
    ax=None,
    scale: float = 1e3,
    units: str = "mOD",
):
    """Transient spectra at the given delays, coloured along a ramp (early = light, late = dark)."""
    ax = _axes(ax)
    delays = list(delays)
    for t, color in zip(delays, series_colors(len(delays), ordered=True)):
        wl, s = data.spectrum(t, width)
        for rows in _contiguous_blocks(wl):
            ax.plot(
                wl[rows],
                scale * s[rows],
                color=color,
                linewidth=1.5,
                label=f"{t:g} ps" if rows.start == 0 else None,
            )
    _zero_line(ax)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(rf"$\Delta A$ ({units})")
    _legend(ax, len(delays))
    return ax


def plot_chirp(wavelengths, t0, model=None, ax=None):
    """Time-zero estimates versus wavelength and the fitted dispersion curve."""
    ax = _axes(ax)
    ok = np.isfinite(t0)
    ax.plot(
        np.asarray(wavelengths)[ok],
        np.asarray(t0)[ok],
        "o",
        color=CATEGORICAL[0],
        markersize=5,
        markeredgecolor="white",
        label="time-zero estimates",
    )
    if model is not None:
        wl = np.linspace(np.min(wavelengths), np.max(wavelengths), 200)
        ax.plot(wl, model(wl), color=CATEGORICAL[1], linewidth=1.5, label="polynomial fit")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Time zero (ps)")
    _legend(ax, 2 if model is not None else 1)
    return ax


def plot_das(
    result: GlobalFitResult, ax=None, eads: bool = False, scale: float = 1e3, units: str = "mOD"
):
    """Decay-associated (or evolution-associated) spectra from a global fit."""
    ax = _axes(ax)
    spectra = result.eads() if eads else result.das
    labels = result.labels
    if eads:
        labels = [f"species {chr(65 + i)} ({lab.split('= ')[-1]})" for i, lab in enumerate(labels)]
    colors = series_colors(len(spectra))
    for spec, label, color in zip(spectra, labels, colors):
        for rows in _contiguous_blocks(result.wavelengths):
            ax.plot(
                result.wavelengths[rows],
                scale * spec[rows],
                color=color,
                linewidth=1.5,
                label=label if rows.start == 0 else None,
            )
    _zero_line(ax)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(f"{'EADS' if eads else 'DAS'} ({units})")
    _legend(ax, len(spectra))
    return ax


def plot_singular_values(data: TAData, n: int = 15, ax=None):
    """Scree plot of the singular values of the TA matrix (components above the noise floor)."""
    ax = _axes(ax)
    _, s, _ = data.svd()
    k = np.arange(1, min(n, s.size) + 1)
    ax.semilogy(
        k,
        s[: k.size],
        "o-",
        color=CATEGORICAL[0],
        markersize=5,
        linewidth=1.0,
        markeredgecolor="white",
    )
    ax.set_xlabel("Component")
    ax.set_ylabel("Singular value")
    return ax
