"""Readers and writers for spectra and transient-absorption data files.

The text reader sniffs the delimiter (tab, semicolon, comma or whitespace) and
decimal separator (``.`` or ``,``) and skips header/footer lines, so exports
from most spectrometer software (Cary, Shimadzu, PerkinElmer, Ocean Optics,
plain CSV/TXT) load without options.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

from .spectrum import Spectrum
from .transient import TAData, average_scans

__all__ = [
    "load_table",
    "read_spectrum",
    "read_spectra",
    "write_spectrum",
    "detect_ta_format",
    "read_harbor",
    "read_harbor_spectrum",
    "read_harpia",
    "read_helios",
    "read_helios_raw",
    "read_ta_matrix",
    "read_ta_scan",
    "read_ta",
    "write_ta_matrix",
    "write_ta_xyz",
]

_DELIMITERS = ("\t", ";", ",", None)  # None = any run of whitespace


# --------------------------------------------------------------------------
# Generic numeric tables
# --------------------------------------------------------------------------
def _split(line: str, delimiter):
    parts = line.split(delimiter) if delimiter is not None else line.split()
    parts = [p.strip() for p in parts]
    while parts and parts[-1] == "":
        parts.pop()
    return parts


def _to_floats(parts, decimal):
    try:
        if decimal == ",":
            return [float(p.replace(",", ".")) for p in parts]
        return [float(p) for p in parts]
    except ValueError:
        return None


def _numeric(line, delimiter, decimal):
    parts = _split(line, delimiter)
    if len(parts) < 2:
        return None
    return _to_floats(parts, decimal)


def _sniff(lines):
    """Pick the (delimiter, decimal) pair that parses the most lines into >= 2 numbers."""
    nonempty = [ln for ln in lines if ln.strip()]
    sample = nonempty[:300] + nonempty[-50:]
    best = None
    for delimiter in _DELIMITERS:
        for decimal in (".", ","):
            if decimal == delimiter:
                continue
            counts = Counter()
            for line in sample:
                values = _numeric(line, delimiter, decimal)
                if values is not None:
                    counts[len(values)] += 1
            if counts:
                ncols, nrows = counts.most_common(1)[0]
                score = (nrows, ncols)
                if best is None or score > best[0]:
                    best = (score, delimiter, decimal)
    if best is None:
        raise ValueError("no numeric data with at least two columns found")
    return best[1], best[2]


def load_table(path) -> tuple[np.ndarray, list[str]]:
    """Read the numeric block of a delimited text file.

    Returns the table as a 2-D float array and the column names taken from the
    header line directly above the data (empty if there is none).
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    delimiter, decimal = _sniff(lines)

    def is_data(line):
        return _numeric(line, delimiter, decimal) is not None

    first = next(i for i, line in enumerate(lines) if is_data(line))
    last = next(i for i in range(len(lines) - 1, -1, -1) if is_data(lines[i]))
    header = []
    for line in reversed(lines[:first]):
        if line.strip():
            header = _split(line, delimiter)
            break

    block = lines[first : last + 1]
    if decimal == ",":
        block = [line.replace(",", ".") for line in block]
    try:  # fast path for clean rectangular blocks (e.g. large TA matrices)
        table = np.loadtxt(block, delimiter=delimiter, ndmin=2)
    except ValueError:
        rows = [v for v in (_numeric(line, delimiter, ".") for line in block) if v is not None]
        ncols = Counter(len(r) for r in rows).most_common(1)[0][0]
        table = np.array([r for r in rows if len(r) == ncols], dtype=float)
    if len(header) != table.shape[1]:
        header = []
    return table, header


# --------------------------------------------------------------------------
# Spectra
# --------------------------------------------------------------------------
def read_spectrum(
    path,
    x_col: int = 0,
    y_col: int = 1,
    name: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
) -> Spectrum:
    """Read one spectrum (two columns of a delimited text file)."""
    table, header = load_table(path)
    if max(x_col, y_col) >= table.shape[1]:
        raise ValueError(f"{path}: has only {table.shape[1]} columns")
    return Spectrum(
        table[:, x_col],
        table[:, y_col],
        name=name or Path(path).stem,
        x_label=x_label or (header[x_col] if header else "Wavelength (nm)"),
        y_label=y_label or (header[y_col] if header else "Absorbance"),
        meta={"source": str(path)},
    )


def read_spectra(path, layout: str = "auto") -> list[Spectrum]:
    """Read every spectrum stored in a multi-column file.

    ``layout`` is ``'xyy'`` (one x column followed by several y columns),
    ``'xyxy'`` (x/y column pairs, as in Cary multi-sample exports) or
    ``'auto'`` (``'xyxy'`` when every even column repeats the first one).
    """
    table, header = load_table(path)
    stem = Path(path).stem
    ncols = table.shape[1]
    if layout == "auto":
        pairs = ncols >= 4 and ncols % 2 == 0
        layout = (
            "xyxy"
            if pairs and all(np.allclose(table[:, j], table[:, 0]) for j in range(2, ncols, 2))
            else "xyy"
        )
    if layout == "xyxy":
        cols = [(j, j + 1) for j in range(0, ncols - 1, 2)]
    elif layout == "xyy":
        cols = [(0, j) for j in range(1, ncols)]
    else:
        raise ValueError("layout must be 'auto', 'xyy' or 'xyxy'")
    spectra = []
    for k, (xc, yc) in enumerate(cols):
        name = stem if len(cols) == 1 else f"{stem}[{k}]"
        if header:
            # Use a column title as the name unless it is a generic axis label.
            for candidate in (header[yc], header[xc]):
                if candidate and not _is_generic_label(candidate):
                    name = candidate
                    break
        x_label = header[xc] if header and _is_generic_label(header[xc]) else "Wavelength (nm)"
        spectra.append(
            Spectrum(
                table[:, xc],
                table[:, yc],
                name=name,
                x_label=x_label,
                meta={"source": str(path), "column": yc},
            )
        )
    return spectra


def _is_generic_label(label: str) -> bool:
    label = label.strip().lower()
    axis_words = ("wavelength", "(nm)", "energy", "(ev)", "cm-1", "wavenumber")
    return label in ("x", "y", "a", "abs", "absorbance", "t", "%t", "abs.") or any(
        w in label for w in axis_words
    )


def write_spectrum(path, spectrum: Spectrum, delimiter: str = ",") -> None:
    """Write a spectrum as two labelled columns."""
    header = f"{spectrum.x_label}{delimiter}{spectrum.y_label}"
    np.savetxt(
        path,
        np.column_stack([spectrum.x, spectrum.y]),
        delimiter=delimiter,
        header=header,
        comments="",
        fmt="%.8g",
    )


# --------------------------------------------------------------------------
# Transient absorption
# --------------------------------------------------------------------------
def _finite_matrix(wl, t, dA):
    """Drop wavelengths/delays without any finite value and zero the remaining NaN/inf."""
    finite = np.isfinite(dA)
    rows = finite.any(axis=1) & np.isfinite(wl)
    cols = finite.any(axis=0) & np.isfinite(t)
    return rows, cols, np.nan_to_num(dA[np.ix_(rows, cols)], nan=0.0, posinf=0.0, neginf=0.0)


def _clean(data: TAData) -> TAData:
    rows, cols, dA = _finite_matrix(data.wavelengths, data.delays, data.dA)
    if rows.all() and cols.all() and np.isfinite(data.dA).all():
        return data
    std = None if data.std is None else np.nan_to_num(data.std[np.ix_(rows, cols)])
    return data.copy(wavelengths=data.wavelengths[rows], delays=data.delays[cols], dA=dA, std=std)


def read_ta_matrix(
    path,
    transpose: bool = False,
    time_scale: float = 1.0,
    dA_scale: float = 1.0,
    name: str | None = None,
    clean: bool = True,
) -> TAData:
    """Read a TA matrix file.

    Layout: the first row holds the delays (after a corner cell), the first
    column the wavelengths and the rest ``dA`` - the layout of the example
    files (Pharos-based TA setup), of HELIOS exports and of
    :func:`write_ta_matrix`. Use ``transpose=True`` for files with delays down
    the first column, ``time_scale`` to convert delays to ps (e.g. ``1e-3``
    for fs) and ``dA_scale`` to convert signals to OD (``1e-3`` for mOD).
    With ``clean`` (default), wavelengths or delays without any finite value
    are dropped and the remaining NaN/inf values become zero.
    """
    table, _ = load_table(path)
    if transpose:
        table = table.T
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError(f"{path}: not a TA matrix (shape {table.shape})")
    with np.errstate(invalid="ignore"):
        dA = table[1:, 1:] * dA_scale
    data = TAData(
        table[1:, 0],
        table[0, 1:] * time_scale,
        dA,
        name=name or Path(path).stem,
        meta={"source": [str(path)]},
    )
    return _clean(data) if clean else data


# --------------------------------------------------------------------------
# Instrument formats
# --------------------------------------------------------------------------
def read_helios(path, name: str | None = None, clean: bool = True) -> TAData:
    """Read a HELIOS (Ultrafast Systems) dA matrix: wavelengths (nm) down the
    first column, delays (ps) along the first row, dA in OD.

    Detector pixels without data (all NaN) are dropped (see :func:`read_ta_matrix`).
    """
    return read_ta_matrix(path, name=name, clean=clean)


def read_helios_raw(
    pump_on, pump_off, delay_tolerance: float = 0.05, name: str | None = None
) -> TAData:
    """dA = -log10(I_on / I_off) from HELIOS raw probe-intensity files.

    ``pump_on``/``pump_off`` are the ``..._RAW_Signal Pump on.dat`` and
    ``..._RAW_Signal Pump off.dat`` files of one scan (delays down the first
    column), or equal-length lists of them. Several scans are averaged; their
    measured delays may differ by up to ``delay_tolerance`` ps.
    """
    if isinstance(pump_on, (str, Path)):
        pump_on, pump_off = [pump_on], [pump_off]
    if len(pump_on) != len(pump_off):
        raise ValueError("give as many pump-off files as pump-on files")
    scans = []
    for on_path, off_path in zip(pump_on, pump_off):
        on, _ = load_table(on_path)
        off, _ = load_table(off_path)
        if on.shape != off.shape or not np.allclose(on[0], off[0]):
            raise ValueError(f"{on_path} and {off_path} have different axes")
        with np.errstate(divide="ignore", invalid="ignore"):
            dA = -np.log10(on[1:, 1:] / off[1:, 1:]).T
        stem = Path(on_path).stem.replace("_RAW_Signal Pump on", "")
        meta = {"source": [str(on_path), str(off_path)]}
        scans.append(TAData(on[0, 1:], on[1:, 0], dA, name=stem, meta=meta))
    data = _clean(average_scans(scans, delay_tolerance=delay_tolerance))
    return data.copy(name=name) if name else data


_TIME_UNITS = {"fs": 1e-3, "ps": 1.0, "ns": 1e3, "us": 1e6, "µs": 1e6}


def read_harpia(path, name: str | None = None, clean: bool = True) -> TAData:
    """Read a HARPIA (Light Conversion) ``*_matrix.dat`` export.

    Layout: ``XAxisTitle``/``YAxisTitle`` lines, then wavelengths along the
    first row and one row per delay. dA is stored in mOD and converted to OD;
    the delay unit is taken from ``YAxisTitle`` (ps if not stated).
    """
    time_scale = 1.0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for _ in range(4):
            line = fh.readline()
            if line.startswith("YAxisTitle") and "(" in line:
                unit = line[line.rfind("(") + 1 : line.rfind(")")].strip()
                time_scale = _TIME_UNITS.get(unit, 1.0)
    return read_ta_matrix(
        path, transpose=True, time_scale=time_scale, dA_scale=1e-3, name=name, clean=clean
    )


def _harbor_table(path):
    """Header row (pixel numbers) and data rows of a HARBOR file, with the first channel's width."""
    table, _ = load_table(path)
    header = table[0, 1:]
    restarts = np.flatnonzero(np.diff(header) < 0)  # pixel numbering restarts at each channel
    n_pix = int(restarts[0]) + 1 if restarts.size else header.size
    return header[:n_pix], table[1:], n_pix


def _common_pixels(axes) -> np.ndarray:
    """Pixels present in every file (the channel width is 580 or 600 depending on the readout)."""
    common = axes[0]
    for px in axes[1:]:
        common = np.intersect1d(common, px)
    if common.size == 0:
        raise ValueError("the files share no pixels")
    return common


def read_harbor_spectrum(path, channel: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pixel numbers and the row-averaged signal of one channel of a HARBOR file.

    Used for static spectra such as the filter measurements of the wavelength
    calibration (:func:`spectroscopy_toolset.calibration.calibrate_harbor`).
    """
    pixels, rows, n_pix = _harbor_table(path)
    block = rows[:, 1 + channel * n_pix : 1 + (channel + 1) * n_pix]
    with np.errstate(invalid="ignore"):
        return pixels, np.nanmean(np.where(np.isfinite(block), block, np.nan), axis=0)


def _harbor_sweeps(rows, sweeps: str):
    """Split the data rows of a scan into the forward sweep and the reversed backward sweep."""
    t = rows[:, 0]
    half = t.size // 2
    fwd, bwd = rows[:half], rows[half:][::-1]
    step = np.median(np.abs(np.diff(t[:half]))) if half > 1 else 0.0
    paired = t.size % 2 == 0 and half > 1 and np.allclose(fwd[:, 0], bwd[:, 0], atol=step)
    if not paired:
        if sweeps != "both":
            raise ValueError("the file holds a single sweep; use sweeps='both'")
        return [rows]
    return {"both": [fwd, bwd], "forward": [fwd], "backward": [bwd]}[sweeps]


def read_harbor(
    paths,
    calibration=None,
    pixel_range: tuple[float, float] | None = None,
    sweeps: str = "both",
    channel: int = 0,
    dA_scale: float = 1e-3,
    name: str | None = None,
) -> TAData:
    """Read and average HARBOR TA scans.

    Each file holds one scan: a header row of pixel numbers (1-600, repeated
    for every recorded channel) and one row per delay with the measured delay
    (fs) followed by the channels. The first ``channel`` holds dA in mOD. The
    delay stage runs forward and then back, so the second half of the rows is
    the backward sweep.

    ``sweeps='both'`` treats every forward and backward sweep of every file as
    one measurement; the result is their mean with the standard error in
    ``std``, and delays are the mean of the measured delays (the spread is
    stored in ``meta['delay_spread_ps']``). Use ``'forward'`` or
    ``'backward'`` to keep one direction.

    Without ``calibration`` the wavelength axis holds pixel numbers.
    ``calibration`` may be a :class:`~spectroscopy_toolset.calibration.PrismCalibration`,
    the path of one saved as JSON, or a text file / array with one wavelength
    per pixel (for the pixels in ``pixel_range`` or for all pixels).
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = list(paths)
    if not paths:
        raise ValueError("no files given")
    if sweeps not in ("both", "forward", "backward"):
        raise ValueError("sweeps must be 'both', 'forward' or 'backward'")
    tables = [_harbor_table(path) for path in paths]
    pixels = _common_pixels([px for px, _, _ in tables])
    blocks = []
    for path, (px, rows, n_pix) in zip(paths, tables):
        cols = 1 + channel * n_pix + np.searchsorted(px, pixels)
        for sweep in _harbor_sweeps(rows, sweeps):
            if blocks and sweep.shape[0] != blocks[0].shape[0]:
                raise ValueError(f"{path} has a different number of delays than {paths[0]}")
            blocks.append(np.column_stack([sweep[:, 0], sweep[:, cols]]))
    stack = np.stack(blocks)  # (sweep, delay, 1 + pixel)
    delays_all = stack[:, :, 0] * 1e-3  # fs -> ps
    signal = np.where(np.isfinite(stack[:, :, 1:]), stack[:, :, 1:], np.nan) * dA_scale

    keep = np.ones(pixels.size, dtype=bool)
    if pixel_range is not None:
        keep &= (pixels >= min(pixel_range)) & (pixels <= max(pixel_range))
    wavelengths = _harbor_wavelengths(calibration, pixels, keep)
    keep &= np.isfinite(wavelengths) & np.isfinite(signal).any(axis=(0, 1))
    signal = signal[:, :, keep]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN entries become 0 below
        mean = np.nanmean(signal, axis=0)
        n = np.isfinite(signal).sum(axis=0)
        sem = np.nanstd(signal, axis=0, ddof=1) / np.sqrt(n) if len(blocks) > 1 else None
    meta = {
        "source": [str(p) for p in paths],
        "n_scans": len(blocks),
        "delay_spread_ps": float(np.ptp(delays_all, axis=0).max()),
        "pixels": pixels[keep],
        "wavelength_axis": "pixel" if calibration is None else "nm",
    }
    return TAData(
        wavelengths=wavelengths[keep],
        delays=delays_all.mean(axis=0),
        dA=np.nan_to_num(mean.T),
        std=None if sem is None else np.nan_to_num(sem.T),
        name=name
        or Path(paths[0]).stem + (f" (mean of {len(blocks)} sweeps)" if len(blocks) > 1 else ""),
        meta=meta,
    )


def _harbor_wavelengths(calibration, pixels, keep) -> np.ndarray:
    if calibration is None:
        return pixels.astype(float)
    if isinstance(calibration, (str, Path)):
        if Path(calibration).suffix.lower() == ".json":
            from .calibration import PrismCalibration

            calibration = PrismCalibration.load(calibration)
        else:  # one wavelength per line, or (pixel, wavelength) columns
            try:
                calibration = load_table(calibration)[0][:, -1]
            except ValueError:
                calibration = np.loadtxt(calibration, ndmin=1)
    if callable(calibration):
        return np.asarray(calibration(pixels), dtype=float)
    wl = np.asarray(calibration, dtype=float).ravel()
    out = np.full(pixels.size, np.nan)
    if wl.size == pixels.size:
        out[:] = wl
    elif wl.size == keep.sum():
        out[keep] = wl
    else:
        raise ValueError(
            f"the calibration has {wl.size} wavelengths for {pixels.size} pixels "
            f"({keep.sum()} in pixel_range)"
        )
    return out


def detect_ta_format(path) -> str:
    """Guess the TA file format: ``'scan'``, ``'harpia'``, ``'harbor'`` or ``'matrix'``."""
    path = Path(path)
    if path.suffix.lower() == ".scan":
        return "scan"
    with open(path, encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
    if first.startswith("XAxisTitle"):
        return "harpia"
    if first.startswith("Pump-probe"):
        raise ValueError(
            f"{path.name} is a raw HARPIA data file; load the '*_matrix.dat' export instead"
        )
    tokens = first.split()
    head = _to_floats(tokens[1:4], ".") if len(tokens) > 600 else None
    if head == [1.0, 2.0, 3.0]:  # scan number, then pixel numbers 1, 2, 3, ...
        return "harbor"
    return "matrix"


def _vector(entry) -> np.ndarray:
    arr = np.asarray(entry, dtype=float)
    return arr.ravel() if arr.ndim <= 1 or arr.shape[0] == 1 else arr[0]


def read_ta_scan(path, name: str | None = None) -> TAData:
    """Read the JSON ``.scan`` format: ``[[wavelengths], [delays in fs], dA[delay][wavelength], [background]]``.

    Delays are converted from fs to ps; NaN values are replaced by zero.
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        raw = [raw[k] for k in sorted(raw, key=lambda k: int(k))]
    wl = _vector(raw[0])
    delays = 1e-3 * _vector(raw[1])
    dA = np.nan_to_num(np.asarray(raw[2], dtype=float))
    if dA.shape == (delays.size, wl.size):
        dA = dA.T
    meta = {"source": [str(path)]}
    if len(raw) > 3:
        meta["background"] = _vector(raw[3])
    return TAData(wl, delays, dA, name=name or Path(path).stem, meta=meta)


def read_ta(paths, format: str = "auto", delay_tolerance: float = 0.0, **kwargs) -> TAData:
    """Read one or several TA scans and average them.

    ``format`` is ``'auto'`` (see :func:`detect_ta_format`), ``'matrix'``
    (:func:`read_ta_matrix`, also HELIOS exports), ``'scan'``, ``'harpia'`` or
    ``'harbor'``. ``kwargs`` go to the reader: ``transpose``/``time_scale``/
    ``dA_scale`` for matrices, ``calibration``/``pixel_range``/``sweeps`` for
    HARBOR files. Several scans must share their axes (up to
    ``delay_tolerance`` ps of delay jitter, see :func:`average_scans`); the
    result carries the standard error of the mean in ``std``.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = list(paths)
    if not paths:
        raise ValueError("no files given")
    formats = [detect_ta_format(p) if format == "auto" else format for p in paths]
    if len(set(formats)) > 1:
        raise ValueError(f"cannot average files of different formats: {sorted(set(formats))}")
    fmt = formats[0]
    if fmt == "harbor":
        return read_harbor(paths, **kwargs)
    readers = {
        "matrix": read_ta_matrix,
        "helios": read_helios,
        "scan": read_ta_scan,
        "harpia": read_harpia,
    }
    if fmt not in readers:
        raise ValueError(f"unknown TA format {fmt!r}")
    if len(paths) == 1:
        return readers[fmt](paths[0], **kwargs)
    if fmt != "scan":  # clean after averaging: scans may flag different pixels as NaN
        kwargs = {**kwargs, "clean": False}
    scans = [readers[fmt](p, **kwargs) for p in paths]
    return _clean(average_scans(scans, delay_tolerance=delay_tolerance))


def write_ta_matrix(path, data: TAData, delimiter: str = "\t", fmt: str = "%.6e") -> None:
    """Write a TA matrix (delays in the first row, wavelengths in the first column)."""
    table = np.zeros((data.wavelengths.size + 1, data.delays.size + 1))
    table[0, 1:] = data.delays
    table[1:, 0] = data.wavelengths
    table[1:, 1:] = data.dA
    np.savetxt(path, table, delimiter=delimiter, fmt=fmt)


def write_ta_xyz(path, data: TAData, scale: float = 1e3, fmt: str = "%.6g") -> None:
    """Write ``wavelength delay scale*dA`` triplets.

    One block per wavelength separated by blank lines - the grid format read
    by gnuplot ``splot``/``pm3d`` and by Origin's XYZ import. ``scale=1e3``
    writes mOD.
    """
    n_t = data.delays.size
    with open(path, "w", encoding="utf-8") as fh:
        for wl, row in zip(data.wavelengths, data.dA):
            block = np.column_stack([np.full(n_t, wl), data.delays, scale * row])
            np.savetxt(fh, block, fmt=fmt, delimiter="\t")
            fh.write("\n")
