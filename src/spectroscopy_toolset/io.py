"""Readers and writers for spectra and transient-absorption data files.

The text reader sniffs the delimiter (tab, semicolon, comma or whitespace) and
decimal separator (``.`` or ``,``) and skips header/footer lines, so exports
from most spectrometer software (Cary, Shimadzu, PerkinElmer, Ocean Optics,
plain CSV/TXT) load without options.
"""

from __future__ import annotations

import json
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
def read_ta_matrix(
    path, transpose: bool = False, time_scale: float = 1.0, name: str | None = None
) -> TAData:
    """Read a TA matrix file.

    Layout: the first row holds the delays (after a corner cell), the first
    column the wavelengths and the rest ``dA`` - the layout of the example
    files (Pharos-based TA setup) and of :func:`write_ta_matrix`.
    Use ``transpose=True`` for files with delays down the first column, and
    ``time_scale`` to convert delays to ps (e.g. ``1e-3`` for fs).
    """
    table, _ = load_table(path)
    if transpose:
        table = table.T
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError(f"{path}: not a TA matrix (shape {table.shape})")
    return TAData(
        wavelengths=table[1:, 0],
        delays=table[0, 1:] * time_scale,
        dA=table[1:, 1:],
        name=name or Path(path).stem,
        meta={"source": [str(path)]},
    )


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


def read_ta(paths, **kwargs) -> TAData:
    """Read one or several TA scans and average them.

    ``.scan`` files are read with :func:`read_ta_scan`, anything else with
    :func:`read_ta_matrix` (``kwargs`` are passed on). Several scans must share
    their axes; the result carries the standard error of the mean in ``std``.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = list(paths)
    if not paths:
        raise ValueError("no files given")
    scans = [
        read_ta_scan(p) if Path(p).suffix.lower() == ".scan" else read_ta_matrix(p, **kwargs)
        for p in paths
    ]
    return scans[0] if len(scans) == 1 else average_scans(scans)


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
