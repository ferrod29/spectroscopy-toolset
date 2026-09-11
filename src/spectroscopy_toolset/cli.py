"""Command-line interface: ``spectro <command> ...`` (``spectro -h`` for help).

Commands
--------
uvvis       plot/process steady-state spectra, optionally list peaks
deconvolve  fit overlapping bands with Gaussian/Lorentzian/Voigt profiles
tauc        optical band gap from a Tauc plot
calibrate   Beer-Lambert calibration curve with LOD/LOQ
ta          TA map, kinetic traces (optionally fitted) and transient spectra
ta-global   global analysis (DAS/EADS) of TA data
ta-export   write processed TA data as a matrix or xyz triplets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__, uvvis
from . import io as stio
from .spectrum import Spectrum
from .transient import TAData, estimate_chirp, fit_chirp


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _figure_output(fig, out: str | None, suffix: str, saved: list[str]) -> None:
    if out is None:
        return
    path = Path(f"{out}_{suffix}.png") if suffix else Path(out)
    if path.suffix == "":
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    saved.append(str(path))


def _finish(args, saved: list[str]) -> None:
    import matplotlib.pyplot as plt

    if getattr(args, "out", None) is None:
        plt.show()
    else:
        for path in saved:
            print(f"saved {path}")
        plt.close("all")


def _setup_matplotlib(args) -> None:
    if getattr(args, "out", None) is not None:
        import matplotlib

        matplotlib.use("Agg")


# --------------------------------------------------------------------------
# UV-vis commands
# --------------------------------------------------------------------------
def _process_spectrum(spec: Spectrum, args) -> Spectrum:
    if args.range:
        spec = spec.crop(*args.range)
    if args.baseline == "als":
        spec = spec.subtract_baseline("als", lam=args.lam, p=args.p)
    elif args.baseline == "poly":
        regions = [tuple(r) for r in args.region] if args.region else None
        spec = spec.subtract_baseline("poly", degree=args.degree, regions=regions)
    elif args.baseline == "offset":
        if args.offset_at is None:
            raise SystemExit("--baseline offset needs --offset-at X or --offset-at X0 X1")
        at = args.offset_at[0] if len(args.offset_at) == 1 else tuple(args.offset_at)
        spec = spec.subtract_baseline("offset", at=at)
    if args.smooth:
        spec = spec.smooth(args.smooth)
    return spec


def _load_spectra(files) -> list[Spectrum]:
    spectra = []
    for f in files:
        spectra.extend(stio.read_spectra(f))
    return spectra


def cmd_uvvis(args) -> int:
    from . import plotting

    spectra = [_process_spectrum(s, args) for s in _load_spectra(args.files)]
    saved: list[str] = []
    ax = plotting.plot_spectra(
        spectra, normalize=args.normalize, energy=args.energy, ordered=args.ordered
    )
    _figure_output(ax.figure, args.out, "spectra" if args.peaks else "", saved)
    if args.save_processed:
        for spec in spectra:
            path = Path(args.save_processed) / f"{spec.name}_processed.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            spec.save(path)
            saved.append(str(path))
    if args.peaks:
        for spec in spectra:
            shown = spec.to_energy() if args.energy else spec
            peaks = shown.find_peaks(prominence=args.prominence, valleys=args.valleys)
            print(f"\n{spec.name}: {len(peaks)} peak(s)")
            print(
                peaks.drop(columns="index").to_string(
                    index=False, float_format=lambda v: f"{v:.5g}"
                )
            )
            ax = plotting.plot_peaks(shown, peaks)
            ax.set_title(spec.name, loc="left", fontsize="medium")
            _figure_output(ax.figure, args.out, f"peaks_{spec.name}", saved)
    _finish(args, saved)
    return 0


def cmd_deconvolve(args) -> int:
    from . import plotting

    spec = stio.read_spectrum(args.file)
    if args.energy:
        spec = spec.to_energy()
    result = uvvis.fit_peaks(
        spec.x,
        spec.y,
        centers=args.centers,
        n_peaks=args.n_peaks,
        shape=args.shape,
        baseline=args.baseline,
        x_range=tuple(args.range) if args.range else None,
    )
    print(result)
    saved: list[str] = []
    ax, _ = plotting.plot_fit(result, xlabel=spec.x_label, ylabel=spec.y_label)
    _figure_output(ax.figure, args.out, "", saved)
    _finish(args, saved)
    return 0


def cmd_tauc(args) -> int:
    from . import plotting

    spec = stio.read_spectrum(args.file)
    result = uvvis.tauc_bandgap(
        spec.x,
        spec.y,
        transition=args.transition,
        fit_range=tuple(args.range) if args.range else None,
        thickness_cm=args.thickness,
    )
    print(result)
    saved: list[str] = []
    ax = plotting.plot_tauc(result)
    _figure_output(ax.figure, args.out, "", saved)
    _finish(args, saved)
    return 0


def cmd_calibrate(args) -> int:
    from . import plotting

    if args.table:
        table, _ = stio.load_table(args.table)
        conc, absorb = table[:, 0], table[:, 1]
    else:
        if not args.conc or not args.abs or len(args.conc) != len(args.abs):
            raise SystemExit(
                "give --conc and --abs with the same number of values, or --table FILE"
            )
        conc, absorb = np.array(args.conc), np.array(args.abs)
    result = uvvis.calibration_curve(
        conc, absorb, path_length=args.path_length, through_origin=args.origin
    )
    print(result)
    if args.unknown:
        for a in args.unknown:
            print(f"  A = {a:.4g} -> c = {float(result.concentration(a)):.4g}")
    saved: list[str] = []
    ax = plotting.plot_calibration(result, unit=args.unit)
    _figure_output(ax.figure, args.out, "", saved)
    _finish(args, saved)
    return 0


# --------------------------------------------------------------------------
# Transient-absorption commands
# --------------------------------------------------------------------------
def _load_ta(args) -> TAData:
    data = stio.read_ta(args.files, transpose=args.transpose, time_scale=args.time_scale)
    if args.zero_first:
        data = data.shift_time(data.delays[0])
    if args.t0 is not None:
        data = data.shift_time(args.t0)
    if args.exclude:
        data = data.exclude_wavelengths(*[tuple(r) for r in args.exclude])
    if args.wl_range:
        data = data.crop(wl=tuple(args.wl_range))
    if args.background_before is not None:
        data = data.subtract_background(args.background_before)
    if args.chirp:
        window = tuple(args.chirp_window) if args.chirp_window else None
        wl, t0 = estimate_chirp(data, window=window, method=args.chirp_method)
        model = fit_chirp(wl, t0, order=args.chirp_order)
        print(f"chirp: {model}")
        data = data.correct_chirp(model)
        data.meta["chirp_estimates"] = (wl, t0)
    if args.bin:
        data = data.bin_wavelengths(width=args.bin)
    if args.t_range:
        data = data.crop(t=tuple(args.t_range))
    return data


def cmd_ta(args) -> int:
    from . import plotting

    data = _load_ta(args)
    print(
        f"{data.name}: {data.shape[0]} wavelengths x {data.shape[1]} delays "
        f"({data.wavelengths[0]:.1f}-{data.wavelengths[-1]:.1f} nm, {data.delays[0]:.3g}-{data.delays[-1]:.4g} ps)"
    )
    saved: list[str] = []
    ax = plotting.plot_ta_map(data, linthresh=args.linthresh, vmax=args.vmax)
    _figure_output(ax.figure, args.out, "map", saved)
    if args.chirp and "chirp_estimates" in data.meta:
        wl, t0 = data.meta["chirp_estimates"]
        ax = plotting.plot_chirp(wl, t0, data.meta.get("chirp_model"))
        _figure_output(ax.figure, args.out, "chirp", saved)
    if args.kinetics:
        fits = None
        if args.fit:
            fits = []
            for wl in args.kinetics:
                result = data.fit_kinetics(
                    wl, args.width, n_exp=args.fit, step=args.step, artifact=args.artifact
                )
                print(f"\n--- {wl:g} nm ---\n{result}")
                fits.append(result)
        ax = plotting.plot_kinetics(
            data, args.kinetics, args.width, fits=fits, linthresh=args.linthresh
        )
        _figure_output(ax.figure, args.out, "kinetics", saved)
    if args.spectra:
        ax = plotting.plot_ta_spectra(data, args.spectra, args.delay_width)
        _figure_output(ax.figure, args.out, "spectra", saved)
    _finish(args, saved)
    return 0


def cmd_ta_global(args) -> int:
    from . import plotting

    data = _load_ta(args)
    result = data.global_fit(
        n_exp=args.n_exp, step=args.step, irf=not args.no_irf, artifact=args.artifact
    )
    print(result)
    saved: list[str] = []
    ax = plotting.plot_das(result)
    _figure_output(ax.figure, args.out, "das", saved)
    if args.eads:
        ax = plotting.plot_das(result, eads=True)
        _figure_output(ax.figure, args.out, "eads", saved)
    residual = data.copy(
        wavelengths=result.wavelengths,
        delays=result.delays,
        dA=result.residuals,
        std=None,
        name=f"{data.name} - residuals",
    )
    ax = plotting.plot_ta_map(residual, linthresh=args.linthresh)
    _figure_output(ax.figure, args.out, "residuals", saved)
    _finish(args, saved)
    return 0


def cmd_ta_export(args) -> int:
    data = _load_ta(args)
    if not args.xyz and not args.matrix:
        raise SystemExit("give --xyz FILE and/or --matrix FILE")
    if args.xyz:
        stio.write_ta_xyz(args.xyz, data, scale=args.scale)
        print(f"saved {args.xyz}")
    if args.matrix:
        stio.write_ta_matrix(args.matrix, data)
        print(f"saved {args.matrix}")
    return 0


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
def _ta_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("TA input and pre-processing (applied in this order)")
    g.add_argument(
        "files",
        nargs="+",
        help="TA matrix files (.dat/.txt/.csv) or .scan files; several are averaged",
    )
    g.add_argument("--transpose", action="store_true", help="delays run down the first column")
    g.add_argument(
        "--time-scale", type=float, default=1.0, help="multiply delays by this (e.g. 1e-3 for fs)"
    )
    g.add_argument("--zero-first", action="store_true", help="shift delays so the first one is 0")
    g.add_argument("--t0", type=float, help="subtract this time zero from the delays")
    g.add_argument(
        "--exclude",
        nargs=2,
        type=float,
        action="append",
        metavar=("MIN", "MAX"),
        help="drop a wavelength band, e.g. pump scatter (repeatable)",
    )
    g.add_argument("--wl-range", nargs=2, type=float, metavar=("MIN", "MAX"))
    g.add_argument(
        "--background-before",
        type=float,
        metavar="T",
        help="subtract the mean signal at delays < T (pre-time-zero background)",
    )
    g.add_argument("--chirp", action="store_true", help="estimate and correct the chirp")
    g.add_argument(
        "--chirp-window",
        nargs=2,
        type=float,
        metavar=("T0", "T1"),
        help="delay window containing the coherent artefact",
    )
    g.add_argument(
        "--chirp-method",
        choices=["onset", "max", "derivative"],
        default="onset",
        help="time-zero estimator (default: half-rise onset)",
    )
    g.add_argument("--chirp-order", type=int, default=2)
    g.add_argument(
        "--bin", type=float, metavar="NM", help="average wavelengths in bins of this width"
    )
    g.add_argument("--t-range", nargs=2, type=float, metavar=("MIN", "MAX"))
    return p


def _uvvis_processing(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("processing")
    g.add_argument("--range", nargs=2, type=float, metavar=("MIN", "MAX"), help="crop the x axis")
    g.add_argument("--baseline", choices=["none", "als", "poly", "offset"], default="none")
    g.add_argument("--lam", type=float, default=1e5, help="ALS smoothness")
    g.add_argument("--p", type=float, default=0.01, help="ALS asymmetry")
    g.add_argument("--degree", type=int, default=1, help="polynomial baseline degree")
    g.add_argument(
        "--region",
        nargs=2,
        type=float,
        action="append",
        metavar=("MIN", "MAX"),
        help="band-free region for the polynomial baseline (repeatable)",
    )
    g.add_argument(
        "--offset-at",
        nargs="+",
        type=float,
        metavar="X",
        help="subtract y at X (or its mean over X0 X1)",
    )
    g.add_argument("--smooth", type=int, metavar="POINTS", help="Savitzky-Golay window")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spectro", description="UV-vis and transient-absorption analysis"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    out_help = "save figures (PNG) using this path/prefix instead of showing them"

    p = sub.add_parser("uvvis", help="plot and process steady-state spectra")
    p.add_argument("files", nargs="+")
    _uvvis_processing(p)
    p.add_argument("--normalize", choices=["max", "absmax", "minmax", "area"])
    p.add_argument("--energy", action="store_true", help="plot versus photon energy (eV)")
    p.add_argument("--ordered", action="store_true", help="colour spectra along a ramp (series)")
    p.add_argument("--peaks", action="store_true", help="find and list peaks")
    p.add_argument("--prominence", type=float, help="minimum peak prominence")
    p.add_argument("--valleys", action="store_true", help="find minima instead of maxima")
    p.add_argument("--save-processed", metavar="DIR", help="write processed spectra as CSV")
    p.add_argument("--out", help=out_help)
    p.set_defaults(func=cmd_uvvis)

    p = sub.add_parser("deconvolve", help="band deconvolution")
    p.add_argument("file")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--n-peaks", type=int)
    group.add_argument("--centers", nargs="+", type=float)
    p.add_argument("--shape", choices=["gaussian", "lorentzian", "voigt"], default="gaussian")
    p.add_argument("--baseline", choices=["none", "constant", "linear"], default="none")
    p.add_argument("--range", nargs=2, type=float, metavar=("MIN", "MAX"))
    p.add_argument("--energy", action="store_true", help="fit on a photon-energy axis (eV)")
    p.add_argument("--out", help=out_help)
    p.set_defaults(func=cmd_deconvolve)

    p = sub.add_parser("tauc", help="optical band gap from a Tauc plot")
    p.add_argument("file")
    p.add_argument("--transition", choices=sorted(uvvis.TAUC_EXPONENTS), default="direct")
    p.add_argument(
        "--range", nargs=2, type=float, metavar=("EMIN", "EMAX"), help="linear region in eV"
    )
    p.add_argument(
        "--thickness", type=float, metavar="CM", help="film thickness for alpha = ln10 A / d"
    )
    p.add_argument("--out", help=out_help)
    p.set_defaults(func=cmd_tauc)

    p = sub.add_parser("calibrate", help="Beer-Lambert calibration curve")
    p.add_argument("--conc", nargs="+", type=float, help="standard concentrations")
    p.add_argument("--abs", nargs="+", type=float, help="standard absorbances")
    p.add_argument("--table", help="two-column file: concentration, absorbance")
    p.add_argument("--path-length", type=float, default=1.0, help="cuvette path length (cm)")
    p.add_argument("--origin", action="store_true", help="force the line through the origin")
    p.add_argument("--unknown", nargs="+", type=float, help="absorbances of unknowns to quantify")
    p.add_argument("--unit", default="M", help="concentration unit for the axis label")
    p.add_argument("--out", help=out_help)
    p.set_defaults(func=cmd_calibrate)

    ta_parent = _ta_parent()
    p = sub.add_parser("ta", parents=[ta_parent], help="TA map, kinetics and spectra")
    p.add_argument(
        "--kinetics", nargs="+", type=float, metavar="NM", help="wavelengths of kinetic traces"
    )
    p.add_argument("--width", type=float, default=0.0, help="average kinetics over +/- width/2 nm")
    p.add_argument(
        "--fit", type=int, metavar="N", help="fit kinetics with N exponentials (Gaussian IRF)"
    )
    p.add_argument("--step", action="store_true", help="add a long-lived component to fits")
    p.add_argument(
        "--artifact",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="model the coherent artefact with N Gaussian-derivative terms",
    )
    p.add_argument(
        "--spectra", nargs="+", type=float, metavar="PS", help="delays of transient spectra"
    )
    p.add_argument(
        "--delay-width", type=float, default=0.0, help="average spectra over +/- width/2 ps"
    )
    p.add_argument(
        "--linthresh", type=float, metavar="PS", help="symmetric-log delay axis, linear below this"
    )
    p.add_argument("--vmax", type=float, help="colour limit of the map (mOD)")
    p.add_argument("--out", help=out_help)
    p.set_defaults(func=cmd_ta)

    p = sub.add_parser("ta-global", parents=[ta_parent], help="global analysis (DAS/EADS)")
    p.add_argument("--n-exp", type=int, default=2)
    p.add_argument("--step", action="store_true", help="add a long-lived component")
    p.add_argument(
        "--no-irf", action="store_true", help="fit without IRF convolution (t >= t0 only)"
    )
    p.add_argument(
        "--artifact",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="model the coherent artefact with N Gaussian-derivative terms",
    )
    p.add_argument("--eads", action="store_true", help="also plot evolution-associated spectra")
    p.add_argument("--linthresh", type=float, metavar="PS")
    p.add_argument("--out", help=out_help)
    p.set_defaults(func=cmd_ta_global)

    p = sub.add_parser("ta-export", parents=[ta_parent], help="export processed TA data")
    p.add_argument("--xyz", help="write 'wavelength delay value' triplets (gnuplot/Origin)")
    p.add_argument("--matrix", help="write the processed matrix")
    p.add_argument(
        "--scale", type=float, default=1e3, help="value multiplier for --xyz (1e3 = mOD)"
    )
    p.set_defaults(func=cmd_ta_export)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _setup_matplotlib(args)
    try:
        return args.func(args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
