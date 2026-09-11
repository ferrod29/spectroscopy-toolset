import numpy as np
import pytest

from spectroscopy_toolset import io as stio
from spectroscopy_toolset import models
from spectroscopy_toolset.cli import main


@pytest.fixture
def spectrum_file(tmp_path):
    x = np.linspace(300, 800, 501)
    y = models.gaussian(x, 1.0, 420, 25) + 0.6 * models.gaussian(x, 1.0, 520, 35) + 0.02
    path = tmp_path / "spec.csv"
    np.savetxt(path, np.c_[x, y], delimiter=",", header="Wavelength (nm),Abs", comments="")
    return path


@pytest.fixture
def ta_file(tmp_path, synthetic_ta):
    path = tmp_path / "ta.dat"
    stio.write_ta_matrix(path, synthetic_ta)
    return path


def test_uvvis_command(tmp_path, spectrum_file, capsys):
    out = tmp_path / "fig"
    assert (
        main(
            [
                "uvvis",
                str(spectrum_file),
                "--baseline",
                "offset",
                "--offset-at",
                "800",
                "--peaks",
                "--save-processed",
                str(tmp_path / "proc"),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert (tmp_path / "fig_spectra.png").exists()
    assert (tmp_path / "fig_peaks_spec.png").exists()
    assert (tmp_path / "proc" / "spec_processed.csv").exists()
    assert "peak(s)" in capsys.readouterr().out


def test_deconvolve_tauc_calibrate(tmp_path, spectrum_file, capsys):
    assert (
        main(
            [
                "deconvolve",
                str(spectrum_file),
                "--n-peaks",
                "2",
                "--baseline",
                "constant",
                "--out",
                str(tmp_path / "dec.png"),
            ]
        )
        == 0
    )
    assert (tmp_path / "dec.png").exists()
    assert (
        main(["tauc", str(spectrum_file), "--range", "2.2", "2.6", "--out", str(tmp_path / "tauc")])
        == 0
    )
    assert (tmp_path / "tauc.png").exists()
    assert (
        main(
            [
                "calibrate",
                "--conc",
                "0",
                "1",
                "2",
                "3",
                "--abs",
                "0.01",
                "0.21",
                "0.4",
                "0.61",
                "--unknown",
                "0.3",
                "--out",
                str(tmp_path / "cal"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Eg" in out and "LOD" in out and "-> c =" in out


def test_ta_commands(tmp_path, ta_file, capsys):
    prefix = str(tmp_path / "ta")
    assert (
        main(
            [
                "ta",
                str(ta_file),
                "--background-before",
                "-0.5",
                "--bin",
                "10",
                "--kinetics",
                "520",
                "620",
                "--fit",
                "2",
                "--spectra",
                "0.5",
                "10",
                "--linthresh",
                "1",
                "--out",
                prefix,
            ]
        )
        == 0
    )
    for suffix in ("map", "kinetics", "spectra"):
        assert (tmp_path / f"ta_{suffix}.png").exists()
    assert main(["ta-global", str(ta_file), "--n-exp", "3", "--eads", "--out", prefix]) == 0
    assert (tmp_path / "ta_das.png").exists() and (tmp_path / "ta_eads.png").exists()
    assert "tau3" in capsys.readouterr().out
    assert (
        main(
            [
                "ta-export",
                str(ta_file),
                "--exclude",
                "500",
                "520",
                "--xyz",
                str(tmp_path / "x.xyz"),
                "--matrix",
                str(tmp_path / "m.dat"),
            ]
        )
        == 0
    )
    exported = stio.read_ta(tmp_path / "m.dat")
    assert not np.any((exported.wavelengths >= 500) & (exported.wavelengths <= 520))


def test_errors_are_reported(tmp_path, capsys):
    assert main(["uvvis", str(tmp_path / "missing.csv"), "--out", str(tmp_path / "x")]) == 1
    assert "error" in capsys.readouterr().err
