import json

import numpy as np
import pytest

import spectroscopy_toolset as st
from conftest import EXAMPLE_DATA
from spectroscopy_toolset import io as stio


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_tab_separated_with_header(tmp_path):
    f = write(tmp_path / "a.txt", "Wavelength (nm)\tAbs\n400\t0.1\n401\t0.2\n402\t0.3\n")
    table, header = stio.load_table(f)
    assert table.shape == (3, 2)
    assert header == ["Wavelength (nm)", "Abs"]


def test_cary_like_csv_descending_with_trailing_commas(tmp_path):
    text = "Sample1,\nWavelength (nm),Abs,\n802,0.05,\n801,0.06,\n800,0.07,\n\nMethod: scan\nAvg time 0.1\n"
    spec = st.read_spectrum(write(tmp_path / "cary.csv", text))
    np.testing.assert_allclose(spec.x, [800, 801, 802])
    np.testing.assert_allclose(spec.y, [0.07, 0.06, 0.05])
    assert spec.x_label == "Wavelength (nm)"


def test_semicolon_decimal_comma_with_metadata(tmp_path):
    text = "Instrument: X\nDate: today\nnm;A\n300,5;0,125\n301,0;0,250\n301,5;0,375\n"
    table, header = stio.load_table(write(tmp_path / "eu.csv", text))
    np.testing.assert_allclose(table, [[300.5, 0.125], [301.0, 0.25], [301.5, 0.375]])
    assert header == ["nm", "A"]


def test_whitespace_decimal_comma(tmp_path):
    table, _ = stio.load_table(write(tmp_path / "ws.txt", "1,5 2,5\n3,5 4,5\n5,5 6,5\n"))
    np.testing.assert_allclose(table, [[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])


def test_no_numeric_data(tmp_path):
    with pytest.raises(ValueError):
        stio.load_table(write(tmp_path / "bad.txt", "hello\nworld\n"))


def test_read_spectra_layouts(tmp_path):
    xyy = write(tmp_path / "xyy.csv", "nm,Sample A,Sample B\n400,1,2\n401,3,4\n402,5,6\n")
    spectra = st.read_spectra(xyy)
    assert [s.name for s in spectra] == ["Sample A", "Sample B"]
    np.testing.assert_allclose(spectra[1].y, [2, 4, 6])
    xyxy = write(
        tmp_path / "xyxy.csv", "Wavelength (nm),Abs,Wavelength (nm),Abs\n400,1,400,2\n401,3,401,4\n"
    )
    spectra = st.read_spectra(xyxy)
    assert len(spectra) == 2
    assert [s.name for s in spectra] == ["xyxy[0]", "xyxy[1]"]


def test_spectrum_roundtrip(tmp_path):
    spec = st.Spectrum([3, 1, 2], [30, 10, 20], name="s", y_label="Abs")
    path = tmp_path / "s.csv"
    spec.save(path)
    back = st.read_spectrum(path)
    np.testing.assert_allclose(back.x, [1, 2, 3])
    np.testing.assert_allclose(back.y, [10, 20, 30])
    assert back.y_label == "Abs"


def test_ta_matrix_roundtrip_and_transpose(tmp_path, synthetic_ta):
    path = tmp_path / "ta.dat"
    synthetic_ta.save(path)
    back = st.read_ta(path)
    np.testing.assert_allclose(back.dA, synthetic_ta.dA, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(back.delays, synthetic_ta.delays)
    table, _ = stio.load_table(path)
    np.savetxt(tmp_path / "ta_t.dat", table.T, delimiter="\t")
    flipped = st.read_ta_matrix(tmp_path / "ta_t.dat", transpose=True)
    np.testing.assert_allclose(flipped.dA, back.dA)


def test_read_ta_averages_scans(tmp_path, synthetic_ta):
    a = synthetic_ta.copy(dA=synthetic_ta.dA + 1e-4)
    b = synthetic_ta.copy(dA=synthetic_ta.dA - 1e-4)
    a.save(tmp_path / "a.dat")
    b.save(tmp_path / "b.dat")
    avg = st.read_ta([tmp_path / "a.dat", tmp_path / "b.dat"])
    np.testing.assert_allclose(avg.dA, synthetic_ta.dA, atol=1e-9)
    np.testing.assert_allclose(avg.std, 1e-4, rtol=1e-3)  # std (ddof=1) sqrt(2)e-4 / sqrt(2 scans)
    assert avg.meta["n_scans"] == 2

    shifted = synthetic_ta.shift_time(-1.9)
    shifted.save(tmp_path / "c.dat")
    with pytest.raises(ValueError, match="offset by"):
        st.read_ta([tmp_path / "a.dat", tmp_path / "c.dat"])


def test_read_ta_scan_json(tmp_path):
    wl = [400.0, 410.0, 420.0]
    t_fs = [-100.0, 0.0, 100.0, 1000.0]
    dA = [[i * 10 + j for j in range(3)] for i in range(4)]  # [delay][wavelength]
    dA[0][0] = None
    path = tmp_path / "x.scan"
    path.write_text(json.dumps([[wl], [t_fs], dA, [[0.0, 0.1, 0.2]]]))
    data = st.read_ta(path)
    assert data.shape == (3, 4)
    np.testing.assert_allclose(data.delays, [-0.1, 0.0, 0.1, 1.0])
    assert data.dA[0, 0] == 0.0  # None/NaN replaced
    assert data.dA[2, 3] == 32
    np.testing.assert_allclose(data.meta["background"], [0.0, 0.1, 0.2])


def test_write_ta_xyz(tmp_path):
    data = st.TAData([500, 510], [0.0, 1.0, 2.0], [[1e-3, 2e-3, 3e-3], [4e-3, 5e-3, 6e-3]])
    path = tmp_path / "x.xyz"
    data.save_xyz(path)
    blocks = path.read_text().strip().split("\n\n")
    assert len(blocks) == 2
    first = np.loadtxt(blocks[0].splitlines())
    np.testing.assert_allclose(first, [[500, 0, 1], [500, 1, 2], [500, 2, 3]])


@pytest.mark.skipif(
    not (EXAMPLE_DATA / "TestData_1.dat").exists(), reason="example data not present"
)
def test_example_data_loads():
    data = st.read_ta(EXAMPLE_DATA / "TestData_1.dat")
    assert data.shape == (2068, 576)
    assert data.wavelengths[0] == pytest.approx(431.783521)
    assert data.delays[0] == pytest.approx(-285.5)
