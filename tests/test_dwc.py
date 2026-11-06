from pathlib import Path

import numpy as np
import pytest

from tadpole.io.dwc import centred_delays, read_dwc, write_dwc

REAL = Path(r"C:\Users\Daniel\Documents\Lillypad Comparison\2026-10-22_Lillypad.dwc")


def test_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    T = rng.uniform(0, 4000, (37, 11))
    wl = np.linspace(380, 420, 37)
    p = tmp_path / "t.dwc"
    write_dwc(p, T, wl, 2.5)
    raw = read_dwc(p)
    np.testing.assert_allclose(raw.trace, T, rtol=1e-6)
    np.testing.assert_allclose(raw.wavelengths_nm, wl, rtol=1e-6)
    np.testing.assert_allclose(raw.delays_fs, centred_delays(11, 2.5))
    assert raw.delay_step_fs == pytest.approx(2.5)


def test_labview_number_format(tmp_path):
    p = tmp_path / "t.dwc"
    write_dwc(p, np.array([[184.3471, 0.0]]), [400.0], 1.0)
    text = p.read_bytes().decode("ascii")
    assert "\r\n" in text
    assert "1.843471E+2" in text and "E+02" not in text


def test_malformed(tmp_path):
    p = tmp_path / "bad.dwc"
    p.write_text("Number of delay points = 3\nNumber of wavelength points = 2\n"
                 "Delay increment = 1\n\n[Wavelength vector]\n1\t2\n\n[Data array]\n1\t2\t3\n")
    with pytest.raises(ValueError, match="expected 2×3"):
        read_dwc(p)


@pytest.mark.skipif(not REAL.exists(), reason="reference Lillypad file not on this machine")
def test_reads_real_lillypad_file():
    raw = read_dwc(REAL)
    assert raw.trace.shape == (3953, 201)
    assert raw.delay_step_fs == pytest.approx(1.0)
    assert np.all(np.diff(raw.wavelengths_nm) > 0)
