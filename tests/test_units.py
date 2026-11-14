import cv2
import numpy as np
import pytest

from weldfusion.features import laser_on_segment, melt_pool_frame, spectral_features
from weldfusion.readers import _num


def test_laser_on_segment_picks_longest_run():
    an = np.r_[np.zeros(10), np.full(5, 3.0), np.zeros(5), np.full(50, 3.0), np.zeros(10)]
    assert laser_on_segment(an, 0.5) == slice(20, 70)


def test_spectral_peak_matches_sine():
    fs = 50_000.0
    t = np.arange(20_000) / fs
    x = np.sin(2 * np.pi * 3_000 * t)
    out = spectral_features(x, fs, 1024, [0, 500, 2000, 8000, 25000])
    assert out["spec_peak_hz"] == pytest.approx(3_000, abs=fs / 1024)
    assert out["band_2000_8000"] > 0.95


def test_melt_pool_geometry_of_ellipse():
    frame = np.zeros((32, 32), np.float32)
    cv2.ellipse(frame, (16, 16), (8, 4), 0, 0, 360, 200.0, -1)
    out = melt_pool_frame(frame, threshold=60, up=4)
    assert out["area_px"] == pytest.approx(np.pi * 8 * 4, rel=0.2)
    assert out["aspect_ratio"] == pytest.approx(2.0, rel=0.2)
    assert out["centroid_x"] == pytest.approx(16, abs=1)


def test_protocol_power_parsing():
    assert _num("0,6*2,5") == pytest.approx(1.5)
    assert _num(2.5) == pytest.approx(2.5)
