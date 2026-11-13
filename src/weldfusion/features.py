"""Time- and frequency-domain features from photodiode signals (SciPy) and melt-pool
geometry/intensity features from coaxial camera frames (OpenCV)."""

from __future__ import annotations

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy import signal, stats

from .paths import Paths
from .readers import PHOTODIODE_CHANNELS

META_COLUMNS = (
    "weld_id",
    "source",
    "quality_label",
    "laser_power_w",
    "gap_um",
    "penetration_um",
    "throat_um",
)


def laser_on_segment(an_power: np.ndarray, fraction: float) -> slice:
    """Longest contiguous run where the analogue laser power command is on."""
    on = an_power > fraction * np.percentile(an_power, 99)
    if not on.any():
        return slice(0, len(an_power))
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(np.int8), 0]))
    starts, ends = edges[::2], edges[1::2]
    k = int(np.argmax(ends - starts))
    return slice(int(starts[k]), int(ends[k]))


def spectral_features(x: np.ndarray, fs: float, nperseg: int, bands: list[float]) -> dict:
    freqs, psd = signal.welch(x - x.mean(), fs=fs, nperseg=min(nperseg, len(x)))
    total = psd.sum()
    out: dict[str, float] = {}
    if total <= 0:
        for lo, hi in zip(bands[:-1], bands[1:], strict=True):
            out[f"band_{int(lo)}_{int(hi)}"] = np.nan
        return out | {"spec_centroid_hz": np.nan, "spec_peak_hz": np.nan, "spec_entropy": np.nan}
    for lo, hi in zip(bands[:-1], bands[1:], strict=True):
        out[f"band_{int(lo)}_{int(hi)}"] = psd[(freqs >= lo) & (freqs < hi)].sum() / total
    p = psd / total
    out["spec_centroid_hz"] = float((freqs * p).sum())
    out["spec_peak_hz"] = float(freqs[1:][np.argmax(psd[1:])])
    out["spec_entropy"] = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(p)))
    return out


def channel_features(x: np.ndarray, fs: float, cfg: dict) -> dict:
    rms = float(np.sqrt(np.mean(x**2)))
    mean = float(x.mean())
    p01, p05, p95, p99 = np.percentile(x, [1, 5, 95, 99])
    out = {
        "mean": mean,
        "std": float(x.std()),
        "rms": rms,
        "cv": float(x.std() / mean) if mean else np.nan,
        "skew": float(stats.skew(x)),
        "kurtosis": float(stats.kurtosis(x)),
        "p05": float(p05),
        "p95": float(p95),
        "range_p01_p99": float(p99 - p01),
        "crest": float(x.max() / rms) if rms else np.nan,
        "roughness": float(np.mean(np.abs(np.diff(x)))),
    }
    return out | spectral_features(x, fs, cfg["welch_nperseg"], cfg["bands_hz"])


def photodiode_features(seg: np.ndarray, fs: float, cfg: dict) -> dict:
    """Features of one (n, 3) block of plasma / temperature / back-reflection samples."""
    feats: dict[str, float] = {}
    for c, name in enumerate(PHOTODIODE_CHANNELS):
        for k, v in channel_features(seg[:, c], fs, cfg).items():
            feats[f"pd_{name}_{k}"] = v
    pairs = ((0, 1), (0, 2), (1, 2))
    for a, b in pairs:
        na, nb = PHOTODIODE_CHANNELS[a], PHOTODIODE_CHANNELS[b]
        feats[f"pd_corr_{na}_{nb}"] = float(np.corrcoef(seg[:, a], seg[:, b])[0, 1])
        mb = seg[:, a].mean()
        feats[f"pd_ratio_{nb}_{na}"] = float(seg[:, b].mean() / mb) if mb else np.nan
    return feats


def melt_pool_frame(frame: np.ndarray, threshold: float, up: int) -> dict:
    """Geometry and intensity of the melt pool in one coaxial camera frame."""
    img = np.clip(frame, 0, None).astype(np.float32)
    big = cv2.resize(img, None, fx=up, fy=up, interpolation=cv2.INTER_LINEAR)
    mask = (big > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hot = img > threshold
    out = {
        "max_intensity": float(img.max()),
        "mean_intensity": float(img.mean()),
        "pool_intensity": float(img[hot].mean()) if hot.any() else 0.0,
        "pool_integral": float(img[hot].sum()),
        "area_px": 0.0,
        "perimeter_px": np.nan,
        "circularity": np.nan,
        "major_axis_px": np.nan,
        "minor_axis_px": np.nan,
        "aspect_ratio": np.nan,
        "centroid_x": np.nan,
        "centroid_y": np.nan,
    }
    if not contours:
        return out
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    out["area_px"] = area / up**2
    out["perimeter_px"] = perim / up
    out["circularity"] = 4 * np.pi * area / perim**2 if perim else np.nan
    m = cv2.moments(c)
    if m["m00"]:
        out["centroid_x"] = m["m10"] / m["m00"] / up
        out["centroid_y"] = m["m01"] / m["m00"] / up
    if len(c) >= 5:
        (_, _), (ax1, ax2), _ = cv2.fitEllipse(c)
        major, minor = max(ax1, ax2) / up, min(ax1, ax2) / up
        out["major_axis_px"], out["minor_axis_px"] = major, minor
        out["aspect_ratio"] = major / minor if minor else np.nan
    return out


def camera_features(frames: np.ndarray, fs: float, cfg: dict) -> tuple[dict, pd.DataFrame]:
    per_frame = pd.DataFrame(
        [melt_pool_frame(f, cfg["melt_pool_threshold"], cfg["upsample"]) for f in frames]
    )
    feats: dict[str, float] = {"cam_pool_present_frac": float((per_frame.area_px > 0).mean())}
    for col in per_frame.columns:
        feats[f"cam_{col}_mean"] = float(per_frame[col].mean())
        feats[f"cam_{col}_std"] = float(per_frame[col].std())
    bands = [0, 10, 50, fs / 2]
    spec = spectral_features(per_frame.pool_integral.to_numpy(), fs, 256, bands)
    feats |= {f"cam_integral_{k}": v for k, v in spec.items()}
    return feats, per_frame


def run(paths: Paths, params: dict) -> None:
    cfg = params["features"]
    welds = pd.read_parquet(paths.fused / "welds.parquet")
    out = paths.features
    (out / "camera_examples").mkdir(parents=True, exist_ok=True)
    windows, weld_rows, cam_rows = [], [], []
    with h5py.File(paths.fused / "welds.h5", "r") as h5:
        for w in welds.itertuples(index=False):
            meta = {k: getattr(w, k) for k in META_COLUMNS}
            if w.has_photodiode:
                ds = h5[f"photodiode/{w.weld_id}"]
                data, fs = ds[:], float(ds.attrs["fs_hz"])
                seg = data[laser_on_segment(data[:, 3], cfg["laser_on_fraction"])][:, :3]
                weld_rows.append(
                    meta | {"on_duration_ms": len(seg) / fs * 1e3}
                    | photodiode_features(seg, fs, cfg)
                )
                win = int(round(cfg["window_ms"] / 1000 * fs))
                for i in range(len(seg) // win):
                    block = seg[i * win : (i + 1) * win]
                    windows.append(meta | {"window": i} | photodiode_features(block, fs, cfg))
            if w.has_camera:
                frames = h5[f"camera/{w.weld_id}"][:]
                on = h5[f"camera_labels/{w.weld_id}"][:] != 0
                feats, per_frame = camera_features(frames[on], float(w.fs_hz), cfg)
                row = meta | {"focal_offset_mm": w.focal_offset_mm, "n_frames": int(on.sum())}
                cam_rows.append(row | feats)
                mean_img = np.clip(frames[on].mean(axis=0), 0, None)
                img = cv2.resize(
                    mean_img / max(float(mean_img.max()), 1e-6) * 255,
                    (256, 256),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.uint8)
                cv2.imwrite(
                    str(out / "camera_examples" / f"{w.weld_id}.png"),
                    cv2.applyColorMap(img, cv2.COLORMAP_INFERNO),
                )
    pd.DataFrame(windows).to_parquet(out / "photodiode_windows.parquet", index=False)
    pd.DataFrame(weld_rows).to_parquet(out / "photodiode_welds.parquet", index=False)
    pd.DataFrame(cam_rows).to_parquet(out / "camera_welds.parquet", index=False)
    print(
        f"[features] {len(weld_rows)} photodiode welds -> {len(windows)} windows; "
        f"{len(cam_rows)} camera welds"
    )
