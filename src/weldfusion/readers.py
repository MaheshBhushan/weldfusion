"""Readers for the raw files exactly as published on Zenodo."""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from openpyxl import load_workbook

PHOTODIODE_CHANNELS = ("plasma", "temperature", "back_reflection")
_HEADER_LINES = 9  # "Key: value" lines before the column header


def read_photodiode(path: Path) -> tuple[dict, pd.DataFrame]:
    """Read one Precitec LWM export (`<id>.csv`, optionally gzip-compressed).

    Column names differ between dataset A/B ("P Eps") and C ("Plasma"), so the raw
    channels are taken by position: Time, P Raw, T Raw, BR Raw, AN Power.
    """
    meta: dict[str, str] = {}
    df = pd.read_csv(path, sep=";", skiprows=_HEADER_LINES)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for _ in range(_HEADER_LINES):
            key, _, value = fh.readline().partition(":")
            meta[key.strip()] = value.strip()
    cols = df.columns
    out = pd.DataFrame(
        {
            "time_s": df[cols[1]].to_numpy(float),
            "plasma": df[cols[2]].to_numpy(float),
            "temperature": df[cols[4]].to_numpy(float),
            "back_reflection": df[cols[6]].to_numpy(float),
            "an_power": df[cols[8]].to_numpy(float),
        }
    )
    rate = re.match(r"([\d.]+)\s*kHz", meta.get("Sample Rate", ""))
    meta["fs_hz"] = float(rate.group(1)) * 1000 if rate else 1.0 / np.median(np.diff(out.time_s))
    return meta, out


def read_quality_table(xlsx: Path) -> pd.DataFrame:
    """Per-weld process parameters and quality labels ("Statistic featuress.xlsx")."""
    frames = []
    for sheet, df in pd.read_excel(xlsx, sheet_name=None).items():
        df = df.dropna(subset=["Label"])
        lower = {c.lower(): c for c in df.columns}

        def col(prefix: str, _lower=lower, _df=df) -> pd.Series:
            match = [orig for low, orig in _lower.items() if low.startswith(prefix)]
            return _df[match[0]].astype(float) if match else pd.Series(np.nan, index=_df.index)

        frames.append(
            pd.DataFrame(
                {
                    "weld_id": df["Signal ID"].astype(int),
                    "subset": sheet,
                    "laser_power_w": col("laser power").round(1),
                    "gap_um": col("part-to-part gap").fillna(col("effective gap")),
                    "penetration_um": col("weld pen"),
                    "throat_um": col("thick. at throat"),
                    "quality_label": df["Label"].astype(str).str.strip(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def find_signal(raw_dir: Path, subset: str, weld_id: int) -> Path | None:
    folder = raw_dir / f"Signals - Dataset {subset}"
    for name in (f"{weld_id}.csv", f"{weld_id}.csv.gz"):
        if (folder / name).exists():
            return folder / name
    return None


def read_camera(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a coaxial IR camera recording: frames (T, 32, 32) and per-frame label codes.

    The HDF5 stores one dataset per pixel named "[row, col]" holding its time series.
    """
    with h5py.File(path, "r") as h5:
        names = [n for n in h5.keys() if n.startswith("[")]
        idx = [tuple(int(v) for v in n.strip("[]").split(",")) for n in names]
        rows = max(i for i, _ in idx) + 1
        cols = max(j for _, j in idx) + 1
        t = h5[names[0]].shape[0]
        frames = np.empty((t, rows, cols), dtype=np.float32)
        for name, (i, j) in zip(names, idx, strict=True):
            frames[:, i, j] = h5[name][:]
        labels = h5["label/label"][:, 0].astype(int)
    return frames, labels


def camera_trial(path: Path) -> int:
    return int(re.search(r"trial_(\d+)", path.name).group(1))


def _num(text: object) -> float:
    s = str(text).replace(",", ".")
    m = re.match(r"\s*([\d.]+)\s*(?:\*\s*([\d.]+))?", s)
    if not m:
        return float("nan")
    return float(m.group(1)) * (float(m.group(2)) if m.group(2) else 1.0)


def read_knaak_protocol(xlsx: Path) -> pd.DataFrame:
    """Parse the `trial_N` rows of the German welding protocol (laser power, focal offset)."""
    ws = load_workbook(xlsx, read_only=True, data_only=True).worksheets[0]
    rows = []
    for r in ws.iter_rows(values_only=True):
        name = str(r[0] or "")
        m = re.match(r"trial_(\d+)\b", name)
        if not m:
            continue
        remark = str(r[42] or "") if len(r) > 42 else ""
        z = re.search(r"z\s*=\s*([+-]?\d+(?:[.,]\d+)?)", remark)
        gap = re.search(r"(\d+,\d+)\s*mm\s*spalt", str(r[1] or ""))
        rows.append(
            {
                "trial": int(m.group(1)),
                "laser_power_w": round(_num(r[10]) * 1000, 1),
                "focal_offset_mm": float(z.group(1).replace(",", ".")) if z else np.nan,
                "gap_um": float(gap.group(1).replace(",", ".")) * 1000 if gap else np.nan,
                "sheets": str(r[1] or ""),
                "protocol_note": name,
            }
        )
    return pd.DataFrame(rows)
