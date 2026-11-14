"""Build the small committed test fixture (tests/fixtures/raw) from the downloaded raw data.

Photodiode: 9 real welds (3 per quality class), CSV files gzip-compressed but unmodified.
Quality table: the real xlsx reduced to those 9 rows.
Camera: 3 real trials, every 20th frame kept, stored in the original per-pixel HDF5 layout
(the subsampling lowers the effective frame rate, so camera spectral features on the fixture
are only a code-path check).
Protocol: the real protocol workbook (copied unchanged).

Usage: python scripts/make_fixture.py [data/raw] [tests/fixtures/raw]
"""

from __future__ import annotations

import gzip
import shutil
import sys
from pathlib import Path

import h5py
import pandas as pd

from weldfusion.readers import read_camera

WELDS = {"A": [507, 511, 517], "C": [666, 667, 669, 671, 675, 676]}
CAMERA_TRIALS = [3, 9, 14]
STEP = 20


def main(src: Path, dst: Path) -> None:
    ch_src, ch_dst = src / "chianese2022", dst / "chianese2022"
    if dst.exists():
        shutil.rmtree(dst)
    sheets = pd.read_excel(ch_src / "Statistic featuress.xlsx", sheet_name=None)
    ch_dst.mkdir(parents=True)
    with pd.ExcelWriter(ch_dst / "Statistic featuress.xlsx") as xw:
        for name, df in sheets.items():
            keep = df[df["Signal ID"].isin(WELDS.get(name, []))]
            keep.to_excel(xw, sheet_name=name, index=False)
            folder = ch_dst / f"Signals - Dataset {name}"
            folder.mkdir()
            for weld in keep["Signal ID"].astype(int):
                raw = ch_src / f"Signals - Dataset {name}" / f"{weld}.csv"
                with open(raw, "rb") as fi, gzip.open(folder / f"{weld}.csv.gz", "wb", 9) as fo:
                    shutil.copyfileobj(fi, fo)

    kn_src, kn_dst = src / "knaak2017", dst / "knaak2017"
    kn_dst.mkdir(parents=True)
    shutil.copy2(kn_src / "2017-06-16_Protocol.xlsx", kn_dst)
    for trial in CAMERA_TRIALS:
        name = f"cam 1750-31_trial_{trial}_offset.dat.h5"
        frames, labels = read_camera(kn_src / name)
        frames, labels = frames[::STEP], labels[::STEP]
        with h5py.File(kn_dst / name, "w") as h5:
            for i in range(frames.shape[1]):
                for j in range(frames.shape[2]):
                    h5.create_dataset(f"[{i}, {j}]", data=frames[:, i, j].astype("float32"))
            h5.create_dataset("label/label", data=labels.reshape(-1, 1).astype("int32"))


if __name__ == "__main__":
    args = sys.argv[1:]
    main(Path(args[0] if args else "data/raw"), Path(args[1] if len(args) > 1 else
                                                     "tests/fixtures/raw"))
