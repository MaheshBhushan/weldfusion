"""Fuse both public datasets into one versioned per-weld dataset.

Outputs (data/fused/):
  welds.parquet  one row per weld: source, process parameters, quality labels, modality flags
  welds.h5       per-weld arrays: /photodiode/<weld_id> (N, 4) and /camera/<weld_id> (T, 32, 32)
  manifest.json  schema version, content-derived dataset version, provenance and counts
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .paths import Paths
from .readers import (
    PHOTODIODE_CHANNELS,
    camera_trial,
    find_signal,
    read_camera,
    read_knaak_protocol,
    read_photodiode,
    read_quality_table,
)
from .sources import CHIANESE, KNAAK

PHOTODIODE_COLUMNS = (*PHOTODIODE_CHANNELS, "an_power")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _chianese_rows(raw: Path, h5: h5py.File) -> list[dict]:
    table = read_quality_table(raw / "Statistic featuress.xlsx")
    rows = []
    for weld_id, grp in table.groupby("weld_id", sort=True):
        first = grp.iloc[0]
        # Welds 340-344 are listed in both subset A and B; they must agree.
        same = grp[["laser_power_w", "gap_um", "quality_label"]].nunique(dropna=False)
        if (same > 1).any():
            raise ValueError(f"Conflicting duplicate rows for weld {weld_id}")
        path = next(
            (p for p in (find_signal(raw, s, weld_id) for s in grp.subset) if p is not None), None
        )
        if path is None:
            raise FileNotFoundError(f"No signal file for weld {weld_id} in {raw}")
        meta, sig = read_photodiode(path)
        key = f"CH-{weld_id}"
        ds = h5.create_dataset(
            f"photodiode/{key}",
            data=sig[list(PHOTODIODE_COLUMNS)].to_numpy(np.float32),
            compression="gzip",
            compression_opts=4,
        )
        ds.attrs["fs_hz"] = meta["fs_hz"]
        ds.attrs["columns"] = list(PHOTODIODE_COLUMNS)
        rows.append(
            {
                "weld_id": key,
                "source": CHIANESE.key,
                "source_doi": CHIANESE.doi,
                "subset": "+".join(sorted(grp.subset.unique())),
                "material": "Cu-to-steel thin-foil battery tab (remote laser welding)",
                "laser_power_w": first.laser_power_w,
                "gap_um": first.gap_um,
                "focal_offset_mm": np.nan,
                "penetration_um": first.penetration_um,
                "throat_um": first.throat_um,
                "quality_label": first.quality_label,
                "has_photodiode": True,
                "has_camera": False,
                "n_samples": len(sig),
                "fs_hz": meta["fs_hz"],
                "recorder_config": meta.get("Configuration", ""),
                "recorded_at": meta.get("Date", ""),
                "raw_file": str(path.relative_to(raw.parent)),
                "raw_sha256": _sha256(path),
            }
        )
    return rows


def _knaak_rows(raw: Path, h5: h5py.File) -> list[dict]:
    protocol = read_knaak_protocol(raw / "2017-06-16_Protocol.xlsx").set_index("trial")
    rows = []
    for path in sorted(raw.glob("*.h5"), key=camera_trial):
        trial = camera_trial(path)
        frames, labels = read_camera(path)
        key = f"KN-T{trial:02d}"
        h5.create_dataset(f"camera/{key}", data=frames, compression="gzip", compression_opts=4)
        h5.create_dataset(f"camera_labels/{key}", data=labels.astype(np.int16))
        p = protocol.loc[trial]
        on = labels != 0
        rows.append(
            {
                "weld_id": key,
                "source": KNAAK.key,
                "source_doi": KNAAK.doi,
                "subset": "trial",
                "material": p.sheets,
                "laser_power_w": p.laser_power_w,
                "gap_um": p.gap_um,
                "focal_offset_mm": p.focal_offset_mm,
                "penetration_um": np.nan,
                "throat_um": np.nan,
                "quality_label": None,
                "has_photodiode": False,
                "has_camera": True,
                "n_samples": int(on.sum()),
                "fs_hz": 500.0,
                "recorder_config": f"camera label code {int(labels[on][0])}",
                "recorded_at": "2017-06-16",
                "raw_file": str(path.relative_to(raw.parent)),
                "raw_sha256": _sha256(path),
            }
        )
    return rows


def run(paths: Paths, params: dict) -> pd.DataFrame:
    out = paths.fused
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(out / "welds.h5", "w") as h5:
        rows = _chianese_rows(paths.raw / CHIANESE.key, h5)
        if (paths.raw / KNAAK.key).exists():
            rows += _knaak_rows(paths.raw / KNAAK.key, h5)
    welds = pd.DataFrame(rows)
    welds.to_parquet(out / "welds.parquet", index=False)

    schema = params["fuse"]["schema_version"]
    digest = hashlib.sha256(f"schema={schema}".encode())
    for weld_id, sha in sorted(zip(welds.weld_id, welds.raw_sha256, strict=True)):
        digest.update(f"{weld_id}:{sha}".encode())
    manifest = {
        "schema_version": schema,
        "dataset_version": digest.hexdigest()[:16],
        "n_welds": len(welds),
        "welds_per_source": welds.source.value_counts().sort_index().to_dict(),
        "quality_labels": welds.quality_label.value_counts().sort_index().to_dict(),
        "modalities": {
            "photodiode": int(welds.has_photodiode.sum()),
            "coaxial_camera": int(welds.has_camera.sum()),
            "process_parameters": int(welds.laser_power_w.notna().sum()),
            "quality_label": int(welds.quality_label.notna().sum()),
            "photodiode_and_camera": int((welds.has_photodiode & welds.has_camera).sum()),
        },
        "sources": [
            {
                "key": s.key,
                "doi": s.doi,
                "license": s.license,
                "citation": s.citation,
                "modalities": list(s.modalities),
            }
            for s in (CHIANESE, KNAAK)
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[fuse] {len(welds)} welds, dataset_version={manifest['dataset_version']}")
    return welds
