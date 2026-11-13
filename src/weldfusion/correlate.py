"""Correlation of weld-level features with seam quality and process parameters."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from .paths import Paths

# Ordinal encoding of the seam quality classes by heat input:
# lack of connection < sound weld < over-penetration.
QUALITY_ORDINAL = {"LoC": 0, "SW": 1, "OP": 2}


def _feature_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in df.columns if c.startswith(prefix)]


def spearman_table(
    df: pd.DataFrame, features: list[str], target: str, modality: str
) -> pd.DataFrame:
    rows = []
    y = df[target].to_numpy(float)
    for f in features:
        x = df[f].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 4 or np.ptp(x[ok]) == 0:
            continue
        rho, p = stats.spearmanr(x[ok], y[ok])
        rows.append(
            {"modality": modality, "feature": f, "target": target, "n": int(ok.sum()),
             "spearman_rho": float(rho), "p_value": float(p)}
        )
    out = pd.DataFrame(rows)
    if len(out):
        out["q_value_bh"] = stats.false_discovery_control(out.p_value.to_numpy())
    return out


def kruskal_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        groups = [g[f].dropna().to_numpy() for _, g in df.groupby("quality_label")]
        groups = [g for g in groups if len(g)]
        if len(groups) < 2 or np.ptp(np.concatenate(groups)) == 0:
            continue
        h, p = stats.kruskal(*groups)
        rows.append({"feature": f, "kruskal_h": float(h), "kruskal_p": float(p)})
    return pd.DataFrame(rows, columns=["feature", "kruskal_h", "kruskal_p"])


def run(paths: Paths, params: dict) -> dict:
    top_k = params["correlate"]["top_k"]
    pd_welds = pd.read_parquet(paths.features / "photodiode_welds.parquet")
    pd_welds["quality_ordinal"] = pd_welds.quality_label.map(QUALITY_ORDINAL)
    pd_feats = _feature_columns(pd_welds, "pd_") + ["laser_power_w", "gap_um"]

    quality = spearman_table(pd_welds, pd_feats, "quality_ordinal", "photodiode")
    quality = quality.merge(kruskal_table(pd_welds, pd_feats), on="feature", how="left")
    penetration = spearman_table(pd_welds, pd_feats, "penetration_um", "photodiode")

    tables = [quality, penetration]
    cam_path = paths.features / "camera_welds.parquet"
    cam = pd.read_parquet(cam_path)
    if len(cam):
        tables.append(
            spearman_table(cam, _feature_columns(cam, "cam_"), "laser_power_w", "coaxial_camera")
        )
    table = pd.concat(tables, ignore_index=True)
    table["abs_rho"] = table.spearman_rho.abs()
    table = table.sort_values(["target", "abs_rho"], ascending=[True, False])
    paths.reports.mkdir(parents=True, exist_ok=True)
    table.drop(columns="abs_rho").to_csv(paths.reports / "correlation.csv", index=False)

    def top(target: str) -> list[dict]:
        t = table[table.target == target].head(top_k)
        return [
            {"feature": r.feature, "n": r.n, "rho": round(r.spearman_rho, 3),
             "q": float(f"{r.q_value_bh:.3g}")}
            for r in t.itertuples()
        ]

    summary = {
        "n_photodiode_welds": len(pd_welds),
        "n_welds_with_penetration": int(pd_welds.penetration_um.notna().sum()),
        "n_camera_welds": len(cam),
        "n_features_significant_q05_quality": int(
            (table[table.target == "quality_ordinal"].q_value_bh < 0.05).sum()
        ),
        "top_quality": top("quality_ordinal"),
        "top_penetration": top("penetration_um"),
        "top_camera_laser_power": top("laser_power_w"),
    }
    (paths.reports / "correlation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[correlate] {len(table)} feature/target pairs; "
        f"{summary['n_features_significant_q05_quality']} quality correlations with q<0.05"
    )
    return summary
