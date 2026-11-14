"""End-to-end run of fuse -> features -> correlate -> train -> export on the committed fixture."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook
from pptx import Presentation
from sklearn.model_selection import StratifiedGroupKFold

from weldfusion.cli import run_all
from weldfusion.paths import Paths, load_params

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "raw"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    params = load_params(ROOT / "params.yaml")
    params["train"] |= {"cv_folds": 3, "rf_trees": 50}
    paths = Paths.at(tmp_path_factory.mktemp("run"), FIXTURE)
    run_all(paths, params, fetch=False)
    return paths, params


def test_fused_dataset(run):
    paths, _ = run
    welds = pd.read_parquet(paths.fused / "welds.parquet")
    manifest = json.loads((paths.fused / "manifest.json").read_text())
    assert manifest["modalities"]["photodiode"] == 9
    assert manifest["modalities"]["coaxial_camera"] == 3
    assert manifest["modalities"]["photodiode_and_camera"] == 0
    assert manifest["quality_labels"] == {"LoC": 3, "OP": 3, "SW": 3}
    assert welds.weld_id.is_unique
    assert len(manifest["dataset_version"]) == 16
    cam = welds[welds.has_camera]
    assert cam.quality_label.isna().all()
    assert set(cam.laser_power_w) == {1500.0, 2500.0, 3500.0}


def test_features(run):
    paths, _ = run
    win = pd.read_parquet(paths.features / "photodiode_windows.parquet")
    welds = pd.read_parquet(paths.features / "photodiode_welds.parquet")
    cam = pd.read_parquet(paths.features / "camera_welds.parquet")
    assert win.weld_id.nunique() == 9 and len(win) > 9
    assert np.isfinite(welds.filter(like="pd_plasma_").to_numpy()).all()
    assert (cam.cam_area_px_mean > 0).all()


def test_folds_never_share_welds(run):
    paths, params = run
    win = pd.read_parquet(paths.features / "photodiode_windows.parquet")
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=params["train"]["seed"])
    for tr, te in cv.split(win, win.quality_label, win.weld_id):
        assert not set(win.weld_id.iloc[tr]) & set(win.weld_id.iloc[te])


def test_metrics_and_predictions(run):
    paths, _ = run
    metrics = json.loads((paths.reports / "metrics.json").read_text())
    assert metrics["n_welds"] == 9
    assert len(metrics["results"]) == 6
    for r in metrics["results"]:
        assert 0 <= r["weld_heldout_weld"]["macro_f1"] <= 1
    oof = pd.read_csv(paths.reports / "oof_weld_predictions.csv")
    assert len(oof) == 9 and oof.weld_id.is_unique
    assert (paths.models / "defect_classifier.joblib").exists()


def test_exports_open(run):
    paths, _ = run
    wb = load_workbook(paths.reports / "weldfusion_summary.xlsx")
    assert {"About", "Welds", "Correlation", "Model results", "Confusion matrix"} <= set(
        wb.sheetnames
    )
    assert wb["Welds"].max_row == 13  # header + 12 welds
    assert wb["Model results"]._charts
    prs = Presentation(paths.reports / "weldfusion_summary.pptx")
    assert len(prs.slides) == 7
    titles = [s.shapes.title.text for s in prs.slides]
    assert "Limitations" in titles
