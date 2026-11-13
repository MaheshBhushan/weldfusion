"""Defect classifiers evaluated on held-out welds.

Each weld contributes several photodiode windows. Cross-validation folds are built with
StratifiedGroupKFold on `weld_id`, so every window of a test weld is unseen during training.
Weld-level predictions average the class probabilities of the weld's windows.
A window-level StratifiedKFold run (which lets windows of the same weld land in train and
test) is reported next to it only to show how much such leakage inflates the scores.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .paths import Paths

PROCESS = ["laser_power_w", "gap_um"]


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    pd_cols = [c for c in df.columns if c.startswith("pd_")]
    return {
        "process_only": PROCESS,
        "photodiode": pd_cols,
        "photodiode+process": pd_cols + PROCESS,
    }


def models(params: dict) -> dict:
    seed = params["seed"]
    return {
        "logreg": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=params["rf_trees"],
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1,
            ),
        ),
    }


def oof_probabilities(model, X, y, splits, n_classes: int) -> np.ndarray:
    proba = np.full((len(y), n_classes), np.nan)
    for train_idx, test_idx in splits:
        m = clone(model).fit(X[train_idx], y[train_idx])
        proba[test_idx] = m.predict_proba(X[test_idx])
    return proba


def weld_level(df: pd.DataFrame, proba: np.ndarray, classes: list[str]) -> pd.DataFrame:
    p = pd.DataFrame(proba, columns=[f"p_{c}" for c in classes])
    p["weld_id"] = df.weld_id.to_numpy()
    agg = p.groupby("weld_id", sort=True).mean()
    truth = df.groupby("weld_id", sort=True).quality_label.first()
    out = agg.assign(
        true_label=truth,
        pred_label=[classes[i] for i in agg.to_numpy().argmax(axis=1)],
        n_windows=df.groupby("weld_id", sort=True).size(),
    )
    return out.reset_index()


def scores(y_true, y_pred, classes: list[str]) -> dict:
    rec = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "recall": {c: round(float(r), 4) for c, r in zip(classes, rec, strict=True)},
    }


def run(paths: Paths, params: dict) -> dict:
    cfg = params["train"]
    df = pd.read_parquet(paths.features / "photodiode_windows.parquet")
    df = df.dropna(subset=["quality_label"]).reset_index(drop=True)
    classes = sorted(df.quality_label.unique())
    y_lab = df.quality_label.to_numpy()
    y = np.searchsorted(classes, y_lab)
    groups = df.weld_id.to_numpy()

    welds_per_class = df.groupby("quality_label").weld_id.nunique()
    n_splits = int(min(cfg["cv_folds"], welds_per_class.min()))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])
    grouped = list(sgkf.split(np.zeros(len(y)), y, groups))
    for train_idx, test_idx in grouped:  # the property the README relies on
        assert not set(groups[train_idx]) & set(groups[test_idx])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])
    leaky = list(skf.split(np.zeros(len(y)), y))

    results, best = [], None
    for fs_name, cols in feature_sets(df).items():
        X = df[cols].to_numpy(float)
        for m_name, model in models(cfg).items():
            proba = oof_probabilities(model, X, y, grouped, len(classes))
            win_pred = np.array(classes)[proba.argmax(axis=1)]
            welds = weld_level(df, proba, classes)
            leak_proba = oof_probabilities(model, X, y, leaky, len(classes))
            leak_pred = np.array(classes)[leak_proba.argmax(axis=1)]
            row = {
                "feature_set": fs_name,
                "model": m_name,
                "n_features": len(cols),
                "window_heldout_weld": scores(y_lab, win_pred, classes),
                "weld_heldout_weld": scores(welds.true_label, welds.pred_label, classes),
                "window_leaky_split": scores(y_lab, leak_pred, classes),
            }
            results.append(row)
            print(
                f"[train] {fs_name:20s} {m_name:14s} weld macro-F1 "
                f"{row['weld_heldout_weld']['macro_f1']:.3f} "
                f"(leaky window split {row['window_leaky_split']['macro_f1']:.3f})"
            )
            uses_pd = fs_name != "process_only"
            score = row["weld_heldout_weld"]["macro_f1"]
            if uses_pd and (best is None or score > best[0]):
                best = (score, fs_name, m_name, cols, welds)

    _, fs_name, m_name, cols, welds = best
    paths.reports.mkdir(parents=True, exist_ok=True)
    paths.models.mkdir(parents=True, exist_ok=True)
    welds.assign(feature_set=fs_name, model=m_name).to_csv(
        paths.reports / "oof_weld_predictions.csv", index=False
    )
    cm = confusion_matrix(welds.true_label, welds.pred_label, labels=classes)
    pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                 columns=[f"pred_{c}" for c in classes]).to_csv(
        paths.reports / "confusion_matrix.csv"
    )

    final = clone(models(cfg)[m_name]).fit(df[cols].to_numpy(float), y)
    joblib.dump({"model": final, "features": cols, "classes": classes},
                paths.models / "defect_classifier.joblib")
    rf = clone(models(cfg)["random_forest"]).fit(df[cols].to_numpy(float), y)
    imp = pd.DataFrame({"feature": cols, "importance": rf[-1].feature_importances_})
    imp.sort_values("importance", ascending=False).to_csv(
        paths.reports / "feature_importance.csv", index=False
    )

    metrics = {
        "split": f"StratifiedGroupKFold(n_splits={n_splits}) grouped by weld_id",
        "n_welds": int(df.weld_id.nunique()),
        "n_windows": len(df),
        "welds_per_class": {k: int(v) for k, v in welds_per_class.items()},
        "classes": classes,
        "best": {"feature_set": fs_name, "model": m_name},
        "results": results,
    }
    (paths.reports / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
