"""Excel workbook (openpyxl) and PowerPoint summary (python-pptx) of a pipeline run."""

from __future__ import annotations

import json

import pandas as pd
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches, Pt

from .paths import Paths

XLSX_NAME = "weldfusion_summary.xlsx"
PPTX_NAME = "weldfusion_summary.pptx"

LIMITATIONS = [
    "No public dataset found that records photodiode signals and coaxial camera images on the "
    "same welds; the two modalities come from different experiments and are never joined "
    "row-wise or used together in one model.",
    "Seam quality labels (SW / LoC / OP) exist only for the photodiode welds. The camera welds "
    "carry process parameters (laser power, focal offset) but no quality label, so camera "
    "features are correlated with laser power only.",
    "The camera is a 32x32-pixel NIR sensor (1750 nm) at about 500 frames/s; melt-pool geometry "
    "is therefore coarse.",
    "142 photodiode welds with three classes is a small sample; held-out scores have wide "
    "uncertainty.",
]


def results_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for r in metrics["results"]:
        rows.append(
            {
                "feature_set": r["feature_set"],
                "model": r["model"],
                "n_features": r["n_features"],
                "weld_macro_f1": r["weld_heldout_weld"]["macro_f1"],
                "weld_balanced_accuracy": r["weld_heldout_weld"]["balanced_accuracy"],
                "weld_accuracy": r["weld_heldout_weld"]["accuracy"],
                "window_macro_f1": r["window_heldout_weld"]["macro_f1"],
                "leaky_window_macro_f1": r["window_leaky_split"]["macro_f1"],
                **{f"weld_recall_{c}": v for c, v in r["weld_heldout_weld"]["recall"].items()},
            }
        )
    return pd.DataFrame(rows)


def _load(paths: Paths) -> dict:
    rep, feat = paths.reports, paths.features
    return {
        "manifest": json.loads((paths.fused / "manifest.json").read_text()),
        "welds": pd.read_parquet(paths.fused / "welds.parquet"),
        "pd_welds": pd.read_parquet(feat / "photodiode_welds.parquet"),
        "cam_welds": pd.read_parquet(feat / "camera_welds.parquet"),
        "correlation": pd.read_csv(rep / "correlation.csv"),
        "corr_summary": json.loads((rep / "correlation_summary.json").read_text()),
        "metrics": json.loads((rep / "metrics.json").read_text()),
        "confusion": pd.read_csv(rep / "confusion_matrix.csv", index_col=0),
        "oof": pd.read_csv(rep / "oof_weld_predictions.csv"),
        "importance": pd.read_csv(rep / "feature_importance.csv"),
    }


def _style_sheet(ws) -> None:
    header_fill = PatternFill("solid", fgColor="DDE4EE")
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
    ws.freeze_panes = "B2"
    if ws.max_row > 1:
        ws.auto_filter.ref = ws.dimensions
    for i, col in enumerate(ws.iter_cols(min_row=1, max_row=min(ws.max_row, 200)), start=1):
        width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
        ws.column_dimensions[get_column_letter(i)].width = min(max(10, width + 2), 60)


def write_xlsx(paths: Paths, d: dict) -> None:
    m = d["manifest"]
    about = [("dataset_version", m["dataset_version"]), ("schema_version", m["schema_version"]),
             ("n_welds", m["n_welds"])]
    about += [(f"welds[{k}]", v) for k, v in m["welds_per_source"].items()]
    about += [(f"modality[{k}]", v) for k, v in m["modalities"].items()]
    for s in m["sources"]:
        about += [(f"{s['key']}.doi", s["doi"]), (f"{s['key']}.license", s["license"]),
                  (f"{s['key']}.modalities", ", ".join(s["modalities"])),
                  (f"{s['key']}.citation", s["citation"])]
    about += [(f"limitation {i + 1}", t) for i, t in enumerate(LIMITATIONS)]
    results = results_table(d["metrics"])

    out = paths.reports / XLSX_NAME
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        sheets = {
            "About": pd.DataFrame(about, columns=["key", "value"]),
            "Welds": d["welds"],
            "Photodiode features": d["pd_welds"],
            "Camera features": d["cam_welds"],
            "Correlation": d["correlation"],
            "Model results": results,
            "Held-out predictions": d["oof"],
            "Feature importance": d["importance"],
        }
        for name, frame in sheets.items():
            frame.to_excel(xw, sheet_name=name, index=False)
        d["confusion"].to_excel(xw, sheet_name="Confusion matrix")
        for ws in xw.book.worksheets:
            _style_sheet(ws)
        ws = xw.book["Model results"]
        chart = BarChart()
        chart.title = "Held-out weld macro-F1 (grouped by weld) vs leaky window split"
        chart.y_axis.title = "macro-F1"
        chart.y_axis.scaling.min, chart.y_axis.scaling.max = 0, 1
        n = len(results)
        for col in ("weld_macro_f1", "leaky_window_macro_f1"):
            idx = results.columns.get_loc(col) + 1
            chart.add_data(Reference(ws, min_col=idx, min_row=1, max_row=n + 1),
                           titles_from_data=True)
        ws.cell(row=1, column=results.shape[1] + 2, value="config")
        for i, r in enumerate(results.itertuples(), start=2):
            ws.cell(row=i, column=results.shape[1] + 2, value=f"{r.feature_set} / {r.model}")
        chart.set_categories(Reference(ws, min_col=results.shape[1] + 2, min_row=2,
                                       max_row=n + 1))
        chart.width, chart.height = 22, 10
        ws.add_chart(chart, f"A{n + 4}")


def _title(slide, text: str) -> None:
    slide.shapes.title.text = text
    slide.shapes.title.text_frame.paragraphs[0].font.size = Pt(28)


def _table(slide, df: pd.DataFrame, top: float, font: int = 11, left: float = 0.4,
           width: float = 9.2) -> None:
    rows, cols = df.shape[0] + 1, df.shape[1]
    shape = slide.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width),
                                   Inches(0.3 * rows))
    t = shape.table
    for j, c in enumerate(df.columns):
        t.cell(0, j).text = str(c)
    for i, row in enumerate(df.itertuples(index=False), start=1):
        for j, v in enumerate(row):
            t.cell(i, j).text = f"{v:.3f}" if isinstance(v, float) else str(v)
    for cell in (t.cell(i, j) for i in range(rows) for j in range(cols)):
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(font)


def _bullets(slide, lines: list[str], top: float, size: int = 14) -> None:
    box = slide.shapes.add_textbox(Inches(0.5), Inches(top), Inches(9), Inches(7.5 - top - 0.3))
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"- {line}"
        p.font.size = Pt(size)


def write_pptx(paths: Paths, d: dict) -> None:
    m, metrics = d["manifest"], d["metrics"]
    prs = Presentation()
    blank_title = prs.slide_layouts[5]

    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = "WeldFusion: laser weld quality from fused sensor data"
    s.placeholders[1].text = (
        f"Dataset version {m['dataset_version']} | {m['n_welds']} welds | "
        f"held-out evaluation grouped by weld"
    )

    s = prs.slides.add_slide(blank_title)
    _title(s, "Data sources and modalities (all real, public, CC-BY-4.0)")
    welds = d["welds"]
    src = pd.DataFrame(
        [
            {
                "source": x["key"],
                "DOI": x["doi"],
                "welds": int((welds.source == x["key"]).sum()),
                "modalities": ", ".join(x["modalities"]),
            }
            for x in m["sources"]
        ]
    )
    _table(s, src, 1.5, font=12)
    _bullets(s, [f"{k}: {v} welds" for k, v in m["modalities"].items()], 3.2)

    s = prs.slides.add_slide(blank_title)
    _title(s, "Photodiode features vs seam quality (Spearman, weld level)")
    top = d["correlation"].query("target == 'quality_ordinal'").head(12)
    cd = CategoryChartData()
    cd.categories = [f.removeprefix("pd_") for f in top.feature]
    cd.add_series("Spearman rho (LoC < SW < OP)", [round(v, 3) for v in top.spearman_rho])
    chart = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.4), Inches(1.4),
                               Inches(9.2), Inches(5.6), cd).chart
    chart.has_legend = False
    chart.value_axis.minimum_scale, chart.value_axis.maximum_scale = -1, 1
    chart.category_axis.tick_labels.font.size = Pt(10)

    s = prs.slides.add_slide(blank_title)
    _title(s, "Defect classifiers on held-out welds")
    res = results_table(metrics)[
        ["feature_set", "model", "weld_macro_f1", "weld_balanced_accuracy",
         "window_macro_f1", "leaky_window_macro_f1"]
    ]
    _table(s, res, 1.4, font=11)
    _bullets(
        s,
        [
            f"{metrics['split']}; {metrics['n_welds']} welds, {metrics['n_windows']} windows",
            "leaky_window_macro_f1: windows split without grouping (same weld in train and "
            "test), shown only to quantify leakage",
        ],
        4.2,
        size=12,
    )

    s = prs.slides.add_slide(blank_title)
    best = metrics["best"]
    _title(s, f"Confusion matrix: {best['feature_set']} / {best['model']} (weld level)")
    _table(s, d["confusion"].reset_index(names=""), 1.6, font=14, width=7)

    cam = d["cam_welds"]
    if len(cam):
        s = prs.slides.add_slide(blank_title)
        _title(s, "Coaxial NIR camera: mean melt-pool image per weld")
        ranked = cam.sort_values("laser_power_w")
        picks = ranked.iloc[[0, len(ranked) // 2, len(ranked) - 1]].drop_duplicates("weld_id")
        for i, r in enumerate(picks.itertuples()):
            img = paths.features / "camera_examples" / f"{r.weld_id}.png"
            s.shapes.add_picture(str(img), Inches(0.5 + 3.1 * i), Inches(1.5), Inches(2.8))
            box = s.shapes.add_textbox(Inches(0.5 + 3.1 * i), Inches(4.35), Inches(2.8),
                                       Inches(0.4))
            box.text_frame.text = f"{r.weld_id}: {r.laser_power_w:.0f} W"
        lines = [f"{t['feature']}: rho={t['rho']} (n={t['n']})"
                 for t in d["corr_summary"]["top_camera_laser_power"][:4]]
        _bullets(s, ["Strongest camera-feature correlations with laser power:"] + lines, 4.9,
                 size=12)

    s = prs.slides.add_slide(blank_title)
    _title(s, "Limitations")
    _bullets(s, LIMITATIONS, 1.4, size=14)

    prs.save(paths.reports / PPTX_NAME)


def run(paths: Paths, params: dict) -> None:
    d = _load(paths)
    write_xlsx(paths, d)
    write_pptx(paths, d)
    print(f"[export] wrote {paths.reports / XLSX_NAME} and {paths.reports / PPTX_NAME}")
