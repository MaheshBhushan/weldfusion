# WeldFusion

Sensor data fusion for laser weld quality. WeldFusion downloads public laser-welding
monitoring data, fuses it into one versioned per-weld dataset, extracts time- and
frequency-domain features (SciPy) and melt-pool image features (OpenCV), correlates them
with seam quality, trains scikit-learn defect classifiers evaluated on held-out welds, and
exports an Excel workbook (openpyxl) and a PowerPoint summary (python-pptx). DVC tracks every
pipeline stage.

## Data: what is real and what is missing

I could not find a public dataset that records photodiode signals, coaxial camera images and
quality labels on the same welds. WeldFusion therefore uses two real, openly licensed Zenodo
datasets and keeps track of which modality each weld actually has. No modality is
synthesised.

| Source | DOI | Licence | Welds | Real modalities |
|---|---|---|---|---|
| Chianese et al. 2022, remote laser welding of Cu-to-steel thin-foil battery tabs | [10.5281/zenodo.6732794](https://doi.org/10.5281/zenodo.6732794) | CC-BY-4.0 | 142 | process parameters (laser power, part-to-part gap), 3 photodiode channels at 50 kHz (plasma, temperature, back-reflection) plus the analogue laser-power signal, seam quality label (SW sound weld / LoC lack of connection / OP over-penetration), penetration depth and throat thickness for subset C (86 welds) |
| Knaak 2017, laser welding of zinc-coated steel overlap joints | [10.5281/zenodo.1049434](https://doi.org/10.5281/zenodo.1049434) | CC-BY-4.0 | 16 | coaxial NIR camera (32x32 px, 1750 nm, ~500 frames/s), process parameters from the published protocol (laser power 1.5-3.5 kW, focal offset for 5 trials) |

Subsets A and B of the Chianese data share welds 340-344; they are de-duplicated (the rows
agree) so 146 listed signals become 142 unique welds. The camera welds have no quality label,
so camera features are correlated with laser power only and are not used by the defect
classifier. Rows from the two sources are never joined to each other.

## Architecture

```
weldfusion download   Zenodo -> data/downloads (MD5 verified) -> data/raw (RAR/ZIP extracted)
weldfusion fuse       data/raw -> data/fused/{welds.parquet, welds.h5, manifest.json}
weldfusion features   data/fused -> data/features/{photodiode_windows, photodiode_welds, camera_welds}.parquet
weldfusion correlate  -> reports/correlation.csv, reports/correlation_summary.json
weldfusion train      -> models/defect_classifier.joblib, reports/metrics.json, predictions, confusion matrix
weldfusion export     -> reports/weldfusion_summary.xlsx, reports/weldfusion_summary.pptx
```

- `fuse` writes one row per weld (source, DOI, process parameters, labels, modality flags,
  SHA-256 of the raw file) and stores the arrays in HDF5 (`/photodiode/<weld_id>`,
  `/camera/<weld_id>`). `manifest.json` carries `dataset_version`, a hash over the schema
  version and every raw-file hash, so any change in the input data changes the version.
  `dvc.lock` pins the hashes of every stage output.
- `features`: the laser-on segment is found from the analogue laser-power signal and cut into
  50 ms windows. Per channel: mean, std, RMS, CV, skewness, kurtosis, percentiles, range,
  crest factor, roughness, Welch-PSD band-power fractions (0-0.5, 0.5-2, 2-8, 8-25 kHz),
  spectral centroid, peak frequency and spectral entropy, plus inter-channel correlations and
  ratios (66 features). Camera frames are upsampled, thresholded and contoured with OpenCV:
  melt-pool area, perimeter, circularity, fitted-ellipse axes, centroid and intensity per
  frame, aggregated per weld, plus the spectrum of the melt-pool intensity over time.
- `correlate`: Spearman correlation of weld-level features with ordinal seam quality
  (LoC < SW < OP), with penetration depth, and (camera) with laser power; Kruskal-Wallis
  across classes; Benjamini-Hochberg q-values.
- `train`: logistic regression and random forest on three feature sets (process parameters
  only, photodiode only, photodiode + process). Folds come from `StratifiedGroupKFold`
  grouped by `weld_id`, so all windows of a test weld are unseen in training (asserted in
  code and in the tests). Weld predictions average window probabilities. A window-level
  `StratifiedKFold` without grouping is also run, only to show the effect of leakage.

## Quickstart

Requires Python 3.11+ and one of `unrar`, `7z` or `bsdtar` to unpack the RAR archives.
The download is about 300 MB.

```
uv venv && uv pip install -e ".[dev,dvc]"
source .venv/bin/activate
dvc repro            # download -> fuse -> features -> correlate -> train -> export
```

Without DVC the same chain is `weldfusion all`. Single stages: `weldfusion <stage>`.

## Tests

```
pytest -q
ruff check src tests scripts
```

The tests run the whole fuse-to-export chain on a small committed fixture
(`tests/fixtures/raw`, about 5.6 MB) built from the real data by `scripts/make_fixture.py`:
9 unmodified photodiode recordings (3 per class, gzip-compressed), the matching rows of the
quality table, and 3 camera recordings with every 20th frame kept. They check the fused
counts, that no fold shares a weld between train and test, the metrics file, and that the
.xlsx and .pptx open with openpyxl and python-pptx. CI runs the same on GitHub Actions.

## Measured results

From `dvc repro` on the full data (dataset_version `37893e166cf15792`, 142 labelled welds:
73 LoC, 40 OP, 29 SW, 759 windows), 5-fold `StratifiedGroupKFold` by weld, seed 42. Numbers
are in `reports/metrics.json` and `reports/correlation_summary.json`.

Held-out welds (weld-level, window probabilities averaged):

| Feature set | Model | Weld macro-F1 | Weld balanced acc. | Window macro-F1 (grouped) | Window macro-F1 (leaky split) |
|---|---|---|---|---|---|
| process only | logistic regression | 0.601 | 0.606 | 0.606 | 0.600 |
| process only | random forest | 0.856 | 0.845 | 0.866 | 0.945 |
| photodiode | logistic regression | 0.862 | 0.857 | 0.813 | 0.822 |
| photodiode | random forest | 0.906 | 0.885 | 0.872 | 0.899 |
| photodiode + process | logistic regression | 0.875 | 0.883 | 0.840 | 0.862 |
| photodiode + process | random forest | 0.924 | 0.908 | 0.909 | 0.926 |

Confusion matrix of the best model (photodiode + process, random forest), held-out welds:

| | pred LoC | pred OP | pred SW |
|---|---|---|---|
| true LoC | 71 | 1 | 1 |
| true OP | 3 | 37 | 0 |
| true SW | 5 | 0 | 24 |

Correlation (Spearman, weld level):

- 49 photodiode/process features correlate with ordinal seam quality at q < 0.05 (n = 142).
  Strongest: temperature-channel roughness rho = 0.811, temperature mean 0.809, plasma mean
  0.803.
- Penetration depth (n = 86, subset C): plasma 95th percentile rho = 0.755, plasma RMS 0.751,
  back-reflection/plasma ratio -0.746.
- Camera vs laser power (n = 16): mean melt-pool minor axis rho = 0.894 (q = 9.2e-5), area
  0.765, perimeter 0.752, circularity -0.714.

The process-only random forest is strong because laser power largely defines the classes in
this experiment; the leaky window split inflates it most (0.866 grouped vs 0.945 leaky),
which is why all reported scores use the weld-grouped split.

To reproduce: `dvc repro` (or `weldfusion all`), then read `reports/metrics.json`.

## Limitations

- Photodiode and camera data come from different experiments, materials and machines; there is
  no weld with both modalities, so no model fuses them. The "fusion" is a common per-weld
  schema with provenance and modality flags.
- The camera data has no seam-quality labels; its features are only related to laser power.
- The camera is 32x32 pixels, so melt-pool geometry is coarse.
- 142 labelled welds from one lab; the scores have wide uncertainty and do not transfer to
  other setups without retraining.
- The .xlsx and .pptx are checked by reloading them with openpyxl and python-pptx; they were
  not opened in Microsoft Office on this machine.
- No DVC remote is configured; `dvc.lock` records hashes and the data is re-created from
  Zenodo by `dvc repro`.

## Data licences and citation

Both datasets are CC-BY-4.0. Cite:

- Chianese, G., Sun, T., Franciosa, P., Ceglarek, D., Patalano, S. (2022). Using photodiodes
  and supervised machine learning for automatic classification of weld defects in laser
  welding of thin foils copper-to-steel battery tabs [Data set]. Zenodo.
  https://doi.org/10.5281/zenodo.6732794
- Knaak, C. (2017). Laser welding IR data for ML [Data set]. Zenodo.
  https://doi.org/10.5281/zenodo.1049434

The fixture in `tests/fixtures/raw` is a subset of these datasets under the same licence.
Code: MIT.
