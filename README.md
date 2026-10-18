# BTP1: NASA airfoil prediction and 2D/3D design sample

This is a separate sample project built from [NASA Airfoil Learning](https://github.com/nasa/airfoil-learning). It trains forward predictors of **Cl (lift), Cd (drag) and Cm (pitching moment)** using airfoil coordinates, angle of attack, Reynolds number and Ncrit. A desktop form searches section shapes and displays that same section as a 2D airfoil and a 3D wing.

The GitHub copy includes source, configuration, processed data, fitted models, generated geometry, experiment records and XFLR5 comparisons. It excludes `.venv`, the large raw ZIP, the separate `upstream` checkout and transient runtime logs. To use it on another computer, install Python 3.13, create a virtual environment and install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe src\design_app.py
```

Run these commands from the cloned project directory. Historical JSON records contain paths from the original `C:\BTP1` workspace; generate a fresh design if the clone is stored elsewhere. The original app model remains the default; experimental bundles and their reports are under `results/experiments`. The latest experimental-model XFLR5 comparison is under `results/xflr5/experimental_model_backtest`.

## Open the airfoil visualizer

Open `C:\BTP1` in VS Code and press **Ctrl+Shift+B**, or double-click **run_app.bat**. This opens the dedicated 2D visualizer, following BTP's **Generate shape** workflow. From PowerShell:

```powershell
cd C:\BTP1
.\.venv\Scripts\python.exe src\design_app.py
```

Enter angle of attack, Reynolds number, Ncrit, lift/drag targets and thickness limits, then click **Generate shape**. The cross-section and predicted Cl/Cd/Cm update together. Pan, zoom and save the preview with its toolbar. **Open saved results** opens the images, DAT/CSV coordinates and design provenance under `results/2d`. The most recent result loads on startup. Changed inputs are flagged until you generate again. Configuration is saved separately in `config/design_conditions.json`.

For the combined visualizer, click **Open 2D + 3D visualizer** in the 2D app to carry over your current section inputs, run **run_combined_app.bat**, or choose **Terminal > Run Task > BTP1 NASA: Open 2D + 3D generator**. Enter section targets and wing dimensions, then click **Generate 2D + 3D**. The section, wing top view and 3D mesh update together from the same generated airfoil. Drag the 3D preview to rotate it, or choose **Perspective**, **Top** or **Front**. Both panels have pan/zoom/save toolbars. Combined images, DAT/CSV coordinates and STL/OBJ exports remain under `results/generated`.

Starting inputs are examples: alpha 4 degrees, Re 100,000, Ncrit 9, desired Cl 0.6 +/- 0.05, maximum Cd 0.03, section thickness 8–18%, span 1.8 m, root chord 0.30 m and tip chord 0.15 m. A result reports either a predicted section match or no predicted match. Cm is predicted and displayed, but is not a search target in this sample.

## Data and models

The raw NASA ZIP is stored under `data/raw`; its SHA-256, URL and source revision are in `data/raw/source.json`. The upstream repository and its documentation are under `upstream/airfoil-learning`. The pipeline reads JSON directly without using the upstream normalized pickle dataset. Source labels are XFOIL results; upstream generation specifies Mach 0.

Each shape is represented by upper and lower ordinates on 41 cosine-spaced chord locations (82 geometry features). Source surfaces are interpolated with PCHIP. Three more model features are alpha in degrees, log10(Re), and Ncrit. This representation replaces the old BTP dataset's eight CST coefficients; the old predictor is not reused.

All rows for a shape remain in one split. Identical names and geometry vectors within Euclidean distance `1e-4` are connected before a fixed seed 70/15/15 group split. This does not guarantee independent airfoil families. Rejected shapes, bad polar rows, duplicate rows, coverage and label conflicts are recorded under `results`.

Four candidates are compared for each target: an arithmetic mean baseline, Random Forest, XGBoost, and a two-hidden-layer MLP (64,64). A fixed seed sample of up to 80,000 training rows limits CPU cost; the full held-out validation and test sets are evaluated. Feature and target scalers fit only training data. MLP early stopping uses validation airfoils. Drag is trained in log space for learned models and transformed back for inference and metrics. Models are selected by validation RMSE independently per target; test data does not select the winner.

The selected bundle is `models/airfoil_baseline.joblib`. All fitted candidates are saved in `models/candidates`. Load only trusted local joblib files. `models/model_card.json` records the exact feature schema, package versions, training sample size, source/split checksums and limits. `requirements.lock.txt` pins this run's packages.

## Reproduce the pipeline

To run a separate accuracy experiment using every training row, without replacing the current app model:

```powershell
.\.venv\Scripts\python.exe src\improve_model.py --epochs 100
```

The experiment keeps the existing geometry-held-out splits, fits scalers on training rows only, and selects each target against the current model using validation RMSE. It evaluates the locked selections on the existing test set afterward; this reused test set is not a new independent confirmation set. Models, histories, full-envelope and example-region scores, and a report are saved under `results/experiments`; `results/experiments/latest.json` identifies the completed run. No image-derived or XFLR5 verification labels are added to training. Use `src/predict.py ... --model <experiment-folder>/airfoil_candidate.joblib` to try a trusted experiment bundle. The desktop apps continue using `models/airfoil_baseline.joblib`.

The original pipeline below replaces model and metric files. Preserve them before rerunning it. Run **train_models.bat**, or:

```powershell
cd C:\BTP1
.\.venv\Scripts\python.exe src\download_data.py
.\.venv\Scripts\python.exe src\prepare_data.py
.\.venv\Scripts\python.exe src\train.py
.\.venv\Scripts\python.exe src\generate_combined.py
.\.venv\Scripts\python.exe src\verify.py
.\.venv\Scripts\python.exe src\write_report.py
.\.venv\Scripts\python.exe src\combined_app.py --smoke-test
```

Existing raw data is checksum-verified instead of downloaded again. To recreate the environment, use Python 3.13 and `python -m venv .venv`, then `.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt`. This environment currently uses the Python runtime installed under `C:\BTP\.python-runtime`; install Python independently when moving this project to another computer.

## Predict a CSV

```powershell
.\.venv\Scripts\python.exe src\predict.py data\processed\example_inputs.csv --output results\example_predictions.csv
```

Input CSVs need the exact 85-feature schema in `data/processed/schema.json`. Example inputs are held-out source rows, for demonstrating inference; they are not additional independent evaluation. The callable `common.feature_rows` constructs rows from the saved geometry format and operating conditions. Inference reports individual feature-range flags. The design form rejects operating conditions outside observed training ranges.

## Review outputs

| File | Contents |
|---|---|
| `results/FIRST_MODEL_REPORT.md` | Actual data counts, selected models and test errors |
| `results/data_audit.json` | Quality checks, coverage and group split counts |
| `results/rejected_shapes.csv` | Excluded geometries and reasons |
| `results/rejected_polars.csv` | Excluded invalid coefficient/condition records |
| `results/label_conflicts.csv` | Disagreements for identical shapes and conditions |
| `results/validation_metrics.csv` | Every model's validation MAE/RMSE/R2 |
| `results/test_metrics.csv` | Selected models and the mean reference on unseen airfoils |
| `results/test_predictions.csv` | Every test prediction and residual |
| `results/plots/` | Test parity/residual and held-out polar plots |
| `results/verification.json` | Software and mesh verification |
| `results/generated/latest.json` | Most recent generated design |

Each generated directory contains `airfoil.png`, `combined_preview.png`, `airfoil.dat`, `airfoil_coordinates.csv`, `airfoil_shape.npy`, `wing_mm.stl`, `wing_m.obj`, `wing_mesh.npz`, `spanwise_geometry.csv`, ranked candidates and `design.json`. DAT/CSV geometry is normalized by chord. STL coordinates are numerical millimetres (STL stores no unit metadata); OBJ and NPZ coordinates are metres. The wing loft preserves a finite trailing-edge thickness where present.

## Interpretation

MAE and RMSE use coefficient units; lower is better. R2 compares against a constant reference and may be negative. These scores are not flight-performance percentages. Full-envelope test errors include difficult operating points and do not establish accuracy for every condition.

The section generator selects training shapes and convex coordinate blends, checks non-intersection/thickness on a dense chord grid, then ranks surrogate predictions. It is a bounded search, not a trained conditional generative network. Search provenance names the parent shapes and blend weight; selection can return a known shape, so novelty is explicitly reported. The search uses training geometries only. Shape feature bounds do not establish valid joint geometry/condition coverage.

The 3D result is a symmetric tapered wing loft with sweep, dihedral and quarter-chord twist. Its span and reference area/aspect ratio come from the chosen dimensions. It uses the same airfoil section at all span stations, scaled and twisted. The learned Cl/Cd/Cm values describe only the section at the input conditions, not the whole wing or each twisted span station. No matched 3D aerodynamic/structural labels exist in this source. Whole-wing performance, structural strength, material, mass and flight feasibility require additional work. No new CFD or experimental validation is claimed.
