# Full training data experiment

Production files were preserved. Selection used validation airfoils only. The test set is the same previously reported set, not a new independent confirmation set.

Training rows: 418,184 (baseline: 80,000).

| Target | Baseline test RMSE | Selected test RMSE | Change in RMSE |
|---|---:|---:|---:|
| Cl | 0.101064 | 0.105171 | +4.1% |
| Cd | 0.011482 | 0.012305 | +7.2% |
| Cm | 0.013537 | 0.013537 | +0.0% |

## Fixed example operating region

Descriptive subset fixed in the original baseline report: Re=100,000, Ncrit=9, alpha from -4 to 8 degrees, across 95 held-out airfoils. This subset did not select the models.

| Target | Baseline regional RMSE | Selected regional RMSE | Change in RMSE |
|---|---:|---:|---:|
| Cl | 0.127635 | 0.111381 | -12.7% |
| Cd | 0.011704 | 0.007979 | -31.8% |
| Cm | 0.015682 | 0.015682 | +0.0% |

## Interpretation

The new candidates lowered validation RMSE and full-test MAE for Cl and Cd, but increased full-envelope test RMSE. Their larger errors therefore remain a concern. Keep the bundle experimental; this is not an overall upgrade. The original app model remains active. The reduced regional errors do not guarantee accuracy at every operating point or on generated blends.

The original generated section now has candidate estimates Cl=0.564735 and Cd=0.015562, compared with baseline Cl=0.599744 and Cd=0.016281. The model was not fitted to that design's XFLR5 results; no new solver verification of the candidate model is claimed.

`training_source.py` preserves the exact experiment source matching the manifest's code checksum. The maintained script in `src/improve_model.py` includes later report-formatting improvements.

## Validation selections

{
  "Cl": "full_training_mlp",
  "Cd": "full_training_mlp",
  "Cm": "existing_mlp"
}

The candidate bundle has the same 85-feature inference interface. This experiment does not prove agreement with a new solver or generated blends. No image/XFLR5 labels were included in training. Panel-converged original-geometry solver checks and additional unseen geometries remain necessary.

See validation_metrics.csv, test_metrics.csv, example_region_metrics.csv and existing_design_predictions.json. Negative RMSE change means lower error; it is not an accuracy percentage.
