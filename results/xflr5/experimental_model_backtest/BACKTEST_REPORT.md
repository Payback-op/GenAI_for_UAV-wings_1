# Experimental model versus fresh XFLR5 analysis

The experimental bundle was used to select the airfoil before any solver outcomes were read. It selected GOE 766, a known training-catalog geometry, not a novel blend. The production model was preserved.

Conditions: alpha 4 degrees, Re 100,000, Mach 0, Ncrit 9, forced transition at x/c=1 on both surfaces. XFLR5 v6.62 used Type 1 analysis.

| Coefficient | Experimental model | XFLR5, 280 panels | Absolute difference | Relative difference |
|---|---:|---:|---:|---:|
| Cl | 0.600100 | 0.610800 | 0.010700 | 1.75% |
| Cd | 0.019121 | 0.019460 | 0.000339 | 1.74% |
| Cm | -0.022819 | -0.013100 | 0.009719 | 74.19% |

Percentages use absolute XFLR5 values as the denominator. A large percentage for a small moment coefficient should be read with its absolute difference.

## Numerical refinement

| Panels | Cl at 4 deg | Cd at 4 deg | Cm at 4 deg |
|---|---:|---:|---:|
| 81 | 0.586900 | 0.016840 | -0.008200 |
| 160 | 0.619500 | 0.019540 | -0.014800 |
| 240 | 0.613000 | 0.019480 | -0.013500 |
| 280 | 0.610800 | 0.019460 | -0.013100 |

Each refinement was made independently from the original 81-point export with XFLR5 Global Panel Refinement, retaining its other defaults. The rejected 300-panel foil has no analysis results. See backtest_summary.json for 240-to-280 changes; this finite check does not prove zero discretization error.

The finest solver result meets the requested lift range 0.55-0.65 and drag limit 0.03 at four degrees.

## Scope

These are discrepancies for one selected, already-known airfoil. They do not establish the overall error rate, generalization to new shapes, or physical flight performance. The experimental bundle uses newly trained Cl and Cd predictors and retains the original Cm predictor. No XFLR5 outcomes were used to retrain, retune, or reselect this airfoil.

The full sweeps cover -4 to 10 degrees in 0.5 degree steps. Only stored, converged polar points are compared; missing angles are listed in backtest_summary.json. Comparison plots use the unchanged exported geometry as model input and XFLR5 refinement of that export as solver geometry.

Files: airfoil.dat and airfoil_coordinates.csv (generated section); design.json (frozen model prediction and provenance); polar_*.txt (native XFLR5 exports); solver_model_comparison.csv (all matched predictions and residuals); design_point_panel_study.csv (point errors); model_backtest.xfl (saved XFLR5 project).