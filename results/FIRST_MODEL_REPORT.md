# BTP1 NASA: first model results

Completed on 15 October 2026. Dataset: https://github.com/nasa/airfoil-learning

Source repository revision: `f211e62fab6be125657d49c58fdce1bdeb0d12a8`. Raw archive SHA-256: `6c052006c8002c6852c4e118c85dcc8a1676d1645f47153efbcda5a761b11e39`.

The raw archive contains 825 airfoil files and 768,542 polar records. The pipeline retained 596,843 rows from 627 airfoils. It excluded 195 geometries (171,529 rows) and 170 invalid polar records. Some source surfaces cross or do not share a leading-edge point on the normalized chord grid. These were excluded rather than modified; see rejected_shapes.csv. Empty valid airfoil files do not contribute training rows.

Exact duplicate rows removed: 0. Identical-input label conflicts found: 0. These checks do not establish label fidelity or experimental validity.

| Split | Rows | Airfoil groups |
|---|---:|---:|
| train | 418,184 | 438 |
| validation | 89,026 | 94 |
| test | 89,633 | 95 |

Fitting used 80,000 fixed seed training rows out of 418,184 available. All validation and test rows were scored. Every geometry stays in one split; related families may still occur in different splits.

Inputs: 82 sampled airfoil ordinates, alpha, log10(Re), Ncrit. Outputs: Cl, Cd, Cm. Candidate families: mean reference, Random Forest, XGBoost, 64/64 MLP. Selection uses validation RMSE per target. MLP scalers fit training data only, with validation-airfoil early stopping. Learned drag models use log/exp transforms.

## Full-envelope unseen-airfoil test

| Target | Model | MAE | RMSE | R2 |
|---|---|---:|---:|---:|
| Cl | mlp | 0.062475 | 0.101064 | 0.9819 |
| Cl | mean_baseline | 0.646910 | 0.753861 | -0.0082 |
| Cd | mlp | 0.005454 | 0.011482 | 0.8547 |
| Cd | mean_baseline | 0.023070 | 0.030128 | -0.0001 |
| Cm | mlp | 0.008698 | 0.013537 | 0.9225 |
| Cm | mean_baseline | 0.040516 | 0.048957 | -0.0137 |

MAE and RMSE are dimensionless coefficient errors, not percentages. These test results were not used to select models.

## Fixed example operating region

The following descriptive subset uses Re=100,000, Ncrit=9 and alpha from -4 to +8 degrees. It is reported after model selection without selecting a model on it; it does not replace the complete test evaluation.

Subset: 4,173 rows across 95 held-out airfoils.

| Target | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Cl | 0.084168 | 0.127635 | 0.9249 |
| Cd | 0.005164 | 0.011704 | 0.6678 |
| Cm | 0.010300 | 0.015682 | 0.8913 |

## Generator and verification

The desktop generator searches training catalog shapes and convex coordinate blends to match target section Cl/Cd and thickness. Cm is displayed. It exports the same section as 2D coordinates/images and a symmetric tapered 3D wing with editable sweep, dihedral and twist. A saved result states predicted match or no match and records the parent geometry, nearest training distance and novelty on the sampled grid.

The 3D stage is a parametric geometry generator, not a learned 3D aerodynamic model. STL uses numerical millimetres; OBJ/NPZ use metres. Finite trailing-edge thickness is preserved.

Software verification checks split and checksum integrity, selected-model inference reproduction, drag inverse transforms, prediction range flags, exact section recovery at every wing station, mesh closure/orientation/volume, STL/OBJ round trips, impossible-target no-match reporting and invalid operating-condition rejection. See verification.json; the GUI also has a --smoke-test.

Predictions remain source-XFOIL surrogates. No new CFD, wind-tunnel testing, whole-wing aerodynamic/structural model, material/mass assessment or flight validation was performed. Generated coordinate blends can be outside the source joint distribution even when individual feature ranges pass. Inspect heldout_polars.png and target residual plots before using the surrogate for candidate rankings.
