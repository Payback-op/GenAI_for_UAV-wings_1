"""Create a results summary from measured artifacts, with no hard-coded scores."""
import json
import pandas as pd
from common import ROOT, TARGETS
from train import metrics

def main():
    card=json.loads((ROOT/"models/model_card.json").read_text())
    audit=json.loads((ROOT/"results/data_audit.json").read_text())
    scores=pd.read_csv(ROOT/"results/test_metrics.csv")
    preds=pd.read_csv(ROOT/"results/test_predictions.csv")
    text=["# BTP1 NASA: first model results","",
        "Completed on 15 October 2026. Dataset: https://github.com/nasa/airfoil-learning", "",
        f"Source repository revision: `{card['source_revision']}`. Raw archive SHA-256: `{card['raw_sha256']}`.","",
        f"The raw archive contains {audit['raw_airfoils']:,} airfoil files and {audit['raw_rows']:,} polar records. "
        f"The pipeline retained {audit['usable_rows']:,} rows from {audit['usable_airfoils']:,} airfoils. "
        f"It excluded {audit['rejected_shapes']:,} geometries ({audit['rows_in_rejected_shapes']:,} rows) and {audit['invalid_polar_rows']:,} invalid polar records. "
        "Some source surfaces cross or do not share a leading-edge point on the normalized chord grid. These were excluded rather than modified; see rejected_shapes.csv. "
        "Empty valid airfoil files do not contribute training rows.","",
        f"Exact duplicate rows removed: {audit['exact_duplicate_rows_removed']}. Identical-input label conflicts found: {audit['conflicting_input_combinations']}. "
        "These checks do not establish label fidelity or experimental validity.","",
        "| Split | Rows | Airfoil groups |","|---|---:|---:|"]
    for name in ["train","validation","test"]:
        p=audit["splits"][name]; text.append(f"| {name} | {p['rows']:,} | {p['groups']:,} |")
    text += ["",f"Fitting used {card['training_rows_used']:,} fixed seed training rows out of {card['training_rows_available']:,} available. "
        "All validation and test rows were scored. Every geometry stays in one split; related families may still occur in different splits.","",
        "Inputs: 82 sampled airfoil ordinates, alpha, log10(Re), Ncrit. Outputs: Cl, Cd, Cm. "
        "Candidate families: mean reference, Random Forest, XGBoost, 64/64 MLP. Selection uses validation RMSE per target. "
        "MLP scalers fit training data only, with validation-airfoil early stopping. Learned drag models use log/exp transforms.","",
        "## Full-envelope unseen-airfoil test","",
        "| Target | Model | MAE | RMSE | R2 |","|---|---|---:|---:|---:|"]
    for r in scores.to_dict("records"):
        text.append(f"| {r['target']} | {r['model']} | {r['MAE']:.6f} | {r['RMSE']:.6f} | {r['R2']:.4f} |")
    text += ["","MAE and RMSE are dimensionless coefficient errors, not percentages. These test results were not used to select models.","",
        "## Fixed example operating region","",
        "The following descriptive subset uses Re=100,000, Ncrit=9 and alpha from -4 to +8 degrees. "
        "It is reported after model selection without selecting a model on it; it does not replace the complete test evaluation.",""]
    subset=preds[(preds.Re==100000)&(preds.Ncrit==9)&preds.alpha.between(-4,8)]
    subgroup=[]
    text += [f"Subset: {len(subset):,} rows across {subset.shape_id.nunique():,} held-out airfoils.","",
        "| Target | MAE | RMSE | R2 |","|---|---:|---:|---:|"]
    for t in TARGETS:
        s=metrics(subset[t],subset[t+"_predicted"])
        subgroup.append({"target":t,"rows":len(subset),"region":"Re=100000,Ncrit=9,alpha[-4,8]",**s})
        text.append(f"| {t} | {s['MAE']:.6f} | {s['RMSE']:.6f} | {s['R2']:.4f} |")
    pd.DataFrame(subgroup).to_csv(ROOT/"results/example_region_metrics.csv",index=False)
    text += ["","## Generator and verification","",
        "The desktop generator searches training catalog shapes and convex coordinate blends to match target section Cl/Cd and thickness. "
        "Cm is displayed. It exports the same section as 2D coordinates/images and a symmetric tapered 3D wing with editable sweep, dihedral and twist. "
        "A saved result states predicted match or no match and records the parent geometry, nearest training distance and novelty on the sampled grid.","",
        "The 3D stage is a parametric geometry generator, not a learned 3D aerodynamic model. "
        "STL uses numerical millimetres; OBJ/NPZ use metres. Finite trailing-edge thickness is preserved.","",
        "Software verification checks split and checksum integrity, selected-model inference reproduction, drag inverse transforms, "
        "prediction range flags, exact section recovery at every wing station, mesh closure/orientation/volume, STL/OBJ round trips, "
        "impossible-target no-match reporting and invalid operating-condition rejection. See verification.json; the GUI also has a --smoke-test.","",
        "Predictions remain source-XFOIL surrogates. No new CFD, wind-tunnel testing, whole-wing aerodynamic/structural model, material/mass assessment or flight validation was performed. "
        "Generated coordinate blends can be outside the source joint distribution even when individual feature ranges pass. "
        "Inspect heldout_polars.png and target residual plots before using the surrogate for candidate rankings.",""]
    (ROOT/"results/FIRST_MODEL_REPORT.md").write_text("\n".join(text),encoding="utf-8")
    print("Wrote results/FIRST_MODEL_REPORT.md")

if __name__=="__main__": main()
