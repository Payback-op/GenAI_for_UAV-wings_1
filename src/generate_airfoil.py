"""Generate a 2D section using the saved NASA predictor and geometry search."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT, X, save_json, sha256
from generate_combined import draw_section, search, validate_airfoil
from predict import load_bundle


def generate(conditions, progress=print, model_path=None, output_folder=None, publish_latest=True):
    model_path = ROOT / "models/airfoil_baseline.joblib" if model_path is None else Path(model_path)
    bundle = load_bundle(model_path)
    validate_airfoil(conditions, bundle)
    progress("Searching airfoil shapes with the NASA prediction model...")
    shape, predictions, matched, thickness, provenance, ranked = search(
        {"airfoil": conditions}, bundle
    )
    progress("Drawing the airfoil and saving coordinates...")
    folder = Path(output_folder) if output_folder else ROOT / "results/2d" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    np.save(folder / "airfoil_shape.npy", shape)
    pd.DataFrame({"x_over_chord": X, "upper_y": shape[:len(X)],
                  "lower_y": shape[len(X):]}).to_csv(folder / "airfoil_coordinates.csv", index=False)
    ring = np.column_stack([np.r_[X[::-1], X[1:]],
                            np.r_[shape[:len(X)][::-1], shape[len(X):][1:]]])
    np.savetxt(folder / "airfoil.dat", ring, header="BTP1 NASA airfoil", comments="", fmt="%.10f")
    ranked.to_csv(folder / "ranked_candidates.csv", index=False)
    summary = {
        "version": "btp1-nasa-2d-v1",
        "status": "predicted_match" if matched else "no_predicted_match",
        "conditions": conditions, "predictions": predictions,
        "maximum_thickness_ratio": thickness, "geometry_provenance": provenance,
        "predicted_matching_candidates": int(ranked.predicted_match.sum()),
        "candidate_count_note": "Matching count covers the saved top 100 candidates only.",
        "output_folder": str(folder.resolve()),
        "predictor_path": str(model_path.resolve()),
        "predictor_sha256": sha256(model_path),
        "source_sha256": bundle["metadata"]["raw_sha256"],
        "export_units": "Coordinates normalized by chord",
        "physical_requirements_verified": False,
        "limitations": ["Section coefficients are surrogate estimates, not physical validation.",
                        "Search selects training shapes and coordinate blends.",
                        "Individual feature bounds do not establish joint operating coverage."]
    }
    fig, ax = plt.subplots(figsize=(11, 4))
    draw_section(ax, shape)
    p = predictions
    fig.suptitle(f"BTP1 | {'Predicted target match' if matched else 'No predicted match - closest candidate'}\n"
                 f"Cl {p['Cl']:.4f} | Cd {p['Cd']:.5f} | Cm {p['Cm']:.4f}")
    fig.tight_layout(rect=(0, 0, 1, .88))
    fig.savefig(folder / "airfoil.png", dpi=150)
    plt.close(fig)
    save_json(folder / "design.json", summary)
    if publish_latest:
        save_json(ROOT / "results/2d/latest.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", type=Path, default=ROOT / "config/design_conditions.json")
    parser.add_argument("--model", type=Path, help="Trusted local bundle; defaults to the existing app model")
    parser.add_argument("--output", type=Path, help="New output directory")
    parser.add_argument("--no-latest", action="store_true", help="Preserve the app's latest design pointer")
    args = parser.parse_args()
    print(json.dumps(generate(json.loads(args.conditions.read_text()), model_path=args.model,
                              output_folder=args.output, publish_latest=not args.no_latest), indent=2))


if __name__ == "__main__":
    main()
