"""Audit every raw polar, group duplicate/near-duplicate shapes, then split."""
import json
import zipfile
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupShuffleSplit
from common import ROOT, X, SHAPE_FEATURES, FEATURES, TARGETS, SEED, save_json, sha256
from geometry import source_shape

def main():
    archive = ROOT / "data/raw/airfoil-learning-dataset.zip"
    source = json.loads((archive.parent / "source.json").read_text())
    if sha256(archive) != source["sha256"]:
        raise ValueError("Raw archive checksum mismatch")
    shapes, rows, rejected, bad_shapes = [], [], [], []
    raw_rows = 0
    with zipfile.ZipFile(archive) as z:
        files = sorted(i for i in z.namelist() if i.endswith(".json"))
        for i, member in enumerate(files):
            record = json.loads(z.read(member))
            raw_rows += len(record.get("polars", []))
            try:
                shape = source_shape(record)
            except (ValueError, KeyError, TypeError) as error:
                bad_shapes.append({"member": member, "reason": str(error), "rows": len(record.get("polars", []))})
                continue
            shape_id = len(shapes)
            shapes.append({"shape_id": shape_id, "name": record["name"], "source_member": member, **dict(zip(SHAPE_FEATURES, shape))})
            for j, polar in enumerate(record["polars"]):
                try:
                    values = {k: float(polar[k]) for k in ["alpha", "Re", "Ncrit"] + TARGETS}
                    if not np.isfinite(list(values.values())).all():
                        raise ValueError("nonfinite values")
                    if values["Re"] <= 0 or values["Cd"] <= 0 or values["Ncrit"] <= 0:
                        raise ValueError("nonpositive Re, Cd or Ncrit")
                except (ValueError, KeyError, TypeError) as error:
                    rejected.append({"source_member": member, "polar_index": j, "reason": str(error)})
                    continue
                rows.append({"source_member": member, "polar_index": j, "shape_id": shape_id, **values})
            if (i+1) % 50 == 0:
                print(f"Read {i+1}/{len(files)} airfoils, {len(rows):,} usable polars", flush=True)
    shape_df = pd.DataFrame(shapes)
    df = pd.DataFrame(rows)
    # All aliases and shapes within Euclidean distance 1e-4 on the sampled grid are connected.
    parent = np.arange(len(shapes))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)
    def union(i,j):
        a,b = find(i),find(j)
        parent[max(a,b)] = min(a,b)
    pairs = sorted(cKDTree(shape_df[SHAPE_FEATURES]).query_pairs(1e-4))
    for a,b in pairs:
        union(a,b)
    for _, part in shape_df.groupby("name"):
        for b in part.shape_id.iloc[1:]:
            union(int(part.shape_id.iloc[0]), int(b))
    shape_df["geometry_group"] = [f"g{find(i):04d}" for i in range(len(shapes))]
    df = df.merge(shape_df[["shape_id", "name", "geometry_group"]], on="shape_id", validate="many_to_one")
    df["log10_Re"] = np.log10(df.Re)
    duplicate = df.duplicated(["shape_id", "alpha", "Re", "Ncrit"] + TARGETS)
    duplicate_count = int(duplicate.sum())
    df = df[~duplicate].reset_index(drop=True)
    # Conflicting exact geometry/condition labels are reported and retained within one split.
    exact = shape_df.groupby(SHAPE_FEATURES, dropna=False).ngroup()
    exact_map = dict(zip(shape_df.shape_id, exact))
    df["exact_shape_id"] = df.shape_id.map(exact_map)
    conflict = df.groupby(["exact_shape_id", "alpha", "Re", "Ncrit"])[TARGETS].agg(["min", "max", "count"])
    conflict.columns = ["_".join(c) for c in conflict.columns]
    mask = np.zeros(len(conflict), bool)
    for t in TARGETS:
        mask |= (conflict[t+"_max"] > conflict[t+"_min"]).to_numpy()
    conflict = conflict[mask].reset_index()
    tv, test = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=SEED).split(df, groups=df.geometry_group))
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.15/.85, random_state=SEED+1).split(df.iloc[tv], groups=df.iloc[tv].geometry_group))
    df["split"] = "test"
    df.loc[tv[tr], "split"] = "train"
    df.loc[tv[va], "split"] = "validation"
    shape_df["split"] = shape_df.geometry_group.map(df.groupby("geometry_group").split.first())
    assert df.groupby("geometry_group").split.nunique().max() == 1
    for folder in ["data/processed", "data/splits", "results"]:
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    shape_df.to_csv(ROOT/"data/processed/shapes.csv", index=False)
    df.to_csv(ROOT/"data/processed/polars.csv", index=False)
    df[["source_member", "polar_index", "shape_id", "geometry_group", "split"]].to_csv(ROOT/"data/splits/membership.csv", index=False)
    pd.DataFrame(rejected, columns=["source_member", "polar_index", "reason"]).to_csv(ROOT/"results/rejected_polars.csv", index=False)
    pd.DataFrame(bad_shapes, columns=["member", "reason", "rows"]).to_csv(ROOT/"results/rejected_shapes.csv", index=False)
    conflict.to_csv(ROOT/"results/label_conflicts.csv", index=False)
    audit = {"source_sha256":source["sha256"], "raw_airfoils":len(files), "raw_rows":raw_rows,
        "usable_airfoils":int(df.shape_id.nunique()), "usable_rows":len(df), "rejected_shapes":len(bad_shapes),
        "rows_in_rejected_shapes":sum(r["rows"] for r in bad_shapes), "invalid_polar_rows":len(rejected),
        "exact_duplicate_rows_removed":duplicate_count, "near_geometry_pairs":len(pairs),
        "geometry_groups":int(df.geometry_group.nunique()), "conflicting_input_combinations":len(conflict),
        "conflict_policy":"Retain source labels, report disagreements, keep identical geometry within one split",
        "near_geometry_tolerance":1e-4, "grouping_limit":"Sampled-coordinate proximity; related airfoil families are not guaranteed disjoint",
        "ranges":{c:[float(df[c].min()),float(df[c].max())] for c in ["alpha","Re","Ncrit"]+TARGETS},
        "splits":{s:{"rows":len(p),"airfoils":int(p.shape_id.nunique()),"groups":int(p.geometry_group.nunique())} for s,p in df.groupby("split")}}
    save_json(ROOT/"results/data_audit.json", audit)
    save_json(ROOT/"data/processed/schema.json", {"features":FEATURES,"targets":TARGETS,"chord_grid":X.tolist(),
        "shape_representation":"PCHIP sampled source upper/suction and lower/pressure ordinates; 41 cosine-spaced x/c locations each",
        "units":{"alpha":"degrees","Re":"dimensionless, log10 transformed for model","Ncrit":"dimensionless","Cl/Cd/Cm":"dimensionless"},
        "pressure_coefficient":"Present in raw data but not predicted by this baseline"})
    example = df[df.split=="test"].head(5).merge(shape_df[["shape_id"]+SHAPE_FEATURES],on="shape_id")
    example[FEATURES].to_csv(ROOT/"data/processed/example_inputs.csv",index=False)
    print(json.dumps(audit, indent=2), flush=True)

if __name__ == "__main__":
    main()
