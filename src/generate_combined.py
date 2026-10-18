"""Surrogate-guided coordinate search, synchronized 2D and parametric 3D output."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from common import ROOT, X, FEATURES, SHAPE_FEATURES, feature_rows, save_json, sha256
from geometry import geometry_checks, evaluate_surfaces
from predict import load_bundle, predict_frame
from wing_geometry import validate_wing, build_mesh, write_stl, write_obj

def validate_airfoil(a,bundle=None):
    bundle=load_bundle() if bundle is None else bundle
    for k in ["angle_of_attack_deg","reynolds_number","ncrit","target_cl","cl_tolerance","maximum_cd","minimum_thickness_ratio","maximum_thickness_ratio"]:
        if isinstance(a.get(k),bool) or not isinstance(a.get(k),(float,int)) or not np.isfinite(a[k]):
            raise ValueError(f"{k} must be a finite number")
    for key,feature in [("angle_of_attack_deg","alpha"),("reynolds_number","Re"),("ncrit","Ncrit")]:
        lo,hi=bundle["metadata"]["condition_bounds"][feature]
        if not lo<=a[key]<=hi:
            raise ValueError(f"{key} must be within the training range {lo:g} to {hi:g}")
    if a["cl_tolerance"]<=0 or a["maximum_cd"]<=0:
        raise ValueError("Lift tolerance and maximum drag must be positive")
    if not .01<=a["minimum_thickness_ratio"]<a["maximum_thickness_ratio"]<=.4:
        raise ValueError("Thickness limits must increase and lie between 1% and 40%")
    count=a.get("candidate_count")
    if isinstance(count,bool) or not isinstance(count,int) or not 100<=count<=50000:
        raise ValueError("candidate_count must be an integer between 100 and 50000")

def validate_combined(c,bundle=None):
    validate_airfoil(c["airfoil"],bundle)
    validate_wing(c["wing"])

def search(c,bundle):
    a=c["airfoil"]
    catalog=pd.read_csv(ROOT/"data/processed/shapes.csv")
    catalog=catalog[catalog.split=="train"].reset_index(drop=True)
    base=catalog[SHAPE_FEATURES].to_numpy()
    rng=np.random.default_rng(a.get("random_seed",42))
    count=a["candidate_count"]
    p=rng.integers(len(base),size=count)
    q=rng.integers(len(base),size=count)
    weights=rng.uniform(.1,.9,size=(count,1))
    # Smooth convex blends of training geometries; no held-out geometry in search.
    mixed=weights*base[p]+(1-weights)*base[q]
    candidates=np.vstack([base,mixed])
    valid,thickness,camber=geometry_checks(candidates,a["minimum_thickness_ratio"],a["maximum_thickness_ratio"])
    candidate_ids=np.flatnonzero(valid)
    candidates=candidates[valid]; thickness=thickness[valid]; camber=camber[valid]
    if not len(candidates):
        raise ValueError("No geometrically valid candidates within the requested thickness limits")
    inputs=feature_rows(candidates,a["angle_of_attack_deg"],a["reynolds_number"],a["ncrit"])
    predictions=predict_frame(inputs,bundle)
    within=~predictions.outside_training_bounds.to_numpy()
    if not within.any():
        raise ValueError("All candidates exceed saved model feature ranges at these conditions")
    candidates=candidates[within]; predictions=predictions[within].reset_index(drop=True)
    thickness=thickness[within]; camber=camber[within]; candidate_ids=candidate_ids[within]
    cl_error=np.abs(predictions.Cl.to_numpy()-a["target_cl"])
    drag_excess=np.maximum(predictions.Cd.to_numpy()-a["maximum_cd"],0)
    match=(cl_error<=a["cl_tolerance"])&(drag_excess<=0)
    penalty=np.maximum(cl_error/a["cl_tolerance"]-1,0)+drag_excess/a["maximum_cd"]
    # Prefer a match, then lift accuracy; no claim of a global aerodynamic optimum.
    order=np.lexsort((predictions.Cd.to_numpy(),cl_error,penalty,~match))
    best=int(order[0]); cid=int(candidate_ids[best])
    nearest_distance,nearest=cKDTree(base).query(candidates[best])
    if cid<len(base):
        provenance={"method":"training catalog geometry","parents":[catalog.iloc[cid]["name"]],"blend_weight":1.0}
    else:
        j=cid-len(base)
        provenance={"method":"convex blend of two training geometries","parents":[catalog.iloc[p[j]]["name"],catalog.iloc[q[j]]["name"]],"blend_weight":float(weights[j,0])}
    provenance.update({"nearest_training_shape":catalog.iloc[int(nearest)]["name"],"sampled_coordinate_distance":float(nearest_distance),"novel_on_saved_grid":bool(nearest_distance>1e-7)})
    ranked=predictions.copy()
    ranked["candidate_id"]=candidate_ids; ranked["lift_absolute_error"]=cl_error; ranked["predicted_match"]=match
    ranked["thickness_ratio"]=thickness; ranked["score"]=penalty
    return candidates[best],{t:float(predictions.iloc[best][t]) for t in ["Cl","Cd","Cm"]},bool(match[best]),float(thickness[best]),provenance,ranked.iloc[order].head(100)

def draw_section(ax,shape,predictions=None):
    ax.clear()
    x,u,l=evaluate_surfaces(shape,np.linspace(0,1,401)); u,l=u[0],l[0]
    ax.fill_between(x,l,u,color="#93c5fd",alpha=.8)
    ax.plot(x,u,color="#1d4ed8"); ax.plot(x,l,color="#1d4ed8")
    ax.set(xlabel="x / chord",ylabel="y / chord",title="2D airfoil section")
    ax.set_aspect("equal",adjustable="datalim"); ax.grid(alpha=.2)

def draw_planform(ax,v):
    ax.clear()
    rings=v.reshape(len(np.unique(v[:,1])),-1,3)
    low,high=rings[:,:,0].min(axis=1),rings[:,:,0].max(axis=1)
    ys=rings[:,0,1]
    ax.fill_between(ys,low,high,color="#93c5fd",alpha=.8)
    ax.plot(ys,low,color="#1d4ed8"); ax.plot(ys,high,color="#1d4ed8")
    ax.set(xlabel="Span y (m)",ylabel="Chord direction x (m)",title="Wing top view")
    ax.set_aspect("equal",adjustable="datalim"); ax.invert_yaxis(); ax.grid(alpha=.2)

def draw_wing(ax,v,f):
    ax.clear()
    ax.add_collection3d(Poly3DCollection(v[f],facecolors="#60a5fa",linewidth=0,shade=True))
    lo,hi=v.min(axis=0),v.max(axis=0)
    centre=(lo+hi)/2; extent=np.maximum(hi-lo,.03)
    for setter,i in [(ax.set_xlim,0),(ax.set_ylim,1),(ax.set_zlim,2)]:
        setter(centre[i]-extent[i]*.55,centre[i]+extent[i]*.55)
    ax.set_box_aspect(extent,zoom=1.15); ax.view_init(elev=28,azim=-125)
    ax.set_axis_off(); ax.set_title("3D wing from the same airfoil")

def generate_combined(c,progress=print,output_folder=None,publish_latest=True):
    bundle=load_bundle(); validate_combined(c,bundle)
    progress("Searching training shapes and smooth coordinate blends...")
    shape,pred,matched,thickness,provenance,ranked=search(c,bundle)
    progress("Building the matching 3D wing and saving outputs...")
    v,f,stations,dimensions,checks=build_mesh(c["wing"],shape)
    folder=Path(output_folder) if output_folder else ROOT/"results/generated"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True,exist_ok=False)
    np.save(folder/"airfoil_shape.npy",shape)
    coords=pd.DataFrame({"x_over_chord":X,"upper_y":shape[:len(X)],"lower_y":shape[len(X):]})
    coords.to_csv(folder/"airfoil_coordinates.csv",index=False)
    ring=np.column_stack([np.r_[X[::-1],X[1:]],np.r_[shape[:len(X)][::-1],shape[len(X):][1:]]])
    np.savetxt(folder/"airfoil.dat",ring,header="BTP1 NASA coordinate design",comments="",fmt="%.10f")
    np.savez_compressed(folder/"wing_mesh.npz",vertices_m=v,faces=f)
    pd.DataFrame(stations).to_csv(folder/"spanwise_geometry.csv",index=False)
    ranked.to_csv(folder/"ranked_candidates.csv",index=False)
    write_stl(folder/"wing_mm.stl",v,f); write_obj(folder/"wing_m.obj",v,f)
    summary={"version":"btp1-nasa-v1","status":"predicted_match" if matched else "no_predicted_match",
        "conditions":c,"predictions":pred,"maximum_thickness_ratio":thickness,"geometry_provenance":provenance,
        "dimensions":dimensions,"mesh_checks":checks,"output_folder":str(folder.resolve()),
        "predictor_sha256":sha256(ROOT/"models/airfoil_baseline.joblib"),"source_sha256":bundle["metadata"]["raw_sha256"],
        "generator_type":"Surrogate-guided training-shape selection/blending plus parametric wing loft; no trained 3D model",
        "export_units":{"airfoil.dat":"chord normalized","wing_mm.stl":"numerical millimetres","wing_m.obj":"metres"},
        "complete_wing_aerodynamic_predictions":None,"physical_requirements_verified":False,
        "limitations":["Section coefficient target match is a model estimate","Geometry blending is not a learned conditional neural generator",
            "Within individual feature ranges does not establish a supported joint operating point","No full-wing, structural, material, mass or flight validation"]}
    save_json(folder/"design.json",summary)
    fig=plt.figure(figsize=(14,8))
    gs=fig.add_gridspec(2,2,width_ratios=[1,1.3])
    draw_section(fig.add_subplot(gs[0,0]),shape)
    draw_planform(fig.add_subplot(gs[1,0]),v)
    draw_wing(fig.add_subplot(gs[:,1],projection="3d"),v,f)
    title="Predicted section match" if matched else "No predicted target match - closest candidate"
    fig.suptitle(f"BTP1 NASA | {title}\nCl {pred['Cl']:.4f} | Cd {pred['Cd']:.5f} | Cm {pred['Cm']:.4f}")
    fig.text(.5,.02,"Section surrogate estimates; 3D geometry only. Whole-wing performance remains unverified.",ha="center",color="#9a3412")
    fig.tight_layout(rect=(0,.05,1,.9)); fig.savefig(folder/"combined_preview.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4)); draw_section(ax,shape); fig.tight_layout(); fig.savefig(folder/"airfoil.png",dpi=150); plt.close(fig)
    if publish_latest: save_json(ROOT/"results/generated/latest.json",summary)
    return summary

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--conditions",type=Path,default=ROOT/"config/combined_conditions.json")
    args=parser.parse_args()
    result=generate_combined(json.loads(args.conditions.read_text()))
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
