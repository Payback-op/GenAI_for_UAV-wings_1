"""Check split integrity, saved inference, section/wing identity and export topology."""
from copy import deepcopy
import json
from pathlib import Path
import struct
import tempfile
import numpy as np
import pandas as pd
from common import ROOT, X, SHAPE_FEATURES, FEATURES, TARGETS, feature_rows, save_json, sha256
from predict import load_bundle, predict_frame
from generate_combined import generate_combined, validate_combined
from wing_geometry import check_mesh, build_mesh

def main():
    checks={}
    bundle=load_bundle(); card=bundle["metadata"]
    shapes=pd.read_csv(ROOT/"data/processed/shapes.csv")
    polars=pd.read_csv(ROOT/"data/processed/polars.csv")
    assert card["membership_sha256"]==sha256(ROOT/"data/splits/membership.csv")
    assert card["polars_sha256"]==sha256(ROOT/"data/processed/polars.csv")
    assert card["shapes_sha256"]==sha256(ROOT/"data/processed/shapes.csv")
    assert polars.groupby("geometry_group").split.nunique().max()==1
    assert polars.groupby("exact_shape_id").split.nunique().max()==1
    checks["checksum_and_geometry_split_integrity"]=True
    sample=pd.read_csv(ROOT/"data/splits/training_sample.csv")
    membership=sample.merge(polars[["source_member","polar_index","split"]],on=["source_member","polar_index"],validate="one_to_one")
    assert len(sample)==card["training_rows_used"] and membership.split.eq("train").all()
    checks["training_sample_uses_training_airfoils_only"]=True
    test=polars[polars.split=="test"].head(100).merge(shapes[["shape_id"]+SHAPE_FEATURES],on="shape_id")
    result=predict_frame(test[FEATURES],bundle)
    saved=pd.read_csv(ROOT/"results/test_predictions.csv").merge(test[["source_member","polar_index"]],on=["source_member","polar_index"],validate="one_to_one")
    for t in TARGETS:
        np.testing.assert_allclose(result[t],saved[t+"_predicted"],rtol=1e-6,atol=1e-8)
    assert (result.Cd>0).all()
    checks["saved_test_inference_reproduced_in_coefficient_units"]=True
    validations=pd.read_csv(ROOT/"results/validation_metrics.csv")
    for t in TARGETS:
        winner=validations[validations.target==t].sort_values("RMSE").iloc[0].model
        assert winner==card["selected_models"][t]
    checks["selection_matches_validation_rmse"]=True
    bad=test[FEATURES].head(1).copy(); bad.loc[:,"alpha"]=np.nan
    try: predict_frame(bad,bundle)
    except ValueError: pass
    else: raise AssertionError("Nonfinite input accepted")
    bad.loc[:,"alpha"]=card["training_bounds"]["alpha"][1]+1
    assert predict_frame(bad,bundle).outside_training_bounds.iloc[0]
    checks["invalid_inputs_and_range_flags"]=True
    latest=json.loads((ROOT/"results/generated/latest.json").read_text())
    folder=Path(latest["output_folder"])
    shape=np.load(folder/"airfoil_shape.npy")
    c=latest["conditions"]; a=c["airfoil"]
    pred=predict_frame(feature_rows(shape,a["angle_of_attack_deg"],a["reynolds_number"],a["ncrit"]),bundle)
    for t in TARGETS: np.testing.assert_allclose(pred[t].iloc[0],latest["predictions"][t],rtol=1e-10,atol=1e-10)
    matches=abs(latest["predictions"]["Cl"]-a["target_cl"])<=a["cl_tolerance"] and latest["predictions"]["Cd"]<=a["maximum_cd"]
    assert matches==(latest["status"]=="predicted_match")
    checks["generated_section_predictions_and_status_reproduced"]=True
    with np.load(folder/"wing_mesh.npz") as mesh: v,f=mesh["vertices_m"],mesh["faces"]
    checks["mesh"]=check_mesh(v,f)
    assert checks["mesh"]["closed_manifold"] and checks["mesh"]["consistent_orientation"]
    assert checks["mesh"]["degenerate_faces"]==0 and checks["mesh"]["signed_volume_m3"]>0
    expected_v,expected_f,*_=build_mesh(c["wing"],shape)
    np.testing.assert_allclose(v,expected_v,rtol=0,atol=0); np.testing.assert_array_equal(f,expected_f)
    # Invert the scale, sweep, dihedral and twist at every station to recover the exact section.
    finite_te=abs(shape[len(X)-1]-shape[-1])>1e-9
    ring_x=np.r_[X[::-1],X[1:] if finite_te else X[1:-1]]
    ring_z=np.r_[shape[:len(X)][::-1],shape[len(X):][1:] if finite_te else shape[len(X):][1:-1]]
    stations=pd.read_csv(folder/"spanwise_geometry.csv")
    rings=v.reshape(len(stations),-1,3)
    for ring,s in zip(rings,stations.to_dict("records")):
        twist=np.deg2rad(s["twist_deg"])
        dx=ring[:,0]-s["leading_edge_datum_x_m"]-.25*s["chord_m"]
        z=ring[:,2]-s["dihedral_datum_z_m"]
        recovered_x=(np.cos(twist)*dx-np.sin(twist)*z)/s["chord_m"]+.25
        recovered_z=(np.sin(twist)*dx+np.cos(twist)*z)/s["chord_m"]
        np.testing.assert_allclose(recovered_x,ring_x,atol=1e-12)
        np.testing.assert_allclose(recovered_z,ring_z,atol=1e-12)
    checks["all_wing_stations_use_exact_2d_section"]=True
    # Independently check a constant-twist loft against its analytical prismoid volume.
    analytic_c=deepcopy(c["wing"])
    analytic_c["root_twist_deg"]=0; analytic_c["tip_twist_deg"]=0
    av,af,_,dims,ac=build_mesh(analytic_c,shape)
    section_area=np.trapezoid(shape[:len(X)]-shape[len(X):],X)
    root,tip=analytic_c["root_chord_m"],analytic_c["tip_chord_m"]
    expected_volume=section_area*analytic_c["span_m"]*(root*root+root*tip+tip*tip)/3
    np.testing.assert_allclose(ac["signed_volume_m3"],expected_volume,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(np.ptp(av[:,1]),analytic_c["span_m"],atol=1e-12)
    np.testing.assert_allclose(dims["reference_planform_area_m2"],analytic_c["span_m"]*(root+tip)/2,atol=1e-12)
    checks["analytical_constant_twist_volume_span_and_area"]=True
    with (folder/"wing_mm.stl").open("rb") as stream:
        stream.seek(80); count=struct.unpack("<I",stream.read(4))[0]
        assert count==len(f)
        records=np.frombuffer(stream.read(),dtype=np.dtype([("normal","<f4",3),("vertices","<f4",(3,3)),("attribute","<u2")]))
        np.testing.assert_allclose(records["vertices"],v[f]*1000,atol=.0001,rtol=1e-6)
    obj_v=[];obj_f=[]
    for line in (folder/"wing_m.obj").read_text().splitlines():
        if line.startswith("v "): obj_v.append([float(x) for x in line.split()[1:]])
        if line.startswith("f "): obj_f.append([int(x)-1 for x in line.split()[1:]])
    np.testing.assert_allclose(obj_v,v,atol=1e-10);np.testing.assert_array_equal(obj_f,f)
    checks["stl_millimetre_and_obj_metre_roundtrip"]=True
    impossible=deepcopy(c); impossible["airfoil"]["target_cl"]=100.0
    impossible["airfoil"]["candidate_count"]=100; impossible["wing"]["span_stations"]=5
    with tempfile.TemporaryDirectory(prefix="btp1-verification-") as temp:
        summary=generate_combined(impossible,progress=lambda m:None,output_folder=Path(temp)/"no_match",publish_latest=False)
        assert summary["status"]=="no_predicted_match"
    bad=deepcopy(c); bad["airfoil"]["reynolds_number"]=1
    try: validate_combined(bad,bundle)
    except ValueError: pass
    else: raise AssertionError("Out-of-range Reynolds number accepted")
    bad=deepcopy(c);bad["wing"]["tip_chord_m"]=-1
    try: validate_combined(bad,bundle)
    except ValueError: pass
    else: raise AssertionError("Negative wing chord accepted")
    checks["no_match_and_invalid_condition_paths"]=True
    save_json(ROOT/"results/verification.json",{"passed":True,"checks":checks})
    print(json.dumps(checks,indent=2))

if __name__=="__main__":main()
