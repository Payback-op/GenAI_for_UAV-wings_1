"""Saved-model CSV inference with input range flags."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from common import ROOT, FEATURES, TARGETS

def load_bundle(path=None):
    """Load a trusted local bundle; preserve the existing default model."""
    bundle=joblib.load(ROOT/"models/airfoil_baseline.joblib" if path is None else path)
    if bundle["metadata"]["features"] != FEATURES:
        raise ValueError("Model schema mismatch")
    return bundle

def predict_frame(frame,bundle=None):
    bundle=load_bundle() if bundle is None else bundle
    missing=set(FEATURES)-set(frame.columns)
    if missing:
        raise ValueError(f"Missing model inputs: {sorted(missing)}")
    x=frame[FEATURES].apply(pd.to_numeric,errors="raise")
    if len(x)==0 or not np.isfinite(x.to_numpy()).all():
        raise ValueError("Expected nonempty finite numeric inputs")
    flags=[]
    for row in x.to_numpy():
        flags.append([c for c,v in zip(FEATURES,row) if not bundle["metadata"]["training_bounds"][c][0]-1e-12<=v<=bundle["metadata"]["training_bounds"][c][1]+1e-12])
    result=pd.DataFrame({t:bundle["models"][t].predict(x) for t in TARGETS},index=frame.index)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Model returned nonfinite predictions")
    result["outside_training_bounds"]=[bool(f) for f in flags]
    result["outside_features"]=[json.dumps(f) for f in flags]
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output",default=str(ROOT/"results/predictions.csv"))
    parser.add_argument("--model",help="Trusted local model bundle; omit to use the existing app model")
    args=parser.parse_args()
    frame=pd.read_csv(args.input)
    result=predict_frame(frame,load_bundle(args.model))
    from pathlib import Path
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    pd.concat([frame,result.add_prefix("predicted_")],axis=1).to_csv(path,index=False)
    print(result.to_string(index=False))

if __name__=="__main__":
    main()
