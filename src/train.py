"""Compare CPU models; select using validation airfoils, evaluate fixed test airfoils."""
import argparse
from copy import deepcopy
import importlib.metadata
import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import platform
import time
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor
from common import ROOT, FEATURES, SHAPE_FEATURES, TARGETS, SEED, save_json, sha256

def metrics(y, pred):
    return {"MAE":float(mean_absolute_error(y,pred)), "RMSE":float(np.sqrt(mean_squared_error(y,pred))), "R2":float(r2_score(y,pred))}

def fit_mlp(x, y, xv, yv, target, max_epochs):
    """External geometry-held-out early stopping; scalers fit on training data only."""
    scaler = StandardScaler().fit(x)
    xx, vv = scaler.transform(x), scaler.transform(xv)
    target_scaler = StandardScaler().fit(y.reshape(-1,1))
    yy = target_scaler.transform(y.reshape(-1,1)).ravel()
    model = MLPRegressor(hidden_layer_sizes=(64,64), batch_size=512, learning_rate_init=.001,
        alpha=.01, random_state=SEED, early_stopping=False)
    best, best_loss, stale, history = None, float("inf"), 0, []
    for epoch in range(max_epochs):
        model.partial_fit(xx, yy)
        p = target_scaler.inverse_transform(model.predict(vv).reshape(-1,1)).ravel()
        if target == "Cd":
            p = np.exp(p)
        loss = metrics(yv,p)["RMSE"]
        history.append({"epoch":epoch+1,"validation_RMSE":loss})
        if loss < best_loss:
            best_loss, best, stale = loss, deepcopy(model), 0
        else:
            stale += 1
        if stale >= 7:
            break
    regressor = make_pipeline(scaler, best)
    wrapped = TransformedTargetRegressor(regressor=regressor, transformer=target_scaler)
    # Explicitly install the fitted pipeline and fitted target scaler.
    wrapped.regressor_, wrapped.transformer_ = regressor, target_scaler
    wrapped._training_dim = 1
    if target == "Cd":
        outer = TransformedTargetRegressor(regressor=wrapped, func=np.log, inverse_func=np.exp)
        from sklearn.preprocessing import FunctionTransformer
        outer.regressor_, outer.transformer_ = wrapped, FunctionTransformer(np.log,np.exp,validate=True).fit(np.exp(y).reshape(-1,1))
        outer._training_dim = 1
        wrapped = outer
    return wrapped, history

def run_signature(args):
    return {"max_train_rows":args.max_train_rows,"mlp_epochs":args.mlp_epochs,"seed":SEED,
        "training_code_sha256":sha256(ROOT/"src/train.py"),"features":FEATURES,
        "polars_sha256":sha256(ROOT/"data/processed/polars.csv"),
        "shapes_sha256":sha256(ROOT/"data/processed/shapes.csv"),
        "membership_sha256":sha256(ROOT/"data/splits/membership.csv"),
        "packages":{p:importlib.metadata.version(p) for p in ["numpy","scikit-learn","xgboost"]}}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--max-train-rows",type=int,default=80000)
    parser.add_argument("--mlp-epochs",type=int,default=50)
    parser.add_argument("--resume",action="store_true",help="Reuse tree checkpoints only when their run manifest matches current data/settings")
    args=parser.parse_args()
    if args.max_train_rows < 100 or args.mlp_epochs < 1:
        parser.error("Use at least 100 training rows and one MLP epoch")
    polars=pd.read_csv(ROOT/"data/processed/polars.csv")
    shapes=pd.read_csv(ROOT/"data/processed/shapes.csv")
    df=polars.merge(shapes[["shape_id"]+SHAPE_FEATURES],on="shape_id",validate="many_to_one")
    parts={s:df[df.split==s].copy() for s in ["train","validation","test"]}
    for a,b in [("train","validation"),("train","test"),("validation","test")]:
        if not set(parts[a].geometry_group).isdisjoint(parts[b].geometry_group):
            raise ValueError("Geometry leakage")
    train=parts["train"]
    if len(train)>args.max_train_rows:
        train=train.sample(args.max_train_rows,random_state=SEED).sort_index()
    # MLP early stopping uses a fixed validation subset; all candidates' final validation scores use every validation row.
    validation_subset=parts["validation"].sample(min(20000,len(parts["validation"])),random_state=SEED)
    out=ROOT/"results"
    (out/"plots").mkdir(parents=True,exist_ok=True)
    (ROOT/"models/candidates").mkdir(parents=True,exist_ok=True)
    signature=run_signature(args)
    manifest=ROOT/"models/candidates/run_manifest.json"
    if args.resume:
        if not manifest.exists() or json.loads(manifest.read_text())!=signature:
            raise ValueError("Cannot resume: checkpoint data/settings manifest is missing or differs")
    else:
        save_json(manifest,signature)
    scores, test_scores, selected, chosen_models, history = [],[],{},{},{}
    predictions=parts["test"][["source_member","polar_index","shape_id","name","geometry_group","alpha","Re","Ncrit"]+TARGETS].copy()
    for target in TARGETS:
        candidates={
            "mean_baseline":DummyRegressor(),
            "random_forest":RandomForestRegressor(n_estimators=100,max_depth=20,min_samples_leaf=3,max_features=.7,n_jobs=4,random_state=SEED),
            "xgboost":XGBRegressor(n_estimators=400,max_depth=6,learning_rate=.05,subsample=.85,colsample_bytree=.9,tree_method="hist",n_jobs=4,random_state=SEED),
            "mlp":None}
        fitted={}
        for name,model in candidates.items():
            start=time.perf_counter()
            print(f"Training {target}/{name} on {len(train):,} rows...",flush=True)
            with threadpool_limits(limits=4):
                checkpoint=ROOT/f"models/candidates/{target}_{name}.joblib"
                if args.resume and name!="mlp" and checkpoint.exists():
                    model=joblib.load(checkpoint)
                    print(f"Restored {target}/{name} from this run's checkpoint",flush=True)
                elif name=="mlp":
                    y=train[target].to_numpy()
                    model,hist=fit_mlp(train[FEATURES],np.log(y) if target=="Cd" else y,
                        validation_subset[FEATURES],validation_subset[target].to_numpy(),target,args.mlp_epochs)
                    history[target]=hist
                else:
                    if target=="Cd" and name!="mean_baseline":
                        model=TransformedTargetRegressor(regressor=model,func=np.log,inverse_func=np.exp)
                    model.fit(train[FEATURES],train[target])
                pred=model.predict(parts["validation"][FEATURES])
            if not np.isfinite(pred).all():
                raise ValueError(f"Nonfinite {target}/{name} predictions")
            score={"target":target,"model":name,**metrics(parts["validation"][target],pred),"training_seconds":time.perf_counter()-start}
            scores.append(score)
            fitted[name]=model
            joblib.dump(model,ROOT/f"models/candidates/{target}_{name}.joblib",compress=3)
            print(f"{target}/{name}: validation RMSE={score['RMSE']:.6f}, {score['training_seconds']:.1f}s",flush=True)
        chosen=min([s for s in scores if s["target"]==target],key=lambda s:s["RMSE"])["model"]
        selected[target]=chosen
        chosen_models[target]=fitted[chosen]
        for name in dict.fromkeys([chosen,"mean_baseline"]):
            pred=fitted[name].predict(parts["test"][FEATURES])
            test_scores.append({"target":target,"model":name,**metrics(parts["test"][target],pred)})
        pred=chosen_models[target].predict(parts["test"][FEATURES])
        predictions[target+"_predicted"]=pred
        predictions[target+"_residual"]=pred-parts["test"][target].to_numpy()
        y=parts["test"][target].to_numpy()
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].scatter(y,pred,s=3,alpha=.15,rasterized=True)
        lo,hi=min(y.min(),pred.min()),max(y.max(),pred.max())
        axes[0].plot([lo,hi],[lo,hi],"k--")
        axes[0].set(xlabel=f"Source {target}",ylabel=f"Predicted {target}",title="Unseen-airfoil test set")
        axes[1].scatter(y,pred-y,s=3,alpha=.15,rasterized=True)
        axes[1].axhline(0,color="black",linestyle="--")
        axes[1].set(xlabel=f"Source {target}",ylabel="Prediction minus source",title="Residuals")
        fig.tight_layout(); fig.savefig(out/f"plots/{target}_test.png",dpi=150); plt.close(fig)
    pd.DataFrame(scores).to_csv(out/"validation_metrics.csv",index=False)
    pd.DataFrame(test_scores).to_csv(out/"test_metrics.csv",index=False)
    predictions.to_csv(out/"test_predictions.csv",index=False)
    save_json(out/"mlp_training_history.json",history)
    source=json.loads((ROOT/"data/raw/source.json").read_text())
    card={"features":FEATURES,"targets":TARGETS,"selected_models":selected,"seed":SEED,
        "training_rows_available":len(parts["train"]),"training_rows_used":len(train),
        "training_sample_policy":"Fixed seed row sample of training airfoils only, bounded for laptop CPU; evaluation uses all held-out rows",
        "training_bounds":{c:[float(train[c].min()),float(train[c].max())] for c in FEATURES},
        "condition_bounds":{c:[float(train[c].min()),float(train[c].max())] for c in ["alpha","Re","Ncrit"]},
        "selection_rule":"Lowest full validation RMSE independently per target including mean; no test tuning and no train+validation refit",
        "mlp":"64,64 hidden layers; train-only feature/target StandardScaler; geometry-held-out validation early stopping; Cd trained in log space",
        "drag_transform":"log(Cd), inverse exp at inference except arithmetic mean baseline",
        "raw_sha256":source["sha256"],"source_revision":source["revision"],
        "membership_sha256":sha256(ROOT/"data/splits/membership.csv"),
        "polars_sha256":sha256(ROOT/"data/processed/polars.csv"),"shapes_sha256":sha256(ROOT/"data/processed/shapes.csv"),
        "python":platform.python_version(),"packages":{p:importlib.metadata.version(p) for p in ["numpy","pandas","scipy","scikit-learn","xgboost","matplotlib","joblib"]},
        "source":source,"data_audit":json.loads((out/"data_audit.json").read_text()),
        "limitations":["XFOIL source labels, not independent experiments","2D Cl/Cd/Cm only; no Cp model or whole-wing aerodynamic predictor",
            "Coordinate-proximity groups do not guarantee unrelated airfoil families","Feature bounds are insufficient to prove physical validity or supported condition combinations",
            "No Mach input; upstream generation code uses Mach 0","Surrogate search is exploratory; extreme conditions/stall and generated shapes need independent checking",
            "3D output is a parametric loft without material, structural, mass or flight labels"]}
    joblib.dump({"models":chosen_models,"metadata":card},ROOT/"models/airfoil_baseline.joblib",compress=3)
    save_json(ROOT/"models/model_card.json",card)
    train[["source_member","polar_index","shape_id"]].to_csv(ROOT/"data/splits/training_sample.csv",index=False)
    # Plots use the first four alphabetical test names and their most populated Re/Ncrit polar.
    names=sorted(parts["test"].name.unique())[:4]
    fig,axes=plt.subplots(3,len(names),figsize=(14,9),squeeze=False)
    for col,name in enumerate(names):
        p=parts["test"][parts["test"].name==name]
        re,ncrit=p.groupby(["Re","Ncrit"]).size().idxmax()
        p=p[(p.Re==re)&(p.Ncrit==ncrit)].sort_values("alpha")
        for row,t in enumerate(TARGETS):
            ax=axes[row,col]
            ax.plot(p.alpha,p[t],".-",label="Source XFOIL")
            ax.plot(p.alpha,chosen_models[t].predict(p[FEATURES]),"--",label="Model")
            ax.set(xlabel="Alpha (deg)",ylabel=t,title=f"{name.split(' - ')[0]}\nRe {re:g}, Ncrit {ncrit:g}")
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out/"plots/heldout_polars.png",dpi=150); plt.close(fig)
    print(pd.DataFrame(test_scores).to_string(index=False),flush=True)

if __name__=="__main__":
    main()
