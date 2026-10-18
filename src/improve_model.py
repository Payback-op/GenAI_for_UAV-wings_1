"""Isolated full-training-data experiment; never replaces the production bundle.

Select against the existing baseline using full validation RMSE, then evaluate
the locked selections once on the unchanged test airfoils. No XFLR5 check data
is added to training. Run with the project's virtual environment.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from common import ROOT, FEATURES, SHAPE_FEATURES, TARGETS, SEED, save_json, sha256, feature_rows
from train import metrics


def fit_candidate(train, validation, target, max_epochs, folder):
    """Train-only scaling; geometry-held-out early stopping in physical units."""
    scaler = StandardScaler().fit(train[FEATURES])
    x = scaler.transform(train[FEATURES]).astype(np.float32)
    xv = scaler.transform(validation[FEATURES]).astype(np.float32)
    y = train[target].to_numpy()
    yt = np.log(y) if target == 'Cd' else y
    ys = StandardScaler().fit(yt.reshape(-1, 1))
    yy = ys.transform(yt.reshape(-1, 1)).ravel().astype(np.float32)
    yv = validation[target].to_numpy()
    net = MLPRegressor(hidden_layer_sizes=(64, 64), batch_size=1024,
                       learning_rate_init=.0005, alpha=.01,
                       random_state=SEED, early_stopping=False)
    best, best_rmse, stale, history = None, float('inf'), 0, []
    started = time.perf_counter()
    for epoch in range(1, max_epochs+1):
        net.partial_fit(x, yy)
        pred = ys.inverse_transform(net.predict(xv).reshape(-1, 1)).ravel()
        if target == 'Cd':
            pred = np.exp(pred)
        if not np.isfinite(pred).all():
            raise ValueError(f'Nonfinite candidate predictions for {target}')
        rmse = metrics(yv, pred)['RMSE']
        history.append({'epoch': epoch, 'validation_RMSE': rmse})
        if rmse < best_rmse:
            best_rmse, best, stale = rmse, deepcopy(net), 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0 or stale >= 12:
            print(f'{target}: epoch {epoch}, validation RMSE {rmse:.6f}, '
                  f'best {best_rmse:.6f}, {time.perf_counter()-started:.1f}s', flush=True)
            save_json(folder/f'{target}_history.json', history)
        if stale >= 12:
            break
    pipeline = make_pipeline(scaler, best)
    wrapped = TransformedTargetRegressor(regressor=pipeline, transformer=ys)
    wrapped.regressor_, wrapped.transformer_, wrapped._training_dim = pipeline, ys, 1
    if target == 'Cd':
        outer = TransformedTargetRegressor(regressor=wrapped, func=np.log, inverse_func=np.exp)
        outer.regressor_ = wrapped
        outer.transformer_ = FunctionTransformer(np.log, np.exp, validate=True).fit(y.reshape(-1, 1))
        outer._training_dim = 1
        wrapped = outer
    save_json(folder/f'{target}_history.json', history)
    return wrapped, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder = Path(args.output) if args.output else ROOT/'results/experiments'/f'full_training_{stamp}'
    folder.mkdir(parents=True, exist_ok=False)
    baseline_path = ROOT/'models/airfoil_baseline.joblib'
    original_sha = sha256(baseline_path)
    baseline = joblib.load(baseline_path)
    polars = pd.read_csv(ROOT/'data/processed/polars.csv')
    shapes = pd.read_csv(ROOT/'data/processed/shapes.csv')
    data = polars.merge(shapes[['shape_id']+SHAPE_FEATURES], on='shape_id', validate='many_to_one')
    parts = {s: data[data.split.eq(s)].copy() for s in ['train', 'validation', 'test']}
    groups = {s: set(p.geometry_group) for s, p in parts.items()}
    assert not groups['train'] & groups['validation']
    assert not groups['train'] & groups['test']
    assert not groups['validation'] & groups['test']
    manifest = {
        'baseline_sha256': original_sha, 'code_sha256': sha256(Path(__file__)),
        'membership_sha256': sha256(ROOT/'data/splits/membership.csv'),
        'polars_sha256': sha256(ROOT/'data/processed/polars.csv'),
        'shapes_sha256': sha256(ROOT/'data/processed/shapes.csv'),
        'seed': SEED, 'max_epochs': args.epochs,
        'candidate': '64/64 MLP; batch 1024; learning rate 0.0005; L2 0.01; patience 12',
        'split_counts': {s: {'rows': len(p), 'airfoils': p.shape_id.nunique()} for s, p in parts.items()},
        'selection': 'Per-target full validation RMSE against unchanged baseline, before any test evaluation',
        'test_note': 'Same previously reported test set; not a new independent confirmation set',
        'packages': {p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','joblib']},
        'production_replaced': False,
    }
    (folder/'training_source.py').write_text(Path(__file__).read_text())
    save_json(folder/'manifest.json', manifest)
    print(f'Experiment: {folder}', flush=True)
    print(f'Train: {len(parts["train"]):,}; validation: {len(parts["validation"]):,}; '
          'test excluded from selection.', flush=True)
    selected, selections, validation_scores, histories = {}, {}, [], {}
    with threadpool_limits(limits=4):
        for target in TARGETS:
            base = baseline['models'][target]
            old_pred = base.predict(parts['validation'][FEATURES])
            old_score = metrics(parts['validation'][target], old_pred)
            print(f'{target}: baseline validation RMSE {old_score["RMSE"]:.6f}', flush=True)
            candidate, hist = fit_candidate(parts['train'], parts['validation'], target, args.epochs, folder)
            histories[target] = hist
            candidate_file = folder/f'{target}_full_training.joblib'
            joblib.dump(candidate, candidate_file, compress=3)
            # Verify persistence and finite predictions on a real validation batch.
            reloaded = joblib.load(candidate_file)
            check = parts['validation'][FEATURES].iloc[:128]
            np.testing.assert_allclose(reloaded.predict(check), candidate.predict(check), rtol=1e-7, atol=1e-8)
            pred = reloaded.predict(parts['validation'][FEATURES])
            new_score = metrics(parts['validation'][target], pred)
            improved = new_score['RMSE'] < old_score['RMSE']
            name = 'full_training_mlp' if improved else 'existing_mlp'
            selected[target] = reloaded if improved else base
            selections[target] = name
            for model_name, score in [('existing_mlp', old_score), ('full_training_mlp', new_score)]:
                validation_scores.append({'target': target, 'model': model_name, **score})
            pd.DataFrame(validation_scores).to_csv(folder/'validation_metrics.csv', index=False)
            print(f'{target}: candidate {new_score["RMSE"]:.6f}; selected {name}', flush=True)
        # Freeze decisions before reading test targets for evaluation.
        save_json(folder/'selection.json', selections)
        test_scores, predictions = [], parts['test'][['shape_id','name','alpha','Re','Ncrit']+TARGETS].copy()
        region = parts['test'].Re.eq(100000) & parts['test'].Ncrit.eq(9) & parts['test'].alpha.between(-4, 8)
        region_scores = []
        for target in TARGETS:
            for name, model in [('existing_mlp', baseline['models'][target]), ('selected', selected[target])]:
                pred = model.predict(parts['test'][FEATURES])
                test_scores.append({'target': target, 'model': name, **metrics(parts['test'][target], pred)})
                region_scores.append({'target': target, 'model': name,
                                      **metrics(parts['test'].loc[region,target], pred[region])})
                predictions[f'{target}_{name}'] = pred
        pd.DataFrame(test_scores).to_csv(folder/'test_metrics.csv', index=False)
        pd.DataFrame(region_scores).to_csv(folder/'example_region_metrics.csv', index=False)
        predictions.to_csv(folder/'test_predictions.csv', index=False)
        metadata = deepcopy(baseline['metadata'])
        metadata.update({'selected_models': selections, 'experiment': manifest,
                         'training_rows_used': len(parts['train']),
                         'training_rows_used_per_target': {t: len(parts['train']) if selections[t]=='full_training_mlp'
                                                          else baseline['metadata']['training_rows_used'] for t in TARGETS},
                         'training_sample_policy': 'All training rows for new candidates; retained baseline uses its saved 80k sample',
                         'selection_rule': manifest['selection'],
                         'mlp': manifest['candidate'],
                         'training_bounds': {c: [float(parts['train'][c].min()),float(parts['train'][c].max())] for c in FEATURES},
                         'condition_bounds': {c: [float(parts['train'][c].min()),float(parts['train'][c].max())] for c in ['alpha','Re','Ncrit']}})
        bundle = {'models': selected, 'metadata': metadata}
        joblib.dump(bundle, folder/'airfoil_candidate.joblib', compress=3)
        save_json(folder/'model_card.json', metadata)
        # Descriptive prediction only; the generated design is never a tuning label.
        design = json.loads((ROOT/'results/generated/latest.json').read_text())
        shape = np.load(Path(design['output_folder'])/'airfoil_shape.npy')
        cond = design['conditions']['airfoil']
        inputs = feature_rows(shape, cond['angle_of_attack_deg'], cond['reynolds_number'], cond['ncrit'])
        design_predictions = {t: {'baseline': float(baseline['models'][t].predict(inputs)[0]),
                                   'candidate': float(selected[t].predict(inputs)[0])} for t in TARGETS}
        save_json(folder/'existing_design_predictions.json', design_predictions)
    assert sha256(baseline_path) == original_sha, 'Production model changed during experiment'
    validation_table, test_table = pd.DataFrame(validation_scores), pd.DataFrame(test_scores)
    report = ['# Full training data experiment', '',
              'Production files were preserved. Selection used validation airfoils only. The test set is the same previously reported set, not a new independent confirmation set.', '',
              f'Training rows: {len(parts["train"]):,} (baseline: {baseline["metadata"]["training_rows_used"]:,}).', '',
              '| Target | Baseline test RMSE | Selected test RMSE | Change in RMSE |',
              '|---|---:|---:|---:|']
    for target in TARGETS:
        p = test_table[test_table.target.eq(target)].set_index('model')
        old, new = p.loc['existing_mlp','RMSE'], p.loc['selected','RMSE']
        report.append(f'| {target} | {old:.6f} | {new:.6f} | {(new/old-1)*100:+.1f}% |')
    regional = pd.DataFrame(region_scores)
    report += ['', '## Fixed example operating region', '',
               'Descriptive subset fixed in the original baseline report: Re=100,000, Ncrit=9, alpha from -4 to 8 degrees. These test results do not select models.', '',
               '| Target | Baseline regional RMSE | Selected regional RMSE | Change in RMSE |',
               '|---|---:|---:|---:|']
    for target in TARGETS:
        p = regional[regional.target.eq(target)].set_index('model')
        old, new = p.loc['existing_mlp','RMSE'], p.loc['selected','RMSE']
        report.append(f'| {target} | {old:.6f} | {new:.6f} | {(new/old-1)*100:+.1f}% |')
    worse = [t for t in TARGETS if test_table[(test_table.target==t)&(test_table.model=='selected')].RMSE.iloc[0]
             > test_table[(test_table.target==t)&(test_table.model=='existing_mlp')].RMSE.iloc[0]]
    report += ['', '## Interpretation', '',
               ('Full-envelope test RMSE increased for '+', '.join(worse)+'. Keep this bundle experimental; it is not an overall upgrade.'
                if worse else 'No target had increased full-envelope test RMSE. Further solver checks are still required.'), '',
               'The production model is unchanged. A lower regional error does not establish accuracy at every point in that region.']
    report += ['', '## Validation selections', '', json.dumps(selections, indent=2), '',
               'The candidate bundle has the same 85-feature inference interface. This experiment does not prove agreement with a new solver or generated blends. No image/XFLR5 labels were included in training. Panel-converged original-geometry solver checks and additional unseen geometries remain necessary.', '',
               'See validation_metrics.csv, test_metrics.csv, example_region_metrics.csv and existing_design_predictions.json. Negative RMSE change means lower error; it is not an accuracy percentage.']
    (folder/'REPORT.md').write_text('\n'.join(report))
    save_json(ROOT/'results/experiments/latest.json', {'output_folder': str(folder.resolve()), 'selections': selections, 'production_replaced': False})
    print(test_table.to_string(index=False), flush=True)
    print(f'Complete. Production preserved. Report: {folder/"REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
