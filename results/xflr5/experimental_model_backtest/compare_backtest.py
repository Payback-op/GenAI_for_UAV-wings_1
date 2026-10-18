"""Compare the frozen experimental-model design to native XFLR5 polar exports."""
from pathlib import Path
import io
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from common import feature_rows, sha256
from predict import load_bundle, predict_frame

folder = Path(__file__).resolve().parent
design = json.loads((folder/'design.json').read_text())
assert sha256(Path(design['predictor_path'])) == design['predictor_sha256']
bundle = load_bundle(design['predictor_path'])
shape = np.load(folder/'airfoil_shape.npy')
a = design['conditions']
panels = [81, 160, 240, 280]
all_frames, study, sweep_scores = [], [], []
for count in panels:
    label = 'original81' if count == 81 else f'refined{count}'
    path = folder/f'polar_{label}.txt'
    lines = path.read_text().splitlines()
    assert f'Experimental_Model_{"Original81" if count==81 else "Refined"+str(count)}' in '\n'.join(lines)
    assert 'Re =     0.100 e 6' in '\n'.join(lines) and 'Ncrit =   9.000' in '\n'.join(lines)
    start = next(i for i, line in enumerate(lines) if line.strip().startswith('-------'))+1
    data = np.loadtxt(io.StringIO('\n'.join(lines[start:])))
    frame = pd.DataFrame(data[:, :8], columns=['alpha_deg','Cl','Cd','Cdp','Cm','top_xtr','bottom_xtr','Cp_min'])
    assert frame.alpha_deg.is_unique and np.isfinite(data).all() and (frame.Cd>0).all()
    assert 4 in frame.alpha_deg.values
    inputs = pd.concat([feature_rows(shape, alpha, a['reynolds_number'], a['ncrit']) for alpha in frame.alpha_deg], ignore_index=True)
    pred = predict_frame(inputs, bundle)
    frame['panels'] = count
    for target in ['Cl','Cd','Cm']:
        frame[f'{target}_model'] = pred[target].to_numpy()
        frame[f'{target}_signed_error'] = frame[f'{target}_model']-frame[target]
        frame[f'{target}_absolute_error'] = frame[f'{target}_signed_error'].abs()
        row = frame[frame.alpha_deg.eq(4)].iloc[0]
        study.append({'panels': count, 'coefficient': target, 'model': float(row[f'{target}_model']),
                      'xflr5': float(row[target]), 'absolute_error': float(row[f'{target}_absolute_error']),
                      'relative_absolute_error_percent': float(100*row[f'{target}_absolute_error']/abs(row[target]))})
        err = frame[f'{target}_signed_error'].to_numpy()
        sweep_scores.append({'panels': count, 'target': target, 'converged_points': len(frame),
                             'MAE': float(np.abs(err).mean()), 'RMSE': float(np.sqrt(np.mean(err**2)))})
    all_frames.append(frame)
combined = pd.concat(all_frames, ignore_index=True)
combined.to_csv(folder/'solver_model_comparison.csv', index=False)
study = pd.DataFrame(study)
study.to_csv(folder/'design_point_panel_study.csv', index=False)
pd.DataFrame(sweep_scores).to_csv(folder/'sweep_error_metrics.csv', index=False)
fine = all_frames[-1]
at4 = fine[fine.alpha_deg.eq(4)].iloc[0]
previous = all_frames[-2].loc[lambda p:p.alpha_deg.eq(4)].iloc[0]
changes = {t: {'absolute': float(abs(at4[t]-previous[t])),
               'percent_of_280_panel_value': float(100*abs(at4[t]-previous[t])/abs(at4[t]))}
           for t in ['Cl','Cd','Cm']}
summary = {
    'reference': 'XFLR5 v6.62, 280-panel refinement of the original exported foil',
    'model_path': design['predictor_path'], 'model_sha256': design['predictor_sha256'],
    'conditions': {'alpha_deg': 4, 'Re': 100000, 'Mach': 0, 'Ncrit': 9, 'xtr_top': 1, 'xtr_bottom': 1},
    'geometry_provenance': design['geometry_provenance'],
    'at_design_point': study[study.panels.eq(280)].to_dict('records'),
    'panel_change_240_to_280': changes,
    'solver_target_match_at_four_degrees': bool(abs(at4.Cl-a['target_cl'])<=a['cl_tolerance'] and at4.Cd<=a['maximum_cd']),
    'converged_counts': {str(n):len(p) for n,p in zip(panels,all_frames)},
    'missing_requested_angles': {str(n): sorted(set(np.arange(-4,10.01,.5))-set(p.alpha_deg)) for n,p in zip(panels,all_frames)},
    'error_definition': 'Absolute(model minus XFLR5) / absolute(XFLR5) * 100 at the specified point',
    'limits': ['Single selected catalog airfoil, present in the training split',
               'Fresh solver consistency check, not experimental validation or overall model accuracy',
               'Refinement interpolates the original geometry; finite discretization sensitivity remains',
               'Cm in the experimental bundle is the retained baseline predictor',
               '300-panel initialization was rejected by XFLR5; no results from that foil are used'],
}
(folder/'backtest_summary.json').write_text(json.dumps(summary, indent=2))
plt.rcParams.update({'font.size':10, 'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(2, 2, figsize=(12,8))
for ax,t,name in zip(axes.flat[:3], ['Cl','Cd','Cm'], ['Lift coefficient Cl','Drag coefficient Cd','Moment coefficient Cm']):
    ax.plot(fine.alpha_deg, fine[f'{t}_model'], '--', color='#1d4ed8', label='Experimental model')
    ax.plot(fine.alpha_deg, fine[t], '-o', markersize=3, color='#ea580c', label='XFLR5 (280 panels)')
    ax.axvline(4, color='#64748b', linestyle=':', linewidth=1)
    ax.set(xlabel='Angle of attack (degrees)',ylabel=name); ax.grid(alpha=.2); ax.legend(fontsize=9)
ax = axes.flat[3]
width=.24; positions=np.arange(3)
for j,count in enumerate([160,240,280]):
    vals=study[study.panels.eq(count)].set_index('coefficient').loc[['Cl','Cd','Cm'],'relative_absolute_error_percent']
    ax.bar(positions+(j-1)*width, vals, width, label=f'{count} panels')
ax.set_xticks(positions,['Cl','Cd','Cm']); ax.set(ylabel='Absolute error relative to XFLR5 (%)',title='At 4 degrees'); ax.legend(); ax.grid(axis='y',alpha=.2)
fig.suptitle('Experimental model back-test | selected GOE 766 airfoil\nRe = 100,000 | Mach = 0 | Ncrit = 9 | exact export, no image tracing')
fig.text(.5,.015,'One training-catalog shape: these errors are not an overall accuracy estimate.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.93)); fig.savefig(folder/'backtest_comparison.png',dpi=160); plt.close(fig)
report = ['# Experimental model versus fresh XFLR5 analysis', '',
          'The experimental bundle was used to select the airfoil before any solver outcomes were read. It selected GOE 766, a known training-catalog geometry, not a novel blend. The production model was preserved.', '',
          'Conditions: alpha 4 degrees, Re 100,000, Mach 0, Ncrit 9, forced transition at x/c=1 on both surfaces. XFLR5 v6.62 used Type 1 analysis.', '',
          '| Coefficient | Experimental model | XFLR5, 280 panels | Absolute difference | Relative difference |',
          '|---|---:|---:|---:|---:|']
for r in summary['at_design_point']:
    report.append(f"| {r['coefficient']} | {r['model']:.6f} | {r['xflr5']:.6f} | {r['absolute_error']:.6f} | {r['relative_absolute_error_percent']:.2f}% |")
report += ['', 'Percentages use absolute XFLR5 values as the denominator. A large percentage for a small moment coefficient should be read with its absolute difference.', '',
           '## Numerical refinement', '', '| Panels | Cl at 4 deg | Cd at 4 deg | Cm at 4 deg |', '|---|---:|---:|---:|']
for count,p in zip(panels,all_frames):
    r=p[p.alpha_deg.eq(4)].iloc[0]
    report.append(f'| {count} | {r.Cl:.6f} | {r.Cd:.6f} | {r.Cm:.6f} |')
report += ['', 'Each refinement was made independently from the original 81-point export with XFLR5 Global Panel Refinement, retaining its other defaults. The rejected 300-panel foil has no analysis results. See backtest_summary.json for 240-to-280 changes; this finite check does not prove zero discretization error.', '',
           f"The finest solver result {'meets' if summary['solver_target_match_at_four_degrees'] else 'does not meet'} the requested lift range 0.55-0.65 and drag limit 0.03 at four degrees.", '',
           '## Scope', '',
           'These are discrepancies for one selected, already-known airfoil. They do not establish the overall error rate, generalization to new shapes, or physical flight performance. The experimental bundle uses newly trained Cl and Cd predictors and retains the original Cm predictor. No XFLR5 outcomes were used to retrain, retune, or reselect this airfoil.', '',
           'The full sweeps cover -4 to 10 degrees in 0.5 degree steps. Only stored, converged polar points are compared; missing angles are listed in backtest_summary.json. Comparison plots use the unchanged exported geometry as model input and XFLR5 refinement of that export as solver geometry.', '',
           'Files: airfoil.dat and airfoil_coordinates.csv (generated section); design.json (frozen model prediction and provenance); polar_*.txt (native XFLR5 exports); solver_model_comparison.csv (all matched predictions and residuals); design_point_panel_study.csv (point errors); model_backtest.xfl (saved XFLR5 project).']
(folder/'BACKTEST_REPORT.md').write_text('\n'.join(report))
print(json.dumps(summary,indent=2))
