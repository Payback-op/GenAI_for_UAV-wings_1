"""Summarize XFLR5's exported results without substituting model predictions."""
from pathlib import Path
import io
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
lines = (folder/'image_airfoil_polar.txt').read_text().splitlines()
start = next(i for i, line in enumerate(lines) if line.strip().startswith('-------')) + 1
values = np.loadtxt(io.StringIO('\n'.join(lines[start:])))
assert values.shape[0] == 29
assert np.allclose(values[:, 0], np.arange(-4, 10.01, .5))
assert np.isfinite(values).all() and (values[:, 2] > 0).all()
# The first eight columns have explicit headers in the XFLR5 export.
frame = pd.DataFrame(values[:, :8], columns=['alpha_deg','Cl','Cd','Cdp','Cm','top_transition_x_over_c','bottom_transition_x_over_c','Cp_min'])
frame['Cl_over_Cd'] = frame.Cl/frame.Cd
frame.to_csv(folder/'image_airfoil_polar.csv', index=False)
at4 = frame.loc[frame.alpha_deg.eq(4)].iloc[0]
best = frame.loc[frame.Cl_over_Cd.idxmax()]
result = {
    'software': 'XFLR5 v6.62 / XFOIL',
    'foil': 'Image_Airfoil_Digitized', 'source': 'source_image.png',
    'Re': 100000, 'Mach': 0, 'Ncrit': 9,
    'forced_transition_top_bottom': [1, 1],
    'converged_points': len(frame), 'alpha_range_deg': [-4, 10], 'alpha_step_deg': .5,
    'at_four_degrees': at4.to_dict(),
    'best_lift_to_drag_among_sampled_points': best.to_dict(),
    'prior_section_targets_at_four_degrees': {'Cl_min': .55, 'Cl_max': .65, 'Cd_max': .03},
    'prior_targets_met_at_four_degrees': bool(.55 <= at4.Cl <= .65 and at4.Cd <= .03),
    'limits': ['Digitized and smoothed image geometry', 'Single discretization; panel convergence not established',
               'XFOIL calculation, not experimental validation', 'Section coefficients, not whole-wing performance',
               'Best sampled lift-to-drag is not a global optimum; no stall angle established'],
}
(folder/'analysis_summary.json').write_text(json.dumps(result, indent=2))
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, y, label in zip(axes.flat, ['Cl','Cd','Cm','Cl_over_Cd'], ['Lift coefficient Cl','Drag coefficient Cd','Pitching moment Cm','Lift / drag Cl/Cd']):
    ax.plot(frame.alpha_deg, frame[y], '-o', color='#1d4ed8', markersize=3)
    ax.axvline(4, linestyle='--', color='#64748b', linewidth=1)
    ax.scatter([4], [at4[y]], color='#dc2626', zorder=5)
    ax.set(xlabel='Angle of attack (degrees)', ylabel=label)
    ax.grid(alpha=.2)
axes[0, 0].axhspan(.55, .65, color='#16a34a', alpha=.12, label='Prior lift target')
axes[0, 0].legend(loc='upper left')
axes[0, 1].axhline(.03, color='#16a34a', linestyle=':', label='Prior drag limit')
axes[0, 1].legend(loc='upper left')
fig.suptitle('Image-derived airfoil | XFLR5 / XFOIL\nRe = 100,000 | Mach = 0 | Ncrit = 9 | red marker: 4 degrees')
fig.text(.5, .015, 'Digitized geometry approximation; section analysis. Panel sensitivity and experimental validation remain open.', ha='center', fontsize=9)
fig.tight_layout(rect=(0,.04,1,.93))
fig.savefig(folder/'performance_curves.png', dpi=160)
plt.close(fig)
report = f'''# Analysis of the airfoil in image.png

The image outline was digitized using its labeled x/chord and y/chord axes, smoothed, and imported into XFLR5 as `Image_Airfoil_Digitized`. This is an image-derived approximation, not a recovery of the original numerical geometry. See `trace_check.png` for the fitted outline over the source image and `digitization.json` for calibration and fit errors.

XFLR5 v6.62 calculated a Type 1 polar at Re = 100,000, Mach = 0 and Ncrit = 9, with forced transition locations 1.0 on both surfaces. These are the previous comparison conditions; the image does not specify them. All 29 requested points, -4 to 10 degrees at 0.5 degree intervals, produced stored polar results.

At 4 degrees: Cl = {at4.Cl:.4f}, Cd = {at4.Cd:.5f}, Cm = {at4.Cm:.4f}, and Cl/Cd = {at4.Cl_over_Cd:.2f}. Against the previous design targets, drag is below 0.03, while lift is below the 0.55 lower bound. The requested lift range was 0.55 to 0.65.

The largest Cl/Cd among the sampled points is {best.Cl_over_Cd:.2f} at {best.alpha_deg:g} degrees. This is not a global optimum or a determination of stall.

Open `image_airfoil_analysis.xfl` in XFLR5 for the saved foils and polar. The project also retains the earlier coordinate-based foil for reference. `image_airfoil.dat` contains only the traced image geometry. `image_airfoil_polar.txt` is the native numerical export; `image_airfoil_polar.csv` is its first eight named columns plus calculated Cl/Cd. `performance_curves.png` plots the exported coefficients.

Results depend on image calibration, tracing/smoothing and numerical paneling. Panel convergence has not been established. These are XFOIL section calculations, not experiments or full-wing aerodynamic validation.
'''
(folder/'ANALYSIS.md').write_text(report)
print(json.dumps(result, indent=2))
