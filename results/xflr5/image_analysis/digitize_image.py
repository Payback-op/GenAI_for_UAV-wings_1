"""Digitize the supplied, axis-calibrated airfoil plot; never modify its source."""
from pathlib import Path
import hashlib
import json
import math
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SOURCE = Path(r'C:\Users\vaibhav\Desktop\image.png')
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
rgb = np.asarray(Image.open(SOURCE).convert('RGB'))
# Read from the plot's labeled grid: x/c=0 at 131.5, x/c=1 at 904.0;
# y/c=0 at 215.5, adjacent 0.05 ticks separated by 38.625 pixels.
X0, X1, Y0, YSCALE = 131.5, 904.0, 215.5, 772.5
mask = (rgb[:, :, 2] > 150) & (rgb[:, :, 0] < 90) & (rgb[:, :, 1] < 160)
rows = []
for col in range(134, 895):
    ys = np.flatnonzero(mask[145:260, col]) + 145
    if len(ys) < 2:
        continue
    # Separate upper and lower strokes at their largest vertical gap.
    gaps = np.diff(ys)
    if gaps.max() < 3:
        continue
    cut = np.argmax(gaps) + 1
    upper, lower = ys[:cut].mean(), ys[cut:].mean()
    rows.append([(col-X0)/(X1-X0), (Y0-upper)/YSCALE, (Y0-lower)/YSCALE])
raw = np.asarray(rows)

def basis(x, degree=10):
    x = np.asarray(x)
    return np.column_stack([
        np.sqrt(x)*(1-x)*math.comb(degree, k)*x**k*(1-x)**(degree-k)
        for k in range(degree+1)
    ])

# Class/shape fit makes the nose smooth while avoiding pixel stair-stepping.
# Both endpoints lie on the plotted chord; the image resolves no TE gap.
coef = np.linalg.lstsq(basis(raw[:, 0]), raw[:, 1:], rcond=None)[0]
x = (1-np.cos(np.linspace(0, np.pi, 81)))/2
surfaces = basis(x) @ coef
dense_x = np.linspace(0, 1, 2001)
dense = basis(dense_x) @ coef
gap = dense[:, 0] - dense[:, 1]
assert gap.min() >= -1e-8, 'Crossed surfaces'
assert np.isfinite(surfaces).all()
residual = (basis(raw[:, 0]) @ coef - raw[:, 1:])*YSCALE
fit_rms = np.sqrt(np.mean(residual**2, axis=0))
assert fit_rms.max() < 0.6, 'Trace needs inspection'
ring = np.column_stack([
    np.r_[x[::-1], x[1:]],
    np.r_[surfaces[::-1, 0], surfaces[1:, 1]],
])
np.savetxt(OUT/'image_airfoil.dat', ring, header='Image_Airfoil_Digitized', comments='', fmt='%.9f')
np.savetxt(OUT/'digitized_surfaces.csv', np.column_stack([x, surfaces]), delimiter=',',
           header='x_over_chord,upper_y_over_chord,lower_y_over_chord', comments='')
np.savetxt(OUT/'raw_pixel_trace.csv', raw, delimiter=',', header='x_over_chord,upper_y_over_chord,lower_y_over_chord', comments='')
shutil.copyfile(SOURCE, OUT/'source_image.png')
metadata = {
    'source': str(SOURCE),
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'method': 'Axis-calibrated blue-stroke extraction and degree-10 CST least-squares smoothing',
    'calibration_pixels': {'x_zero': X0, 'x_one': X1, 'y_zero': Y0, 'pixels_per_unit_y': YSCALE},
    'trace_columns': len(raw), 'coordinate_rows_in_dat': len(ring),
    'fit_rms_pixels_upper_lower': fit_rms.tolist(),
    'max_abs_fit_residual_pixels': float(np.abs(residual).max()),
    'max_thickness_ratio': float(gap.max()),
    'max_camber_ratio': float(((dense[:, 0]+dense[:, 1])/2).max()),
    'approx_one_pixel_chord_fraction': 1/YSCALE,
    'limitations': ['Image-derived approximation, not original CAD coordinates',
                    'Axis calibration and line width limit geometric precision',
                    'Closed trailing edge assumed because no finite gap is resolved in the image'],
    'analysis_conditions': {'alpha_deg': 4, 'Re': 100000, 'Mach': 0, 'Ncrit': 9, 'xtr_top': 1, 'xtr_bottom': 1},
    'condition_note': 'Reused prior comparison conditions; these cannot be inferred from the image',
}
(OUT/'digitization.json').write_text(json.dumps(metadata, indent=2))
fig, ax = plt.subplots(figsize=(12, 5.4))
ax.imshow(rgb)
px = X0+dense_x*(X1-X0)
ax.plot(px, Y0-dense[:, 0]*YSCALE, color='#ef4444', linewidth=1, linestyle='--', label='Digitized outline')
ax.plot(px, Y0-dense[:, 1]*YSCALE, color='#ef4444', linewidth=1, linestyle='--')
ax.set_xlim(95, 940); ax.set_ylim(300, 130)
ax.legend(); ax.set_title('Image tracing check: dashed red reconstruction over original blue outline')
ax.set_axis_off(); fig.tight_layout(); fig.savefig(OUT/'trace_check.png', dpi=150); plt.close(fig)
print(json.dumps(metadata, indent=2))
