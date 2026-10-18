"""Interpolate source surfaces and check geometry on a dense chord grid."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from common import X

def source_shape(record):
    surfaces = []
    for side in ["ss", "ps"]:
        x, y = np.asarray(record["x" + side], float), np.asarray(record["y" + side], float)
        if x.ndim != 1 or x.shape != y.shape or len(x) < 4 or not np.isfinite([x,y]).all():
            raise ValueError("Malformed source coordinates")
        order = np.argsort(x)
        x, y = x[order], y[order]
        if np.any(np.diff(x) <= 0) or x[0] > 1e-6 or x[-1] < 1-1e-6:
            raise ValueError("Coordinates must uniquely cover the unit chord")
        surfaces.append(PchipInterpolator(x, y, extrapolate=False)(X))
    shape = np.concatenate(surfaces)
    valid, _, _ = geometry_checks(shape[None, :], 0.001, 0.5)
    if not valid[0]:
        raise ValueError("Crossing or unsupported source geometry")
    return shape

def evaluate_surfaces(shapes, x=None):
    shapes = np.atleast_2d(np.asarray(shapes, float))
    x = X if x is None else np.asarray(x)
    n = len(X)
    return x, PchipInterpolator(X, shapes[:, :n], axis=1)(x), PchipInterpolator(X, shapes[:, n:], axis=1)(x)

def geometry_checks(shapes, minimum=0.01, maximum=0.4):
    shapes = np.atleast_2d(np.asarray(shapes, float))
    _, upper, lower = evaluate_surfaces(shapes, np.linspace(0, 1, 401))
    gap = upper - lower
    thickness = gap.max(axis=1)
    camber = np.abs((upper+lower)/2).max(axis=1)
    valid = np.isfinite(shapes).all(axis=1) & (gap.min(axis=1) >= -1e-7) & (thickness >= minimum) & (thickness <= maximum) & (camber <= 0.2)
    # A shared leading edge is required for a closed loft; finite trailing-edge thickness is preserved.
    valid &= np.abs(upper[:,0]-lower[:,0]) <= 1e-6
    return valid, thickness, camber
