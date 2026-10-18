"""Shared schema; geometry uses sampled coordinates, not the old CST coefficients."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
X = (1 - np.cos(np.linspace(0, np.pi, 41))) / 2
SHAPE_FEATURES = [f"upper_{i:02d}" for i in range(len(X))] + [f"lower_{i:02d}" for i in range(len(X))]
FEATURES = SHAPE_FEATURES + ["alpha", "log10_Re", "Ncrit"]
TARGETS = ["Cl", "Cd", "Cm"]
DATA_URL = "https://nasa-public-data.s3.amazonaws.com/plot3d_utilities/airfoil-learning-dataset.zip"

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def feature_rows(shapes, alpha, reynolds, ncrit):
    import pandas as pd
    shapes = np.atleast_2d(np.asarray(shapes, dtype=float))
    if shapes.shape[1] != len(SHAPE_FEATURES) or not np.isfinite(shapes).all():
        raise ValueError("Expected finite upper/lower ordinates on the saved 41-point chord grid.")
    if not np.isfinite([alpha, reynolds, ncrit]).all() or reynolds <= 0:
        raise ValueError("Conditions must be finite and Reynolds number must be positive.")
    return pd.DataFrame(np.column_stack([shapes, np.tile([alpha, np.log10(reynolds), ncrit], (len(shapes), 1))]), columns=FEATURES)
