"""Closed-form rectangular Procrustes front-end for revision experiments.

Standard alignment primitive, not a new algorithm. It is an equal-information
stress test for the established 011 downstream estimator: paired old/new feature
anchors in, old->new feature map out. It never sees labels, revision requests,
or evaluation outcomes.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ProcrustesMap:
    old_mean: np.ndarray
    new_mean: np.ndarray
    rotation: np.ndarray
    scale: float
    pilot_bias: np.ndarray
    mode: str

def _matrix(x: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite nonempty 2-D array")
    return a

def fit_rectangular_procrustes(old_fit, new_fit, *, mode="isometry",
                               old_pilot=None, new_pilot=None) -> ProcrustesMap:
    x = _matrix(old_fit, "old_fit")
    y = _matrix(new_fit, "new_fit")
    if x.shape[0] != y.shape[0]:
        raise ValueError("old_fit and new_fit must contain the same paired rows")
    if x.shape[1] > y.shape[1]:
        raise ValueError("This 011 front-end is restricted to old_dim <= new_dim")
    if x.shape[0] < 2:
        raise ValueError("At least two paired anchors are required")
    if mode not in {"isometry", "similarity"}:
        raise ValueError("mode must be 'isometry' or 'similarity'")

    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    u, svals, vt = np.linalg.svd(xc.T @ yc, full_matrices=False)
    r = u @ vt
    if mode == "isometry":
        scale = 1.0
    else:
        denom = float(np.sum(xc * xc))
        if denom <= np.finfo(np.float64).tiny:
            raise ValueError("old_fit has no centered variation")
        scale = float(np.sum(svals) / denom)
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("invalid fitted global scale")

    bias = np.zeros(y.shape[1], dtype=np.float64)
    if (old_pilot is None) != (new_pilot is None):
        raise ValueError("Provide both pilot arrays or neither")
    if old_pilot is not None:
        xp = _matrix(old_pilot, "old_pilot")
        yp = _matrix(new_pilot, "new_pilot")
        if xp.shape[0] != yp.shape[0] or xp.shape[1] != x.shape[1] or yp.shape[1] != y.shape[1]:
            raise ValueError("pilot dimensions must match fit spaces and pairing")
        base = my + scale * (xp - mx) @ r
        bias = (yp - base).mean(0)

    for a in (mx, my, r, bias):
        a.setflags(write=False)
    return ProcrustesMap(mx, my, r, scale, bias, mode)

def apply_procrustes(model: ProcrustesMap, old_features: np.ndarray) -> np.ndarray:
    x = _matrix(old_features, "old_features")
    if x.shape[1] != model.rotation.shape[0]:
        raise ValueError("old feature dimension does not match fitted map")
    z = model.new_mean + model.pilot_bias + model.scale * (x - model.old_mean) @ model.rotation
    if not np.isfinite(z).all():
        raise FloatingPointError("nonfinite mapped feature")
    return z

def geometry_error(model: ProcrustesMap) -> float:
    r = model.rotation
    return float(np.linalg.norm(r @ r.T - np.eye(r.shape[0]), ord="fro"))
