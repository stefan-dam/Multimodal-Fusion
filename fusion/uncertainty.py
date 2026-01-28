from typing import List, Sequence, Optional
import numpy as np
from .expected_vad import VAD


def entropy_from_probs(probs: Sequence[float]) -> float:
    p = np.array(probs, dtype=np.float32)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def vad_ensemble_variance(vads: Sequence[VAD]) -> float:
    
    X = np.stack([v.as_np() for v in vads], axis=0)  # (M, 3)
    var = np.var(X, axis=0)  # (3,)
    return float(np.mean(var))


def l2_distance_vad(v1: VAD, v2: VAD) -> float:
    x1, x2 = v1.as_np(), v2.as_np()
    return float(np.linalg.norm(x1 - x2, ord=2))
