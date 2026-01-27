from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np

@dataclass(frozen=True)
class VAD:
    v: float
    a: float
    d: float
    
    def as_np(self) -> np.ndarray:
        return np.array([self.v, self.a, self.d], dtype=np.float32)
    
def softmax(logits: Sequence[float]) -> np.ndarray:
    x = np.array(logits, dtype=np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)
    
def expected_vad_from_probs(probs: Sequence[float], anchors: Sequence[VAD]) -> VAD:
    p = np.array(probs, dtype=np.float32)
    A = np.array([anchor.as_np() for anchor in anchors], axis=0)
    vad = p @ A
    vad = np.clip(vad, 0.0, 1.0)
    return VAD(float(vad[0]), float(vad[1]), float(vad[2]))
    
def expected_vad_from_logits(logits: Sequence[float], anchors: Sequence[VAD]) -> VAD:
    p = softmax(logits)
    return expected_vad_from_probs(p, anchors)