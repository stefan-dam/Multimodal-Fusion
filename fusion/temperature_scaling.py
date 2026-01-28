from __future__ import annotations
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

def softmax_temp(logits: np.ndarray, temperature: float) -> np.ndarray:
    x = logits / max(temperature, 1e-6)
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=-1, keepdims=True) + 1e-12)

def nll_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    N = probs.shape[0]
    p = probs[np.arange(N), labels]
    return float(-np.mean(np.log(np.clip(p, 1e-12, 1.0))))

@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def transform_logits(self, logits: np.darray) -> np.ndarray:
        return softmax_temp(logits, self.temperature)
    
    @staticmethod
    def fit(
        logits: np.ndarray,
        labels: np.ndarray,
        T_min: float = 0.5,
        T_max: float = 5.0,
        steps: int = 200,
    ) -> "TemperatureScaler":
        assert logits.ndim == 2
        assert labels.ndim == 1
        Ts = np.linspace(T_min, T_max, steps, dtype=np.float32)
        best_T, best_loss = 1.0, float("inf")
        
        for T in Ts:
            probs = softmax_temp(logits, float(T))
            loss = nll_loss(probs, labels)
            if loss < best_loss:
                best_loss = loss
                best_T = float(T)
                
        return TemperatureScaler(temperature=best_T)
    