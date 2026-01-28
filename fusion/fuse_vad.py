from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .expected_vad import VAD
from .uncertainty import l2_distance_vad


@dataclass
class FusionResult:
    vad_fused: VAD
    w_text: float
    w_audio: float
    disagreement: float
    incongruent: bool


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def static_fuse(v_text: VAD, v_audio: VAD, w_text: float = 0.5) -> FusionResult:
    w_text = _clamp01(w_text)
    w_audio = 1.0 - w_text

    vf = VAD(
        v=w_text * v_text.v + w_audio * v_audio.v,
        a=w_text * v_text.a + w_audio * v_audio.a,
        d=w_text * v_text.d + w_audio * v_audio.d,
    )

    delta = l2_distance_vad(v_text, v_audio)
    return FusionResult(vad_fused=vf, w_text=w_text, w_audio=w_audio,
                        disagreement=delta, incongruent=False)


def dynamic_fuse(
    v_text: VAD,
    v_audio: VAD,
    entropy_text: float,
    entropy_audio: float,
    var_text: float = 0.0,
    var_audio: float = 0.0,
    quality_text: float = 1.0,
    quality_audio: float = 1.0,
    alpha: float = 1.0,   # entropy weight
    beta: float = 1.0,    # variance weight
    gamma: float = 2.0,   # quality penalty weight (strong)
    disagreement_thresh: float = 0.35,
    entropy_confident_thresh: float = 1.2,
) -> FusionResult:

    qt = _clamp01(quality_text)
    qa = _clamp01(quality_audio)

    r_text = -alpha * entropy_text - beta * var_text - gamma * (1.0 - qt)
    r_audio = -alpha * entropy_audio - beta * var_audio - gamma * (1.0 - qa)

    # softmax over two values
    mx = max(r_text, r_audio)
    et = np.exp(r_text - mx)
    ea = np.exp(r_audio - mx)
    w_text = float(et / (et + ea + 1e-12))
    w_audio = 1.0 - w_text

    vf = VAD(
        v=w_text * v_text.v + w_audio * v_audio.v,
        a=w_text * v_text.a + w_audio * v_audio.a,
        d=w_text * v_text.d + w_audio * v_audio.d,
    )

    delta = l2_distance_vad(v_text, v_audio)

    
    incongruent = (delta > disagreement_thresh) and \
                  (entropy_text < entropy_confident_thresh) and \
                  (entropy_audio < entropy_confident_thresh)

    return FusionResult(vad_fused=vf, w_text=w_text, w_audio=w_audio,
                        disagreement=delta, incongruent=incongruent)
