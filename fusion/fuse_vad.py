import numpy as np

def fuse_static(v_text, v_audio, w_text: float = 0.5):
    
    if not (0.0 <= w_text <= 1.0):
        raise ValueError("w_text must be in [0, 1]")

    vt = np.asarray(v_text, dtype=float).reshape(3)
    va = np.asarray(v_audio, dtype=float).reshape(3)

    return w_text * vt + (1.0 - w_text) * va


def _base_confidence(entropy_val: float, var_val: float = 0.0, eps: float = 1e-8) -> float:
    return 1.0 / (entropy_val + var_val + eps)


def fuse_dynamic(
    v_text,
    v_audio,
    ent_text: float,
    ent_audio: float,
    var_text: float = 0.0,
    var_audio: float = 0.0,
    asr_conf: float = None,
    snr: float = None,
    eps: float = 1e-8,
):
    vt = np.asarray(v_text, dtype=float).reshape(3)
    va = np.asarray(v_audio, dtype=float).reshape(3)

    c_text = 1.0 / (ent_text + var_text + eps)
    c_audio = 1.0 / (ent_audio + var_audio + eps)

    if asr_conf is not None:
        asr_conf = float(asr_conf)
        asr_conf = max(0.0, min(1.0, asr_conf))
        c_text *= asr_conf

    if snr is not None:
        snr = float(snr)
        snr_scaled = (snr - 5.0) / (20.0 - 5.0)
        snr_scaled = max(0.0, min(1.0, snr_scaled))
        c_audio *= snr_scaled

    denom = c_text + c_audio
    if denom <= 0:
        w_text, w_audio = 0.5, 0.5
    else:
        w_text = c_text / denom
        w_audio = c_audio / denom

    fused = w_text * vt + w_audio * va
    
    return fused, float(w_text), float(w_audio)


if __name__ == "__main__":

    v_text = np.array([0.6, 0.4, 0.5])
    v_audio = np.array([0.2, 0.8, 0.6])

    print("Static:", fuse_static(v_text, v_audio, w_text=0.7))

    fused, wt, wa = fuse_dynamic(
        v_text, v_audio,
        ent_text=0.6, ent_audio=1.2,
        var_text=0.01, var_audio=0.02,
        asr_conf=0.9, snr=12.0
    )
    print("Dynamic fused:", fused, "w_text:", wt, "w_audio:", wa)

