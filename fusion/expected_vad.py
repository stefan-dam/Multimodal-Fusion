import numpy as np

def expected_vad(probs, labels, vad_map):
    p = np.asarray(probs, dtype=float).reshape(-1)
    if len(p) != len(labels):
        raise ValueError("probs and labels length mismatch")

    s = float(p.sum())
    if s <= 0:
        raise ValueError("probs must sum to a positive value")
    p = p / s

    missing = [lab for lab in labels if lab not in vad_map]
    if missing:
        raise ValueError(f"Missing VAD mapping for labels: {missing}")

    vad_mat = np.array([vad_map[lab] for lab in labels], dtype=float)
    return p @ vad_mat

if __name__ == "__main__":

    labels = ["joy", "sadness", "anger"]
    vad_map = {
        "joy": (0.9, 0.6, 0.6),
        "sadness": (0.2, 0.4, 0.3),
        "anger": (0.1, 0.8, 0.7),
    }
    
    probs = [0.7, 0.2, 0.1]
    print(expected_vad(probs, labels, vad_map))
