import numpy as np

def entropy(probs, eps: float = 1e-12) -> float:

    p = np.asarray(probs, dtype=float).reshape(-1)
    s = float(p.sum())
    if s <= 0:
        raise ValueError("probs must sum to a positive value")
    p = p / s
    p = np.clip(p, eps, 1.0)

    return float(-np.sum(p * np.log(p)))


def max_prob(probs) -> float:

    p = np.asarray(probs, dtype=float).reshape(-1)
    s = float(p.sum())
    if s <= 0:
        raise ValueError("probs must sum to a positive value")
    p = p / s

    return float(np.max(p))


def ensemble_mean_var(probs_list, eps: float = 1e-12):
    
    P = np.asarray(probs_list, dtype=float)
    if P.ndim != 2:
        raise ValueError("probs_list must have shape (M, K)")

    # normalize each member just in case
    sums = P.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("Each probs vector must sum to a positive value")
    P = P / sums

    mean_probs = P.mean(axis=0)
    var_per_class = P.var(axis=0)
    var_scalar = float(np.mean(var_per_class))
    
    return mean_probs, var_scalar


if __name__ == "__main__":
    probs = [0.7, 0.2, 0.1]
    print("entropy:", entropy(probs))
    print("max_prob:", max_prob(probs))

    probs_list = [
        [0.7, 0.2, 0.1],
        [0.6, 0.25, 0.15],
        [0.72, 0.18, 0.10],
    ]
    mean_p, var_s = ensemble_mean_var(probs_list)
    print("ensemble mean:", mean_p, "var_scalar:", var_s)

