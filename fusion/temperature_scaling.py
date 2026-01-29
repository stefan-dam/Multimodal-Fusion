import numpy as np

def softmax_with_temperature(logits, T: float = 1.0):

    if T <= 0:
        raise ValueError("Temperature T must be > 0")

    z = np.asarray(logits, dtype=float)

    # numerical stability
    z = z / T
    z = z - np.max(z)

    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


if __name__ == "__main__":
    
    logits = [2.0, 1.0, 0.1]
    probs_T1 = softmax_with_temperature(logits, T=1.0)
    probs_T2 = softmax_with_temperature(logits, T=2.0)

    print("T=1.0:", probs_T1, "sum=", probs_T1.sum())
    print("T=2.0:", probs_T2, "sum=", probs_T2.sum())

