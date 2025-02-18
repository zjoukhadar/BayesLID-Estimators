import numpy as np

def arithmetic_pooling(estimates):
    """
    Linear (Arithmetic) pooling: compute the simple arithmetic mean of the estimates.
    This corresponds to the arithmetic mean point estimate in Table 2 of the paper.
    """
    return np.mean(estimates)

def harmonic_pooling(estimates):
    """
    Logarithmic (Harmonic) pooling: compute the harmonic mean of the estimates.
    This is appropriate for estimators derived under Stein loss (and also for pooling MLE
    estimates when using logarithmic pooling as shown in the paper).
    """
    estimates = np.array(estimates)
    return len(estimates) / np.sum(1.0 / estimates)

def geometric_pooling(estimates):
    """
    Geometric pooling: compute the geometric mean of the estimates.
    This pooling method is used for the Brown loss estimator, as per Table 2 of the paper.
    """
    estimates = np.array(estimates)
    return np.exp(np.mean(np.log(estimates + 1e-12)))  # small epsilon avoids log(0)