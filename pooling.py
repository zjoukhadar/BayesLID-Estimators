import numpy as np

def arithmetic_pooling(estimates):
    """Arithmetic pooling (linear pooling): simple average of estimates."""
    return np.mean(estimates)

def harmonic_pooling(estimates):
    """Harmonic pooling (logarithmic pooling): harmonic mean of estimates."""
    estimates = np.array(estimates)
    return len(estimates) / np.sum(1.0 / estimates)

def geometric_pooling(estimates):
    """Geometric pooling: exponentiated average of the logarithm of estimates."""
    estimates = np.array(estimates)
    return np.exp(np.mean(np.log(estimates + 1e-12)))  # small epsilon to avoid log(0)