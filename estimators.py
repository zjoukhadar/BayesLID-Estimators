import numpy as np
from scipy.special import digamma

def compute_lid_estimators(neighbor_distances):
    """
    Given an array of k nearest neighbor distances (sorted in ascending order)
    for a single sample, compute:
      - MLE (also the quadratic-loss estimator)
      - Stein loss estimator
      - Brown loss estimator

    The beta parameter is computed as:
        beta = -sum( ln(x_i / w) )  for i=1,...,k,
    where w is the maximum distance among the k neighbors.

    Returns a dictionary with the estimators and beta.
    """
    k = len(neighbor_distances)
    w = neighbor_distances[-1]
    beta = -np.sum(np.log(neighbor_distances / w))
    mle = k / beta
    stein = (k - 1) / beta if k > 1 else mle
    brown = np.exp(digamma(k)) / beta
    return {"mle": mle, "ql": mle, "stein": stein, "brown": brown, "beta": beta}