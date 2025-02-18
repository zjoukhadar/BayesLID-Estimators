import numpy as np
from scipy.special import digamma

def compute_lid_estimators(neighbor_distances):
    """
    Compute the local intrinsic dimensionality (LID) estimators for a single sample
    based on its sorted k–nearest neighbor distances.

    According to the paper:
      - The Maximum Likelihood Estimator (MLE) (also the Quadratic loss estimator)
        is given by:
            Î_MLE = k / (-Σ_{i=1}^k ln(x_i / w))
      - The Stein loss estimator is:
            Î_Stein = (k - 1) / (-Σ_{i=1}^k ln(x_i / w))
      - The Brown loss estimator is:
            Î_Brown = exp(ψ(k)) / (-Σ_{i=1}^k ln(x_i / w))
    where w is the maximum (i.e., k-th) neighbor distance.

    Parameters:
      neighbor_distances : array-like of shape (k,)
          Sorted distances from the query point to its k nearest neighbors.

    Returns:
      A dictionary containing:
        - "mle"   : MLE / Quadratic loss estimator
        - "stein" : Stein loss estimator
        - "brown" : Brown loss estimator
        - "beta"  : The computed beta value (β = -Σ ln(x_i/w))
    """
    k = len(neighbor_distances)
    w = neighbor_distances[-1]
    # Compute beta = - sum_{i=1}^{k} ln(x_i/w)
    beta = -np.sum(np.log(neighbor_distances / w))
    mle = k / beta
    stein = (k - 1) / beta if k > 1 else mle
    brown = np.exp(digamma(k)) / beta
    return {"mle": mle, "ql": mle, "stein": stein, "brown": brown, "beta": beta}