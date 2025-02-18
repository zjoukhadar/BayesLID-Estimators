def one_step_update(prev_beta, current_beta, k, tau):
    """
    One-Step Prior Method:
      new_alpha = (tau + 1) * k
      new_beta = tau * prev_beta + current_beta
      LID estimator = new_alpha / new_beta

    Returns:
      - The updated one-step estimate.
      - The updated beta to be used in the next iteration.
    """
    new_alpha = (tau + 1) * k
    new_beta = tau * prev_beta + current_beta
    return new_alpha / new_beta, new_beta

def accumulative_update(accum_alpha, accum_beta, current_beta, k, tau, epoch):
    """
    Accumulative Prior Method with exponential decay:
      Update the accumulated gamma parameters with a decaying weight:
          accum_alpha += tau^epoch * k
          accum_beta  += tau^epoch * current_beta
      LID estimator = accum_alpha / accum_beta

    Returns:
      - The updated accumulative estimate.
      - The new accum_alpha and accum_beta values.
    """
    weight = tau ** epoch
    accum_alpha += weight * k
    accum_beta += weight * current_beta
    return accum_alpha / accum_beta, accum_alpha, accum_beta