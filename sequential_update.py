def one_step_update(prev_beta, current_beta, k, tau):
    """
    One-Step Prior Method for sequential LID estimation (see Equation (15) in the paper).

    The update rules are:
       new_alpha = (tau + 1) * k
       new_beta  = tau * prev_beta + current_beta
       Î_1S    = new_alpha / new_beta

    Parameters:
      prev_beta   : float
          The beta value (–Σ ln(x_i/w)) from the previous epoch.
      current_beta: float
          The beta value computed from the current mini-batch.
      k           : int
          Number of nearest neighbors.
      tau         : float
          Temperature parameter controlling the weighting.

    Returns:
      A tuple (one_step_estimate, new_beta)
    """
    new_alpha = (tau + 1) * k
    new_beta = tau * prev_beta + current_beta
    return new_alpha / new_beta, new_beta

def accumulative_update(accum_alpha, accum_beta, current_beta, k, tau, epoch):
    """
    Accumulative Prior Method for sequential LID estimation (see Equation (16) in the paper).

    The accumulative update is defined as:
       accum_alpha_t = Σ_{j=0}^{t} tau^(t-j) * k   (since alpha_j = k for each epoch)
       accum_beta_t  = Σ_{j=0}^{t} tau^(t-j) * beta_j
       Î_Acc_t      = accum_alpha_t / accum_beta_t

    Here we simulate this by updating the accumulated alpha and beta with a weight = tau^epoch.

    Parameters:
      accum_alpha  : float
          The accumulated alpha value from previous epochs.
      accum_beta   : float
          The accumulated beta value from previous epochs.
      current_beta : float
          The beta value computed from the current mini-batch.
      k            : int
          Number of nearest neighbors.
      tau          : float
          Exponential decay factor.
      epoch        : int
          Current epoch index (starting from 0).

    Returns:
      A tuple (accum_estimate, updated accum_alpha, updated accum_beta)
    """
    weight = tau ** epoch  # weight for the current epoch
    accum_alpha += weight * k
    accum_beta += weight * current_beta
    return accum_alpha / accum_beta, accum_alpha, accum_beta