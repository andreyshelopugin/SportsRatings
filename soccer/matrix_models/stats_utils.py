from math import exp

import numpy as np


def win_tie_loss_probs(score_matrix: np.array) -> tuple[float, float, float]:
    """
    The win probability of the home team can be calculated as the sum of the probabilities in the bottom triangle.
    Sum of diagonal probabilities is probability of tie.
    Sum of top triangle probabilities is win probability of away team.
    """
    bottom_triangle = np.tril(score_matrix, -1).sum()
    diag = np.trace(score_matrix)
    upper_triangle = 1 - bottom_triangle - diag

    return bottom_triangle, diag, upper_triangle


def poisson_pmf(lamb: float, end: int) -> np.array:
    """Calculate Poisson Probability Mass Function."""

    factorials = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800,
                  479001600, 6227020800, 87178291200, 1307674368000, 20922789888000,
                  355687428096000, 6402373705728000, 121645100408832000]

    return exp(-lamb) * np.array([pow(lamb, k) / factorials[k] for k in range(end)])
