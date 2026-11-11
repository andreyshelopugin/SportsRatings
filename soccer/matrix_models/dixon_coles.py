import numpy as np

from soccer.matrix_models.matrix_models import MatrixModels
from soccer.matrix_models.stats_utils import poisson_pmf


class DixonColes(MatrixModels):
    """Dixon-Coles bivariate Poisson model for football score prediction.

    Implements the Dixon-Coles (1997) adjustment to independent Poisson models,
    correcting for the empirical observation that low-scoring outcomes (0-0, 1-0,
    0-1, 1-1) occur more/less frequently than predicted by independent Poissons.

    The model applies a correlation parameter (rho/dc_param) to adjust probabilities
    for these four specific scorelines, addressing the underestimation of draws in
    naive Poisson models.

    Reference:
        Dixon, M.J. and Coles, S.G. (1997). "Modelling Association Football Scores
        and Inefficiencies in the Football Betting Market". Journal of the Royal
        Statistical Society: Series C (Applied Statistics), 46(2), 265-280.
        https://doi.org/10.1111/1467-9876.00065

    dc_param: the correlation parameter rho, fitted by fit (W/D/L loss on the eval set of the lgbm).
    """

    def __init__(self, dc_param=-0.06):
        super().__init__()
        self.dc_param = dc_param

    def score_matrix_model(self, m1: float, m2: float) -> np.ndarray:
        """Dixon-Coles model adjusting Poisson probabilities with a correction factor."""

        # assumption 1: poisson distribution for each team's goals
        x = poisson_pmf(m1, end=self.score_matrix_size)  # P(home goals = i)
        y = poisson_pmf(m2, end=self.score_matrix_size)  # P(away goals = j)

        # assumption 2: independence between home and away goals
        # under independence: P(i, j) = P(i) * P(j)
        score_matrix = np.outer(x, y)

        # Dixon-Coles correction: relax independence for low-scoring outcomes
        # (adjustment applied below for 0-0, 1-0, 0-1, 1-1)

        if (1 - m1 * m2 * self.dc_param) < 0.001:
            return score_matrix
        else:
            score_matrix[0][0] *= (1 - m1 * m2 * self.dc_param)
            score_matrix[0][1] *= (1 + m1 * self.dc_param)
            score_matrix[1][0] *= (1 + m2 * self.dc_param)
            score_matrix[1][1] *= (1 - self.dc_param)
            score_matrix /= score_matrix.sum()
            return score_matrix

    def fit(self):
        """rho by a grid search on the W/D/L loss (as Glicko), the last 300 000 train matches."""

        train_matches, _ = self.get_data()
        train_matches = train_matches.tail(300_000)

        best_loss = np.inf
        best_param = None

        # -0.06 was the boundary of the previous grid and the estimate with the new lgbm (10.11)
        dc_params = np.linspace(-0.12, -0.02, 101)
        for i, dc_param in enumerate(dc_params):
            self.dc_param = dc_param

            loss = self.outcome_loss(self.score_matrix_model, train_matches,
                                   desc=f'dixon-coles {i + 1}/{len(dc_params)} rho={dc_param:.4f}')

            if loss < best_loss:
                best_loss = loss
                best_param = dc_param

        print(f'dixon-coles best rho: {best_param:.5f}, train loss: {best_loss:.6f}')
        self.dc_param = best_param
