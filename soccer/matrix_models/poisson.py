import numpy as np

from soccer.matrix_models.matrix_models import MatrixModels
from soccer.matrix_models.stats_utils import poisson_pmf


class Poisson(MatrixModels):
    """Assumes goals in soccer follow a Poisson distribution and are independent between teams."""

    def score_matrix_model(self, m1: float, m2: float) -> np.ndarray:
        """Score matrix based on independent Poisson distributions."""
        x = poisson_pmf(m1, end=self.score_matrix_size)
        y = poisson_pmf(m2, end=self.score_matrix_size)
        return np.outer(x, y)
