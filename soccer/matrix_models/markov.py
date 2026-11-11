import numpy as np

from soccer.matrix_models.matrix_models import MatrixModels


class Markov(MatrixModels):
    """Continuous-time Markov chain model.
    Scoring rates depend on current score difference (comeback effect).
    Low-score states have adjusted rates scaled by expected goals (analogous to Dixon-Coles correction)."""

    def __init__(self):
        super().__init__()
        self.big_score_matrix_size = self.score_matrix_size * self.score_matrix_size

        # fitted on the W/D/L loss on the eval set of the lgbm (fit), the loss is flat: not fully converged
        self.low_scoring_00 = 0.025
        self.low_scoring_10_01 = 0.085
        self.low_scoring_11 = 0.04

        self.three_goals_lead_param = 1.23
        self.two_goals_lead_param = 0.96
        self.one_goal_lead_param = 0.80
        self.one_goals_trail_param = 1.205
        self.two_goals_trail_param = 0.93
        self.three_goals_trail_param = 0.855

        # the expected goals of the chain are rescaled to the ones of the lgbm (score_matrix_model)
        self.mean_matching_iterations = 3

        self._precompute_indices()

    def _precompute_indices(self):
        """Precompute state indices and transitions."""
        n = self.score_matrix_size

        self._home_comeback = np.ones(self.big_score_matrix_size)
        self._away_comeback = np.ones(self.big_score_matrix_size)

        # Store which states get low-scoring adjustments (type and index)
        self._low_scoring_states = {}

        self._home_goal_from = []
        self._away_goal_from = []

        for i in range(n):
            for j in range(n):
                idx = i * n + j
                diff = min(max(i - j, -3), 3)

                comeback = {
                    -3: (self.three_goals_trail_param, self.three_goals_lead_param),
                    -2: (self.two_goals_trail_param, self.two_goals_lead_param),
                    -1: (self.one_goals_trail_param, self.one_goal_lead_param),
                    0: (1.0, 1.0),
                    1: (self.one_goal_lead_param, self.one_goals_trail_param),
                    2: (self.two_goals_lead_param, self.two_goals_trail_param),
                    3: (self.three_goals_lead_param, self.three_goals_trail_param),
                }

                self._home_comeback[idx] = comeback[diff][0]
                self._away_comeback[idx] = comeback[diff][1]

                if (i, j) == (0, 0):
                    self._low_scoring_states[idx] = 'low_scoring_00'
                elif (i, j) in [(1, 0), (0, 1)]:
                    self._low_scoring_states[idx] = 'low_scoring_10_01'
                elif (i, j) == (1, 1):
                    self._low_scoring_states[idx] = 'low_scoring_11'

                if i + 1 < n:
                    self._home_goal_from.append((idx, (i + 1) * n + j))
                if j + 1 < n:
                    self._away_goal_from.append((idx, i * n + j + 1))

    def score_matrix_model(self, m1: float, m2: float) -> np.ndarray:
        """Score matrix with the same expected goals as the lgbm: the comeback and low-scoring multipliers change
        the expected goals of the chain (+14% with the fitted params), the model must change the shape only.
        The base rates are rescaled by a fixed point: rate *= target / expected, 3 iterations (|error| < 1e-3)."""
        goals = np.arange(self.score_matrix_size)

        rate_1, rate_2 = m1, m2
        score_matrix = self._chain_score_matrix(rate_1, rate_2)
        for _ in range(self.mean_matching_iterations):
            expected_1 = score_matrix.sum(axis=1) @ goals
            expected_2 = score_matrix.sum(axis=0) @ goals

            rate_1 *= m1 / expected_1
            rate_2 *= m2 / expected_2

            score_matrix = self._chain_score_matrix(rate_1, rate_2)

        return score_matrix

    def _chain_score_matrix(self, m1: float, m2: float) -> np.ndarray:
        """Score matrix after the time T = 1: the distribution of the chain started at 0-0.
        Uniformization instead of expm(Q): P = I + Q / L, L >= max exit rate,
            p(1) = sum_k exp(-L) L^k / k! * p(0) P^k,
        the same result (1e-16) about 30 times faster: a goal moves the mass to the neighbour state, so p P is two shifts
        of the score grid, the rate matrix Q is not built."""
        n = self.score_matrix_size

        # rates out of every state (home goal / away goal), low-scoring corrections scaled by m1 * m2 as in Dixon-Coles
        low_scale = m1 * m2
        low_scoring_mult = np.ones(self.big_score_matrix_size)
        for idx, param_name in self._low_scoring_states.items():
            param = getattr(self, param_name)
            if param_name == 'low_scoring_10_01':
                low_scoring_mult[idx] = 1 - param * low_scale
            else:
                low_scoring_mult[idx] = 1 + param * low_scale

        # 1 - low_scoring_10_01 * m1 * m2 < 0 in high scoring matches: negative rates
        low_scoring_mult = np.maximum(low_scoring_mult, 0.05)

        home_rate = (m1 * self._home_comeback * low_scoring_mult).reshape(n, n)
        away_rate = (m2 * self._away_comeback * low_scoring_mult).reshape(n, n)

        # no goals beyond the grid (as in Q): the mass stays at the edge
        home_rate[-1, :] = 0
        away_rate[:, -1] = 0

        exit_rate = home_rate + away_rate
        uniform_rate = max(exit_rate.max(), 1e-12)

        stay = 1 - exit_rate / uniform_rate
        home_move = home_rate / uniform_rate
        away_move = away_rate / uniform_rate

        state = np.zeros((n, n))
        state[0, 0] = 1.

        weight = np.exp(-uniform_rate)
        result = weight * state
        total_weight = weight

        k = 0
        while total_weight < 1 - 1e-15 or k < uniform_rate:
            k += 1
            moved = state * stay
            moved[1:, :] += state[:-1, :] * home_move[:-1, :]
            moved[:, 1:] += state[:, :-1] * away_move[:, :-1]
            state = moved

            weight *= uniform_rate / k
            result += weight * state
            total_weight += weight

            if k > 200:
                break

        return result

    def fit(self, n_iterations=10):
        """Comeback and low-scoring params by coordinate descent on the W/D/L loss (as Glicko)."""

        train_matches, test_matches = self.get_data()

        best_loss = self.outcome_loss(self.score_matrix_model, train_matches, desc='markov initial loss')
        print()
        print(f"Initial loss: {best_loss:.6f}")
        print()

        params_to_fit = [
            'low_scoring_00',
            'low_scoring_10_01',
            'low_scoring_11',
            'one_goal_lead_param',
            'one_goals_trail_param',
            'two_goals_lead_param',
            'two_goals_trail_param',
            'three_goals_lead_param',
            'three_goals_trail_param',
        ]

        delta = 0.005

        for iteration in range(n_iterations):
            is_improved = False

            for param_name in params_to_fit:
                current = getattr(self, param_name)

                best_candidate_loss = best_loss
                best_candidate_val = current

                for candidate in [current - delta, current + delta]:
                    setattr(self, param_name, candidate)
                    self._precompute_indices()

                    loss = self.outcome_loss(self.score_matrix_model, train_matches,
                                           desc=f'markov iter {iteration} {param_name}={candidate:.4f}')

                    if loss < best_candidate_loss - 0.00000001:
                        best_candidate_loss = loss
                        best_candidate_val = candidate
                        break

                if best_candidate_val != current:
                    setattr(self, param_name, best_candidate_val)
                    self._precompute_indices()
                    best_loss = best_candidate_loss
                    is_improved = True
                    print(f"  {param_name}={best_candidate_val:.4f}, loss={best_loss:.6f}")
                else:
                    setattr(self, param_name, current)
                    self._precompute_indices()

            if not is_improved:
                print(f"Converged at iteration {iteration}")
                break

            print()
            print(f"Iter {iteration}, loss: {best_loss:.6f} | "
                  f"s00={self.low_scoring_00:.3f} s10_01={self.low_scoring_10_01:.3f} s11={self.low_scoring_11:.3f} "
                  f"trail1={self.one_goals_trail_param:.3f} lead1={self.one_goal_lead_param:.3f} "
                  f"trail2={self.two_goals_trail_param:.3f} lead2={self.two_goals_lead_param:.3f} "
                  f"trail3={self.three_goals_trail_param:.3f} lead3={self.three_goals_lead_param:.3f}")
            print()

        test_loss = self.outcome_loss(self.score_matrix_model, test_matches, desc='markov test loss')
        print(f"\nTrain loss: {best_loss:.6f}")
        print(f"Test loss:  {test_loss:.6f}")

        print("\nFinal params:")
        print(f"  low_scoring_00: {self.low_scoring_00}")
        print(f"  low_scoring_10_01: {self.low_scoring_10_01}")
        print(f"  low_scoring_11: {self.low_scoring_11}")
        print(f"  one_goal_lead: {self.one_goal_lead_param}")
        print(f"  one_goal_trail: {self.one_goals_trail_param}")
        print(f"  two_goals_lead: {self.two_goals_lead_param}")
        print(f"  two_goals_trail: {self.two_goals_trail_param}")
        print(f"  three_goals_lead: {self.three_goals_lead_param}")
        print(f"  three_goals_trail: {self.three_goals_trail_param}")
