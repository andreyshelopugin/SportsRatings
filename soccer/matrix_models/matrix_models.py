import gc

import pandas as pd
from tqdm import tqdm

from config import config
from soccer.matrix_models.stats_utils import win_tie_loss_probs
from utils.metrics import three_outcomes_log_loss_string_outcome


class MatrixModels:
    """Score matrix models on the goals expectations of the lgbm: the baselines of the paper."""

    def __init__(self, score_matrix_size=15, train_test_split_season=2021, fit_on_eval_set=True):
        self.train_test_split_season = train_test_split_season

        # the goals expectations of the lgbm are overfitted on its train set, so the params are fitted on the eval set
        # of the lgbm (10% of the train seasons, used only for the early stopping), see diagnostics/lgbm_overfit_check.py
        self.fit_on_eval_set = fit_on_eval_set
        self.score_matrix_size = score_matrix_size

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Train and test matches with the goals expectations of the lgbm.
        The goals expectations are for 90 minutes, so the outcome is of 90 minutes too (as the Glicko test loss)."""

        matches = pd.read_feather(config.soccer['matches'], columns=['match_id', 'season', 'country', 'tournament', 'source',
                                                                     'home_score', 'away_score', 'outcome_regular_time'])
        matches = matches.rename(columns={'outcome_regular_time': 'outcome'})

        matches['tournament'] = matches['country'] + ' ' + matches['tournament']

        goals_expectations = pd.read_feather(config.soccer['outcomes']['lgbm_predictions'],
                                             columns=['match_id', 'home_goals_expectations', 'away_goals_expectations'])

        home_goals_expectations = dict(zip(goals_expectations['match_id'], goals_expectations['home_goals_expectations']))
        away_goals_expectations = dict(zip(goals_expectations['match_id'], goals_expectations['away_goals_expectations']))

        matches['home_goals_expectations'] = matches['match_id'].map(home_goals_expectations)
        matches['away_goals_expectations'] = matches['match_id'].map(away_goals_expectations)

        train_matches = matches[matches['season'] < self.train_test_split_season]

        if self.fit_on_eval_set:
            eval_ids = pd.read_feather(config.soccer['outcomes']['datasets']['eval'], columns=['match_id'])['match_id']
            train_matches = train_matches[train_matches['match_id'].isin(set(eval_ids))]
        test_matches = matches[matches['season'] >= self.train_test_split_season]

        del matches, goals_expectations, home_goals_expectations, away_goals_expectations
        gc.collect()

        return train_matches, test_matches

    @staticmethod
    def outcome_loss(model, matches: pd.DataFrame, desc: str = 'outcome loss') -> float:
        """W/D/L log loss."""
        outcomes = matches['outcome'].values
        m1 = matches['home_goals_expectations'].values
        m2 = matches['away_goals_expectations'].values

        loss_sum = 0
        for i in tqdm(range(len(outcomes)), desc=desc, leave=False):
            score_matrix = model(m1[i], m2[i])
            w, d, l = win_tie_loss_probs(score_matrix)
            loss_sum += three_outcomes_log_loss_string_outcome(w, d, l, outcomes[i])

        return loss_sum / len(outcomes)
