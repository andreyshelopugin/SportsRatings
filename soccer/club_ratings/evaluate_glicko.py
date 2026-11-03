import shutil
from pathlib import Path

import pandas as pd

from config import config
from utils.functions import load_pickle
from soccer.club_ratings.train_glicko import TrainGlicko


class EvaluateGlicko(TrainGlicko):
    """Honest evaluation of the model for the paper.
    The production params (league_params.pkl) are fitted on all seasons: the ratings are fresh, but the test loss leaks.
    Here the params are fitted on the train seasons only (< 2021) and saved to evaluation_league_params.pkl,
    then the test loss is calculated on the seasons >= 2021 - the same test matches as the lgbm, Dixon-Coles and Markov.
    The production params are the start point: the fit converges faster, and the difference between the two sets
    of params is the stability check (the table of the estimated params: train fit vs full fit)."""

    def __init__(self, train_test_split_season=2021):
        super().__init__(train_test_split_season=train_test_split_season)

        self.league_params_name = 'evaluation_league_params'

        # the extra time params of the train fit, the production ones are the start point
        self.extra_time_params_name = 'evaluation_extra_time'
        evaluation_extra_time_path = Path(config.soccer['params'][self.extra_time_params_name])
        if evaluation_extra_time_path.exists():
            self.extra_time_params = load_pickle(evaluation_extra_time_path)

    def init_from_production(self, overwrite: bool = False):
        """Copy the production params as the start point of the train fit."""

        production_path = Path(config.soccer['params']['league_params'])
        evaluation_path = Path(config.soccer['params'][self.league_params_name])

        if evaluation_path.exists() and not overwrite:
            print(f'{evaluation_path} exists, the fit continues from it')
            return

        shutil.copy(production_path, evaluation_path)
        print(f'start point: {production_path} -> {evaluation_path}')

    def fit(self, shared_shifts: list, league_fit_iterations: int = 1, extra_time_iterations: int = 30):
        """The same stages as the production fit (TrainGlicko.fit) on the train seasons,
        saved to evaluation_league_params.pkl after every improvement."""

        self.init_from_production()

        # the constraints depend on the seasons: a league with low international history on the train seasons
        # (Canada, since 2019) has a cap of init_mu, the production params can be above it
        self.clip_params_to_constraints()

        super().fit(shared_shifts, league_fit_iterations, extra_time_iterations)

    def evaluate(self) -> pd.DataFrame:
        """Train and test loss of the train-fitted params and, for reference, of the production params
        (their test loss is not honest: the test seasons were in their fit).
        The honest settings as in compare_losses: no loss weights, no discount, all periods."""

        self.world_cup_loss_weight = 1
        self.international_cup_loss_weight = 1
        self.discount_factor = 1
        self.periods_without_loss_calculation = 0

        rows = []
        for params_name in [self.league_params_name, 'league_params']:
            league_params = self.load_params(config.soccer['params'][params_name])
            global_params = self.global_params()

            # train loss
            self._setup_train()
            matches_params = self._match_params(league_params)

            self.extra_time_loss_sum = 0.
            train_loss = self.calculate_loss(league_params, matches_params)
            train_extra_time_loss = self.extra_time_loss_sum
            n_train_matches = len(self.home_away_match_info)

            ratings = self.rate_teams(league_params, matches_params)

            # test loss, leagues that appeared only in test get group params
            self._setup_eval()
            league_params = self._fill_missing_leagues(league_params)
            matches_params = self._match_params(league_params)

            self.extra_time_loss_sum = 0.
            test_loss = self.calculate_test_loss(league_params, ratings, matches_params)
            n_test_matches = len(self.home_away_match_info)
            test_extra_time_loss = self.extra_time_loss_sum / n_test_matches

            rows.append({'params': params_name,
                         'honest_test': params_name == self.league_params_name,
                         'n_train': n_train_matches,
                         'n_test': n_test_matches,
                         'train_loss': round(train_loss / n_train_matches, 6),
                         'test_loss': round(test_loss, 6),
                         'train_loss_90': round((train_loss - train_extra_time_loss) / n_train_matches, 6),
                         'test_loss_90': round(test_loss - test_extra_time_loss, 6)} | global_params)

        report = pd.DataFrame(rows)
        print(report.to_string(index=False))

        return report

    def save_test_predictions(self, path: str = 'data/soccer/tables/glicko_test_predictions_sequential.feather') -> pd.DataFrame:
        """The probabilities of every test match, the same sequential scheme as the test loss of evaluate
        (inside a rating period the ratings are updated after every match, see _loss_inside_period).
        The first version (add_predictions, the ratings at the start of the period) was worse by 0.004.
        Brier, RPS and the paired bootstrap vs the baselines (soccer/matrix_models test_predictions.feather, the same match_id).
        Home-oriented, 90 minutes, the honest settings as in evaluate."""

        self.world_cup_loss_weight = 1
        self.international_cup_loss_weight = 1
        self.discount_factor = 1
        self.periods_without_loss_calculation = 0

        league_params = self.load_params(config.soccer['params'][self.league_params_name])

        # the ratings after the train seasons
        self._setup_train()
        matches_params = self._match_params(league_params)
        ratings = self.rate_teams(league_params, matches_params)

        # the test seasons, the predictions are recorded in _loss_inside_period
        self._setup_eval()
        league_params = self._fill_missing_leagues(league_params)
        matches_params = self._match_params(league_params)

        self.prediction_records = []
        test_loss = self.calculate_test_loss(league_params, ratings, matches_params)
        records, self.prediction_records = self.prediction_records, None

        predictions = pd.DataFrame(records, columns=['match_id', 'glicko_home_win', 'glicko_draw', 'glicko_away_win'])

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        predictions.to_feather(path)
        print(f'test predictions: {len(predictions)} matches, test loss (with extra time) {test_loss:.6f}, saved: {path}')

        return predictions

    # def save_test_predictions_batch(self, path: str = 'data/soccer/tables/glicko_test_predictions.feather') -> pd.DataFrame:
    #     """The probabilities of every test match with the train-fitted params (fixed), the ratings are updated
    #     through the test seasons as in evaluate: Brier, RPS and the paired bootstrap vs the baselines
    #     (soccer/matrix_models test_predictions.feather, the same match_id). Home-oriented, 90 minutes."""

    #     matches = pd.read_feather(config.soccer['glicko_matches'])
    #     self._precalculate_info(matches)

    #     league_params = self._fill_missing_leagues(self.load_params(config.soccer['params'][self.league_params_name]))
    #     matches_params = self._match_params(league_params)

    #     result = self.add_predictions(matches, league_params, matches_params)

    #     predictions = (result
    #                    .loc[(result['is_home'] == True) & (result['season'] >= self.train_test_split_season),
    #                         ['match_id', 'season', 'outcome', 'p_team_win', 'p_draw', 'p_team_lose']]
    #                    .rename(columns={'p_team_win': 'glicko_home_win', 'p_draw': 'glicko_draw',
    #                                     'p_team_lose': 'glicko_away_win'})
    #                    .reset_index(drop=True))

    #     Path(path).parent.mkdir(parents=True, exist_ok=True)
    #     predictions.to_feather(path)
    #     print(f'test predictions: {len(predictions)} matches, saved: {path}')

    #     return predictions


    def compare_params(self) -> pd.DataFrame:
        """League params: train fit vs full fit, for the table of the estimated params and the stability check."""

        evaluation = self.load_params(config.soccer['params'][self.league_params_name])
        production = self.load_params(config.soccer['params']['league_params'])

        rows = []
        for league in sorted(set(evaluation) & set(production)):
            for param_name, value in evaluation[league].items():
                if param_name in production[league]:
                    rows.append({'league': league, 'param': param_name,
                                 'train_fit': value, 'full_fit': production[league][param_name]})

        params = pd.DataFrame(rows)
        params['diff'] = params['train_fit'] - params['full_fit']

        print(params.groupby('param')[['train_fit', 'full_fit', 'diff']].agg(['mean', 'std']).round(4).to_string())

        return params
