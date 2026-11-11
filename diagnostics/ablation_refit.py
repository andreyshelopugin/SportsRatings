"""The ablation of the paper with a refit (ablation.py: without it): for every variant one component
of the model is switched off and the other params are refitted on the train seasons (< 2021), then the test loss.
The refit is the global params + the shared shifts (the same stages as the production fit without the per-league fit,
~1 hour per variant): the other params can compensate the removed component, so the increase of the test loss
is the contribution of the component. 'full' goes through the same refit, the differences are vs it.
Start point: the completed train fit (evaluation_league_params.pkl), every variant has its own params file
data/soccer/params/ablation/<variant>.pkl (the train fit is not changed). The results are saved after every variant.
Run from the root of the project or from diagnostics/ (the paths are from the root), ~6-7 hours."""
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

from config import config
from soccer.club_ratings.evaluate_glicko import EvaluateGlicko
from soccer.club_ratings.train_glicko import SHARED_SHIFTS
from utils.functions import stage
from utils.metrics import three_outcomes_log_loss

warnings.filterwarnings('ignore')

PARAMS_DIR = ROOT / 'data/soccer/params/ablation'
RESULTS = ROOT / 'data/soccer/tables/ablation_refit.csv'

HOME_PARAMS = ['crowd', 'logistic', 'strength', 'scoring']
SCORING_PARAMS = ['scoring', 'scoring_overall']


def set_params(params: dict, values: dict) -> dict:
    """The same values of the params in all leagues."""
    return {league: {**p, **values} for league, p in params.items()}


def mean_init_mu(params: dict) -> dict:
    """All leagues start from the same rating (the mean of init_mu)."""
    value = float(np.mean([p['init_mu'] for p in params.values()]))
    return set_params(params, {'init_mu': value})


# name -> (change of the league params, params excluded from the refit, forced model attributes, refit of the global params)
VARIANTS = {
    'full': (None, [], {}, True),
    'no_scoring_dependence': (lambda p: set_params(p, {name: 0. for name in SCORING_PARAMS}), SCORING_PARAMS, {}, True),
    'no_home_advantage': (lambda p: set_params(p, {name: 0. for name in HOME_PARAMS}), HOME_PARAMS, {}, True),
    'no_league_initialization': (mean_init_mu, [], {}, True),
    'no_promotion_relegation_lift': (lambda p: set_params(p, {'ms_lift_update_mu': 0}), ['ms_lift_update_mu'], {}, True),
    'no_newcomer_offset': (None, [], {'ms_new_team_update_mu': 0}, True),
    # the constraint ms_update_rd >= update_rd of fit_global_params does not allow 0: the global params are not refitted
    'no_season_rd_growth': (None, [], {'ms_update_rd_first': 0, 'ms_update_rd_second': 0}, False),
}


class AblationGlicko(EvaluateGlicko):
    """EvaluateGlicko with its own params file and forced attributes (they survive load_params / set_global_params)."""

    def __init__(self, variant: str):
        super().__init__()

        self.variant = variant
        self.league_params_name = f'ablation_{variant}'
        config.soccer['params'][self.league_params_name] = str(PARAMS_DIR / f'{variant}.pkl')

        self.forced = VARIANTS[variant][2]
        self._force()

    def _force(self):
        for name, value in self.forced.items():
            setattr(self, name, value)

    def set_global_params(self, global_params: dict):
        super().set_global_params(global_params)
        if hasattr(self, 'forced'):
            self._force()


def prepare(variant: str):
    """The params file of the variant: a copy of the completed train fit with the component switched off."""

    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    path = PARAMS_DIR / f'{variant}.pkl'
    shutil.copy(config.soccer['params']['evaluation_league_params'], path)

    model = AblationGlicko(variant)
    league_params = model.load_params(path)

    change = VARIANTS[variant][0]
    if change is not None:
        league_params = change(league_params)

    model.save_params(league_params, path)


def refit(variant: str):
    """Global params + shared shifts, without the switched off params."""

    excluded = VARIANTS[variant][1]
    refit_global = VARIANTS[variant][3]

    if refit_global:
        print(f'\n===== {variant}: global params =====')
        AblationGlicko(variant).fit_global_params()

    for param_name, delta, groups in SHARED_SHIFTS:
        if param_name in excluded:
            continue
        print(f'\n===== {variant}: shared shift {param_name} =====')
        AblationGlicko(variant).fit_shared_shift(param_name, delta=delta, groups=groups)


def test_loss(variant: str) -> dict:
    """Test loss as in EvaluateGlicko.evaluate (sequential, seasons >= 2021, no weights, no discount, 90 minutes)."""

    model = AblationGlicko(variant)
    model.world_cup_loss_weight = 1
    model.international_cup_loss_weight = 1
    model.discount_factor = 1
    model.periods_without_loss_calculation = 0

    league_params = model.load_params(config.soccer['params'][model.league_params_name])

    model._setup_train()
    matches_params = model._match_params(league_params)
    train_loss = model.calculate_loss(league_params, matches_params) / len(model.home_away_match_info)
    ratings = model.rate_teams(league_params, matches_params)

    model._setup_eval()
    league_params = model._fill_missing_leagues(league_params)
    matches_params = model._match_params(league_params)

    model.prediction_records = []
    model.calculate_test_loss(league_params, ratings, matches_params)
    records = pd.DataFrame(model.prediction_records, columns=['match_id', 'p_win', 'p_draw', 'p_lose'])
    model.prediction_records = None

    info = pd.DataFrame([(match_id, i.outcome, i.is_domestic) for match_id, i in model.home_away_match_info.items()],
                        columns=['match_id', 'outcome', 'is_domestic'])
    records = records.merge(info, on='match_id', how='left', validate='1:1')
    records['loss'] = [three_outcomes_log_loss(w, d, l, o) / 3
                       for w, d, l, o in zip(records['p_win'], records['p_draw'], records['p_lose'], records['outcome'])]

    return {'variant': variant, 'n_matches': len(records), 'train_loss': round(train_loss, 6),
            'all': round(records['loss'].mean(), 6),
            'domestic': round(records.loc[records['is_domestic'] == True, 'loss'].mean(), 6),
            'international': round(records.loc[records['is_domestic'] == False, 'loss'].mean(), 6)}


def main():
    rows = []
    for variant in VARIANTS:
        stage(f'ablation with a refit: {variant}')
        prepare(variant)
        refit(variant)

        rows.append(test_loss(variant))
        print(rows[-1], flush=True)

        report = pd.DataFrame(rows)
        for column in ['all', 'domestic', 'international']:
            report[f'{column}_vs_full'] = (report[column] - report.loc[report['variant'] == 'full', column].iloc[0]).round(6)
        report.to_csv(RESULTS, index=False)
        print(report.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
