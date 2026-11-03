"""30.10: the ablation of the paper (knock-out, without a refit): one component of the model is switched off,
the other params are the train-fitted ones (evaluation_league_params.pkl), the test loss as in evaluate
(sequential predictions, seasons >= 2021, no weights, no discount, 90 minutes) for all and international matches.
Without a refit the other params do not compensate the removed component: the differences are upper bounds of the
loss of the full model (lower bounds of the contribution are not given), see the text of the paper.
Run from the root of the project or from diagnostics/ (the paths are from the root)."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

from config import config
from soccer.club_ratings.evaluate_glicko import EvaluateGlicko
from utils.metrics import three_outcomes_log_loss

RESULTS = ROOT / 'data/soccer/tables/ablation.csv'


def no_scoring(params: dict) -> dict:
    """The expected goals enter neither the home advantage nor the draw component."""
    return {league: {**p, 'scoring': 0., 'scoring_overall': 0.} for league, p in params.items()}


def no_home_advantage(params: dict) -> dict:
    """No home advantage at all."""
    return {league: {**p, 'crowd': 0., 'logistic': 0., 'strength': 0., 'scoring': 0.} for league, p in params.items()}


def no_league_init(params: dict) -> dict:
    """All leagues start from the same rating (the mean of init_mu), no league-aware initialization."""
    mean_init_mu = float(np.mean([p['init_mu'] for p in params.values()]))
    return {league: {**p, 'init_mu': mean_init_mu} for league, p in params.items()}


def no_lift(params: dict) -> dict:
    """No rating change at promotion and relegation."""
    return {league: {**p, 'ms_lift_update_mu': 0} for league, p in params.items()}


# name -> (change of the league params, change of the model attributes)
VARIANTS = {
    'full': (None, {}),
    'no_scoring_dependence': (no_scoring, {}),
    'no_home_advantage': (no_home_advantage, {}),
    'no_league_initialization': (no_league_init, {}),
    'no_promotion_relegation_lift': (no_lift, {}),
    'no_newcomer_offset': (None, {'ms_new_team_update_mu': 0}),
    'no_season_rd_growth': (None, {'ms_update_rd_first': 0, 'ms_update_rd_second': 0}),
    'no_extra_time_model': (None, {'use_extra_time': False}),
}


def test_loss(variant: str) -> dict:
    """Test loss of one variant: all and international matches (90 minutes)."""

    change_params, attributes = VARIANTS[variant]

    model = EvaluateGlicko()
    model.world_cup_loss_weight = 1
    model.international_cup_loss_weight = 1
    model.discount_factor = 1
    model.periods_without_loss_calculation = 0

    league_params = model.load_params(config.soccer['params'][model.league_params_name])

    for name, value in attributes.items():
        if name == 'use_extra_time':
            continue
        setattr(model, name, value)

    if change_params is not None:
        league_params = change_params(league_params)

    model._setup_train()
    matches_params = model._match_params(league_params)
    ratings = model.rate_teams(league_params, matches_params)

    model._setup_eval()
    league_params = model._fill_missing_leagues(league_params)
    matches_params = model._match_params(league_params)

    model.prediction_records = []
    model.calculate_test_loss(league_params, ratings, matches_params)
    records = pd.DataFrame(model.prediction_records, columns=['match_id', 'p_win', 'p_draw', 'p_lose'])

    info = pd.DataFrame([(match_id, i.outcome, i.is_domestic) for match_id, i in model.home_away_match_info.items()],
                        columns=['match_id', 'outcome', 'is_domestic'])
    records = records.merge(info, on='match_id', how='left', validate='1:1')

    records['loss'] = [three_outcomes_log_loss(w, d, l, o) / 3
                       for w, d, l, o in zip(records['p_win'], records['p_draw'], records['p_lose'], records['outcome'])]

    return {'variant': variant, 'n_matches': len(records),
            'all': round(records['loss'].mean(), 6),
            'domestic': round(records.loc[records['is_domestic'] == True, 'loss'].mean(), 6),
            'international': round(records.loc[records['is_domestic'] == False, 'loss'].mean(), 6)}


def main():
    rows = []
    for variant in VARIANTS:
        if variant == 'no_extra_time_model':
            # TODO: the 90 minutes outcome does not depend on the extra time params, only the updates of the ET matches
            continue
        rows.append(test_loss(variant))
        print(rows[-1], flush=True)

        report = pd.DataFrame(rows)
        for column in ['all', 'domestic', 'international']:
            report[f'{column}_vs_full'] = (report[column] - report.loc[report['variant'] == 'full', column].iloc[0]).round(6)
        report.to_csv(RESULTS, index=False)

    print(report.to_string(index=False))


if __name__ == '__main__':
    main()
