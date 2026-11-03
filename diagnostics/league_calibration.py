import warnings

import numpy as np
import pandas as pd

from config import config
from soccer.club_ratings.ratings import Ratings
from utils.functions import load_pickle

"""
Does the model follow the level of a league in time? International matches of the clubs of a league:
expected points of the model (3 * p_win + p_draw, ratings before the match) vs actual points, by season.
actual - expected < 0 for several seasons in a row: the league is overrated, its ratings lag behind (Iran, 26.10).
Run:
    .venv_core/bin/python league_calibration.py
"""

LEAGUES = ['Iran. First League', 'Saudi Arabia. First League', 'United Arab Emirates. First League',
           'Qatar. First League', 'Uzbekistan. First League', 'Japan. First League', 'South Korea. First League',
           'Czech Republic. First League', 'Romania. First League', 'Croatia. First League']

FIRST_SEASON = 2016


def points(outcome: pd.Series) -> pd.Series:
    """outcome of the team: 1 - win, 0.5 - draw, 0 - loss"""
    return outcome.map({1.: 3., .5: 1., 0.: 0.})


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', None)

    model = Ratings()

    matches = pd.read_feather(config.soccer['glicko_matches'])
    model._precalculate_info(matches)

    league_params = model._fill_missing_leagues(model.load_params(config.soccer['params']['league_params']))
    matches_params = model._match_params(league_params)

    predictions = model.add_predictions(matches, league_params, matches_params)

    predictions['team_league'] = np.where(predictions['is_home'], predictions['home_team_league'], predictions['away_team_league'])
    predictions['opp_league'] = np.where(predictions['is_home'], predictions['away_team_league'], predictions['home_team_league'])

    international = predictions.loc[(predictions['is_domestic'] == False)
                                    & (predictions['team_league'] != predictions['opp_league'])
                                    & (predictions['season'].astype(str).str[:4].astype(int) >= FIRST_SEASON)].copy()

    international['year'] = international['season'].astype(str).str[:4].astype(int)
    international['expected_points'] = 3 * international['p_team_win'] + international['p_draw']
    international['actual_points'] = points(international['outcome'])
    international['diff'] = international['actual_points'] - international['expected_points']

    international.to_feather('datasets/soccer/international_predictions.feather')

    for league in LEAGUES:
        league_matches = international.loc[international['team_league'] == league]

        by_season = (league_matches
                     .groupby('year')
                     .agg(n=('diff', 'size'),
                          expected=('expected_points', 'mean'),
                          actual=('actual_points', 'mean'),
                          diff=('diff', 'mean'),
                          team_rating=('team_rating', 'mean'),
                          opp_rating=('opp_rating', 'mean'))
                     .round(2))

        print(f'\n===== {league}: {len(league_matches)} international matches, '
              f'actual - expected per match: {league_matches["diff"].mean():+.2f} =====')
        print(by_season.to_string())

    # the last three seasons, by the league of the opponent
    iran = international.loc[(international['team_league'] == 'Iran. First League') & (international['year'] >= 2023)]
    print('\n===== Iran 2023-2025 by the opponent league =====')
    print(iran
          .groupby('opp_league')
          .agg(n=('diff', 'size'), expected=('expected_points', 'sum'), actual=('actual_points', 'sum'),
               team_rating=('team_rating', 'mean'), opp_rating=('opp_rating', 'mean'))
          .assign(diff=lambda df: df['actual'] - df['expected'])
          .sort_values('n', ascending=False)
          .round(1)
          .to_string())
