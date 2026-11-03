"""The held-out calibration of the league levels (Table 8 of the paper): actual - expected score per match
(score: 1 win, 0.5 draw, 0 loss; expected: p_win + 0.5 * p_draw) of the clubs of a league in the international
club matches of the test seasons (>= 2021), from the sequential predictions of the train fit
(evaluation_pipeline.save_test_predictions). A match between two leagues counts for both.
The noise level of the mean of n matches: 0.44 / sqrt(n) (the standard deviation of the score of one match ~0.44).
Result: data/soccer/tables/league_calibration_test.csv.
Run from the root of the project."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import config
from utils.functions import stage

warnings.filterwarnings('ignore')
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', None)

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / 'data/soccer/tables'
PREDICTIONS = TABLES / 'glicko_test_predictions_sequential.feather'
RESULTS = TABLES / 'league_calibration_test.csv'

SCORE_STD = 0.44
MIN_LEAGUE_MATCHES = 20


def main():
    stage('held-out calibration: international club matches of the test seasons by league')

    predictions = pd.read_feather(PREDICTIONS)
    matches = pd.read_feather(config.soccer['glicko_matches'],
                              columns=['match_id', 'is_home', 'outcome', 'is_domestic', 'home_team_league', 'away_team_league'])
    matches = matches.loc[(matches['is_home'] == True) & (matches['is_domestic'] == False)]

    data = predictions.merge(matches, on='match_id', how='inner', validate='1:1')
    expected_home = data['glicko_home_win'] + 0.5 * data['glicko_draw']
    expected_away = data['glicko_away_win'] + 0.5 * data['glicko_draw']

    rows = pd.concat([pd.DataFrame({'league': data['home_team_league'], 'difference': data['outcome'] - expected_home}),
                      pd.DataFrame({'league': data['away_team_league'], 'difference': (1 - data['outcome']) - expected_away})])

    leagues = rows.groupby('league')['difference'].agg(['size', 'mean']).rename(columns={'size': 'n_matches', 'mean': 'difference'})
    leagues['noise'] = SCORE_STD / np.sqrt(leagues['n_matches'])
    leagues = leagues.sort_values('n_matches', ascending=False).round(4)
    leagues.to_csv(RESULTS)

    large = leagues.loc[leagues['n_matches'] >= MIN_LEAGUE_MATCHES]
    print(f'#matches {len(data)}, leagues with >= {MIN_LEAGUE_MATCHES} matches: {len(large)}')
    print(f"mean |difference| {large['difference'].abs().mean():.4f}, "
          f"expected for a calibrated model ~ 0.8 * mean noise = {0.8 * large['noise'].mean():.4f}, "
          f"share of leagues beyond 2 noise levels {(large['difference'].abs() > 2 * large['noise']).mean():.3f}")
    print(large.head(40).to_string())


if __name__ == '__main__':
    main()
