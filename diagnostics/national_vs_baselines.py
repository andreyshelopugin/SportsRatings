"""The national teams table of the paper, as glicko_vs_baselines.py for the clubs:
    1. Glicko with the train-fitted params (data/soccer/params/national_train_fit, matches before 2021-01-01),
       sequential predictions of the test matches (as calculate_test_loss, prediction_records)
    2. Elo with Davidson draws (diagnostics/elo_baseline.run_elo), K, h, nu by a grid on the train matches
    3. Skellam, Dixon-Coles, Markov (compare_models.py, source == national)
    on the common test matches (90 minutes): log loss, Brier, RPS, Glicko + Markov,
    the paired bootstrap of the differences of the log loss, by match type.
Run from the root of the project or from diagnostics/ (the paths are from the root)."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import config

from diagnostics.elo_baseline import run_elo, outcome_log_loss
from diagnostics.glicko_vs_baselines import metrics
from diagnostics.national_ablation import national_params
from national_pipeline import TRAIN_FIT_FOLDER
from soccer.national_ratings.train_glicko import TrainNationalGlicko

BASELINE_PREDICTIONS = ROOT / 'data/soccer/tables/matrix_models/test_predictions.feather'
GLICKO_PREDICTIONS = ROOT / 'data/soccer/tables/national_glicko_test_predictions.feather'
ELO_GRID_RESULTS = ROOT / 'data/soccer/tables/national_elo_grid.csv'
RESULTS = ROOT / 'data/soccer/tables/national_vs_baselines.csv'

SPLIT_DATE = pd.Timestamp('2021-01-01')
BASELINES = ['skellam', 'dixon_coles', 'markov']
N_BOOTSTRAP = 1000

K_GRID = (10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200)
HOME_GRID = (40, 60, 80, 100, 120, 140, 160, 200)
DRAW_GRID = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1)


def glicko_predictions() -> pd.DataFrame:
    """Test predictions of the train fit: the same pass as compare_losses (unweighted), recorded per match."""

    train = TrainNationalGlicko()
    train.match_type_loss_weights = {match_type: 1 for match_type in train.match_type_loss_weights}
    train._setup_train()

    state, country_params = train._initial_state()
    global_params, confederation_params, final_stage_params = train._unflatten(state)
    print('global', global_params, '\nconfederation', confederation_params)

    team_params = train.team_params(global_params, confederation_params, country_params)
    train.loss_filter = None
    train_ratings = train.rate_teams(team_params, train._match_params(team_params, final_stage_params=final_stage_params))

    train._setup_eval()
    team_params = train.team_params(global_params, confederation_params, country_params)
    matches_params = train._match_params(team_params, final_stage_params=final_stage_params)

    train.prediction_records = []
    test_loss = train.calculate_test_loss(team_params, train_ratings, matches_params)
    print(f'glicko test loss (with extra time): {test_loss:.6f}, #matches {len(train.prediction_records)}')

    predictions = pd.DataFrame(train.prediction_records, columns=['match_id', 'glicko_home_win', 'glicko_draw', 'glicko_away_win'])
    predictions.to_feather(GLICKO_PREDICTIONS)

    return predictions


def elo_predictions() -> pd.DataFrame:
    """Elo of the national teams: all teams start from 1500, the grid on the train matches (the first two years skipped)."""

    matches = pd.read_feather(config.soccer['national_glicko_matches'],
                              columns=['match_id', 'team', 'opp_team', 'is_home', 'outcome', 'home_indicator', 'date'])
    matches = matches.loc[matches['is_home'] == True].sort_values(['date', 'match_id']).reset_index(drop=True)

    teams = pd.Index(pd.unique(np.concatenate([matches['team'].to_numpy(), matches['opp_team'].to_numpy()])))
    home = teams.get_indexer(matches['team'])
    away = teams.get_indexer(matches['opp_team'])
    home_indicator = matches['home_indicator'].to_numpy(dtype=float)
    outcome = matches['outcome'].to_numpy(dtype=float)

    is_test = (matches['date'] >= SPLIT_DATE).to_numpy()
    is_train = ~is_test & (matches['date'] >= matches['date'].min() + pd.Timedelta(days=2 * 365)).to_numpy()

    rows = []
    for k, home_advantage, draw in itertools.product(K_GRID, HOME_GRID, DRAW_GRID):
        losses = outcome_log_loss(run_elo(home, away, home_indicator, outcome, len(teams), k, home_advantage, draw), outcome)
        rows.append({'k': k, 'home_advantage': home_advantage, 'draw': draw,
                     'train_loss': losses[is_train].mean(), 'test_loss': losses[is_test].mean()})

    grid = pd.DataFrame(rows).sort_values('train_loss')
    grid.to_csv(ELO_GRID_RESULTS, index=False)
    best = grid.iloc[0]
    print(grid.head(5).to_string(index=False))
    print(f"elo best by the train loss: K {best['k']}, h {best['home_advantage']}, nu {best['draw']}")

    probs = run_elo(home, away, home_indicator, outcome, len(teams), best['k'], best['home_advantage'], best['draw'])

    return pd.DataFrame({'match_id': matches['match_id'], 'elo_home_win': probs[:, 0],
                         'elo_draw': probs[:, 1], 'elo_away_win': probs[:, 2]}).loc[is_test]


def main():
    # the train fit instead of the production params (national_pipeline.train_fit)
    with national_params(TRAIN_FIT_FOLDER):
        glicko = glicko_predictions()
    elo = elo_predictions()

    baselines = pd.read_feather(BASELINE_PREDICTIONS)
    baselines = baselines.loc[baselines['source'] == 'national']

    match_info = pd.read_feather(config.soccer['national_glicko_matches'], columns=['match_id', 'match_type', 'home_is_host',
                                                                                    'away_is_host', 'is_home'])
    match_info = match_info.loc[match_info['is_home'] == True].drop(columns='is_home')

    data = (baselines
            .merge(glicko, on='match_id', how='inner', validate='1:1')
            .merge(elo, on='match_id', how='inner', validate='1:1')
            .merge(match_info, on='match_id', how='left', validate='1:1')
            .reset_index(drop=True))
    print(f'#matches: {len(data)} (baselines {len(baselines)}, glicko {len(glicko)}, elo {len(elo)})')

    actual = np.stack([(data['outcome'] == outcome).to_numpy(dtype=float) for outcome in ['H', 'D', 'A']], axis=1)

    models = ['elo'] + BASELINES + ['glicko']
    probs = {name: data[[f'{name}_home_win', f'{name}_draw', f'{name}_away_win']].to_numpy() for name in models}
    probs['glicko_markov'] = (probs['glicko'] + probs['markov']) / 2
    scores = {name: metrics(p, actual) for name, p in probs.items()}

    is_final = data['match_type'].isin(['world_cup_final_stage', 'final_stage']).to_numpy()
    has_host = (data['home_is_host'] | data['away_is_host']).to_numpy()
    groups = [('all', np.ones(len(data), dtype=bool)),
              ('qualifiers_nations_league', ~is_final),
              ('final_stage', is_final),
              ('final_stage_host', is_final & has_host),
              ('final_stage_no_host', is_final & ~has_host)]

    rng = np.random.default_rng(0)
    rows = []
    for group, mask in groups:
        n = mask.sum()
        seed = int(rng.integers(0, 2 ** 31))
        glicko_log_loss = scores['glicko'][0][mask]

        for name, (log_loss, brier, rps) in scores.items():
            row = {'group': group, 'n_matches': n, 'model': name, 'log_loss': round(log_loss[mask].mean(), 6),
                   'brier': round(brier[mask].mean(), 6), 'rps': round(rps[mask].mean(), 6)}
            if name != 'glicko':
                difference = glicko_log_loss - log_loss[mask]
                sample_rng = np.random.default_rng(seed)
                bootstrap = np.array([difference[sample_rng.integers(0, n, n)].mean() for _ in range(N_BOOTSTRAP)])
                row |= {'glicko_minus_model': round(difference.mean(), 6),
                        'ci_low': round(np.percentile(bootstrap, 2.5), 6),
                        'ci_high': round(np.percentile(bootstrap, 97.5), 6)}
            rows.append(row)

    report = pd.DataFrame(rows)
    report.to_csv(RESULTS, index=False)
    print(report.to_string(index=False))


if __name__ == '__main__':
    main()
