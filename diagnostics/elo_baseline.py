"""30.10: the Elo baseline for the paper: the classical Elo rating with the home advantage and the draws of Davidson,
all clubs start from 1500 (no league-aware initialization, no transitions), as the standard Elo of the literature.
    q_home = 10 ** ((r_home + h * home_indicator) / 400), q_away = 10 ** (r_away / 400)
    P(H) = q_home / D, P(A) = q_away / D, P(D) = nu * sqrt(q_home * q_away) / D, D = q_home + q_away + nu * sqrt(q_home * q_away)
    r_home += K * (S - E), E = P(H) + 0.5 * P(D), S = 1 / 0.5 / 0 (90 minutes)
K, h and nu are chosen by a grid on the train seasons (< 2021, unweighted outcome log loss, as the other baselines),
the test predictions (seasons >= 2021) with the chosen params are saved for glicko_vs_baselines.py.
Run from the root of the project or from diagnostics/ (the paths are from the root)."""
import itertools
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

from config import config

ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS = ROOT / 'data/soccer/tables/elo_test_predictions.feather'
# GRID_RESULTS = ROOT / 'data/soccer/tables/elo_grid.csv'
# 30.10: the second grid in its own file, the first is kept
GRID_RESULTS = ROOT / 'data/soccer/tables/elo_grid_2.csv'

TRAIN_TEST_SPLIT_SEASON = 2021
INIT_RATING = 1500.

# K_GRID = (10, 15, 20, 25, 30, 40)
# HOME_GRID = (40, 60, 80, 100)
# DRAW_GRID = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1)
# 30.10 21:00: the first grid: best K 30, h 100 (the edge), nu 0.8, train 0.588770, test 0.591207 -> extended around it
K_GRID = (30, 35, 40, 50)
HOME_GRID = (100, 120, 140, 160)
DRAW_GRID = (0.75, 0.8, 0.85)


def load_matches() -> pd.DataFrame:
    """One row per match, home-oriented, in the order of the dates."""

    columns = ['match_id', 'team', 'opp_team', 'is_home', 'outcome', 'home_indicator', 'season', 'date', 'is_domestic']
    matches = pd.read_feather(ROOT / config.soccer['glicko_matches'], columns=columns)

    matches = (matches
               .loc[matches['is_home'] == True]
               .sort_values(['date', 'match_id'])
               .reset_index(drop=True))

    return matches


def run_elo(home: np.ndarray, away: np.ndarray, home_indicator: np.ndarray, outcome: np.ndarray,
            n_teams: int, k: float, home_advantage: float, draw: float) -> np.ndarray:
    """Probabilities of all matches (home win, draw, away win) before the match, the ratings are updated after every match."""

    ratings = np.full(n_teams, INIT_RATING)
    probs = np.empty((len(home), 3))

    for i in range(len(home)):
        q_home = 10 ** ((ratings[home[i]] + home_advantage * home_indicator[i]) / 400)
        q_away = 10 ** (ratings[away[i]] / 400)
        q_draw = draw * np.sqrt(q_home * q_away)
        total = q_home + q_away + q_draw

        p_home, p_draw = q_home / total, q_draw / total
        probs[i] = p_home, p_draw, 1 - p_home - p_draw

        change = k * (outcome[i] - p_home - 0.5 * p_draw)
        ratings[home[i]] += change
        ratings[away[i]] -= change

    return probs


def outcome_log_loss(probs: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Mean of the three binary log losses of every match (the same as Glicko and the score matrix models)."""

    actual = np.stack([outcome == 1, outcome == 0.5, outcome == 0], axis=1).astype(float)
    clipped = np.clip(probs, 1e-15, 1 - 1e-15)

    return -(actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped)).sum(axis=1) / 3


def main():
    matches = load_matches()

    teams = pd.Index(pd.unique(np.concatenate([matches['team'].to_numpy(), matches['opp_team'].to_numpy()])))
    home = teams.get_indexer(matches['team'])
    away = teams.get_indexer(matches['opp_team'])
    home_indicator = matches['home_indicator'].to_numpy(dtype=float)
    outcome = matches['outcome'].to_numpy(dtype=float)
    is_test = matches['season'].to_numpy() >= TRAIN_TEST_SPLIT_SEASON

    print(f'#matches: {len(matches)}, train {(~is_test).sum()}, test {is_test.sum()}, teams {len(teams)}')

    rows = []
    for k, home_advantage, draw in itertools.product(K_GRID, HOME_GRID, DRAW_GRID):
        start = time()
        probs = run_elo(home, away, home_indicator, outcome, len(teams), k, home_advantage, draw)
        losses = outcome_log_loss(probs, outcome)

        # the first two seasons are excluded from the train loss, as in Glicko (the ratings are not established)
        is_train = ~is_test & (matches['season'].to_numpy() >= 2002)
        rows.append({'k': k, 'home_advantage': home_advantage, 'draw': draw,
                     'train_loss': losses[is_train].mean(), 'test_loss': losses[is_test].mean()})
        print(rows[-1] | {'seconds': round(time() - start)}, flush=True)

    grid = pd.DataFrame(rows).sort_values('train_loss')
    grid.to_csv(GRID_RESULTS, index=False)
    print(grid.head(10).to_string(index=False))

    best = grid.iloc[0]
    print(f"best by the train loss: K {best['k']}, h {best['home_advantage']}, nu {best['draw']}, "
          f"train {best['train_loss']:.6f}, test {best['test_loss']:.6f}")

    probs = run_elo(home, away, home_indicator, outcome, len(teams), best['k'], best['home_advantage'], best['draw'])

    predictions = pd.DataFrame({'match_id': matches['match_id'], 'elo_home_win': probs[:, 0],
                                'elo_draw': probs[:, 1], 'elo_away_win': probs[:, 2]}).loc[is_test].reset_index(drop=True)
    predictions.to_feather(PREDICTIONS)
    print(f'test predictions: {len(predictions)}, saved: {PREDICTIONS}')


if __name__ == '__main__':
    main()
