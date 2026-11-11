"""The main table of the paper: Glicko (train-fitted params, sequential predictions of EvaluateGlicko.save_test_predictions)
vs Elo (elo_baseline.py) and the score matrix models (soccer/matrix_models/compare_models.py)
on the same test matches (clubs, seasons >= 2021, 90 minutes):
log loss, Brier, RPS for all / domestic / international matches, the average of Glicko and Markov,
the paired bootstrap of the differences of the log loss.
Run from the root of the project or from diagnostics/ (the paths are from the root)."""
from pathlib import Path

import numpy as np
import pandas as pd

from config import config

ROOT = Path(__file__).resolve().parent.parent

GLICKO_PREDICTIONS = ROOT / 'data/soccer/tables/glicko_test_predictions_sequential.feather'
BASELINE_PREDICTIONS = ROOT / 'data/soccer/tables/matrix_models/test_predictions.feather'
RESULTS = ROOT / 'data/soccer/tables/glicko_vs_baselines.csv'
ELO_PREDICTIONS = ROOT / 'data/soccer/tables/elo_test_predictions.feather'

BASELINES = ['skellam', 'dixon_coles', 'markov']
N_BOOTSTRAP = 1000


def metrics(probs: np.ndarray, actual: np.ndarray) -> tuple:
    """Log loss (mean of the three binary log losses), Brier, RPS of every match."""

    clipped = np.clip(probs, 1e-15, 1 - 1e-15)
    log_loss = -(actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped)).sum(axis=1) / 3
    brier = ((probs - actual) ** 2).sum(axis=1)
    rps = (np.cumsum(probs - actual, axis=1)[:, :2] ** 2).sum(axis=1) / 2

    return log_loss, brier, rps


def main():
    glicko = pd.read_feather(GLICKO_PREDICTIONS)
    baselines = pd.read_feather(BASELINE_PREDICTIONS)
    is_domestic = (pd.read_feather(ROOT / config.soccer['glicko_matches'], columns=['match_id', 'is_domestic'])
                   .drop_duplicates('match_id'))

    data = (baselines
            .merge(glicko, on='match_id', how='inner', validate='1:1')
            .merge(is_domestic, on='match_id', how='left', validate='1:1')
            .reset_index(drop=True))

    data = data.merge(pd.read_feather(ELO_PREDICTIONS), on='match_id', how='inner', validate='1:1')
    models = ['elo'] + BASELINES + ['glicko']
    print(f'#matches: {len(data)} (baselines {len(baselines)}, glicko {len(glicko)})')

    actual = np.stack([(data['outcome'] == outcome).to_numpy(dtype=float) for outcome in ['H', 'D', 'A']], axis=1)

    probs = {name: data[[f'{name}_home_win', f'{name}_draw', f'{name}_away_win']].to_numpy()
             for name in models}
    probs['glicko_markov'] = (probs['glicko'] + probs['markov']) / 2

    scores = {name: metrics(p, actual) for name, p in probs.items()}

    rng = np.random.default_rng(0)

    rows = []
    for group, mask in [('all', np.ones(len(data), dtype=bool)),
                        ('domestic', data['is_domestic'].to_numpy() == True),
                        ('international', data['is_domestic'].to_numpy() == False)]:
        n = mask.sum()
        # the same samples for every model (paired): the same seed in every group, the samples are not stored (memory)
        seed = int(rng.integers(0, 2 ** 31))
        glicko_log_loss = scores['glicko'][0][mask]

        for name, (log_loss, brier, rps) in scores.items():
            row = {'group': group, 'n_matches': n, 'model': name,
                   'log_loss': round(log_loss[mask].mean(), 6),
                   'brier': round(brier[mask].mean(), 6),
                   'rps': round(rps[mask].mean(), 6)}

            # glicko - model: < 0 if glicko is better
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
