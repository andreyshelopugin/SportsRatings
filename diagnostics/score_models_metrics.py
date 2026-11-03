"""30.10: Brier score and RPS of the baselines for the paper (Table score_models),
from the probabilities saved by soccer/matrix_models/compare_models.py (test seasons, 90 minutes)."""
from pathlib import Path

import numpy as np
import pandas as pd

# relative to the working directory, as in compare_models.py (PyCharm runs it from soccer/matrix_models)
RESULTS_DIR = Path('data/soccer/tables/matrix_models')
MODELS = ['skellam', 'dixon_coles', 'markov']


def brier_score(probs: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Multiclass Brier score of every match: sum over H, D, A."""

    return ((probs - actual) ** 2).sum(axis=1)


def ranked_probability_score(probs: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """RPS of every match, the outcomes are ordered H, D, A (Constantinou and Fenton, 2012)."""

    cumulative_difference = np.cumsum(probs - actual, axis=1)[:, :2]

    return (cumulative_difference ** 2).sum(axis=1) / 2


def main():
    predictions = pd.read_feather(RESULTS_DIR / 'test_predictions.feather')

    actual = np.stack([(predictions['outcome'] == outcome).to_numpy(dtype=float) for outcome in ['H', 'D', 'A']], axis=1)

    rows = []
    for source, part in [('all', predictions)] + list(predictions.groupby('source')):
        part_actual = actual[part.index]
        row = {'source': source, 'n_matches': len(part)}
        for name in MODELS:
            probs = part[[f'{name}_home_win', f'{name}_draw', f'{name}_away_win']].to_numpy()
            row[f'{name}_brier'] = round(float(brier_score(probs, part_actual).mean()), 6)
            row[f'{name}_rps'] = round(float(ranked_probability_score(probs, part_actual).mean()), 6)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / 'test_brier_rps.csv', index=False)

    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
