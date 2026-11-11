
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from soccer.matrix_models.dixon_coles import DixonColes
from soccer.matrix_models.markov import Markov
from soccer.matrix_models.matrix_models import MatrixModels
from soccer.matrix_models.poisson import Poisson
from soccer.matrix_models.stats_utils import win_tie_loss_probs
from utils.metrics import three_outcomes_log_loss_string_outcome

RESULTS_DIR = Path('data/soccer/tables/matrix_models')


def probabilities(model, matches: pd.DataFrame, desc: str) -> np.ndarray:
    """Home win, draw, away win of every match."""

    m1 = matches['home_goals_expectations'].to_numpy()
    m2 = matches['away_goals_expectations'].to_numpy()

    return np.array([win_tie_loss_probs(model(m1[i], m2[i])) for i in tqdm(range(len(m1)), desc=desc, leave=False)])


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # rho of DixonColes is refitted on the held-out part of the train seasons (a grid search, minutes),
    # the params of Markov are the defaults of the class: their fit (Markov.fit) takes ~1.5 hours
    dixon_coles = DixonColes()
    dixon_coles.fit()
    markov = Markov()

    _, test_matches = MatrixModels().get_data()
    test_matches = test_matches.reset_index(drop=True)

    models = {'skellam': Poisson().score_matrix_model,
              'dixon_coles': dixon_coles.score_matrix_model,
              'markov': markov.score_matrix_model}

    predictions = test_matches[['match_id', 'season', 'source', 'outcome']].copy()
    for name, model in models.items():
        probs = probabilities(model, test_matches, desc=f'{name} test')
        predictions[f'{name}_home_win'] = probs[:, 0]
        predictions[f'{name}_draw'] = probs[:, 1]
        predictions[f'{name}_away_win'] = probs[:, 2]
        predictions[f'{name}_loss'] = [three_outcomes_log_loss_string_outcome(w, d, l, o)
                                       for w, d, l, o in zip(probs[:, 0], probs[:, 1], probs[:, 2],
                                                             predictions['outcome'])]

    predictions.to_feather(RESULTS_DIR / 'test_predictions.feather')

    rows = []
    for source, part in [('all', predictions)] + list(predictions.groupby('source')):
        rows.append({'source': source, 'n_matches': len(part)} |
                    {name: round(part[f'{name}_loss'].mean(), 6) for name in models})
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / 'test_loss.csv', index=False)

    params = pd.DataFrame([{'model': 'dixon_coles', 'param': 'rho', 'value': dixon_coles.dc_param}] +
                          [{'model': 'markov', 'param': name, 'value': round(getattr(markov, name), 6)}
                           for name in ['low_scoring_00', 'low_scoring_10_01', 'low_scoring_11',
                                        'one_goal_lead_param', 'one_goals_trail_param',
                                        'two_goals_lead_param', 'two_goals_trail_param',
                                        'three_goals_lead_param', 'three_goals_trail_param']])
    params.to_csv(RESULTS_DIR / 'params.csv', index=False)

    print('\n===== test outcome log loss (90 minutes) =====')
    print(summary.to_string(index=False))
    print(params.to_string(index=False))
    print(f'saved: {RESULTS_DIR}')


if __name__ == '__main__':
    main()
