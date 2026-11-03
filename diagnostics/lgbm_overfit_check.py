import numpy as np
import pandas as pd

from config import config

# 29.10: are lgbm goals expectations on the train seasons honest enough to fit Dixon-Coles / Markov on them?
# train: the boosting was fitted on it, eval: 10% of the train seasons (only early stopping), test: seasons >= 2021.
# if train ~ eval ~ test, no overfitting and no leakage through the features (rolling stats with the current match).


def poisson_deviance(goals: np.array, lamb: np.array) -> float:
    """Mean Poisson deviance per team-match."""

    log_term = np.where(goals > 0, goals * np.log(np.where(goals > 0, goals, 1) / lamb), 0)

    return float(np.mean(2 * (log_term - (goals - lamb))))


def outcome_log_loss(predictions: pd.DataFrame) -> float:
    """Three outcomes log loss, Skellam probabilities of the lgbm."""

    probability = np.select([predictions['outcome'] == 'H', predictions['outcome'] == 'D'],
                            [predictions['home_win'], predictions['draw']],
                            predictions['away_win'])

    return float(-np.mean(np.log(np.clip(probability, 1e-15, 1))))


def metrics(predictions: pd.DataFrame) -> dict:
    """"""
    goals = np.concatenate([predictions['home_score'].to_numpy(), predictions['away_score'].to_numpy()])
    lamb = np.concatenate([predictions['home_goals_expectations'].to_numpy(),
                           predictions['away_goals_expectations'].to_numpy()])

    return {'n_matches': len(predictions),
            'poisson_deviance': round(poisson_deviance(goals, lamb), 5),
            'mae': round(float(np.mean(np.abs(goals - lamb))), 5),
            'mean_goals': round(float(goals.mean()), 4),
            'mean_expectation': round(float(lamb.mean()), 4),
            'skellam_log_loss': round(outcome_log_loss(predictions), 5)}


def main():
    matches = pd.read_feather(config.soccer['matches'],
                              columns=['match_id', 'season', 'source', 'home_score', 'away_score', 'outcome'])

    predictions = pd.read_feather(config.soccer['outcomes']['lgbm_predictions'],
                                  columns=['match_id', 'home_goals_expectations', 'away_goals_expectations',
                                           'home_win', 'draw', 'away_win'])

    predictions = predictions.merge(matches, how='inner', on='match_id', validate='1:1')

    print(f'#matches without prediction: {len(matches) - len(predictions)}')

    split = dict()
    for set_name in ('train', 'eval', 'test'):
        ids = pd.read_feather(config.soccer['outcomes']['datasets'][set_name], columns=['match_id'])['match_id']
        split.update(dict.fromkeys(ids.unique(), set_name))

    predictions['set'] = predictions['match_id'].map(split)

    print(f'#matches without set: {predictions["set"].isna().sum()}')

    report = []
    for (source, set_name), part in predictions.groupby(['source', 'set']):
        report.append({'source': source, 'set': set_name} | metrics(part))

    # train seasons only, year by year: a train-eval gap inside the same season is overfitting, not drift
    for (set_name, season), part in predictions.loc[predictions['set'] != 'test'].groupby(['set', 'season']):
        if season >= 2016:
            report.append({'source': 'all', 'set': f'{set_name} {season}'} | metrics(part))

    print(pd.DataFrame(report).to_string(index=False))


if __name__ == '__main__':
    main()
