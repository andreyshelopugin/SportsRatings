import gc
import os

import numpy as np
import optuna
import pandas as pd
from scipy.stats import skellam

from catboost_model import CatBoost, CatBoostRegressor, Pool
from config import config
from utils.functions import save_feather


class OutcomesCatBoost(CatBoost):
    """The CatBoost model predicts the number of goals for each team.
    Afterward, we make the assumption that goals in football follow a Poisson distribution.
    This assumption enables us to utilize the probability mass function for a Skellam distribution."""

    def __init__(self, target='score'):
        super().__init__(target=target,
                         fold_count=3,
                         iterations=10_000,
                         learning_rate=0.006551,
                         colsample_bylevel=0.9092111469100367,
                         depth=9,
                         l2_leaf_reg=1.790062,
                         # bagging_temperature=1,
                         random_strength=3.261477,
                         subsample=0.955578,
                         od_wait=100,
                         min_data_in_leaf=29)

        self.loss_function = 'Poisson'
        self.cv_metric = 'test-Poisson-mean'
        self.bootstrap_type = 'Bernoulli'
        self.boosting_type = 'Plain'

        self.train_set_path = config.soccer['outcomes']['datasets']['train']
        self.eval_set_path = config.soccer['outcomes']['datasets']['eval']
        self.test_set_path = config.soccer['outcomes']['datasets']['test']

        self.predictions_path = config.soccer['outcomes']['catboost_predictions']

        self.model_path = config.soccer['outcomes']['catboost_model']

        self.features = [

            'venue', 'is_pandemic', 'is_no_spectators', 'is_different_stadium', 'is_limited_spectators',

            # the same features as in OutcomesLGBM
            'is_extra_time_possible',

            'days_since_last_game', 'games_in_season', 'opp_days_since_last_game', 'opp_games_in_season',

            'avg_scoring_5', 'avg_scoring_10', 'avg_scoring_20', 'avg_scoring_30',
            'avg_scoring_tournament_5', 'avg_scoring_tournament_10', 'avg_scoring_tournament_20', 'avg_scoring_tournament_30',

            'avg_conceded_5', 'avg_conceded_10', 'avg_conceded_20', 'avg_conceded_30',
            'opp_avg_conceded_5', 'opp_avg_conceded_10', 'opp_avg_conceded_20', 'opp_avg_conceded_30',
            'opp_avg_scoring_5', 'opp_avg_scoring_10', 'opp_avg_scoring_20', 'opp_avg_scoring_30',

            'location_mean_score_5', 'location_mean_score_10', 'location_mean_score_20', 'location_mean_score_30',
            'location_median_score_5', 'location_median_score_10', 'location_median_score_20', 'location_median_score_30',

            'location_mean_score_5_conceded', 'location_mean_score_10_conceded', 'location_mean_score_20_conceded', 'location_mean_score_30_conceded',
            'location_median_score_5_conceded', 'location_median_score_10_conceded', 'location_median_score_20_conceded',
            'location_median_score_30_conceded',
            'location_max_score_10',
            'league', 'opp_league', 'tournament', 'tournament_type',
            'leagues_difference'
        ]

        self.cat_features = ['league', 'opp_league', 'tournament', 'tournament_type']

    def optuna_optimization(self, n_trials: int, study_name: str) -> pd.DataFrame:
        """"""

        study = optuna.create_study(study_name=study_name, direction="minimize", load_if_exists=True)

        train = pd.read_feather(self.train_set_path)

        cv_dataset = Pool(data=train.loc[:, self.features],
                          label=train[self.target],
                          cat_features=self.cat_features)

        del train
        gc.collect()

        def objective(trial):
            """boosting_type:
            Ordered — Usually provides better quality on small datasets (< 50k), but it may be slower than the Plain scheme.
            Plain — The classic gradient boosting scheme."""

            params = {
                'loss_function': self.loss_function,
                'od_type': self.od_type,
                'iterations': self.iterations,
                "boosting_type": self.boosting_type,
                'verbose': 500,
                "allow_writing_files": False,
                'task_type': 'GPU',
                'thread_count': 8,

                'od_wait': self.od_wait,
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),

                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.01, 5, log=True),
                "colsample_bylevel": 1.0,
                "depth": trial.suggest_int("depth", 6, 12),

                "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),
                'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 5, 50),
                "random_strength": trial.suggest_float("random_strength", 0.1, 5),

            }

            if params["bootstrap_type"] == "Bayesian":
                params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
            elif params["bootstrap_type"] == "Bernoulli":
                params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

            return self.optuna_cv_score(params, cv_dataset)

        study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True, show_progress_bar=True)

        experiments = study.trials_dataframe()

        experiments = (experiments
                       .drop(columns=['datetime_start', 'datetime_complete', 'state'])
                       .sort_values(['value'], ascending=True))

        experiments.columns = [col.replace('params_', '') for col in experiments.columns]

        experiments_path = config.soccer['outcomes']['experiments']['catboost'] + study_name + '.feather'

        if os.path.exists(experiments_path):
            old_experiments = pd.read_feather(experiments_path)
            experiments = pd.concat([old_experiments, experiments], ignore_index=True)

        save_feather(experiments, experiments_path)

        return experiments

    def predict(self, new_data=None) -> pd.DataFrame:
        """Predicts the outcomes of the football matches."""

        if new_data is None:
            new_data = pd.read_feather(self.test_set_path)

        model = CatBoostRegressor()

        model.load_model(self.model_path)

        new_data['prediction'] = model.predict(new_data.loc[:, self.features])

        home = (new_data
                .loc[(new_data['is_home'] == True), ['match_id', 'prediction']]
                .dropna()
                .rename(columns={'prediction': 'home_goals'}))

        away = (new_data
                .loc[(new_data['is_home'] == False), ['match_id', 'prediction']]
                .dropna()
                .rename(columns={'opponent': 'team', 'prediction': 'away_goals'}))

        home['match_id'] = home['match_id'].to_numpy('int')
        away['match_id'] = away['match_id'].to_numpy('int')

        predictions = home.merge(away, how='inner', on='match_id', validate='1:1')

        # use the assumption that goals are Poisson distributed
        predictions['home_win'] = (predictions
                                   .loc[:, ['home_goals', 'away_goals']]
                                   .apply(lambda df: skellam.pmf(range(1, 30), df['home_goals'], df['away_goals']).sum(), axis=1))

        predictions['draw'] = (predictions
                               .loc[:, ['home_goals', 'away_goals']]
                               .apply(lambda df: skellam.pmf(0, df['home_goals'], df['away_goals']), axis=1))

        predictions['away_win'] = (1 - predictions['home_win'] - predictions['draw'])

        save_feather(predictions, self.predictions_path)

        return predictions

    def outcome_test_metrics(self) -> dict:
        """Calculate test metrics for match outcome predictions (home_win/draw/away_win)."""
        from sklearn.metrics import accuracy_score, brier_score_loss
        from utils.metrics import three_outcomes_log_loss_string_outcome
        from collections import Counter

        # load predictions
        predictions = pd.read_feather(self.predictions_path)
        matches = pd.read_feather(config.soccer['preprocessed_matches'], columns=['match_id', 'outcome'])

        predictions = predictions.merge(matches, how='inner', on='match_id', validate='1:1')

        # create predicted outcome (argmax)
        y_pred_proba = predictions[['home_win', 'draw', 'away_win']].values
        outcome_labels = ['H', 'D', 'A']
        y_pred_class = np.array([outcome_labels[i] for i in y_pred_proba.argmax(axis=1)])
        y_true = predictions['outcome'].values

        # calculate metrics
        accuracy = accuracy_score(y_true, y_pred_class)

        # custom log loss using string labels directly
        custom_logloss = np.mean([
            three_outcomes_log_loss_string_outcome(
                row[0],  # home_win probability
                row[1],  # draw probability
                row[2],  # away_win probability
                y_true[i]  # 'H', 'D', or 'A' directly
            )
            for i, row in enumerate(y_pred_proba)
        ])

        # brier score for each outcome
        brier_home = brier_score_loss(y_true == 'H', y_pred_proba[:, 0])
        brier_draw = brier_score_loss(y_true == 'D', y_pred_proba[:, 1])
        brier_away = brier_score_loss(y_true == 'A', y_pred_proba[:, 2])
        brier_avg = (brier_home + brier_draw + brier_away) / 3

        # baseline: predict most common outcome
        most_common = Counter(y_true).most_common(1)[0][0]
        baseline_preds = np.full(len(y_true), most_common)
        baseline_accuracy = accuracy_score(y_true, baseline_preds)

        # baseline probabilities (historical distribution)
        outcome_counts = Counter(y_true)
        total = len(y_true)
        baseline_probs = np.array([[outcome_counts['H'] / total, outcome_counts['D'] / total, outcome_counts['A'] / total]
                                   for _ in range(len(y_true))])

        # baseline custom log loss
        baseline_custom_logloss = np.mean([three_outcomes_log_loss_string_outcome(row[0], row[1], row[2], y_true[i])
                                           for i, row in enumerate(baseline_probs)])

        # baseline brier
        baseline_brier_home = brier_score_loss(y_true == 'H', baseline_probs[:, 0])
        baseline_brier_draw = brier_score_loss(y_true == 'D', baseline_probs[:, 1])
        baseline_brier_away = brier_score_loss(y_true == 'A', baseline_probs[:, 2])
        baseline_brier_avg = (baseline_brier_home + baseline_brier_draw + baseline_brier_away) / 3

        metrics = {
            "accuracy": accuracy,
            "custom_log_loss": custom_logloss,
            "brier_score": brier_avg,
            "brier_home": brier_home,
            "brier_draw": brier_draw,
            "brier_away": brier_away,
        }

        baseline_metrics = {
            "accuracy": baseline_accuracy,
            "custom_log_loss": baseline_custom_logloss,
            "brier_score": baseline_brier_avg,
            "brier_home": baseline_brier_home,
            "brier_draw": baseline_brier_draw,
            "brier_away": baseline_brier_away,
        }

        # print results
        print("\n" + "=" * 80)
        print("OUTCOME PREDICTION METRICS (CatBoost)")
        print("=" * 80)

        # metrics comparison
        print("\n  Classification Metrics:")
        for m in ["accuracy", "custom_log_loss", "brier_score"]:
            delta = metrics[m] - baseline_metrics[m]
            sign = "+" if delta >= 0 else ""

            # for accuracy, higher is better
            if m == "accuracy":
                improvement = "better" if delta > 0 else "worse"
            else:
                # for loss/brier, lower is better
                improvement = "better" if delta < 0 else "worse"

            print(f"    {m:<20} "
                  f"model = {metrics[m]:>8.5f} | "
                  f"baseline = {baseline_metrics[m]:>8.5f} | "
                  f"Δ = {sign}{delta:>8.5f} ({improvement})")

        return metrics
