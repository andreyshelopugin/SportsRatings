import gc
import os

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from lightgbm import Booster
from scipy.stats import skellam
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from config import config
from lgbm import LightGBM
from utils.functions import save_feather


class OutcomesLGBM(LightGBM):
    """The Light GBM model predicts the number of goals for each team.
    Afterward, we make the assumption that goals in football follow a Poisson distribution.
    This assumption enables us to utilize the probability mass function for a Skellam distribution.

    Extra time matches contain 120' score, so log(exposure) is used as init_score (offset) in training.
    Booster.predict ignores init_score, so predictions are regular time (90') goals expectations."""

    def __init__(self):
        super().__init__(target='score',
                         objective='poisson',
                         metric='poisson',
                         cv_metric='valid poisson-mean',
                         n_estimators=10_000,
                         # optuna 'night' (10.11), trial 42: CV poisson 0.869593 with 8,393 trees,
                         # the previous params (learning_rate 0.006, 146 leaves): 0.869820 with 8,020 trees
                         learning_rate=0.014855,
                         num_leaves=43,
                         feature_fraction=0.273134,
                         lambda_l1=4.145553,
                         lambda_l2=2.550531,
                         bagging_fraction=0.813335,
                         bagging_freq=2,
                         min_data_in_leaf=42,
                         # 63 bins instead of 255: faster histograms, the features are means over a few matches
                         max_bin=63,
                         early_stopping_round=100,
                         cv=3)

        self.features = [

            # venue: +1 at home, 0 neutral, -1 away (instead of is_home, is_neutral, home_indicator)
            'venue', 'is_pandemic', 'is_no_spectators', 'is_different_stadium', 'is_limited_spectators',

            # final_stage is a value of tournament_type (national_final_stage)
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

        # potential data leak here: there's a wrong assumption that the strengths of the leagues remain stable over time.
        # on the one hand, we want to take into account information about opponents' leagues.
        # on the other hand, we're looking into the future when using this information.
        self.cat_features = ['league', 'opp_league', 'tournament', 'tournament_type']
        # exposure is not a feature: it is used as offset (init_score), see LightGBM._init_score
        self.exposure_column = 'exposure'
        self.columns = self.features + [self.target, self.exposure_column]

        self.train_set_path = config.soccer['outcomes']['datasets']['train']
        self.eval_set_path = config.soccer['outcomes']['datasets']['eval']
        self.test_set_path = config.soccer['outcomes']['datasets']['test']

        self.predictions_path = config.soccer['outcomes']['lgbm_predictions']

        self.model_path = config.soccer['outcomes']['lgbm_model']

        if os.path.exists(self.model_path):
            self.model = Booster(model_file=self.model_path)
        else:
            self.model = None

    @staticmethod
    def _experiments(study: optuna.Study, study_name: str) -> pd.DataFrame:
        """All trials of the study, the best first, saved to feather (overwritten: the study holds all trials)."""

        experiments = (study
                       .trials_dataframe()
                       .drop(columns=['datetime_start', 'datetime_complete', 'state'])
                       .sort_values(['value'], ascending=True))

        experiments.columns = [col.replace('params_', '').replace('user_attrs_', '') for col in experiments.columns]

        save_feather(experiments, config.soccer['outcomes']['experiments']['lgbm'] + study_name + '.feather')

        return experiments

    def _cv_dataset(self) -> tuple[lgb.Dataset, list]:
        """The dataset and the folds of the CV on train + eval. The folds are split by match_id:
        the home and the away rows of a match mirror each other, they must be in the same fold."""

        train_set = pd.read_feather(self.train_set_path, columns=self.columns + ['match_id'])
        eval_set = pd.read_feather(self.eval_set_path, columns=self.columns + ['match_id'])
        combined = pd.concat([train_set, eval_set], ignore_index=True)

        folds = list(GroupKFold(n_splits=self.cv).split(combined, groups=combined['match_id']))

        x, y = combined.loc[:, self.features], combined[self.target]

        # 120' score for extra time matches: offset makes the model learn regular time (90') goals
        init_score = self._init_score(combined)

        # feature_pre_filter=False: the bins are built once for all trials, min_data_in_leaf changes between them
        cv_dataset = lgb.Dataset(x, y, init_score=init_score, categorical_feature=self.cat_features, free_raw_data=False,
                                 params={'feature_pre_filter': False, 'max_bin': self.max_bin})

        del train_set, eval_set, combined, x, y, init_score
        gc.collect()

        return cv_dataset, folds

    def cv_score(self) -> tuple[float, int]:
        """The CV loss and the number of trees of the parameters of the model (__init__), on the same folds as optuna_optimization:
        the reference point for the trials."""

        cv_dataset, folds = self._cv_dataset()

        params = {**self.params(), 'verbose': -1}
        params.pop('early_stopping_round')

        cv_results = lgb.cv(params=params,
                            train_set=cv_dataset,
                            folds=folds,
                            callbacks=[lgb.log_evaluation(period=500), lgb.early_stopping(stopping_rounds=self.early_stopping_round)],
                            seed=self.seed)

        n_trees = len(cv_results[self.cv_metric])
        print(f'CV {self.cv_metric}: {cv_results[self.cv_metric][-1]:.6f}, trees: {n_trees}, trees x leaves: {n_trees * self.num_leaves}')

        return cv_results[self.cv_metric][-1], n_trees

    def optuna_optimization(self, n_trials: int, study_name: str) -> pd.DataFrame:
        """Hyperparameters by the CV on train + eval (see _cv_dataset)."""

        study = optuna.create_study(study_name=study_name, direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=self.seed))

        cv_dataset, folds = self._cv_dataset()

        def objective(trial):
            params = {
                'objective': self.objective,
                'metric': self.metric,
                'n_estimators': self.n_estimators,
                'verbose': -1,
                'seed': self.seed,

                'learning_rate': trial.suggest_float('learning_rate', 0.012, 0.1, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.15, 0.7),
                'num_leaves': trial.suggest_int('num_leaves', 15, 150),
                'lambda_l1': trial.suggest_float('lambda_l1', 1.0, 5.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1.0, 5.0, log=True),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.9),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 25, 80),
                'max_bin': self.max_bin,
                'device_type': 'cpu'
            }

            cv_results = lgb.cv(params=params,
                                train_set=cv_dataset,
                                folds=folds,
                                callbacks=[lgb.log_evaluation(period=500), lgb.early_stopping(stopping_rounds=self.early_stopping_round)],
                                seed=self.seed)

            # the cost of the prediction: trees x leaves, to choose a lighter model with almost the same loss
            n_trees = len(cv_results[self.cv_metric])
            trial.set_user_attr('n_trees', n_trees)
            trial.set_user_attr('prediction_cost', n_trees * params['num_leaves'])

            return cv_results[self.cv_metric][-1]

        study.optimize(objective,
                       n_trials=n_trials,
                       gc_after_trial=True,
                       show_progress_bar=True,
                       callbacks=[lambda study, trial: self._experiments(study, study_name)])

        return self._experiments(study, study_name)

    def predict_chunks(self, new_data=None, n_chunks: int = 50) -> pd.DataFrame:
        """Goals expectations (regular time, Booster.predict ignores init_score) and the outcome probabilities
        (Skellam) of all matches, in chunks to save memory."""

        if new_data is None:
            # full dataset
            # is_home is not a feature (venue), but it splits the rows into home and away
            columns = ['match_id', 'is_home'] + self.features
            new_data = pd.read_feather(config.soccer['outcomes']['datasets']['dataset'], columns=columns)

        new_data['match_id'] = new_data['match_id'].to_numpy(dtype='int64')

        unique_match_ids = new_data['match_id'].drop_duplicates().to_numpy()
        match_id_chunks = np.array_split(unique_match_ids, n_chunks)

        prediction_chunks = []

        for match_ids_chunk in tqdm(match_id_chunks, desc='Predicting outcomes'):
            chunk = new_data.loc[new_data['match_id'].isin(match_ids_chunk)].copy()

            chunk['goals_expectations'] = self.model.predict(chunk.loc[:, self.features])

            home = (chunk
                    .loc[chunk['is_home'] == True, ['match_id', 'goals_expectations']]
                    .rename(columns={'goals_expectations': 'home_goals_expectations'}))

            away = (chunk
                    .loc[chunk['is_home'] == False, ['match_id', 'goals_expectations']]
                    .rename(columns={'goals_expectations': 'away_goals_expectations'}))

            home['match_id'] = home['match_id'].to_numpy(dtype='int64')
            away['match_id'] = away['match_id'].to_numpy(dtype='int64')

            predictions_chunk = home.merge(away, how='inner', on='match_id', validate='1:1')

            home_lambda = predictions_chunk['home_goals_expectations'].to_numpy()
            away_lambda = predictions_chunk['away_goals_expectations'].to_numpy()

            # D = home_goals - away_goals
            predictions_chunk['draw'] = skellam.pmf(0, home_lambda, away_lambda)

            predictions_chunk['home_win'] = 1 - skellam.cdf(0, home_lambda, away_lambda)

            predictions_chunk['away_win'] = 1 - predictions_chunk['home_win'] - predictions_chunk['draw']

            prediction_chunks.append(predictions_chunk)

        predictions = pd.concat(prediction_chunks, ignore_index=True)

        del prediction_chunks, new_data
        gc.collect()

        save_feather(predictions, self.predictions_path)

        return predictions

    def outcome_test_metrics(self) -> dict:
        """Calculate test metrics for match outcome predictions (home_win/draw/away_win).
        Predictions are for regular time, so they are compared with the regular time (90') outcome."""
        from collections import Counter

        from sklearn.metrics import accuracy_score, brier_score_loss

        # load predictions (full dataset)
        predictions = pd.read_feather(self.predictions_path,
                                      columns=['match_id', 'home_goals_expectations', 'away_goals_expectations',
                                               'home_win', 'draw', 'away_win'])

        # outcomes from the test set (home rows, one per match): the metrics of the test seasons only
        matches = pd.read_feather(self.test_set_path, columns=['match_id', 'is_home', 'outcome_regular_time'])
        matches = (matches
                   .loc[matches['is_home'] == True, ['match_id', 'outcome_regular_time']]
                   .rename(columns={'outcome_regular_time': 'outcome'}))

        matches['match_id'] = matches['match_id'].to_numpy(dtype='int64')

        predictions = predictions.merge(matches, how='inner', on='match_id', validate='1:1')

        # create predicted outcome (argmax)
        y_pred_proba = predictions[['home_win', 'draw', 'away_win']].values
        outcome_labels = ['H', 'D', 'A']
        y_pred_class = np.array([outcome_labels[i] for i in y_pred_proba.argmax(axis=1)])
        y_true = predictions['outcome'].values

        # calculate metrics
        accuracy = accuracy_score(y_true, y_pred_class)

        # log loss of the three binary outcomes, averaged (as utils.metrics.three_outcomes_log_loss_string_outcome)
        y_one_hot = np.column_stack([y_true == label for label in outcome_labels])

        def three_outcomes_log_loss(probs: np.ndarray) -> float:
            return -np.mean(np.where(y_one_hot, np.log(probs), np.log(1 - probs)))

        custom_logloss = three_outcomes_log_loss(y_pred_proba)

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
        baseline_probs = np.tile([outcome_counts[label] / total for label in outcome_labels], (total, 1))

        baseline_custom_logloss = three_outcomes_log_loss(baseline_probs)

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
        print("OUTCOME PREDICTION METRICS")
        print("=" * 80)

        # metrics comparison
        print(f"\n  Classification Metrics:")
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
