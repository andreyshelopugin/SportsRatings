import gc
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import Dataset, LGBMClassifier, LGBMRegressor, cv as cv_score, Booster
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold


class LightGBM(object):

    def __init__(self, target: str,
                 objective,
                 metric,
                 cv_metric,
                 is_regression=True,
                 n_estimators=5000,
                 learning_rate=0.005,
                 num_leaves=31,
                 feature_fraction=1.0,
                 lambda_l1=0.0,
                 lambda_l2=0.0,
                 bagging_fraction=1.0,
                 bagging_freq=1,
                 min_data_in_leaf=20,
                 max_bin=255,
                 early_stopping_round=50,
                 cv=3,
                 seed=7,
                 device='cpu'):

        self.is_regression = is_regression

        self.train_set_path = ''
        self.eval_set_path = ''
        self.test_set_path = ''

        self.model_path = ''

        self.target = target

        self.objective = objective
        self.metric = metric
        self.cv_metric = cv_metric

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.early_stopping_round = early_stopping_round
        self.num_leaves = num_leaves
        self.feature_fraction = feature_fraction
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.min_data_in_leaf = min_data_in_leaf
        self.max_bin = max_bin

        self.cv = cv
        self.seed = seed

        self.device = device

        self.features = []
        self.cat_features = []
        self.columns = []

        self.weights = None

        # column with the duration of the observation (e.g. 120' matches for goals models).
        # log(exposure) is used as init_score (offset): the model learns the rate per unit of exposure,
        # Booster.predict ignores init_score, so predictions are per unit of exposure
        self.exposure_column = None

    def params(self) -> dict:
        """"""

        params = {
            'device_type': self.device,
            'gpu_platform_id': 0,
            'gpu_device_id': 0,

            'objective': self.objective,
            'metric': self.metric,

            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'early_stopping_round': self.early_stopping_round,
            'random_state': self.seed,
            'verbose': -1,

            'feature_fraction': self.feature_fraction,
            'num_leaves': self.num_leaves,
            'lambda_l1': self.lambda_l1,
            'lambda_l2': self.lambda_l2,
            'bagging_fraction': self.bagging_fraction,
            'bagging_freq': self.bagging_freq,
            'min_data_in_leaf': self.min_data_in_leaf,
            'max_bin': self.max_bin,
        }

        return params

    def _init_score(self, data: pd.DataFrame):
        """Offset on the raw score scale (log for poisson): log(exposure). None if exposure is not used."""
        if self.exposure_column is None:
            return None
        return np.log(data[self.exposure_column].to_numpy())

    def _exposure(self, data: pd.DataFrame) -> np.ndarray:
        """Exposure for comparing predictions with the observed target. Ones if exposure is not used."""
        if self.exposure_column is None:
            return np.ones(data.shape[0])
        return data[self.exposure_column].to_numpy()

    def classifier(self):
        """"""
        return LGBMClassifier(objective=self.objective,
                              n_estimators=self.n_estimators,
                              learning_rate=self.learning_rate,
                              early_stopping_round=self.early_stopping_round,
                              feature_fraction=self.feature_fraction,
                              num_leaves=self.num_leaves,
                              lambda_l1=self.lambda_l1,
                              lambda_l2=self.lambda_l2,
                              bagging_fraction=self.bagging_fraction,
                              bagging_freq=self.bagging_freq,
                              min_data_in_leaf=self.min_data_in_leaf,
                              max_bin=self.max_bin,
                              random_state=self.seed,
                              verbose=-1,
                              device_type=self.device)

    def regression(self):
        """"""
        return LGBMRegressor(objective=self.objective,
                             n_estimators=self.n_estimators,
                             learning_rate=self.learning_rate,
                             early_stopping_round=self.early_stopping_round,
                             feature_fraction=self.feature_fraction,
                             num_leaves=self.num_leaves,
                             lambda_l1=self.lambda_l1,
                             lambda_l2=self.lambda_l2,
                             bagging_fraction=self.bagging_fraction,
                             bagging_freq=self.bagging_freq,
                             min_data_in_leaf=self.min_data_in_leaf,
                             max_bin=self.max_bin,
                             random_state=self.seed,
                             verbose=-1,
                             device_type=self.device)

    def classifier_cv_score(self, params: dict, train_set: Dataset) -> float:
        """
        Run LightGBM CV with multiple metrics, print them,
        and return the primary CV metric.
        """

        metrics = ["auc", "binary_logloss", "binary_error"]

        params["metric"] = metrics

        cv_results = cv_score(params=params,
                              train_set=train_set,
                              stratified=True,
                              nfold=self.cv,
                              seed=self.seed)

        print("\nCV metrics (last iteration):")
        for k in cv_results:
            if k.endswith("-mean"):
                std_key = k.replace("-mean", "-stdv")
                mean = cv_results[k][-1]
                std = cv_results.get(std_key, [None])[-1]

                if std is not None:
                    print(f"  {k:<35} {mean:.6f} ± {std:.6f}")
                else:
                    print(f"  {k:<35} {mean:.6f}")

        return cv_results[self.cv_metric][-1]

    def regression_cv_score(self, params: dict, train_set: Dataset) -> float:
        """"""
        return cv_score(params=params,
                        train_set=train_set,
                        stratified=False,
                        nfold=self.cv,
                        callbacks=[lgb.log_evaluation(period=500), lgb.early_stopping(stopping_rounds=self.early_stopping_round)],
                        seed=self.seed)[self.cv_metric][-1]

    def cross_val_score(self) -> float:
        """"""
        train = pd.read_feather(self.train_set_path, columns=self.columns)

        x, y = train.loc[:, self.features], train[self.target]

        train_set = lgb.Dataset(x, y, init_score=self._init_score(train), categorical_feature=self.cat_features)

        if self.is_regression:
            return self.regression_cv_score(self.params(), train_set)
        else:
            return self.classifier_cv_score(self.params(), train_set)

    def classifier_test_metrics(self):
        """"""

        train_set = pd.read_feather(self.train_set_path, columns=self.columns)
        test_set = pd.read_feather(self.test_set_path, columns=self.columns)
        eval_set = pd.read_feather(self.eval_set_path, columns=self.columns)

        lgbm = LGBMClassifier(**self.params())

        x_train, y_train = train_set.loc[:, self.features], train_set[self.target]
        x_test, y_test = test_set.loc[:, self.features], test_set[self.target]
        x_eval, y_eval = eval_set.loc[:, self.features], eval_set[self.target]

        lgbm.fit(x_train, y_train,
                 categorical_feature=self.cat_features,
                 eval_set=[(x_eval, y_eval)])

        preds = lgbm.predict_proba(x_test)[:, 1]

        p_base = y_train.mean()
        baseline_preds = np.full_like(preds, p_base)

        metrics = {
            "auc": roc_auc_score(y_test, preds),
            "logloss": log_loss(y_test, preds),
            "accuracy": accuracy_score(y_test, preds >= 0.5),
            "binary_error": 1.0 - accuracy_score(y_test, preds >= 0.5),
        }

        baseline_metrics = {
            "auc": 0.5,
            "logloss": log_loss(y_test, baseline_preds),
            "accuracy": accuracy_score(y_test, np.zeros_like(y_test)),
            "binary_error": 1.0 - accuracy_score(y_test, np.zeros_like(y_test)),
        }

        # ---- print results ----
        print("\n=== TEST METRICS ===")
        print(f"train target mean   = {p_base:.5f}")
        print(f"eval target mean    = {y_eval.mean():.5f}")
        print(f"test target mean    = {y_test.mean():.5f}\n")

        for m in ["auc", "logloss", "binary_error"]:
            delta = metrics[m] - baseline_metrics[m]
            sign = "+" if delta >= 0 else ""
            print(f"{m:<15} "
                  f"model = {metrics[m]:.5f} | "
                  f"baseline = {baseline_metrics[m]:.5f} | "
                  f"Δ = {sign}{delta:.5f}")

    def regressor_test_metrics(self):
        """Test metrics of the saved model (save_model): the same model that makes the predictions, no refit."""
        from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

        train_set = pd.read_feather(self.train_set_path, columns=[self.target])
        eval_set = pd.read_feather(self.eval_set_path, columns=[self.target])
        test_set = pd.read_feather(self.test_set_path, columns=self.columns)

        lgbm = Booster(model_file=self.model_path)

        y_train, y_eval = train_set[self.target], eval_set[self.target]
        x_test, y_test = test_set.loc[:, self.features], test_set[self.target]

        test_exposure = self._exposure(test_set)

        # save memory
        del train_set, eval_set, test_set
        gc.collect()

        # predictions are per unit of exposure, the observed target is for the whole exposure
        preds = lgbm.predict(x_test) * test_exposure

        # baseline: predict mean
        baseline_preds = np.full_like(preds, y_train.mean())

        baseline_metrics = {
            "rmse": root_mean_squared_error(y_test, baseline_preds),
            "mae": mean_absolute_error(y_test, baseline_preds),
            "r2": r2_score(y_test, baseline_preds),
        }

        metrics = {
            "rmse": root_mean_squared_error(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
            "r2": r2_score(y_test, preds),
        }

        # ---- print results ----
        print("\n=== TEST METRICS ===")
        print(f"train target mean   = {y_train.mean():.5f}")
        print(f"eval target mean    = {y_eval.mean():.5f}")
        print(f"test target mean    = {y_test.mean():.5f}")
        print(f"train target std    = {y_train.std():.5f}")
        print(f"test target std     = {y_test.std():.5f}\n")

        for m in ["rmse", "mae", "r2"]:
            delta = metrics[m] - baseline_metrics[m]
            sign = "+" if delta >= 0 else ""

            # For R2, higher is better
            if m == "r2":
                improvement = "better" if delta > 0 else "worse"
                print(f"{m:<15} "
                      f"model = {metrics[m]:>8.5f} | "
                      f"baseline = {baseline_metrics[m]:>8.5f} | "
                      f"Δ = {sign}{delta:>8.5f} ({improvement})")
            else:
                # For error metrics, lower is better
                improvement = "better" if delta < 0 else "worse"
                print(f"{m:<15} "
                      f"model = {metrics[m]:>8.5f} | "
                      f"baseline = {baseline_metrics[m]:>8.5f} | "
                      f"Δ = {sign}{delta:>8.5f} ({improvement})")

        return metrics

    def custom_mse_cross_val_score(self) -> float:
        """"""
        train_set = pd.read_feather(self.train_set_path, columns=self.columns)
        eval_set = pd.read_feather(self.eval_set_path)

        x, y = train_set.loc[:, self.features], train_set[self.target]
        x_eval, y_eval = eval_set.loc[:, self.features], eval_set[self.target]

        k_fold = KFold(n_splits=self.cv, shuffle=True, random_state=self.seed)

        weights = self.weights

        lgbm = LGBMRegressor(**self.params())

        cv_score_sum = 0
        for train_index, test_index in k_fold.split(x):
            x_train, x_test = x.loc[train_index], x.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]
            self.weights = weights.loc[train_index]

            lgbm.fit(x_train, y_train,
                     categorical_feature=self.cat_features,
                     eval_set=[(x_eval, y_eval)])

            y_star = lgbm.predict(x_test)

            mse = ((y_test - y_star) ** 2) / weights.loc[test_index]

            cv_score_sum += np.sum(mse)

        return cv_score_sum

    def custom_log_loss_cross_val_score(self) -> float:
        """"""
        train_set = pd.read_feather(self.train_set_path, columns=self.columns)
        eval_set = pd.read_feather(self.eval_set_path, columns=self.columns)

        x, y = train_set.loc[:, self.features], train_set[self.target]
        x_eval, y_eval = eval_set.loc[:, self.features], eval_set[self.target]

        k_fold = KFold(n_splits=self.cv, shuffle=True, random_state=self.seed)

        lgbm = LGBMClassifier(**self.params())

        cv_scores = []
        for train_index, test_index in k_fold.split(x):
            x_train, x_test = x.loc[train_index], x.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]

            lgbm.fit(x_train, y_train,
                     categorical_feature=self.cat_features,
                     eval_set=[(x_eval, y_eval)])

            # a custom objective: the model gives the raw scores (logits), the probabilities are below
            logits = lgbm.predict(x_test, raw_score=True)
            y_star = np.where(logits >= 0,
                              1. / (1. + np.exp(-logits)),
                              np.exp(logits) / (1. + np.exp(logits)))

            ll = log_loss(y_test, y_star) / self.weights.loc[test_index]

            cv_scores.append(np.mean(ll))

        return np.mean(cv_scores)

    def validation(self, is_custom_loss=False) -> pd.DataFrame:
        """
            Check quality on validation set
        """

        test_set = pd.read_feather(self.test_set_path)

        x = test_set.loc[:, self.features]
        y = test_set[self.target]

        lgbm = Booster(model_file=self.model_path)

        if self.is_regression:
            # predictions are per unit of exposure, the observed target is for the whole exposure
            test_set['prediction'] = lgbm.predict(x) * self._exposure(test_set)
            print('MSE: ', mean_squared_error(y, test_set['prediction']))
            print('MAE: ', mean_absolute_error(y, test_set['prediction']))
        else:
            if is_custom_loss:
                logits = lgbm.predict(x)
                preds = np.where(logits >= 0,
                                 1. / (1. + np.exp(-logits)),
                                 np.exp(logits) / (1. + np.exp(logits)))

                test_set['prediction'] = preds

            else:
                test_set['prediction'] = lgbm.predict(x)[:, 1]

            print('LogLoss: ', log_loss(y, test_set['prediction']))
            print('Accuracy: ', accuracy_score(y, test_set['prediction'].round()))

        return test_set

    def feature_importance(self) -> pd.DataFrame:
        """
        """

        train_set = pd.read_feather(self.train_set_path)
        x_train, y_train = train_set.loc[:, self.features], train_set[self.target]

        eval_set = pd.read_feather(self.eval_set_path)
        x_eval, y_eval = eval_set.loc[:, self.features], eval_set[self.target]

        if self.is_regression:
            light_gbm = self.regression()
        else:
            light_gbm = self.classifier()

        eval_init_score = self._init_score(eval_set)

        light_gbm.fit(x_train, y_train,
                      init_score=self._init_score(train_set),
                      categorical_feature=self.cat_features,
                      eval_set=[(x_eval, y_eval)],
                      eval_init_score=None if eval_init_score is None else [eval_init_score])

        feature_importance = (pd.DataFrame(zip(x_train.columns, light_gbm.feature_importances_),
                                           columns=['feature', 'gain'])
                              .sort_values(['gain'], ascending=False)
                              .reset_index(drop=True))

        feature_importance['gain'] = round(100 * feature_importance['gain'] / sum(feature_importance['gain']), 3)

        return feature_importance

    def save_model(self):
        """Fit and save trained model."""
        train_set = pd.read_feather(self.train_set_path, columns=self.columns)
        eval_set = pd.read_feather(self.eval_set_path, columns=self.columns)

        if self.is_regression:
            light_gbm = self.regression()
        else:
            light_gbm = self.classifier()

        x_train = train_set.loc[:, self.features]
        y_train = train_set[self.target]

        x_eval, y_eval = eval_set.loc[:, self.features], eval_set[self.target]

        train_init_score, eval_init_score = self._init_score(train_set), self._init_score(eval_set)

        # save memory
        del train_set, eval_set
        gc.collect()

        light_gbm.fit(x_train, y_train,
                      init_score=train_init_score,
                      categorical_feature=self.cat_features,
                      eval_set=[(x_eval, y_eval)],
                      eval_init_score=None if eval_init_score is None else [eval_init_score],
                      callbacks=[lgb.log_evaluation(period=500)])

        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)

        light_gbm.booster_.save_model(self.model_path, num_iteration=light_gbm.booster_.best_iteration)

        print('lgbm saved')

        del x_train, y_train, x_eval, y_eval, light_gbm
        gc.collect()

    def log_likelihood(self, labels, logits):
        """"""
        # numerically stable sigmoid:
        # preds = np.where(logits >= 0,
        #                  1. / (1. + np.exp(-logits)),
        #                  np.exp(logits) / (1. + np.exp(logits)))

        # clip logits values to avoid overflow
        sigmoid = 1. / (1. + np.exp(-np.clip(logits, -500, 500)))

        gradient = (sigmoid - labels) / self.weights

        # sigma' = sigma * (1 - sigma)
        hessian = sigmoid * (1. - sigmoid) / self.weights

        return gradient, hessian

    def custom_mse(self, y_true, y_pred):
        """"""
        gradient = (y_pred - y_true) / self.weights
        hessian = np.ones_like(y_true) / self.weights
        return gradient, hessian

    @staticmethod
    def custom_eval(labels, logits):
        # numerically stable logsigmoid:
        logsigmoid = np.where(logits >= 0,
                              -np.log(1 + np.exp(-logits)),
                              logits - np.log(1 + np.exp(logits)))

        loss = (-logsigmoid + logits * (1 - labels)).mean()

        return "error", loss, False

    def save_model_with_custom_loss_classification(self):
        """Fit and save trained model with custom loss."""
        train_set = pd.read_feather(self.train_set_path, columns=self.columns)
        eval_set = pd.read_feather(self.eval_set_path)

        self.objective = self.log_likelihood

        light_gbm = self.classifier()

        x_train = train_set.loc[:, self.features]
        y_train = train_set[self.target]

        x_eval, y_eval = eval_set.loc[:, self.features], eval_set[self.target]

        light_gbm.fit(x_train, y_train,
                      categorical_feature=self.cat_features,
                      eval_set=[(x_eval, y_eval)],
                      eval_metric=self.custom_eval)

        joblib.dump(light_gbm, self.model_path)

    def save_model_with_custom_loss_regression(self):
        """Fit and save trained model with custom loss."""
        train_set = pd.read_feather(self.train_set_path, columns=self.columns)
        eval_set = pd.read_feather(self.eval_set_path)

        self.objective = self.custom_mse

        light_gbm = self.regression()

        x_train = train_set.loc[:, self.features]
        y_train = train_set[self.target]

        x_eval, y_eval = eval_set.loc[:, self.features], eval_set[self.target]

        light_gbm.fit(x_train, y_train,
                      categorical_feature=self.cat_features,
                      eval_set=[(x_eval, y_eval)])

        joblib.dump(light_gbm, self.model_path)

    def predict_probability(self, new_data=None, is_custom_loss=False, name: str = None) -> pd.DataFrame:
        """Predict outcomes of football matches."""
        if new_data is None:
            new_data = pd.read_feather(self.test_set_path)

        if name is None:
            name = 'probability'

        lgbm = joblib.load(self.model_path)

        x = new_data.loc[:, self.features]

        if is_custom_loss:
            logits = lgbm.predict_proba(x)
            sigmoids = np.where(logits >= 0,
                                1. / (1. + np.exp(-logits)),
                                np.exp(logits) / (1. + np.exp(logits)))

            new_data[name] = sigmoids

        else:
            new_data[name] = lgbm.predict_proba(x)[:, 1]

        return new_data