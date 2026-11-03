import gc

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier, Pool, cv
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score


class CatBoost(object):

    def __init__(self,
                 target: str,
                 is_regression=True,
                 loss_function='MSE',
                 cv_metric='test-MSE-mean',
                 iterations=10_000,
                 best_iteration=10_000,
                 learning_rate=0.03,
                 depth=8,
                 l2_leaf_reg=3,
                 colsample_bylevel=0.8,
                 bagging_temperature=None,
                 subsample=0.66,
                 random_strength=1,
                 min_data_in_leaf=20,
                 boosting_type='Ordered',
                 bootstrap_type='Bayesian',
                 od_type='Iter',
                 od_wait=100,
                 fold_count=5,
                 seed=7,
                 task_type='GPU'):

        self.task_type = task_type

        self.is_regression = is_regression

        self.train_set_path = ''
        self.eval_set_path = ''
        self.test_set_path = ''

        self.model_path = ''

        self.target = target
        self.loss_function = loss_function
        self.cv_metric = cv_metric

        self.iterations = iterations
        self.best_iteration = best_iteration
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.bagging_temperature = bagging_temperature
        self.subsample = subsample
        self.random_strength = random_strength
        self.min_data_in_leaf = min_data_in_leaf

        if self.task_type == 'GPU':
            self.colsample_bylevel = None
        else:
            self.colsample_bylevel = colsample_bylevel

        self.boosting_type = boosting_type
        self.bootstrap_type = bootstrap_type
        self.od_type = od_type
        self.od_wait = od_wait

        if self.bootstrap_type != "Bayesian":
            self.bagging_temperature = None
        elif self.bootstrap_type != "Bernoulli":
            self.subsample = None

        self.fold_count = fold_count
        self.seed = seed

        self.features = []
        self.cat_features = []

    def params(self) -> dict:
        """"""
        params = {
            "loss_function": self.loss_function,
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            "colsample_bylevel": self.colsample_bylevel,
            "random_strength": self.random_strength,
            "min_data_in_leaf": self.min_data_in_leaf,
            "boosting_type": self.boosting_type,
            "bootstrap_type": self.bootstrap_type,
            "verbose": False,
            "allow_writing_files": False,
            "task_type": self.task_type
        }

        if self.bootstrap_type == "Bayesian":
            params["bagging_temperature"] = self.bagging_temperature
        elif self.bootstrap_type == "Bernoulli":
            params["subsample"] = self.subsample

        return params

    def regressor(self):
        """"""
        # bagging_temperature
        if self.bootstrap_type == "Bayesian":
            return CatBoostRegressor(loss_function=self.loss_function,
                                     iterations=self.best_iteration,
                                     learning_rate=self.learning_rate,
                                     depth=self.depth,
                                     l2_leaf_reg=self.l2_leaf_reg,
                                     colsample_bylevel=self.colsample_bylevel,
                                     bagging_temperature=self.bagging_temperature,
                                     random_strength=self.random_strength,
                                     min_data_in_leaf=self.min_data_in_leaf,
                                     boosting_type=self.boosting_type,
                                     bootstrap_type=self.bootstrap_type,
                                     verbose=False,
                                     allow_writing_files=False,
                                     task_type=self.task_type)

        # subsample
        elif self.bootstrap_type == "Bernoulli":
            return CatBoostRegressor(loss_function=self.loss_function,
                                     iterations=self.best_iteration,
                                     learning_rate=self.learning_rate,
                                     depth=self.depth,
                                     l2_leaf_reg=self.l2_leaf_reg,
                                     colsample_bylevel=self.colsample_bylevel,
                                     subsample=self.subsample,
                                     random_strength=self.random_strength,
                                     min_data_in_leaf=self.min_data_in_leaf,
                                     boosting_type=self.boosting_type,
                                     bootstrap_type=self.bootstrap_type,
                                     verbose=False,
                                     allow_writing_files=False,
                                     task_type=self.task_type)

        else:
            return CatBoostRegressor(loss_function=self.loss_function,
                                     iterations=self.best_iteration,
                                     learning_rate=self.learning_rate,
                                     depth=self.depth,
                                     l2_leaf_reg=self.l2_leaf_reg,
                                     colsample_bylevel=self.colsample_bylevel,
                                     random_strength=self.random_strength,
                                     min_data_in_leaf=self.min_data_in_leaf,
                                     boosting_type=self.boosting_type,
                                     bootstrap_type=self.bootstrap_type,
                                     verbose=False,
                                     allow_writing_files=False,
                                     task_type=self.task_type)

    def classifier(self):
        """"""
        # bagging_temperature
        if self.bootstrap_type == "Bayesian":
            return CatBoostClassifier(loss_function=self.loss_function,
                                      iterations=self.best_iteration,
                                      learning_rate=self.learning_rate,
                                      depth=self.depth,
                                      l2_leaf_reg=self.l2_leaf_reg,
                                      colsample_bylevel=self.colsample_bylevel,
                                      bagging_temperature=self.bagging_temperature,
                                      random_strength=self.random_strength,
                                      min_data_in_leaf=self.min_data_in_leaf,
                                      boosting_type=self.boosting_type,
                                      bootstrap_type=self.bootstrap_type,
                                      verbose=False,
                                      allow_writing_files=False,
                                      task_type=self.task_type)

        # subsample
        elif self.bootstrap_type == "Bernoulli":
            return CatBoostClassifier(loss_function=self.loss_function,
                                      iterations=self.best_iteration,
                                      learning_rate=self.learning_rate,
                                      depth=self.depth,
                                      l2_leaf_reg=self.l2_leaf_reg,
                                      colsample_bylevel=self.colsample_bylevel,
                                      subsample=self.subsample,
                                      random_strength=self.random_strength,
                                      min_data_in_leaf=self.min_data_in_leaf,
                                      boosting_type=self.boosting_type,
                                      bootstrap_type=self.bootstrap_type,
                                      verbose=False,
                                      allow_writing_files=False,
                                      task_type=self.task_type)

        else:
            return CatBoostClassifier(loss_function=self.loss_function,
                                      iterations=self.best_iteration,
                                      learning_rate=self.learning_rate,
                                      depth=self.depth,
                                      l2_leaf_reg=self.l2_leaf_reg,
                                      colsample_bylevel=self.colsample_bylevel,
                                      random_strength=self.random_strength,
                                      min_data_in_leaf=self.min_data_in_leaf,
                                      boosting_type=self.boosting_type,
                                      bootstrap_type=self.bootstrap_type,
                                      verbose=False,
                                      allow_writing_files=False,
                                      task_type=self.task_type)

    def optuna_cv_score(self, params: dict, cv_dataset: Pool) -> float:
        """"""
        params["task_type"] = self.task_type
        result = cv(cv_dataset,
                    params,
                    fold_count=self.fold_count,
                    shuffle=False,
                    seed=self.seed)
        return result[self.cv_metric].tolist()[-1]

    def cross_val_score(self):
        """"""
        train = pd.read_feather(self.train_set_path)
        x, y = train.loc[:, self.features], train[self.target]

        cv_dataset = Pool(data=x,
                          label=y,
                          cat_features=self.cat_features)

        params = self.params()
        result = cv(cv_dataset,
                    params,
                    fold_count=self.fold_count,
                    shuffle=True,
                    seed=self.seed)
        return result[self.cv_metric].tolist()[-1]

    def regressor_test_metrics(self):
        """"""
        from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

        train_set = pd.read_feather(self.train_set_path)
        eval_set = pd.read_feather(self.eval_set_path)
        test_set = pd.read_feather(self.test_set_path)

        model = CatBoostRegressor(
            loss_function=self.loss_function,
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            colsample_bylevel=self.colsample_bylevel,
            bagging_temperature=self.bagging_temperature,
            random_strength=self.random_strength,
            min_data_in_leaf=self.min_data_in_leaf,
            od_type=self.od_type,
            od_wait=self.od_wait,
            boosting_type=self.boosting_type,
            bootstrap_type=self.bootstrap_type,
            verbose=200,
            allow_writing_files=False,
        )

        x_train, y_train = train_set.loc[:, self.features], train_set[self.target]
        x_eval, y_eval = eval_set.loc[:, self.features], eval_set[self.target]
        x_test, y_test = test_set.loc[:, self.features], test_set[self.target]

        train_pool = Pool(x_train, y_train, cat_features=self.cat_features)
        eval_pool = Pool(x_eval, y_eval, cat_features=self.cat_features)

        del train_set, eval_set, test_set
        gc.collect()

        model.fit(train_pool, eval_set=eval_pool)

        preds = model.predict(x_test)

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
        print("\n=== TEST METRICS (CatBoost) ===")
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
            else:
                # For error metrics, lower is better
                improvement = "better" if delta < 0 else "worse"

            print(f"{m:<15} "
                  f"model = {metrics[m]:>8.5f} | "
                  f"baseline = {baseline_metrics[m]:>8.5f} | "
                  f"Δ = {sign}{delta:>8.5f} ({improvement})")

        return metrics

    def validation(self) -> pd.DataFrame:
        """Check model quality on the validation set."""
        validation = pd.read_feather(self.test_set_path)

        x = validation.loc[:, self.features]
        y = validation[self.target]

        if self.is_regression:
            model = CatBoostRegressor()
            model.load_model(self.model_path)
            y_hat = model.predict(x)
            print('MSE:', mean_squared_error(y, y_hat))
            print('MAE:', mean_absolute_error(y, y_hat))
        else:
            model = CatBoostClassifier()
            model.load_model(self.model_path)
            y_proba = model.predict_proba(x)
            y_pred = (y_proba[:, 1] > 0.5).astype(int)
            print('LogLoss:', log_loss(y, y_proba))
            print('Accuracy:', accuracy_score(y, y_pred))

        return validation

    def save_model(self):
        """"""

        train_set = pd.read_feather(self.train_set_path, columns=self.features + [self.target])
        eval_set = pd.read_feather(self.eval_set_path, columns=self.features + [self.target])

        x_train = train_set[self.features]
        y_train = train_set[self.target]

        x_eval = eval_set[self.features]
        y_eval = eval_set[self.target]

        del train_set, eval_set
        gc.collect()

        if self.is_regression:
            model = self.regressor()
        else:
            model = self.classifier()

        train_pool = Pool(x_train, y_train, cat_features=self.cat_features)
        eval_pool = Pool(x_eval, y_eval, cat_features=self.cat_features)

        model.fit(train_pool,
                  eval_set=eval_pool,
                  verbose=200)

        model.save_model(self.model_path)

    def make_predictions(self, data: pd.DataFrame, prediction_name: str = None) -> pd.DataFrame:
        """Load trained model and make predictions on new data."""
        if prediction_name is None:
            prediction_name = 'prediction'

        model = CatBoostRegressor() if self.is_regression else CatBoostClassifier()
        model.load_model(self.model_path)

        if self.is_regression:
            data[prediction_name] = model.predict(data[self.features])
        else:
            data[prediction_name] = model.predict_proba(data[self.features])[:, 1]

        return data
