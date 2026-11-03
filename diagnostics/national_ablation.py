"""The ablation of the national teams (the table of the paper), as diagnostics/ablation.py and ablation_refit.py for the clubs.
Start point: the train fit (national_pipeline.train_fit, matches before 2021-01-01), test loss of the matches since
2021-01-01 (90 minutes) for all matches, qualifiers + Nations League and the final stages.
    knockout: one component is switched off, the other params are kept (~5 min per variant)
    refit:    one component is switched off, the global params and the confederation levels are refitted
              on the train matches (~20-30 min per variant). Every variant works in its own copy of the params
              (data/soccer/params/national_ablation_refit/<variant>), the production params and the train fit are not changed.
Results: data/soccer/tables/national_ablation.csv, national_ablation_refit.csv.
Run from the root of the project."""
import pickle
import shutil
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from config import config
from national_pipeline import COUNTRY_PENALTY, DISCOUNT_PER_YEAR, TRAIN_FIT_FOLDER
from utils.functions import stage

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / 'data/soccer/tables'
REFIT_FOLDER = ROOT / 'data/soccer/params/national_ablation_refit'

HOME_ADVANTAGE_PARAMS = ('crowd', 'logistic', 'strength', 'scoring')
SCORING_PARAMS = ('scoring', 'scoring_overall')
GLOBAL_ITERATIONS = 100


@contextmanager
def national_params(folder: Path):
    """The national params from the folder instead of the production params (the config is read in __init__ of the model)."""

    production = dict(config.soccer['national_params'])
    config.soccer['national_params'] = {name: str(folder / Path(path).name) for name, path in production.items()}
    try:
        yield config.soccer['national_params']
    finally:
        config.soccer['national_params'] = production


def test_matches() -> pd.DataFrame:
    matches = pd.read_feather(config.soccer['national_glicko_matches'], columns=['match_id', 'is_home', 'outcome', 'match_type'])
    return matches.loc[matches['is_home'] == True].drop(columns='is_home')


def test_log_loss(change=None) -> pd.DataFrame:
    """Test loss (90 minutes, all weights 1) of the current national params, the change is applied to the params in memory."""
    from soccer.national_ratings.train_glicko import TrainNationalGlicko

    train = TrainNationalGlicko()
    train.match_type_loss_weights = {match_type: 1 for match_type in train.match_type_loss_weights}
    train._setup_train()

    state, country_params = train._initial_state()
    global_params, confederation_params, final_stage_params = train._unflatten(state)
    if change is not None:
        global_params, confederation_params, country_params = change(global_params, confederation_params, country_params)

    team_params = train.team_params(global_params, confederation_params, country_params)
    train.loss_filter = None
    train_ratings = train.rate_teams(team_params, train._match_params(team_params, final_stage_params=final_stage_params))

    train._setup_eval()
    team_params = train.team_params(global_params, confederation_params, country_params)
    matches_params = train._match_params(team_params, final_stage_params=final_stage_params)

    train.prediction_records = []
    train.calculate_test_loss(team_params, train_ratings, matches_params)

    predictions = (pd.DataFrame(train.prediction_records, columns=['match_id', 'p_home', 'p_draw', 'p_away'])
                   .merge(test_matches(), on='match_id', how='left', validate='1:1'))

    actual = np.stack([predictions['outcome'] == 1, predictions['outcome'] == 0.5, predictions['outcome'] == 0], axis=1).astype(float)
    probs = np.clip(predictions[['p_home', 'p_draw', 'p_away']].to_numpy(), 1e-15, 1 - 1e-15)
    log_loss = -(actual * np.log(probs) + (1 - actual) * np.log(1 - probs)).sum(axis=1) / 3

    is_final = predictions['match_type'].isin(['world_cup_final_stage', 'final_stage']).to_numpy()
    return pd.DataFrame([{'group': group, 'n_matches': mask.sum(), 'log_loss': round(log_loss[mask].mean(), 6)}
                         for group, mask in [('all', np.ones(len(predictions), dtype=bool)),
                                             ('qualifiers_nations_league', ~is_final), ('final_stage', is_final)]])


def with_increase(report: pd.DataFrame) -> pd.DataFrame:
    """The increase of the test loss vs the full model."""
    full = report.loc[report['variant'] == 'full'].set_index('group')['log_loss']
    report['increase'] = (report['log_loss'] - report['group'].map(full)).round(6)
    return report


# knockout: the changes of (global params, confederation params, country params)

def no_confederation_levels(global_params, confederation_params, country_params):
    """All confederations start from the same rating (the mean of init_mu), no confederation-aware initialization."""
    mean_init_mu = float(np.mean([p['init_mu'] for p in confederation_params.values()]))
    confederation_params = {name: {**p, 'init_mu': mean_init_mu} for name, p in confederation_params.items()}
    country_params = {name: {k: v for k, v in p.items() if k != 'init_mu'} for name, p in country_params.items()}
    return global_params, confederation_params, country_params


def no_home_advantage(global_params, confederation_params, country_params):
    """No home advantage at all (the host of a final stage included)."""
    global_params = {**global_params, **{name: 0. for name in HOME_ADVANTAGE_PARAMS}}
    country_params = {name: {k: v for k, v in p.items() if k not in HOME_ADVANTAGE_PARAMS} for name, p in country_params.items()}
    return global_params, confederation_params, country_params


def no_scoring_dependence(global_params, confederation_params, country_params):
    """The expected goals enter neither the home advantage nor the draw component."""
    return {**global_params, **{name: 0. for name in SCORING_PARAMS}}, confederation_params, country_params


def no_country_params(global_params, confederation_params, country_params):
    """The home advantage and the initial rating of the confederation for every country (no country offsets)."""
    return global_params, confederation_params, dict()


KNOCKOUT_VARIANTS = {'full': None,
                     'no_confederation_levels': no_confederation_levels,
                     'no_home_advantage': no_home_advantage,
                     'no_scoring_dependence': no_scoring_dependence,
                     'no_country_params': no_country_params}


def knockout():
    reports = []
    with national_params(TRAIN_FIT_FOLDER):
        for variant, change in KNOCKOUT_VARIANTS.items():
            stage(f'national ablation: {variant}')
            reports.append(test_log_loss(change).assign(variant=variant))
            print(reports[-1].to_string(index=False), flush=True)

    report = with_increase(pd.concat(reports))
    report.to_csv(TABLES / 'national_ablation.csv', index=False)
    print(report.pivot_table(index='variant', columns='group', values='increase').to_string())


# refit: the changes of (model, global params, confederation params), the switched off params are excluded from the fit

def refit_no_confederation_levels(train, global_params, confederation_params):
    """All confederations start from the mean level, the levels are not fitted."""
    mean_init_mu = float(np.mean([p['init_mu'] for p in confederation_params.values()]))
    train.fit_confederation_levels = False
    return global_params, {name: {**p, 'init_mu': mean_init_mu} for name, p in confederation_params.items()}


def refit_no_home_advantage(train, global_params, confederation_params):
    """No home advantage, the other global params are refitted."""
    train.global_fit_params = [name for name in train.global_fit_params if name not in HOME_ADVANTAGE_PARAMS]
    return {**global_params, **{name: 0. for name in HOME_ADVANTAGE_PARAMS}}, confederation_params


def refit_no_scoring_dependence(train, global_params, confederation_params):
    """The expected goals enter neither the home advantage nor the draw component."""
    train.global_fit_params = [name for name in train.global_fit_params if name not in SCORING_PARAMS]
    return {**global_params, **{name: 0. for name in SCORING_PARAMS}}, confederation_params


# full: the same refit without a change (a control: the refit from the fitted params must not change the loss)
REFIT_VARIANTS = {'full': None,
                  'no_confederation_levels': refit_no_confederation_levels,
                  'no_home_advantage': refit_no_home_advantage,
                  'no_scoring_dependence': refit_no_scoring_dependence}


def refit_variant(variant: str, change):
    """A copy of the train fit with the component switched off, then the refit of the global params."""
    from soccer.national_ratings.train_glicko import TrainNationalGlicko

    folder = REFIT_FOLDER / variant
    shutil.copytree(TRAIN_FIT_FOLDER, folder, dirs_exist_ok=True)

    with national_params(folder) as paths:
        fit_settings = dict()
        if change is not None:
            train = TrainNationalGlicko()
            # team_confederation is filled in _setup (_initial_state needs it)
            train._setup_train()
            state, _ = train._initial_state()
            global_params, confederation_params, _ = train._unflatten(state)
            global_params, confederation_params = change(train, global_params, confederation_params)

            with open(paths['global'], 'wb') as file:
                pickle.dump({**global_params, 'rd_growth': state['rd_growth']}, file)
            with open(paths['confederation'], 'wb') as file:
                pickle.dump(confederation_params, file)

            # the same restrictions of the fit for the new model
            fit_settings = {'global_fit_params': train.global_fit_params,
                            'fit_confederation_levels': train.fit_confederation_levels}

        train = TrainNationalGlicko()
        train.discount_per_year = DISCOUNT_PER_YEAR
        train.country_penalty = COUNTRY_PENALTY
        for name, value in fit_settings.items():
            setattr(train, name, value)
        train.fit_global_params(number_iterations=GLOBAL_ITERATIONS)

        return test_log_loss()


def refit():
    reports = []
    for variant, change in REFIT_VARIANTS.items():
        stage(f'national ablation with a refit: {variant}')
        reports.append(refit_variant(variant, change).assign(variant=variant))
        print(reports[-1].to_string(index=False), flush=True)

        # saved after every variant
        pd.concat(reports).to_csv(TABLES / 'national_ablation_refit.csv', index=False)

    report = with_increase(pd.concat(reports))
    report.to_csv(TABLES / 'national_ablation_refit.csv', index=False)
    print(report.pivot_table(index='variant', columns='group', values='increase').to_string())


if __name__ == '__main__':
    knockout()
    refit()
