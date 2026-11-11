"""The national teams pipeline after the club pipeline (run.py): the goals expectations come from the boosting trained
on all matches (clubs + national), so after new lgbm predictions the national glicko matches and params are refitted.
    production: the global params, the country offsets with an L2 penalty, the global params again,
                the normalization and the ratings table (all matches)
    train_fit:  the same on the matches before 2021-01-01, for the evaluation of the paper (TRAIN_FIT_FOLDER)
The final stage params, the dead rubbers, the goal difference, the volatility, the weights of the match types
and the rating periods by the international windows did not reduce the test loss (the list of the paper)."""
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import config
from soccer.national_ratings.preprocessing import NationalPreprocessing
from soccer.national_ratings.ratings import NationalRatings
from soccer.national_ratings.train_glicko import TrainNationalGlicko
from utils.functions import load_pickle, save_pickle, stage

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', None)

# the L2 penalty of the country offsets, chosen by the test loss from 1, 5 and 20
COUNTRY_PENALTY = 5.

# the discount of the loss per year, as for the clubs (0.95 per quarter)
DISCOUNT_PER_YEAR = 0.81

# the production params are fitted on all matches, the train / test split is only for the evaluation
ALL_DATES = '2100-01-01'

# the params fitted on the train matches (before 2021-01-01): the evaluation and the ablation of the paper
TRAIN_FIT_FOLDER = Path(config.soccer['national_params']['global']).parent.with_name('national_train_fit')


def backup_params(label: str):
    """Copy of the folder of the national params before any change."""

    params_folder = Path(config.soccer['national_params']['global']).parent
    backup = params_folder.with_name(f'national_before_{label}_{datetime.now():%Y%m%d_%H%M%S}')
    shutil.copytree(params_folder, backup)
    print(f'backup: {backup}')


def export_table(label: str):
    """Snapshot of national_ratings.csv (NationalRatings.transform overwrites it)."""

    snapshots = Path('data/soccer/tables')
    snapshots.mkdir(parents=True, exist_ok=True)

    path = snapshots / f'national_ratings_{datetime.now():%Y%m%d_%H%M}_{label}.csv'
    shutil.copy('national_ratings.csv', path)

    print(f'table saved to {path}')


def main():
    stage('national glicko matches')
    NationalPreprocessing().transform()
    production()


def train_model() -> TrainNationalGlicko:
    """The train split (default 2021-01-01) with the settings of the production model."""
    train = TrainNationalGlicko()
    train.discount_per_year = DISCOUNT_PER_YEAR
    train.country_penalty = COUNTRY_PENALTY
    return train


def train_fit():
    """The params on the train matches (before 2021-01-01) for the evaluation of the paper, saved to TRAIN_FIT_FOLDER.
    Run after the national glicko matches are built, the production params are restored by production()."""

    backup_params('train_fit')
    save_pickle({'gamma': 1.}, config.soccer['national_params']['final_stage'])

    stage('train fit: global params, country offsets, global params')
    train_model().fit_global_params(number_iterations=100)
    train_model().fit_country_params(number_iterations=20)
    train_model().fit_global_params(number_iterations=100)

    stage('train fit: train and test loss')
    print(train_model().compare_losses())

    params_folder = Path(config.soccer['national_params']['global']).parent
    shutil.copytree(params_folder, TRAIN_FIT_FOLDER, dirs_exist_ok=True)
    print(f'train fit saved to {TRAIN_FIT_FOLDER}')


def production():
    """The production params (all matches) and the ratings table."""

    backup_params('production')

    # the host of a final stage as a usual home team (gamma = 1)
    save_pickle({'gamma': 1.}, config.soccer['national_params']['final_stage'])

    stage(f'1. production: global params on all matches, discount {DISCOUNT_PER_YEAR}')
    train = TrainNationalGlicko(train_test_split_date=ALL_DATES)
    train.discount_per_year = DISCOUNT_PER_YEAR
    train.fit_global_params(number_iterations=100)

    stage(f'2. production: country offsets on all matches, penalty {COUNTRY_PENALTY}')
    train = TrainNationalGlicko(train_test_split_date=ALL_DATES)
    train.discount_per_year = DISCOUNT_PER_YEAR
    train.country_penalty = COUNTRY_PENALTY
    train.fit_country_params(number_iterations=20)

    train = TrainNationalGlicko(train_test_split_date=ALL_DATES)
    train.discount_per_year = DISCOUNT_PER_YEAR
    train.country_penalty = COUNTRY_PENALTY
    train.fit_global_params(number_iterations=100)

    paths = config.soccer['national_params']
    country_params = load_pickle(paths['country'])
    print(f'confederations: {load_pickle(paths["confederation"])}')
    print('country offsets:', {team: country_params[team] for team in ('Morocco', 'Brazil', 'Germany', 'Japan', 'Senegal')
                               if team in country_params})

    stage('3. production: normalization and the ratings table')
    TrainNationalGlicko(train_test_split_date=ALL_DATES).normalize_params()
    NationalRatings().transform()
    export_table('production')


if __name__ == '__main__':
    main()
