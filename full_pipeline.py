"""The stages of the club pipeline: matches -> features -> lgbm -> goals expectations -> glicko_matches -> league params -> ratings.
    rebuild_data:   after a change of the raw data or the preprocessing, the ratings with the current params
    production_fit: the production params of the clubs (all seasons)
Every stage depends on the output of the previous one (match_id, goals expectations),
so after a change of the raw data run from the first stage. run.py: the same stages in one script."""
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import config
from soccer.boostings.features import Features
from soccer.boostings.outcomes_lgbm import OutcomesLGBM
from soccer.club_ratings.preprocessing import ClubPreprocessing
from soccer.club_ratings.ratings import Ratings
from soccer.club_ratings.train_glicko import TrainGlicko
from soccer.preprocessing import MatchesPreprocessor
from utils.functions import stage

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', None)

ALL_SEASONS = 10_000

# the per-league fit of all leagues takes ~20 hours per iteration
LEAGUE_FIT_ITERATIONS = 1

# (param, delta, groups) for TrainGlicko.fit_shared_shift
SHARED_SHIFTS = [('scoring_overall', 0.02, ('all', 'first', 'second')),
                 ('draw_frequency', 0.02, ('all', 'first', 'second')),
                 ('strength_overall', 0.008, ('all', 'first', 'second')),
                 ('scoring', 0.02, ('all', 'first', 'second')),
                 ('crowd', 0.02, ('all', 'first', 'second')),
                 ('logistic', 0.02, ('all', 'first', 'second')),
                 ('strength', 0.008, ('all', 'first', 'second')),
                 ('ms_lift_update_mu', 5, ('all',))]


def export_tables(label: str):
    """Snapshots of the ratings tables: Ratings.transform overwrites club_ratings.csv and league_ratings.csv."""

    timestamp = f'{datetime.now():%Y%m%d_%H%M}'
    snapshots = Path('data/soccer/tables')
    snapshots.mkdir(parents=True, exist_ok=True)

    shutil.copy('club_ratings.csv', snapshots / f'club_ratings_{timestamp}_{label}.csv')
    shutil.copy('league_ratings.csv', snapshots / f'league_ratings_{timestamp}_{label}.csv')

    print(f'tables saved to {snapshots}, label {label}')


def rebuild_data():
    """Stages 1-4 after a change of the raw data or the preprocessing, then the ratings with the current params.
    Then evaluation_pipeline.py and production_fit (can be run in parallel)."""

    stage('1. preprocessing club and national teams')
    MatchesPreprocessor().transform()

    stage('2. features')
    Features().transform()

    stage('3. lgbm training')
    OutcomesLGBM().save_model()

    stage('3. lgbm predictions')
    model = OutcomesLGBM()
    model.predict_chunks()

    stage('4. glicko matches')
    ClubPreprocessing().transform()

    Ratings().transform()
    export_tables('rebuild')


def production_fit():
    """The production params of the clubs (all seasons), ~20 hours.
    All stages of TrainGlicko.fit, the params are saved after every improvement: the fit can be stopped, then 6 alone."""

    params_path = Path(config.soccer['params']['league_params'])
    backup = params_path.with_name(f'league_params_before_pipeline_{datetime.now():%Y%m%d_%H%M%S}.pkl')
    shutil.copy(params_path, backup)
    print(f'backup: {backup}')

    stage('5. fit: global params, shared shifts, country shift, per-league, extra time')
    TrainGlicko(train_test_split_season=ALL_SEASONS).fit(SHARED_SHIFTS, LEAGUE_FIT_ITERATIONS)

    stage('6. normalization and ratings tables')
    TrainGlicko(train_test_split_season=ALL_SEASONS).normalize_params()
    Ratings().transform()
    export_tables('production_fit')


def main():
    # the rebuild only, the fits: production_fit, evaluation_pipeline.py, national_pipeline.py (see paper.py)
    rebuild_data()


if __name__ == '__main__':
    main()
