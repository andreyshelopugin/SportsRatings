"""All tables of the paper from the fitted params (the fits: run.py, evaluation_pipeline.py, national_pipeline.py).
Order of a full rebuild:
    1. run.py (REFIT = True)                matches, features, lgbm, goals expectations, the production params and ratings
    2. evaluation_pipeline.main             train fit of the clubs (< 2021) and the sequential test predictions
    3. national_pipeline.train_fit          train fit of the national teams (the production params: run.py)
    4. soccer/matrix_models/compare_models  Skellam, Dixon-Coles, Markov
    5. paper.main                           the tables below
Results: data/soccer/tables/.
Run from the root of the project. ~10 hours (the ablations with a refit are the longest, they can be commented out)."""
import shutil
import warnings
from pathlib import Path

import pandas as pd

import diagnostics.ablation as ablation
import diagnostics.ablation_refit as ablation_refit
import diagnostics.elo_baseline as elo_baseline
import diagnostics.glicko_vs_baselines as glicko_vs_baselines
import diagnostics.league_calibration_test as league_calibration_test
import diagnostics.national_ablation as national_ablation
import diagnostics.national_extensions as national_extensions
import diagnostics.national_vs_baselines as national_vs_baselines
import diagnostics.params_interpretation as params_interpretation
import diagnostics.score_models_metrics as score_models_metrics
import diagnostics.transition_calibration as transition_calibration
from soccer.club_ratings.ratings import Ratings
from soccer.national_ratings.ratings import NationalRatings
from utils.functions import stage

warnings.filterwarnings('ignore')

TABLES = Path(__file__).resolve().parent / 'data/soccer/tables'


def ratings_tables():
    """The ratings tables of the appendix (production params): clubs, leagues, national teams."""

    stage('ratings: clubs and leagues')
    Ratings().transform()
    shutil.copy('club_ratings.csv', TABLES / 'club_ratings.csv')
    shutil.copy('league_ratings.csv', TABLES / 'league_ratings.csv')

    stage('ratings: national teams')
    NationalRatings().transform()
    shutil.copy('national_ratings.csv', TABLES / 'national_ratings.csv')


def main():
    stage('clubs: Brier score and RPS of the score matrix models')
    score_models_metrics.main()

    stage('clubs: Elo baseline')
    elo_baseline.main()

    stage('clubs: Glicko vs the baselines')
    glicko_vs_baselines.main()

    stage('clubs: ablation')
    ablation.main()

    stage('clubs: ablation with a refit')
    ablation_refit.main()

    league_calibration_test.main()

    stage('clubs: calibration of the season transitions')
    transition_calibration.main()

    stage('national teams: Glicko vs the baselines')
    national_vs_baselines.main()

    national_ablation.knockout()
    national_ablation.refit()

    stage('national teams: the extensions that did not help')
    national_extensions.main()

    params_interpretation.main()
    ratings_tables()


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 5000)
    main()
