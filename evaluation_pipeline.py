"""The evaluation of the club ratings for the paper: the params are fitted on the train seasons only
(< 2021, evaluation_league_params.pkl), the test loss on the seasons >= 2021.
The production params (league_params.pkl, all seasons) are not changed.
Run after the production fit: the start point is a copy of league_params.pkl.

Stages (unneeded ones are commented out in main):
    1. train fit: global params, shared shifts, per-league fit (~20 hours, the params are saved after every improvement,
       the fit can be stopped, then the next stages alone)
    2. train and test loss: train fit vs production
    3. sequential test predictions of the train fit (Brier, RPS, the paired bootstrap vs the baselines, the calibration)
    4. params: train fit vs production"""
import warnings

import pandas as pd

from soccer.club_ratings.evaluate_glicko import EvaluateGlicko
from soccer.club_ratings.train_glicko import LEAGUE_FIT_ITERATIONS, SHARED_SHIFTS
from utils.functions import stage

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)


def main():
    stage('1. train fit: global params, shared shifts, per-league fit')
    EvaluateGlicko().fit(shared_shifts=SHARED_SHIFTS, league_fit_iterations=LEAGUE_FIT_ITERATIONS)

    stage('2. train and test loss: train fit vs production')
    EvaluateGlicko().evaluate()

    stage('3. test predictions of the train fit')
    EvaluateGlicko().save_test_predictions()

    stage('4. params: train fit vs production')
    EvaluateGlicko().compare_params()


if __name__ == '__main__':
    main()
