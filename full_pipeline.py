"""Full rebuild after the raw data or the preprocessing changed:
matches -> features -> lgbm -> goals expectations -> glicko_matches -> league params -> ratings.

Every stage depends on the output of the previous one (match_id, goals expectations),
so after a change of the raw data run from the first stage. Unneeded stages are commented out in main."""
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

# leagues with a doubtful level between countries: refit with stronger time decay
# and higher weights of international matches (the level between countries is identified only by them)
CROOKED_LEAGUES = {f'{country}. {level} League'
                   for country in ['Algeria', 'Morocco', 'Egypt', 'Tunisia', 'Nigeria', 'Ghana', 'Cameroon', 'South Africa',
                                   'Iran', 'Japan', 'Saudi Arabia', 'South Korea', 'Turkey', 'Netherlands', 'Greece']
                   for level in ['First', 'Second']} | {'Benin. First League', 'Libya. First League',
                                                        'Sudan. First League', 'Dr Congo. First League'}
CROOKED_FIT_ITERATIONS = 10

# suspicious countries (26.10): the international matches since 2023 vs the model (Iran, Uzbekistan -; Saudi Arabia,
# Japan, Qatar, Croatia +), the second leagues too high in the table (Spain, England), the newcomers (Algeria, Turkey).
# Fitted first with several iterations, then the other leagues
SUSPICIOUS_LEAGUES = {f'{country}. {level} League'
                      for country in ['Iran', 'Uzbekistan', 'Saudi Arabia', 'Japan', 'Qatar', 'Croatia',
                                      'Spain', 'England', 'Algeria', 'Turkey']
                      for level in ['First', 'Second']}
SUSPICIOUS_FIT_ITERATIONS = 5

# 5b found nothing with the normal steps (26.10): init_mu and lift compensate each other,
# so the suspicious countries are refitted with the pair moved together and with smaller steps (1/3 of _league_moves).
# Second leagues have no ms_lift_update_mu (tied to the first league), for them only init_mu is moved
FINE_INIT_LIFT_MOVES = {'init_mu': {'init_mu': 10},
                        'ms_lift_update_mu': {'ms_lift_update_mu': 3},
                        'init_mu_lift_same': {'init_mu': 10, 'ms_lift_update_mu': 3},
                        'init_mu_lift_opposite': {'init_mu': 10, 'ms_lift_update_mu': -3}}

# the level of a country: init_mu of the first and the second league together (fit_country_shift, 27.10).
# Asia first (the international matches since 2023 vs the model), then South America
COUNTRY_SHIFT_COUNTRIES = {'Iran', 'Uzbekistan', 'Saudi Arabia', 'Japan', 'Qatar', 'United Arab Emirates', 'South Korea',
                           'Iraq', 'China', 'Brazil', 'Argentina'}

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
    """Snapshots of the ratings tables (Ratings.transform overwrites club_ratings.csv and league_ratings.csv)
    and the table of the leagues for the analyst: 'England 1', the ratings of the places 1-10."""

    timestamp = f'{datetime.now():%Y%m%d_%H%M}'
    snapshots = Path('data/soccer/tables')
    snapshots.mkdir(parents=True, exist_ok=True)

    shutil.copy('club_ratings.csv', snapshots / f'club_ratings_{timestamp}_{label}.csv')
    shutil.copy('league_ratings.csv', snapshots / f'league_ratings_{timestamp}_{label}.csv')

    leagues = pd.read_csv('league_ratings.csv')
    leagues['league'] = (leagues['league']
                         .str.replace('. First League', ' 1', regex=False)
                         .str.replace('. Second League', ' 2', regex=False))

    columns = ['#', 'league', 'league_score', 'top_8_weighted', 'top_7', 'n_teams', 'comparable'] + [f'team_{place}' for place in range(1, 11)]
    leagues[columns].rename(columns={'#': 'rank'}).to_csv(snapshots / f'league_ratings_analyst_{timestamp}_{label}.csv', index=False)

    print(f'tables saved to {snapshots}, label {label}')


def main_29_10():
    # stages 1-4 again (28.10): new features venue and tournament_type, Al- names in Flashscore
    # stages 1-4, 5a, 6a are done (26.10): the data, update_rd 14, ms_update_rd_first 16, ms_update_rd_second 34

    # night 28-29.10: stages 1-4 again, the final stages of the national teams with a margin of a day
    # (the finals of the World Cups 2010, 2018, Euro 2008, AFCON 2021 were cut), then the clubs and the national teams

    # 1. matches: neutral venues by the city, namesakes split by the team id
    # stage('1. preprocessing club and national teams')
    # MatchesPreprocessor().transform()

    # stage('2. features')
    # Features().transform()
    # Features().train_validation_test()

    # stage('3. lgbm training')
    # OutcomesLGBM().save_model()

    # regressor_test_metrics was slow (memory after the training), the metrics are recalculated later
    # stage('3. lgbm predictions')
    # model = OutcomesLGBM()
    # model.predict_chunks()

    # stage('4. glicko matches')
    # ClubPreprocessing().transform()

    # 5. league params, a backup before any change
    # params_path = Path(config.soccer['params']['league_params'])
    # backup = params_path.with_name(f'league_params_before_pipeline_{datetime.now():%Y%m%d_%H%M%S}.pkl')
    # shutil.copy(params_path, backup)
    # print(f'backup: {backup}')

    # # 5a. global params: speed of the ratings
    # stage('5a. global params fit')
    # TrainGlicko(train_test_split_season=ALL_SEASONS).fit_global_params()
    #
    # # 6a. tables after the global params: for the analyst in the morning, the league fit below takes ~20 hours
    # stage('6a. normalization and ratings tables after the global params')
    # TrainGlicko(train_test_split_season=ALL_SEASONS).normalize_params()
    # Ratings().transform()
    # export_tables('global_params')

    # 5b. per-league fit with the new global params: the suspicious countries first, their tables, then the others
    # stage('5b. per-league fit: suspicious countries')
    # TrainGlicko(train_test_split_season=ALL_SEASONS).fit_league_params(number_iterations=SUSPICIOUS_FIT_ITERATIONS,
    #                                                                     leagues_to_fit=SUSPICIOUS_LEAGUES)
    #
    # stage('6b. normalization and ratings tables after the suspicious countries')
    # TrainGlicko(train_test_split_season=ALL_SEASONS).normalize_params()
    # Ratings().transform()
    # export_tables('suspicious_fit')
    #
    # ~20 hours, after a check of the init_lift_fit tables
    # stage('5c. per-league fit: other leagues')
    # TrainGlicko(train_test_split_season=ALL_SEASONS).fit_league_params(number_iterations=LEAGUE_FIT_ITERATIONS,
    #                                                                     leagues_to_skip=SUSPICIOUS_LEAGUES)

    # 5d. init_mu and lift of the suspicious countries together, small steps: a plateau, no improvement (26.10)
    # stage('5d. per-league fit: suspicious countries, init_mu + lift with small steps')
    # TrainGlicko(train_test_split_season=ALL_SEASONS).fit_league_params(number_iterations=SUSPICIOUS_FIT_ITERATIONS,
    #                                                                     leagues_to_fit=SUSPICIOUS_LEAGUES,
    #                                                                     moves=FINE_INIT_LIFT_MOVES)

    # night 26.10: the shared shifts of all leagues (the second leagues are too high everywhere, lift), then the other leagues.
    # A warm start: the data is being reloaded, the loss will be recalculated anyway.
    # fit_league_params saves league_params.pkl after every improvement, so the fit can be stopped in the morning
    # for param_name, delta, groups in SHARED_SHIFTS:
    #     stage(f'5. shared shift: {param_name}')
    #     TrainGlicko(train_test_split_season=ALL_SEASONS).fit_shared_shift(param_name, delta=delta, groups=groups)
    #
    # stage('5c. per-league fit: all leagues')
    # TrainGlicko(train_test_split_season=ALL_SEASONS).fit_league_params(number_iterations=LEAGUE_FIT_ITERATIONS)
    #
    # # stage('5. crooked leagues fit')
    # # TrainGlicko(train_test_split_season=ALL_SEASONS).fit_league_params(number_iterations=CROOKED_FIT_ITERATIONS,leagues_to_fit=CROOKED_LEAGUES)
    #
    # # stage('5. stuck leagues fit')
    # # TrainGlicko(train_test_split_season=ALL_SEASONS).fit_stuck_leagues(leagues_to_fit=STUCK_LEAGUES)
    #
    # # stage('5. Iran fit')
    # # TrainGlicko(train_test_split_season=ALL_SEASONS).fit_stuck_leagues(leagues_to_fit=IRAN_LEAGUES)
    #
    # # 6c. normalization (mean rating of active teams = 1500) and the ratings tables
    # stage('6c. normalization and ratings tables after the league fit')
    # TrainGlicko(train_test_split_season=ALL_SEASONS).normalize_params()
    # Ratings().transform()
    # export_tables('night_fit')


    # ==================================================================================================================
    # 27.10: the fit from scratch on the new data (Flashscore re-scrape, QSL 2 duplicates removed, stages 1-4 done).
    # The per-league fit (5c) is skipped: on the old data it gave -1.4 of the loss in 8 hours, the shared shifts gave -33.5

    # 5. league params, a backup before any change
    # params_path = Path(config.soccer['params']['league_params'])
    # backup = params_path.with_name(f'league_params_before_pipeline_{datetime.now():%Y%m%d_%H%M%S}.pkl')
    # shutil.copy(params_path, backup)
    # print(f'backup: {backup}')

    # 5a. global params: speed of the ratings. Done 27.10: unchanged on the new data (14, 16, 34), the tables exported
    # stage('5a. global params fit')
    # TrainGlicko(train_test_split_season=ALL_SEASONS).fit_global_params()
    #
    # stage('6a. normalization and ratings tables after the global params')
    # TrainGlicko(train_test_split_season=ALL_SEASONS).normalize_params()
    # Ratings().transform()
    # export_tables('new_data_global_params')

    # 5e, 6e, 5f, 6f done 28.10: the shared shifts -1.7, the country shift of 11 countries -15.8
    # 5e. shared shifts of all leagues (all / first / second leagues)
    # for param_name, delta, groups in SHARED_SHIFTS:
    #     stage(f'5e. shared shift: {param_name}')
    #     TrainGlicko(train_test_split_season=ALL_SEASONS).fit_shared_shift(param_name, delta=delta, groups=groups)
    #
    # 6e. normalization (mean rating of active teams = 1500) and the ratings tables
    # stage('6e. normalization and ratings tables after the shared shifts')
    # TrainGlicko(train_test_split_season=ALL_SEASONS).normalize_params()
    # Ratings().transform()
    # export_tables('new_data_shared_shifts')
    #
    # 5f. the level of a country: init_mu of both leagues together, steps 50, 30, 10 (~3 min per country and step)
    # stage('5f. country shift: init_mu of the first and the second league together')
    # TrainGlicko(train_test_split_season=ALL_SEASONS).fit_country_shift(countries=COUNTRY_SHIFT_COUNTRIES)
    #
    # stage('6f. normalization and ratings tables after the country shift')
    # TrainGlicko(train_test_split_season=ALL_SEASONS).normalize_params()
    # Ratings().transform()
    # export_tables('new_data_country_shift')

    # # 5g. the level of all other countries, the countries with more international matches first.
    # # A country without international matches is skipped: the loss does not depend on its init_mu (28.10)
    # leagues = pd.read_csv('league_ratings.csv')
    # first_leagues = (leagues
    #                  .loc[leagues['league'].str.endswith('. First League') & (leagues['n_international_country'] > 0)]
    #                  .sort_values('n_international_country', ascending=False))
    # countries = [league.split('.')[0] for league in first_leagues['league']
    #              if league.split('.')[0] not in COUNTRY_SHIFT_COUNTRIES]
    # print(f'countries: {len(countries)}')

    # stage('5g. country shift: all countries')
    # TrainGlicko(train_test_split_season=ALL_SEASONS).fit_country_shift(countries=countries)

    # stage('6g. normalization and ratings tables after the country shift of all countries')
    # TrainGlicko(train_test_split_season=ALL_SEASONS).normalize_params()
    # Ratings().transform()
    # export_tables('new_data_all_country_shift')

    # 5h stopped (28.10): only Argentina moved (+20), the country levels converged after 5g
    # # 5h. a second pass over all countries, Asia included: Iran, Japan, Saudi Arabia still lean one way (28.10)
    # leagues = pd.read_csv('league_ratings.csv')
    # first_leagues = (leagues
    #                  .loc[leagues['league'].str.endswith('. First League') & (leagues['n_international_country'] > 0)]
    #                  .sort_values('n_international_country', ascending=False))
    # countries = [league.split('.')[0] for league in first_leagues['league']]
    # print(f'countries: {len(countries)}')

    # stage('5h. country shift: all countries, the second pass')
    # # one step of 20, repeated in the same direction while the loss decreases: ~3 hours instead of ~12 (28.10)
    # TrainGlicko(train_test_split_season=ALL_SEASONS).fit_country_shift(countries=countries, steps=(20,))

    # stage('6h. normalization and ratings tables after the second pass')
    # TrainGlicko(train_test_split_season=ALL_SEASONS).normalize_params()
    # Ratings().transform()
    # export_tables('new_features_country_shift_2')

    # stage('5a. global params fit (night 28-29.10)')
    # TrainGlicko(train_test_split_season=ALL_SEASONS).fit_global_params()

    # leagues = pd.read_csv('league_ratings.csv')
    # first_leagues = (leagues
    #                  .loc[leagues['league'].str.endswith('. First League') & (leagues['n_international_country'] > 0)]
    #                  .sort_values('n_international_country', ascending=False))
    # countries = [league.split('.')[0] for league in first_leagues['league']]
    # print(f'countries: {len(countries)}')

    # steps 50, 30, 10 as the first pass 5g: the data and the lgbm changed, a coarse step can move again (~11 hours)
    # stage('5i. country shift: all countries, steps 50, 30, 10 (night 28-29.10)')
    # TrainGlicko(train_test_split_season=ALL_SEASONS).fit_country_shift(countries=countries, steps=(50, 30, 10))

    # stage('6i. normalization and ratings tables')
    # TrainGlicko(train_test_split_season=ALL_SEASONS).normalize_params()
    # Ratings().transform()
    # export_tables('night_29_10')

    # the national teams use the goals expectations of the new lgbm: the production fit (national_pipeline.py),
    # run separately in the morning, ~10 minutes
    # stage('7. national teams: production')
    # import national_pipeline
    # national_pipeline.main()

    # morning 29.10: the night run moved nothing (global params 14 / 16 / 34, no country shift accepted).
    # All the league params of the clubs on the new data: the shared shifts, then the per-league fit
    # (~20 hours for all leagues: the params are saved after every improvement, can be stopped, then run 6k alone)
    params_path = Path(config.soccer['params']['league_params'])
    backup = params_path.with_name(f'league_params_before_pipeline_{datetime.now():%Y%m%d_%H%M%S}.pkl')
    shutil.copy(params_path, backup)
    print(f'backup: {backup}')

    for param_name, delta, groups in SHARED_SHIFTS:
        stage(f'5j. shared shift: {param_name}')
        TrainGlicko(train_test_split_season=ALL_SEASONS).fit_shared_shift(param_name, delta=delta, groups=groups)

    stage('6j. normalization and ratings tables after the shared shifts')
    TrainGlicko(train_test_split_season=ALL_SEASONS).normalize_params()
    Ratings().transform()
    export_tables('shared_shifts_29_10')

    stage('5k. per-league fit: all leagues')
    TrainGlicko(train_test_split_season=ALL_SEASONS).fit_league_params(number_iterations=LEAGUE_FIT_ITERATIONS)

    stage('6k. normalization and ratings tables after the per-league fit')
    TrainGlicko(train_test_split_season=ALL_SEASONS).normalize_params()
    Ratings().transform()
    export_tables('league_fit_29_10')


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
    export_tables('rebuild_30_10')


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
