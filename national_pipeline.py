"""National teams rebuild after the club pipeline (full_pipeline.py, 28.10):
the goals expectations come from the boosting trained on all matches (clubs + national),
so after new features or new lgbm predictions the national glicko matches and params are refitted.

Order (see TrainNationalGlicko):
    1. national glicko matches with the new goals expectations
    2. global params + final stage (host / beta), warm start from the saved params (penalty 5, 30.09)
    3. country offsets with L2 shrinkage (penalty 5), then the global params again with alpha
    4. compare_losses: the saved params vs no final stage advantage
    5. normalization and the ratings table

Unneeded stages are commented out in main."""
import itertools
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import config
from soccer.national_ratings.preprocessing import NationalPreprocessing
from soccer.national_ratings.ratings import NationalRatings
from soccer.national_ratings.train_glicko import TrainNationalGlicko, _backup, _restore
from utils.functions import load_pickle, save_pickle, stage

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', None)

# chosen on 30.09 from 1 / 5 / 20: the country offsets are small, the test loss is the best
COUNTRY_PENALTY = 5.

# weights of a dead rubber in the rating update (NationalPreprocessing._dead_rubbers), compared by the test loss (28.10)
DEAD_RUBBER_WEIGHTS = (1., 0.5, 0.25, 0.)

# weights of the goal difference surprise in the rating update (NationalPreprocessing._goal_difference_surprise), 28.10
MARGIN_WEIGHTS = (0., 0.1, 0.2, 0.3, 0.5)

# 19:30: the test loss falls up to 0.5 without a refit, the train loss grows: the other params are fitted for 0.
# Refit of the global params for every margin weight, from the same saved params
REFIT_MARGIN_WEIGHTS = (0., 0.5, 1., 2.)

# 19:33: after the refit the test loss falls up to 2 (0.517981 -> 0.517589), the train loss grows, the refit changes nothing.
# Control: the same mean weight for every match without the score (the mean surprise over the matches is 0.615)
MEAN_GOAL_DIFFERENCE_SURPRISE = 0.615

# 19:35: the same weight 1.3-1.6 for every match is better than the surprise (test and train): the ratings are too slow.
# Glicko-2 volatility instead of a weight: an adaptive speed of the teams with surprising results (28.10)
VOLATILITY_TAUS = (0.2, 0.5, 1., 1.5)

# 19:44: the volatility changes nothing (1-2 matches in a FIFA window, sigma does not move).
# Weights of the match types in the rating update (the K-factor by importance, as World Football Elo), 28.10
# 20:03: the clubs discount the loss by 0.95 per period, 4 periods per year: 0.95 ** 4 = 0.81 per year,
# the national teams by 0.98 per year. The params of the national teams (rd_growth) fit the whole history since 2000,
# the old glory of Brazil and Germany, Morocco too low. Discount per year as for the clubs, refit (28.10)
DISCOUNTS_PER_YEAR = (0.98, 0.9, 0.81)

# 20:07: the discount changes little (test 0.518516 -> 0.518294, rd_growth 7.75 -> 9.25), 0.81 as the clubs.
# The divergence from FIFA is the level of the confederations (inertia, as the leagues).
DISCOUNT_PER_YEAR = 0.81

# 20:10: rating periods as the clubs (4 per year) vs international windows and match days of the final stages
RATING_PERIOD_MODES = ('windows', 'quarters')

# 20:13: the quarters are not worse (test 0.518204 vs 0.518294 of the windows), the same ratings.
# 20:21: without the start levels of the confederations: all teams start from 1500 (28.10)
# 20:24: much worse (test 0.5218 vs 0.5183, New Zealand 8th): the start levels are the main information
# about the confederations, they are kept.

# 20:26: the production ratings: the params are fitted on all matches (as ALL_SEASONS of the clubs),
# the train / test split is only for the evaluation above
ALL_DATES = '2100-01-01'

NATIONS_LEAGUE_UPDATE_WEIGHTS = (0.5, 0.75, 1.)
FINAL_STAGE_UPDATE_WEIGHTS = (1., 1.5, 2.)
WORLD_CUP_UPDATE_WEIGHTS = (1., 2., 3.)

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
    # 28.10, 17:36: stages 1-5 are done (host = 2.53 is worse on the test than no final stage advantage).
    # Now: the matches again (is_dead_rubber), the host variants and the weights of dead rubbers, without a refit

    # 18:17: the matches with is_dead_rubber are built, the dead rubber weight changes nothing (1e-5), weight 1
    # 19:10: again, the Nations League finals are regular matches (not final stages)
    # 19:28: the matches with goal_difference_surprise are built
    # stage('1. national glicko matches (the goal difference surprise)')
    # NationalPreprocessing().transform()

    # stage('2. gamma = 0.5')
    # save_pickle({'gamma': 0.5}, config.soccer['national_params']['final_stage'])
    #
    # gamma_variants = {f'gamma_{gamma}': {'gamma': gamma} for gamma in (0., 0.5, 1.)}
    #
    # stage('4. losses: the saved params vs the gamma variants')
    # TrainNationalGlicko().compare_losses(gamma_variants)

    # # 19:28: without a refit, the test loss 0.517970 (0) -> 0.517843 (0.5)
    # for margin_weight in MARGIN_WEIGHTS:
    #     stage(f'4c. losses: margin weight {margin_weight}')
    #     train = TrainNationalGlicko()
    #     train.margin_weight = margin_weight
    #     train.compare_losses()

    # # every margin weight starts from the same saved params (gamma = 0.5 fixed), the saved params are restored at the end
    # paths = config.soccer['national_params']
    # baseline = _backup(paths, 'before_margin')

    # try:
    #     for margin_weight in REFIT_MARGIN_WEIGHTS:
    #         _restore(baseline)

    #         stage(f'4d. margin weight {margin_weight}: refit of the global params')
    #         train = TrainNationalGlicko()
    #         train.margin_weight = margin_weight
    #         train.fit_global_params(number_iterations=100)

    #         stage(f'4d. margin weight {margin_weight}: losses after the refit')
    #         train = TrainNationalGlicko()
    #         train.margin_weight = margin_weight
    #         train.compare_losses()

    # finally:
    #     _restore(baseline)
    #     print('\nsaved national params restored')

    # for margin_weight in (0.5, 1., 2.):
    #     stage(f'4e. margin weight {margin_weight}: the goal difference surprise')
    #     train = TrainNationalGlicko()
    #     train.margin_weight = margin_weight
    #     train.compare_losses()

    #     uniform_weight = round(1 + margin_weight * MEAN_GOAL_DIFFERENCE_SURPRISE, 4)
    #     stage(f'4e. control for {margin_weight}: the same weight {uniform_weight} for every match')
    #     train = TrainNationalGlicko()
    #     train.uniform_update_weight = uniform_weight
    #     train.compare_losses()

    # stage('4f. base: Glicko without volatility')
    # TrainNationalGlicko().compare_losses()

    # stage('4f. the same weight 1.3075 for every match (the best control of 19:35)')
    # train = TrainNationalGlicko()
    # train.uniform_update_weight = 1.3075
    # train.compare_losses()

    # for tau in VOLATILITY_TAUS:
    #     stage(f'4f. Glicko-2 volatility, tau {tau}')
    #     train = TrainNationalGlicko()
    #     train.use_volatility = True
    #     train.tau = tau
    #     train.compare_losses()

    # rows = []
    # for nations_league, final_stage, world_cup in itertools.product(NATIONS_LEAGUE_UPDATE_WEIGHTS, FINAL_STAGE_UPDATE_WEIGHTS,
    #                                                                 WORLD_CUP_UPDATE_WEIGHTS):
    #     weights = {'qualification': 1., 'nations_league': nations_league, 'final_stage': final_stage,
    #                'world_cup_final_stage': world_cup}
    #     stage(f'4g. update weights {weights}')

    #     train = TrainNationalGlicko()
    #     train.match_type_update_weights = weights
    #     # pivot report: index (split, subset, n_matches), a column per variant
    #     report = train.compare_losses()['saved'].reset_index()

    #     test = report.loc[report['split'] == 'test'].set_index('subset')['saved']
    #     train_loss = report.loc[(report['split'] == 'train') & (report['subset'] == 'all'), 'saved'].iloc[0]
    #     rows.append({'nations_league': nations_league, 'final_stage': final_stage, 'world_cup': world_cup,
    #                  'test_all': test['all'], 'test_regular': test['regular'], 'test_final_no_host': test['final_no_host'],
    #                  'test_final_host': test['final_host'], 'train_all': train_loss})

    # print('\n===== update weights by the match type, base (all 1): test 0.517970, train 0.520794 =====')
    # print(pd.DataFrame(rows).sort_values('test_all').round(6).to_string(index=False))

    # # 19:54: the national model as the club model (28.10). Every extension of today is within the noise
    # # (0.00004 - 0.001 on 3700 test matches): the host of a final stage gets the usual home advantage (gamma = 1),
    # # a neutral venue 0, all match weights 1, no margin, no volatility. The extra time model is kept.
    # stage('2. gamma = 1: the host of a final stage as a usual home team')
    # save_pickle({'gamma': 1.}, config.soccer['national_params']['final_stage'])

    # stage('5. normalization and the ratings table')
    # TrainNationalGlicko().normalize_params()
    # NationalRatings().transform()
    # export_table('as_clubs')

    # # every discount starts from the same saved params (gamma = 1), the saved params are restored at the end:
    # # the best one is saved by hand after the comparison
    # paths = config.soccer['national_params']
    # baseline = _backup(paths, 'before_discount')

    # try:
    #     for discount in DISCOUNTS_PER_YEAR:
    #         _restore(baseline)

    #         stage(f'4h. discount per year {discount}: refit of the global params')
    #         train = TrainNationalGlicko()
    #         train.discount_per_year = discount
    #         train.fit_global_params(number_iterations=100)
    #         print(f'fitted params: {load_pickle(paths["global"])}', flush=True)

    #         stage(f'4h. discount per year {discount}: losses after the refit')
    #         TrainNationalGlicko().compare_losses()

    #         stage(f'4h. discount per year {discount}: ratings')
    #         TrainNationalGlicko().normalize_params()
    #         NationalRatings().transform()
    #         export_table(f'discount_{discount}')

    # finally:
    #     _restore(baseline)
    #     print('\nsaved national params restored')

    # # every mode: the matches with its periods, a refit from the same saved params with the discount 0.81,
    # # the saved params are restored at the end, the matches are rebuilt with the windows (the default)
    # paths = config.soccer['national_params']
    # baseline = _backup(paths, 'before_periods')

    # try:
    #     for mode in RATING_PERIOD_MODES:
    #         _restore(baseline)

    #         stage(f'4i. rating periods {mode}: matches')
    #         preprocessing = NationalPreprocessing()
    #         preprocessing.rating_period_mode = mode
    #         preprocessing.transform()

    #         stage(f'4i. rating periods {mode}: refit of the global params, discount {DISCOUNT_PER_YEAR}')
    #         train = TrainNationalGlicko()
    #         train.discount_per_year = DISCOUNT_PER_YEAR
    #         train.fit_global_params(number_iterations=100)
    #         print(f'fitted params: {load_pickle(paths["global"])}', flush=True)

    #         stage(f'4i. rating periods {mode}: losses after the refit')
    #         TrainNationalGlicko().compare_losses()

    #         stage(f'4i. rating periods {mode}: ratings')
    #         TrainNationalGlicko().normalize_params()
    #         NationalRatings().transform()
    #         export_table(f'periods_{mode}')

    # finally:
    #     _restore(baseline)
    #     print('\nsaved national params restored')

    #     stage('national glicko matches with the windows (the default)')
    #     NationalPreprocessing().transform()

    # paths = config.soccer['national_params']
    # baseline = _backup(paths, 'before_no_init')

    # try:
    #     for fit_levels in (True, False):
    #         label = 'confederation_levels' if fit_levels else 'no_confederation_levels'
    #         _restore(baseline)

    #         if not fit_levels:
    #             # all confederations start from the global init_mu, the init_mu offsets of the countries are dropped
    #             save_pickle(dict(), paths['confederation'])
    #             country_params = load_pickle(paths['country'])
    #             save_pickle({team: {name: value for name, value in params.items() if name != 'init_mu'}
    #                          for team, params in country_params.items()}, paths['country'])

    #         stage(f'4j. {label}: refit of the global params, discount {DISCOUNT_PER_YEAR}')
    #         train = TrainNationalGlicko()
    #         train.discount_per_year = DISCOUNT_PER_YEAR
    #         train.fit_confederation_levels = fit_levels
    #         train.fit_global_params(number_iterations=100)
    #         print(f'fitted params: {load_pickle(paths["global"])}, confederations: {load_pickle(paths["confederation"])}', flush=True)

    #         stage(f'4j. {label}: losses after the refit')
    #         TrainNationalGlicko().compare_losses()

    #         stage(f'4j. {label}: ratings')
    #         TrainNationalGlicko().normalize_params()
    #         NationalRatings().transform()
    #         export_table(label)

    # finally:
    #     _restore(baseline)
    #     print('\nsaved national params restored')

    stage('1. national glicko matches')
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

    # the host of a final stage as a usual home team (gamma = 1, 19:54)
    save_pickle({'gamma': 1.}, config.soccer['national_params']['final_stage'])

    stage(f'2. production: global params on all matches, discount {DISCOUNT_PER_YEAR}')
    train = TrainNationalGlicko(train_test_split_date=ALL_DATES)
    train.discount_per_year = DISCOUNT_PER_YEAR
    train.fit_global_params(number_iterations=100)

    stage(f'3. production: country offsets on all matches, penalty {COUNTRY_PENALTY}')
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

    stage('5. production: normalization and the ratings table')
    TrainNationalGlicko(train_test_split_date=ALL_DATES).normalize_params()
    NationalRatings().transform()
    export_table('production')



if __name__ == '__main__':
    main()
