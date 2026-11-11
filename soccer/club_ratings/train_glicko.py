import gc
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import config
from soccer.club_ratings.ratings import Ratings
from utils.functions import save_pickle, load_pickle

# train_test_split_season of the production params: all seasons are train seasons
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


class TrainGlicko(Ratings):
    """Fit parameters for the Glicko model."""

    def __init__(self, train_test_split_season=2021):
        super().__init__()

        self.loss_epsilon = 0.01
        self.train_test_split_season = train_test_split_season

        # the file of the fitted params: league_params - production (all seasons),
        # EvaluateGlicko overrides it with evaluation_league_params (train seasons only)
        self.league_params_name = 'league_params'
        self.extra_time_params_name = 'extra_time'

        # first two years
        self.periods_without_loss_calculation = 8

        # the goal is the current strength: 4 rating periods per season, the half-life ln(0.5) / ln(0.95) = 13.5 periods
        # = ~3.4 seasons, the last 3 seasons give ~46% of the weight of the loss
        self.discount_factor = 0.95

        # the level between countries is identified only by international matches
        self.world_cup_loss_weight = 10
        self.international_cup_loss_weight = 4

        # candidates are rounded to avoid float drift (0.9960000000000006)
        self.round_digits = 6

        # integer league params: kept integer in fit_league_params
        self.int_params = {'init_mu', 'ms_lift_update_mu'}

        # adaptive step for shared params: doubled after an improvement (up to max_step_factor * delta),
        # halved otherwise (down to delta / min_step_divisor)
        self.max_step_factor = 50
        self.min_step_divisor = 8

        # ms_lift_update_mu is fitted only for first leagues (>= 0),
        # second leagues use -lift of the first league of the country (see Ratings._tie_lift_params)
        self.first_league_constraints = {
            'ms_lift_update_mu': lambda x: x >= 0,
            'crowd': lambda x: x >= 0,
            'logistic': lambda x: x >= 0,
        }

        self.second_league_constraints = {
            'crowd': lambda x: x >= 0,
            'logistic': lambda x: x >= 0,
        }

        self.low_international_history_first_league_constraints = {
            **self.first_league_constraints,
            'init_mu': lambda x: x <= self.low_international_history_init_mu,
        }

        self.low_international_history_second_league_constraints = {
            **self.second_league_constraints,
            'init_mu': lambda x: x <= self.low_international_history_init_mu,
        }

        # strength_overall in extra time is fixed to 1: _d1_d2 is exact only for strength_overall = 1,
        # and in extra time it is collinear with draw_frequency (avg_mu of extra time matches is almost constant)
        self.extra_time_fixed_params = {'strength_overall': 1.}

        self.extra_time_constraints = {
            'kappa': lambda x: x > 0,
            'home_advantage_scale': lambda x: x >= 0,
        }

    def _candidates(self, value: float, step: float) -> list[float]:
        """Rounded candidates: value - step, value + step."""

        return [round(value - step, self.round_digits), round(value + step, self.round_digits)]

    def _next_step(self, step: float, delta: float, improved: bool) -> float:
        """Adaptive step: doubled after an improvement, halved otherwise."""

        if improved:
            return round(min(step * 2, self.max_step_factor * delta), self.round_digits)

        return round(max(step / 2, delta / self.min_step_divisor), self.round_digits)

    def _league_constraints(self, league: str) -> dict:
        """Constraints of a league: first / second league, low international history."""

        if league in self.first_leagues and league in self.leagues_with_low_international_history:
            return self.low_international_history_first_league_constraints
        if league in self.second_leagues and league in self.leagues_with_low_international_history:
            return self.low_international_history_second_league_constraints
        if league in self.first_leagues:
            return self.first_league_constraints
        return self.second_league_constraints

    def _check_constraints(self, league_params: dict):
        """A param outside its constraints is frozen: both candidates value -/+ step are rejected.
        So the fit must start from valid params."""

        violations = [(league, name, params[name])
                      for league, params in league_params.items()
                      for name, check in self._league_constraints(league).items()
                      if name in params and not check(params[name])]

        assert not violations, f'params violate constraints: {violations[:10]} (total {len(violations)}), ' \
                               f'call clip_params_to_constraints'

    def clip_params_to_constraints(self):
        """Move saved league params to the constraints boundary (e.g. after a change of constraints
        or of the leagues with low international history). Uses the seasons of _setup_train."""

        self._setup_train()

        league_params_path = Path(config.soccer['params'][self.league_params_name])
        league_params = self.load_params(league_params_path)

        n_changed = 0
        for league, params in league_params.items():
            low_history = league in self.leagues_with_low_international_history

            bounds = {
                'ms_lift_update_mu': (0, None),
                'crowd': (0, None),
                'logistic': (0, None),
                'init_mu': (None, self.low_international_history_init_mu) if low_history else (None, None),
            }

            for name, (low, high) in bounds.items():
                if name not in params:
                    continue
                value = params[name]
                if low is not None:
                    value = max(value, low)
                if high is not None:
                    value = min(value, high)
                if value != params[name]:
                    print(f'{league}, {name}: {params[name]} -> {value}')
                    params[name] = value
                    n_changed += 1

        self._check_constraints(league_params)
        self.save_params(league_params, league_params_path)
        print(f'clipped params: {n_changed}')

    def _setup_train(self):
        """"""
        matches = pd.read_feather(config.soccer['glicko_matches'])

        matches = matches[matches['season'] < self.train_test_split_season]

        # after the split some teams may have no championship matches
        matches = self._drop_teams_without_championship(matches)

        self._precalculate_info(matches)

        del matches
        gc.collect()

    def _setup_eval(self):
        """"""
        matches = pd.read_feather(config.soccer['glicko_matches'])

        matches = matches[matches['season'] >= self.train_test_split_season]

        # after the split some teams may have no championship matches
        matches = self._drop_teams_without_championship(matches)

        self._precalculate_info(matches)

        del matches
        gc.collect()

    def calculate_loss(self, league_params: dict, matches_params: dict) -> float:
        """Calculate total log loss"""

        ratings = self._rating_initialization(league_params)

        total_loss = 0
        for rating_period_id, team_matches in self.period_team_matches.items():

            # update RD
            self._inflate_rd(ratings, rating_period_id)

            if rating_period_id > self.periods_without_loss_calculation:
                log_loss = self._loss_inside_period(ratings, self.period_matches[rating_period_id], matches_params)

                total_loss += self.discount_factor ** (self.rating_periods - rating_period_id) * log_loss

            ratings = self._rate_teams_inside_period(ratings, team_matches, matches_params)

        return total_loss

    def calculate_test_loss(self, league_params, ratings: dict, matches_params: dict) -> float:
        """Calculate total log loss"""

        # for teams that appeared after train
        new_ratings = self._rating_initialization(league_params)

        for team in new_ratings:
            if team not in ratings:
                ratings[team] = new_ratings[team]

        total_loss = 0
        n_matches = 0
        for rating_period_id, team_matches in self.period_team_matches.items():
            # update RD
            self._inflate_rd(ratings, rating_period_id)

            match_ids = self.period_matches[rating_period_id]
            total_loss += self._loss_inside_period(ratings, match_ids, matches_params)
            n_matches += len(match_ids)

            ratings = self._rate_teams_inside_period(ratings, team_matches, matches_params)

        return total_loss / n_matches

    def compare_losses(self):
        """train_loss / test_loss include the extra time term for extra time matches (nested model).
        train_loss_90 / test_loss_90 are 90 minutes only: comparable with the losses before the five-outcome model
        and with the boostings model. The extra time term is accumulated without discount,
        so the split is valid only with discount_factor = 1 (set below)."""

        self.world_cup_loss_weight = 1
        self.international_cup_loss_weight = 1
        self.discount_factor = 1
        self.periods_without_loss_calculation = 0
        rows = []
        for params_name in ['original_glicko', 'init', 'init_first_second_leagues', 'league_params']:
            # league_params: the model with its global params; the baselines (original Glicko, init params)
            # are league params only, with the default global params of the model (Ratings.__init__)
            if params_name == 'league_params':
                league_params = self.load_params(config.soccer['params'][params_name])
            else:
                self.set_global_params(self.default_global_params)
                league_params = load_pickle(config.soccer['params'][params_name])

            # train loss
            self._setup_train()
            matches_params = self._match_params(league_params)

            self.extra_time_loss_sum = 0.
            train_loss = self.calculate_loss(league_params, matches_params)
            train_extra_time_loss = self.extra_time_loss_sum
            n_train_matches = len(self.home_away_match_info)

            ratings = self.rate_teams(league_params, matches_params)

            # test loss, leagues that appeared only in test get group params
            self._setup_eval()
            league_params = self._fill_missing_leagues(league_params)
            matches_params = self._match_params(league_params)

            self.extra_time_loss_sum = 0.
            test_loss = self.calculate_test_loss(league_params, ratings, matches_params)
            n_test_matches = len(self.home_away_match_info)
            test_extra_time_loss = self.extra_time_loss_sum / n_test_matches

            rows.append({
                'params': params_name,
                'train_loss': round(train_loss / n_train_matches, 6),
                'test_loss': round(test_loss, 6),
                'train_loss_90': round((train_loss - train_extra_time_loss) / n_train_matches, 6),
                'test_loss_90': round(test_loss - test_extra_time_loss, 6),
            })

        print(pd.DataFrame(rows))

    def normalize_params(self):
        """During fitting, init_mu params could be very high or very low.
        This is not a problem as the Glicko rating system is indifferent to the addition of a constant.
        However, it makes the calculated ratings harder to interpret: the average rating of the active teams
        (the ratings table) should be 1500. A shift of all init_mu shifts all ratings by the same constant
        (newcomer anchors are relative), so the loop converges in one or two iterations."""

        league_params_path = Path(config.soccer['params'][self.league_params_name])
        if not league_params_path.exists():
            raise ValueError('call fit_league_params firstly')
        league_params = self.load_params(league_params_path)

        matches = pd.read_feather(config.soccer['glicko_matches'])
        self._precalculate_info(matches)

        # the same teams as in the ratings table (ratings_to_df): active teams of the latest seasons, without Russia
        active_teams = {team for team, league in self._active_team_leagues(matches).items()
                        if self._country(league) != 'Russia'}
        print(f'active teams: {len(active_teams)}')

        while True:
            # leagues that appeared after the train seasons are only used for the average, not saved
            filled_league_params = self._fill_missing_leagues(league_params)
            matches_params = self._match_params(filled_league_params)
            ratings = self.rate_teams(filled_league_params, matches_params)

            avg_mu = np.mean([ratings[team].mu for team in active_teams if team in ratings])
            delta = avg_mu - self.init_mu
            print(f'avg_mu: {avg_mu:.4f}, delta: {delta:.4f}')

            if abs(delta) < 1:
                break

            league_params = {league: {**params, 'init_mu': params['init_mu'] - delta} for league, params in league_params.items()}

        int_params = ['init_mu', 'ms_lift_update_mu']
        float_params = ['crowd', 'logistic', 'strength', 'scoring',
                        'draw_frequency', 'strength_overall', 'scoring_overall']

        for league in league_params:
            for param in int_params:
                if param in league_params[league]:
                    league_params[league][param] = int(round(league_params[league][param]))
            for param in float_params:
                league_params[league][param] = round(league_params[league][param], 6)

        self.save_params(league_params, league_params_path)

    def fit_default_params(self, number_iterations: int):
        """Fit params from original Glicko paper. We only split home field advantage into crowd and logistic.
        All leagues share the same params,
        custom params (ms_update_rd, ms_lift_update_mu, ms_new_team_update_mu, crowd, scoring, scoring_overall) are zero.

        We call it function only for honest comparison of the models."""

        self._setup_train()

        # no RD growth at the season transitions in the original Glicko
        self.ms_update_rd_first = 0
        self.ms_update_rd_second = 0

        league_params_path = Path(config.soccer['params']['original_glicko'])
        if league_params_path.exists():
            league_params = load_pickle(league_params_path)
        else:
            league_params = self._original_glicko_league_params_initialization()

        matches_params = self._match_params(league_params)

        current_loss = self.calculate_loss(league_params, matches_params)

        # only tune params that exist in original Glicko
        # we only split home field advantage into crowd and logistic
        deltas = {
            'crowd': 0.001,
            'logistic': 0.001,
            'strength': 0.001,

            'draw_frequency': 0.001,
            'strength_overall': 0.001,
        }

        constraints = {
            'crowd': lambda x: x >= 0,
            'logistic': lambda x: x >= 0,
        }

        steps = deltas.copy()

        for i in range(number_iterations):
            any_improved = False

            for param_name, delta in deltas.items():
                # all leagues share the same value, so update all at once
                current_value = league_params[next(iter(league_params))][param_name]

                step = steps[param_name]

                improved = False
                for candidate in self._candidates(current_value, step):
                    if param_name in constraints and not constraints[param_name](candidate):
                        continue

                    league_params_copy = {league: {**params, param_name: candidate} for league, params in league_params.items()}

                    matches_params = self._match_params(league_params_copy)

                    loss = self.calculate_loss(league_params_copy, matches_params)

                    if loss < current_loss - self.loss_epsilon:
                        print(f"Loss down by: {current_loss - loss:.2f}, {param_name}={candidate}, step={step}, Iteration: {i}")
                        print()

                        league_params = league_params_copy
                        current_loss = loss

                        save_pickle(league_params, config.soccer['params']['original_glicko'])

                        improved = True
                        break

                steps[param_name] = self._next_step(step, delta, improved)
                any_improved |= improved

            if not any_improved and all(steps[p] <= deltas[p] / self.min_step_divisor for p in deltas):
                print(f'Converged, iteration: {i}')
                break

    def fit_init_league_params(self, number_iterations: int):
        """Fit the same params for all leagues."""

        self._setup_train()

        league_params_path = Path(config.soccer['params']['init'])
        if league_params_path.exists():
            league_params = load_pickle(league_params_path)
        else:
            league_params = self._league_params_initialization()

        matches_params = self._match_params(league_params)

        current_loss = self.calculate_loss(league_params, matches_params)

        # ms_lift_update_mu is not fitted here: it is fitted by groups in fit_first_second_league_params
        # (only for first leagues, second leagues use -lift of the first league)
        deltas = {
            'crowd': 0.01,
            'logistic': 0.01,
            'strength': 0.01,
            'scoring': 0.01,

            'draw_frequency': 0.01,
            'strength_overall': 0.01,
            'scoring_overall': 0.01
        }

        # check constraints using first league as representative
        sample_league = next(iter(league_params))
        league_constraints = self.first_league_constraints if sample_league in self.first_leagues else self.second_league_constraints

        steps = deltas.copy()

        for i in range(number_iterations):
            any_improved = False

            for param_name, delta in deltas.items():
                # all leagues share the same value, so update all at once
                current_value = league_params[sample_league][param_name]

                step = steps[param_name]

                improved = False
                for candidate in self._candidates(current_value, step):
                    if param_name in league_constraints and not league_constraints[param_name](candidate):
                        continue

                    league_params_copy = {league: {**params, param_name: candidate} for league, params in league_params.items()}

                    matches_params = self._match_params(league_params_copy)
                    loss = self.calculate_loss(league_params_copy, matches_params)

                    if loss < current_loss - self.loss_epsilon:
                        print(f"Loss down by: {current_loss - loss:.2f}, {param_name}={candidate}, step={step}, Iteration: {i}")
                        print()

                        league_params = league_params_copy
                        current_loss = loss

                        save_pickle(league_params, config.soccer['params']['init'])

                        improved = True
                        break

                steps[param_name] = self._next_step(step, delta, improved)
                any_improved |= improved

            if not any_improved and all(steps[p] <= deltas[p] / self.min_step_divisor for p in deltas):
                print(f'Converged, iteration: {i}')
                break

    def fit_first_second_league_params(self, number_iterations: int):
        """Fit the same params for first and second leagues separately.
        ms_lift_update_mu is fitted only for first leagues (second leagues use -lift of the first league)."""

        self._setup_train()

        if not Path(config.soccer['params']['init']).exists():
            raise FileNotFoundError("call fit_init_league_params first")

        league_params_path = Path(config.soccer['params']['init_first_second_leagues'])
        if league_params_path.exists():
            league_params = load_pickle(league_params_path)
        else:
            league_params = load_pickle(config.soccer['params']['init'])

        self._check_constraints(league_params)

        matches_params = self._match_params(league_params)

        current_loss = self.calculate_loss(league_params, matches_params)

        deltas = {
            'init_mu': 1,
            'ms_lift_update_mu': 1,

            'crowd': 0.001,
            'logistic': 0.001,
            'strength': 0.001,
            'scoring': 0.001,

            'draw_frequency': 0.001,
            'strength_overall': 0.001,
            'scoring_overall': 0.001
        }

        league_groups = [
            (self.first_leagues - self.leagues_with_low_international_history,
             self.first_league_constraints),

            (self.second_leagues - self.leagues_with_low_international_history,
             self.second_league_constraints),

            (self.first_leagues & self.leagues_with_low_international_history,
             self.low_international_history_first_league_constraints),

            # empty
            (self.second_leagues & self.leagues_with_low_international_history,
             self.low_international_history_second_league_constraints),
        ]

        # separate step for each (group, param)
        steps = {(group_id, param_name): delta for group_id in range(len(league_groups)) for param_name, delta in deltas.items()}

        for i in range(number_iterations):
            any_improved = False

            for param_name, delta in deltas.items():
                for group_id, (leagues, constraints) in enumerate(league_groups):
                    # second leagues have no ms_lift_update_mu, so the subset is empty for them
                    subset = {l for l, p in league_params.items() if l in leagues and param_name in p}

                    if not subset:
                        continue

                    current_value = league_params[next(iter(subset))][param_name]

                    step = steps[(group_id, param_name)]

                    improved = False
                    for candidate in self._candidates(current_value, step):
                        if param_name in constraints and not constraints[param_name](candidate):
                            continue

                        league_params_copy = {league: ({**params, param_name: candidate} if league in subset else params)
                                              for league, params in league_params.items()}

                        matches_params = self._match_params(league_params_copy)
                        loss = self.calculate_loss(league_params_copy, matches_params)

                        if loss < current_loss - self.loss_epsilon:
                            print(f"Loss down by: {current_loss - loss:.2f}, {param_name}={candidate}, "
                                  f"group={group_id}, step={step}, Iteration: {i}")
                            print()

                            league_params = league_params_copy
                            current_loss = loss

                            save_pickle(league_params, config.soccer['params']['init_first_second_leagues'])

                            improved = True
                            break

                    steps[(group_id, param_name)] = self._next_step(step, delta, improved)
                    any_improved |= improved

            if not any_improved and all(step <= deltas[param_name] / self.min_step_divisor
                                        for (_, param_name), step in steps.items()):
                print(f'Converged, iteration: {i}')
                break

    @staticmethod
    def _league_moves() -> dict:
        """Moves of fit_league_params: {move name: {param: delta}}, all params of a move change at once.
        crowd and logistic enter the home advantage as a sum (they differ only in matches without spectators
        or at a different stadium), so they are moved as a sum (home_advantage) and as a share (crowd_share)."""

        return {
            'init_mu': {'init_mu': 30},

            'ms_lift_update_mu': {'ms_lift_update_mu': 10},

            'home_advantage': {'crowd': 0.02, 'logistic': 0.02},
            'crowd_share': {'crowd': 0.02, 'logistic': -0.02},
            'strength': {'strength': 0.008},
            'scoring': {'scoring': 0.02},

            'draw_frequency': {'draw_frequency': 0.02},
            'strength_overall': {'strength_overall': 0.008},
            'scoring_overall': {'scoring_overall': 0.02},
        }

    def _moved_params(self, params: dict, move: dict, direction: int) -> dict:
        """New params of a league after a move in the direction (+1 / -1), integer params stay integer."""

        new_params = dict(params)
        for name, delta in move.items():
            value = params[name] + direction * delta
            new_params[name] = int(round(value)) if name in self.int_params else round(value, self.round_digits)

        return new_params

    def fit_league_params(self, number_iterations: int, top_popular_leagues=1000, loss_epsilon=1.,
                          log_every_seconds=3 * 60, skip_iterations=5, leagues_to_fit: set = None,
                          leagues_to_skip: set = None, moves: dict = None):
        """Fit params independently for each league (fine tuning after the shared shifts).

        Moves (see _league_moves) change one or several params at once. After an improvement the move is repeated
        in the same direction while the loss decreases, then the next move is tried.

        loss_epsilon is higher than self.loss_epsilon: loss is almost flat in params of a single league,
        so tiny improvements are noise and lead to drift of params.

        ms_lift_update_mu is fitted only for first leagues, it is shared with the second leagues of the country
        (_match_params rebuilds their matches too).

        The loss is deterministic: a league is tried again only if some params changed after its last try
        (params_version), a league without improvement is skipped for skip_iterations iterations.
        The fit stops when no league has changes since its last try (converged).

        leagues_to_fit: if passed, only these leagues are fitted (top_popular_leagues is ignored).
        leagues_to_skip: if passed, these leagues are not fitted (e.g. fitted just before).
        Leagues without params get group params (_fill_missing_leagues) and are fitted too.
        """

        self._setup_train()

        league_params_path = Path(config.soccer['params'][self.league_params_name])
        if league_params_path.exists():
            league_params = self.load_params(league_params_path)
        else:
            league_params = load_pickle(config.soccer['params']['init_first_second_leagues'])

        league_params = self._fill_missing_leagues(league_params)

        # integer params may be fractional after the shared shifts (fit_shared_shift halves the step)
        league_params = {league: {name: (int(round(value)) if name in self.int_params else value) for name, value in params.items()}
                         for league, params in league_params.items()}

        self._check_constraints(league_params)

        matches_params = self._match_params(league_params)

        current_loss = self.calculate_loss(league_params, matches_params)
        print(f'Start loss: {current_loss:.2f}', flush=True)

        # custom moves (e.g. larger or smaller steps for a refit of some leagues)
        moves = self._league_moves() if moves is None else moves

        if leagues_to_fit is not None:
            unknown = set(leagues_to_fit) - set(league_params)
            if unknown:
                print(f'leagues without params, skipped: {sorted(unknown)}')

            leagues = [league for league in self.leagues_sorted_by_matches
                       if league in leagues_to_fit
                       and (league in self.first_leagues or league in self.second_leagues)
                       and league in league_params]
        else:
            leagues = [league for league in self.leagues_sorted_by_matches[:top_popular_leagues]
                       if (league in self.first_leagues or league in self.second_leagues) and league in league_params]

        if leagues_to_skip is not None:
            leagues = [league for league in leagues if league not in leagues_to_skip]

        print(f'leagues to fit: {len(leagues)}')

        params_version = 0
        league_tried_version = {league: -1 for league in leagues}
        skip_fit_until = {league: 0 for league in leagues}

        fit_start_time = time.monotonic()
        last_improvement_time = fit_start_time
        last_log_time = fit_start_time
        evaluations_since_improvement = 0

        for iteration in range(number_iterations):

            # leagues with changes after their last try
            changed_since_try = [league for league in leagues if league_tried_version[league] < params_version]
            if not changed_since_try:
                print(f'Converged: no changes since the last try of every league, iteration {iteration}')
                break

            # all remaining candidates are skipped: skipping only postpones them, try them now
            leagues_to_try = [league for league in changed_since_try if iteration >= skip_fit_until[league]]
            if not leagues_to_try:
                leagues_to_try = changed_since_try

            iteration_start_time = time.monotonic()
            iteration_start_loss = current_loss

            n_updated = 0
            n_evaluations = 0

            for league_position, league in enumerate(leagues_to_try, start=1):

                league_constraints = self._league_constraints(league)

                league_tried_version[league] = params_version
                league_is_updated = False

                # second leagues have no ms_lift_update_mu, leagues without newcomers may miss other params
                league_moves = {move_name: move for move_name, move in moves.items()
                                if all(name in league_params[league] for name in move)}

                for move_name, move in league_moves.items():
                    for direction in (-1, 1):
                        moved = False

                        # repeat the move while the loss decreases
                        while True:
                            candidate_params = self._moved_params(league_params[league], move, direction)

                            if not all(league_constraints[name](candidate_params[name])
                                       for name in move if name in league_constraints):
                                break

                            # only one league changes: match params are rebuilt only for its matches
                            # (and for its second leagues, shared ms_lift_update_mu)
                            league_params_copy = {**league_params, league: candidate_params}

                            candidate_matches_params = self._match_params(league_params_copy, matches_params=matches_params,
                                                                          changed_leagues={league})
                            loss = self.calculate_loss(league_params_copy, candidate_matches_params)

                            n_evaluations += 1
                            evaluations_since_improvement += 1

                            now = time.monotonic()
                            if now - last_log_time >= log_every_seconds:
                                print(f'[{self._format_duration(now - fit_start_time)}] no improvement for '
                                      f'{self._format_duration(now - last_improvement_time)} '
                                      f'({evaluations_since_improvement} evaluations), '
                                      f'iteration {iteration}, league {league_position}/{len(leagues_to_try)}: {league}, '
                                      f'move {move_name}, loss {current_loss:.2f}', flush=True)
                                last_log_time = now

                            if loss >= current_loss - loss_epsilon:
                                break

                            changes = ', '.join(f'{name}: {league_params[league][name]} -> {candidate_params[name]}'
                                                for name in move)
                            print(f'Loss down by: {current_loss - loss:.2f}, {league}, {move_name} ({changes}), '
                                  f'iteration: {iteration}', flush=True)

                            # replace, not mutate: match params keep references to league params dicts
                            league_params = league_params_copy
                            matches_params = candidate_matches_params
                            current_loss = loss
                            league_is_updated = True
                            moved = True
                            params_version += 1

                            self.save_params(league_params, config.soccer['params'][self.league_params_name])

                            last_improvement_time = time.monotonic()
                            last_log_time = last_improvement_time
                            evaluations_since_improvement = 0

                        # the other direction makes no sense after an improvement in this one
                        if moved:
                            break

                if league_is_updated:
                    print(league_params[league], flush=True)
                    print()
                    n_updated += 1
                    skip_fit_until[league] = 0
                else:
                    skip_fit_until[league] = iteration + skip_iterations

            # after each iteration make backup
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = Path(config.soccer['params'][self.league_params_name]).parent / f'league_params_{timestamp}.pkl'
            self.save_params(league_params, backup_path)

            iteration_time = time.monotonic() - iteration_start_time
            print(f'Iteration {iteration}: loss {iteration_start_loss:.2f} -> {current_loss:.2f} '
                  f'({iteration_start_loss - current_loss:+.2f}), '
                  f'leagues tried {len(leagues_to_try)}/{len(leagues)}, updated {n_updated}, '
                  f'evaluations {n_evaluations}, time {self._format_duration(iteration_time)}', flush=True)

        print(f'Final loss: {current_loss:.2f}, total time {self._format_duration(time.monotonic() - fit_start_time)}')

        return league_params

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Seconds as h:mm:ss."""
        seconds = int(seconds)
        return f'{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}'

    def fit_extra_time_params(self, number_iterations: int):
        """Fit extra time params (nested model), one set for all leagues. League params are fixed.
        Home advantage in extra time is the league home advantage scaled by home_advantage_scale."""

        self._setup_train()

        league_params = self.load_params(config.soccer['params'][self.league_params_name])

        # fixed params are overwritten: they may differ in the saved pickle
        extra_time_params = {**self.extra_time_params, **self.extra_time_fixed_params}
        self.extra_time_params = extra_time_params
        save_pickle(extra_time_params, config.soccer['params'][self.extra_time_params_name])

        matches_params = self._match_params(league_params, extra_time_params)

        current_loss = self.calculate_loss(league_params, matches_params)

        deltas = {
            'kappa': 0.01,
            'home_advantage_scale': 0.01,

            'draw_frequency': 0.01,
            'scoring_overall': 0.01
        }

        steps = deltas.copy()

        for i in range(number_iterations):
            any_improved = False

            for param_name, delta in deltas.items():
                current_value = extra_time_params[param_name]

                step = steps[param_name]

                improved = False
                for candidate in self._candidates(current_value, step):
                    if param_name in self.extra_time_constraints and not self.extra_time_constraints[param_name](candidate):
                        continue

                    extra_time_params_copy = {**extra_time_params, param_name: candidate}

                    matches_params = self._match_params(league_params, extra_time_params_copy)
                    loss = self.calculate_loss(league_params, matches_params)

                    if loss < current_loss - self.loss_epsilon:
                        print(f"Loss down by: {current_loss - loss:.2f}, {param_name}: {current_value} -> {candidate}, "
                              f"step={step}, Iteration: {i}")
                        print(extra_time_params_copy)
                        print()

                        extra_time_params = extra_time_params_copy
                        self.extra_time_params = extra_time_params
                        current_loss = loss

                        save_pickle(extra_time_params, config.soccer['params'][self.extra_time_params_name])

                        improved = True
                        break

                steps[param_name] = self._next_step(step, delta, improved)
                any_improved |= improved

            if not any_improved and all(steps[p] <= deltas[p] / self.min_step_divisor for p in deltas):
                print(f'Converged, iteration: {i}')
                break

    def check_extra_time_calibration(self):
        """Mean predicted vs actual extra time outcomes on train (home team perspective).
        Predictions use ratings at the start of the period, as in _loss_inside_period."""

        self._setup_train()

        league_params = self.load_params(config.soccer['params'][self.league_params_name])
        matches_params = self._match_params(league_params)

        ratings = self._rating_initialization(league_params)

        outcome_index = {1: 0, 0.5: 1, 0: 2}
        predicted = np.zeros(3)
        actual = np.zeros(3)

        for rating_period_id, team_matches in self.period_team_matches.items():

            # update RD
            self._inflate_rd(ratings, rating_period_id)

            for match_id in self.period_matches[rating_period_id]:
                info = self.home_away_match_info[match_id]

                if not info.is_extra_time:
                    continue

                params = matches_params[match_id]

                season = self.match_season[match_id]
                home_value = self._transition_value(ratings, info.home_team_league, season, params.home_league,
                                                    info.home_season_transition)
                away_value = self._transition_value(ratings, info.away_team_league, season, params.away_league,
                                                    info.away_season_transition)

                home_rating = self._season_update_rating(ratings[info.home_team], params.home_league, info.home_team_league,
                                                         info.home_season_transition, home_value)
                away_rating = self._season_update_rating(ratings[info.away_team], params.away_league, info.away_team_league,
                                                         info.away_season_transition, away_value)

                home_mu = self._convert_into_glicko2(home_rating).mu
                away_mu = self._convert_into_glicko2(away_rating).mu

                predicted += self.probabilities(home_mu, away_mu, info.home_indicator, info.goals_expectations_et,
                                                params.home_extra_time)
                actual[outcome_index[info.extra_time_outcome]] += 1

            ratings = self._rate_teams_inside_period(ratings, team_matches, matches_params)

        n = actual.sum()
        print(f'extra time matches: {int(n)}')
        print(f'predicted: win {predicted[0] / n:.3f}, tie {predicted[1] / n:.3f}, loss {predicted[2] / n:.3f}')
        print(f'actual:    win {actual[0] / n:.3f}, tie {actual[1] / n:.3f}, loss {actual[2] / n:.3f}')

    def fit_global_params(self, number_iterations: int = 10, loss_epsilon: float = 1., deltas: dict = None):
        """Fit the global params of the model (see Ratings.global_params) together, league params are fixed:
            update_rd, ms_update_rd_first, ms_update_rd_second - speed of the ratings.
        Coordinate descent: a move is repeated while the loss decreases, then the other direction and the next param;
        the steps are halved (down to min_deltas) when no param improves.
        Saved to league_params.pkl (with the league params) after every improvement.
        Constraints: ms_update_rd_first, ms_update_rd_second >= update_rd >= 0."""

        self._setup_train()

        path = Path(config.soccer['params'][self.league_params_name])

        # saved as they are, the leagues without params get group params only for the loss
        saved_league_params = self.load_params(path)
        league_params = self._fill_missing_leagues(saved_league_params)

        if deltas is None:
            deltas = {'update_rd': 4, 'ms_update_rd_first': 6, 'ms_update_rd_second': 6}
        deltas = dict(deltas)
        min_deltas = {'update_rd': 1, 'ms_update_rd_first': 1, 'ms_update_rd_second': 1}

        def is_valid(params: dict) -> bool:
            return (params['update_rd'] >= 0
                    and params['ms_update_rd_first'] >= params['update_rd']
                    and params['ms_update_rd_second'] >= params['update_rd'])

        def loss_of(params: dict) -> float:
            self.set_global_params(params)
            return self.calculate_loss(league_params, self._match_params(league_params))

        # the start must satisfy the constraints
        current_params = self.global_params()
        current_params['ms_update_rd_first'] = max(current_params['ms_update_rd_first'], current_params['update_rd'])
        current_params['ms_update_rd_second'] = max(current_params['ms_update_rd_second'], current_params['update_rd'])

        current_loss = loss_of(current_params)
        print(f'Start loss: {current_loss:.2f}, {current_params}', flush=True)

        for iteration in range(number_iterations):
            any_improved = False

            for name, delta in deltas.items():
                for direction in (-1, 1):
                    moved = False

                    while True:
                        candidate_params = {**current_params, name: round(current_params[name] + direction * delta, 3)}
                        if not is_valid(candidate_params):
                            break

                        loss = loss_of(candidate_params)
                        if loss >= current_loss - loss_epsilon:
                            break

                        print(f'Loss down by: {current_loss - loss:.2f}, {name}: {current_params[name]} -> '
                              f'{candidate_params[name]}, iteration: {iteration}', flush=True)

                        current_params = candidate_params
                        current_loss = loss
                        moved = True
                        any_improved = True
                        self.set_global_params(current_params)
                        self.save_params(saved_league_params, path)

                    if moved:
                        break

            if not any_improved:
                if all(deltas[name] <= min_deltas[name] for name in deltas):
                    print(f'Converged, iteration: {iteration}', flush=True)
                    break
                deltas = {name: max(delta / 2, min_deltas[name]) for name, delta in deltas.items()}
                print(f'steps: {deltas}', flush=True)

        self.set_global_params(current_params)
        self.save_params(saved_league_params, path)
        print(f'Global params: {current_params}, loss {current_loss:.2f}', flush=True)

        return current_params

    def fit_country_shift(self, countries: set = None, steps: tuple = (50, 30, 10), loss_epsilon: float = 1.,
                          log_every_seconds=3 * 60):
        """Shift init_mu of the first and the second league of a country together (the level of the country).
        A shift of the first league alone changes the gap to the second league of the same country: the cups between
        them are many and they are against the shift, the international matches are few and they are for it,
        so the per-league fit does not move init_mu (a plateau). A shift of both leagues keeps the domestic
        matches as they are, only the international matches decide.
        For every country every step (coarse to fine) is tried in both directions and repeated while the loss decreases.
        init_mu does not enter the match params, so matches_params are calculated once.
        countries: if None, the countries with international matches, the most first (_international_countries):
        the loss does not depend on init_mu of a country without them.
        Progress is printed every log_every_seconds (one evaluation of the loss takes ~30 seconds)."""

        self._setup_train()

        league_params_path = Path(config.soccer['params'][self.league_params_name])
        league_params = self._fill_missing_leagues(self.load_params(league_params_path))
        self._check_constraints(league_params)

        matches_params = self._match_params(league_params)
        current_loss = self.calculate_loss(league_params, matches_params)
        print(f'Start loss: {current_loss:.2f}')

        if countries is None:
            countries = self._international_countries()

        # a list keeps its order (the countries with more international matches first), a set is sorted
        countries = sorted(countries) if isinstance(countries, set) else list(countries)

        start_time = time.time()
        last_log_time = start_time
        n_evaluations = 0

        for country_position, country in enumerate(countries, start=1):
            country_leagues = [league for league in (f'{country}. First League', f'{country}. Second League')
                               if league in league_params]
            if not country_leagues:
                continue

            total_shift = 0

            for step in steps:
                for direction in (1, -1):
                    while True:
                        shift = direction * step
                        candidate_params = {**league_params,
                                            **{league: {**league_params[league],
                                                        'init_mu': int(round(league_params[league]['init_mu'] + shift))}
                                               for league in country_leagues}}

                        if not all(check(candidate_params[league]['init_mu'])
                                   for league in country_leagues
                                   for name, check in self._league_constraints(league).items()
                                   if name == 'init_mu'):
                            break

                        loss = self.calculate_loss(candidate_params, matches_params)
                        n_evaluations += 1

                        if time.time() - last_log_time >= log_every_seconds:
                            last_log_time = time.time()
                            print(f'[{self._format_duration(last_log_time - start_time)}] {n_evaluations} evaluations, '
                                  f'country {country_position}/{len(countries)}: {country}, shift {shift:+d}, '
                                  f'loss {current_loss:.2f}', flush=True)
                        if loss >= current_loss - loss_epsilon:
                            break

                        total_shift += shift
                        print(f'Loss down by: {current_loss - loss:.2f}, {country} init_mu shift {shift:+d} '
                              f'(total {total_shift:+d}), leagues: {len(country_leagues)}', flush=True)

                        league_params = candidate_params
                        current_loss = loss
                        self.save_params(league_params, league_params_path)

            print(f'{country}: total init_mu shift {total_shift:+d}, loss {current_loss:.2f}', flush=True)

        return league_params

    @staticmethod
    def _international_countries() -> list:
        """Countries by the number of the international matches of their clubs, the most first."""

        matches = pd.read_feather(config.soccer['glicko_matches'], columns=['is_home', 'is_domestic', 'home_team_league', 'away_team_league'])
        international = matches.loc[(matches['is_home'] == True) & (matches['is_domestic'] == False)]

        countries = pd.concat([international['home_team_league'], international['away_team_league']]).str.split('.').str[0]

        return countries.value_counts().index.tolist()

    def fit_from_scratch(self, shared_shifts: list, init_iterations: int = 30, league_fit_iterations: int = 1,
                         extra_time_iterations: int = 30):
        """The first fit, without saved league params:
            1. the same params for all leagues (init_league_params.pkl)
            2. the same params for the first and for the second leagues (init_first_second_leagues_params.pkl)
            3. the start point of the league params, then all stages of fit."""

        print('\n===== fit from scratch: the same params for all leagues =====')
        self.fit_init_league_params(number_iterations=init_iterations)

        print('\n===== fit from scratch: the first and the second leagues =====')
        self.fit_first_second_league_params(number_iterations=init_iterations)

        self.save_params(load_pickle(config.soccer['params']['init_first_second_leagues']), config.soccer['params'][self.league_params_name])

        self.fit(shared_shifts, league_fit_iterations, extra_time_iterations)

    def fit(self, shared_shifts: list, league_fit_iterations: int = 1, extra_time_iterations: int = 30):
        """All stages of the fit, from the global params to the params of individual leagues:
            1. global params (the speed of the ratings)
            2. shared shifts of all leagues, the first and the second leagues
            3. the level of every country: init_mu of its two divisions together
            4. the params of individual leagues
            5. extra time params
        Every stage saves the params after an improvement, so the fit can be stopped and continued.
        The normalization (normalize_params) is a separate step after the fit."""

        print('\n===== fit: global params =====')
        self.fit_global_params()

        for param_name, delta, groups in shared_shifts:
            print(f'\n===== fit: shared shift {param_name} =====')
            self.fit_shared_shift(param_name, delta=delta, groups=groups)

        print('\n===== fit: country shift =====')
        self.fit_country_shift()

        print('\n===== fit: per-league =====')
        self.fit_league_params(number_iterations=league_fit_iterations)

        print('\n===== fit: extra time =====')
        self.fit_extra_time_params(number_iterations=extra_time_iterations)

    def fit_shared_shift(self, param_name: str, delta: float = 0.02, number_iterations: int = 30,
                         groups: tuple = ('all', 'first', 'second')):
        """Shift param_name by the same delta in all leagues of a group (all / first / second leagues).
        Per-league differences are kept. Adaptive step, as in fit_first_second_league_params."""

        self._setup_train()

        league_params_path = Path(config.soccer['params'][self.league_params_name])
        league_params = self._fill_missing_leagues(self.load_params(league_params_path))
        self._check_constraints(league_params)

        matches_params = self._match_params(league_params)
        current_loss = self.calculate_loss(league_params, matches_params)
        print(f'Start loss: {current_loss:.2f}')

        group_leagues = {'all': self.first_leagues | self.second_leagues,
                         'first': self.first_leagues,
                         'second': self.second_leagues}

        for group in groups:
            subset = {league for league in group_leagues[group] if param_name in league_params.get(league, {})}
            step = delta
            total_shift = 0.

            for i in range(number_iterations):
                improved = False

                for shift in (step, -step):
                    league_params_copy = {league: ({**params, param_name: round(params[param_name] + shift, self.round_digits)}
                                                   if league in subset else params)
                                          for league, params in league_params.items()}

                    if not all(check(league_params_copy[league][name])
                               for league in subset
                               for name, check in self._league_constraints(league).items()
                               if name == param_name):
                        continue

                    candidate_matches_params = self._match_params(league_params_copy)
                    loss = self.calculate_loss(league_params_copy, candidate_matches_params)

                    if loss < current_loss - self.loss_epsilon:
                        total_shift += shift
                        print(f'Loss down by: {current_loss - loss:.2f}, {param_name} [{group}] shift {shift:+.4f} '
                              f'(total {total_shift:+.4f}), step={step}, iteration {i}', flush=True)

                        league_params = league_params_copy
                        matches_params = candidate_matches_params
                        current_loss = loss
                        self.save_params(league_params, league_params_path)

                        improved = True
                        break

                step = self._next_step(step, delta, improved)

                if not improved and step <= delta / self.min_step_divisor:
                    print(f'{param_name} [{group}] converged, total shift {total_shift:+.4f}, loss {current_loss:.2f}')
                    break

        return league_params
