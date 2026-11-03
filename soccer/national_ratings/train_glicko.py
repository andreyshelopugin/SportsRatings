import gc
import shutil
from pathlib import Path
from math import sqrt

import numpy as np
import pandas as pd

from config import config
from soccer.national_ratings.ratings import NationalRatings
from utils.functions import save_pickle, load_pickle


class TrainNationalGlicko(NationalRatings):
    """Fit params of the national teams ratings. Order (see NationalRatings):
        1. fit_global_params: global params, init_mu of confederations, rd_growth (final stage params fixed)
        2. fit_global_params(fit_final_stage=True), then compare_losses: gamma is kept
           only if they improve the test loss
        3. fit_country_params: culture-dependent params per country with L2 shrinkage to the confederation level
        4. normalize_params: the mean rating of the active teams is init_mu

    Train / test split by date (not by season: final stages are played in the next year, e.g. Euro 2020 in 2021).
    Loss: the first burn_in_days are skipped, a period is discounted by discount_per_year ** years to the end of train.

    State of the shared params is a flat dict:
        'global.<param>', 'confederation.<confederation>.init_mu', 'final_stage.<param>', 'rd_growth'
    """

    def __init__(self, train_test_split_date='2021-01-01'):
        super().__init__()

        self.train_test_split_date = pd.Timestamp(train_test_split_date)

        self.loss_epsilon = 0.01
        self.burn_in_days = 2 * 365
        self.discount_per_year = 0.98

        self.round_digits = 6

        # adaptive step (as in TrainGlicko): doubled after an improvement, halved otherwise
        self.max_step_factor = 50
        self.min_step_divisor = 8

        self.global_fit_params = ['crowd', 'logistic', 'strength', 'scoring',
                                  'draw_frequency', 'strength_overall', 'scoring_overall']

        self.deltas = {'init_mu': 10,
                       'rd_growth': 1,
                       'gamma': 0.02,
                       'default': 0.01}

        self.constraints = {'crowd': lambda x: x >= 0,
                            'logistic': lambda x: x >= 0,
                            'rd_growth': lambda x: x > 0,
                            # intensity of the support transfers from home matches to a neutral venue (final stage
                            # without host): alpha * (crowd_h - crowd_a), alpha > 1 is allowed
                            'alpha': lambda x: x >= 0,
                            'host': lambda x: x >= 0,
                            # logistic of the host at a final stage: the guests arrive in advance (acclimatization,
                            # no travel before the match), so the effect is not larger than in qualifiers
                            'beta': lambda x: 0 <= x <= 1,
                            # the share of the usual home advantage of the host of a final stage (28.10)
                            'gamma': lambda x: x >= 0}

        # country level: culture-dependent params, L2 shrinkage to the confederation level
        # draw_frequency is not per country: draws are about the nature of football, not a country
        self.country_fit_params = ['crowd', 'logistic', 'init_mu']

        # country params are offsets to the confederation level (NationalRatings.team_params),
        # penalty = country_penalty * sum((offset / scale) ** 2), in loss units
        self.country_penalty = 1.
        self.country_penalty_scales = {'init_mu': 100, 'crowd': 0.1, 'logistic': 0.1}

        # countries with fewer train matches keep the confederation params
        self.min_country_matches = 20

        self.period_loss_weight = None

        # match subsets for the losses report (compare_losses): a predicate on HomeAwayMatchInfo
        self.loss_subsets = {
            'all': lambda info: True,
            'final_host': lambda info: info.stage == 'final' and (info.home_is_host or info.away_is_host),
            'final_no_host': lambda info: info.stage == 'final' and not (info.home_is_host or info.away_is_host),
            'regular': lambda info: info.stage == 'regular',
        }

        # if set, matches outside the subset have zero loss weight (ratings are still updated by them)
        self.loss_filter = None

        # fit init_mu of the confederations (the start levels). False: all teams start from the same init_mu,
        # the levels of the confederations come only from the matches (28.10)
        self.fit_confederation_levels = True

    def _loss_weight(self, info) -> float:
        """Match type weight, zero for matches outside loss_filter."""

        if self.loss_filter is not None and not self.loss_filter(info):
            return 0.
        return super()._loss_weight(info)

    # setup

    def _setup(self, is_train: bool):
        """Precalculate train (date < split) or test (date >= split) matches.
        Loss weight of a period: 0 in the burn-in, discount_per_year ** years to the end otherwise (train),
        1 for test (no burn-in, no discount)."""

        matches = pd.read_feather(config.soccer['national_glicko_matches'])

        is_train_match = matches['date'] < self.train_test_split_date
        matches = matches.loc[is_train_match if is_train else ~is_train_match]

        self._precalculate_info(matches)

        period_start = matches.groupby('rating_period_id')['date'].min()

        if is_train:
            data_start, data_end = matches['date'].min(), matches['date'].max()
            years_to_end = (data_end - period_start).dt.days / 365.25

            weights = np.where(period_start < data_start + pd.Timedelta(days=self.burn_in_days), 0.,
                               self.discount_per_year ** years_to_end)
        else:
            weights = np.ones(len(period_start))

        self.period_loss_weight = dict(zip(period_start.index, weights))

        del matches
        gc.collect()

    def _setup_train(self):
        self._setup(is_train=True)

    def _setup_eval(self):
        self._setup(is_train=False)

    # state of the shared params

    def _initial_state(self) -> tuple[dict, dict]:
        """(state, country_params) from the saved params, defaults for the missing ones.
        A confederation without init_mu starts with the global init_mu."""

        global_params, confederation_params, country_params = self.load_params()

        state = {f'global.{name}': value for name, value in global_params.items() if name != 'init_mu'}

        for confederation in sorted(set(self.team_confederation.values())):
            init_mu = confederation_params.get(confederation, dict()).get('init_mu', global_params['init_mu'])
            state[f'confederation.{confederation}.init_mu'] = init_mu

        for name, value in self.final_stage_params.items():
            state[f'final_stage.{name}'] = value

        state['rd_growth'] = sqrt(self.rd_growth_squared)

        return state, country_params

    def _unflatten(self, state: dict) -> tuple[dict, dict, dict]:
        """state -> (global_params, confederation_params, final_stage_params), sets rd_growth."""

        global_params = {'init_mu': self.init_mu}
        confederation_params = dict()
        final_stage_params = dict()

        for key, value in state.items():
            parts = key.split('.')
            if parts[0] == 'global':
                global_params[parts[1]] = value
            elif parts[0] == 'confederation':
                confederation_params.setdefault(parts[1], dict())[parts[2]] = value
            elif parts[0] == 'final_stage':
                final_stage_params[parts[1]] = value
            elif key == 'rd_growth':
                self.rd_growth_squared = value ** 2

        return global_params, confederation_params, final_stage_params

    def _save_state(self, state: dict, country_params: dict = None):
        """Global params with rd_growth, confederation params, final stage params (and country params)."""

        global_params, confederation_params, final_stage_params = self._unflatten(state)

        save_pickle({**global_params, 'rd_growth': state['rd_growth']}, self.national_params_paths['global'])
        save_pickle(confederation_params, self.national_params_paths['confederation'])
        save_pickle(final_stage_params, self.national_params_paths['final_stage'])

        if country_params is not None:
            save_pickle(country_params, self.national_params_paths['country'])

    @staticmethod
    def _param_name(key: str) -> str:
        """'confederation.Europe.init_mu' -> 'init_mu', 'final_stage.alpha' -> 'alpha'."""
        return key.split('.')[-1]

    # losses

    def calculate_loss(self, team_params: dict, matches_params: dict) -> float:
        """Weighted log loss over the periods (see _setup)."""

        ratings = self._rating_initialization(team_params)

        total_loss = 0
        for rating_period_id, team_matches in self.period_team_matches.items():
            self._inflate_rd(ratings, rating_period_id)

            weight = self.period_loss_weight[rating_period_id]
            if weight > 0:
                total_loss += weight * self._loss_inside_period(ratings, self.period_matches[rating_period_id], matches_params)

            ratings = self._rate_teams_inside_period(ratings, team_matches, matches_params)

        return total_loss

    def calculate_test_loss(self, team_params: dict, ratings: dict, matches_params: dict) -> float:
        """Mean loss per match on test, starting from the train ratings. Teams without train matches start from init_mu."""

        ratings = dict(ratings)
        for team, rating in self._rating_initialization(team_params).items():
            if team not in ratings:
                ratings[team] = rating

        total_loss = 0
        n_matches = 0
        for rating_period_id, team_matches in self.period_team_matches.items():
            self._inflate_rd(ratings, rating_period_id)

            match_ids = self.period_matches[rating_period_id]
            total_loss += self._loss_inside_period(ratings, match_ids, matches_params)
            n_matches += self._n_loss_matches(match_ids)

            ratings = self._rate_teams_inside_period(ratings, team_matches, matches_params)

        return total_loss / n_matches

    def _n_loss_matches(self, match_ids) -> int:
        """Number of matches in the loss (loss_filter)."""

        if self.loss_filter is None:
            return len(match_ids)
        return sum(1 for match_id in match_ids if self.loss_filter(self.home_away_match_info[match_id]))

    def _state_loss(self, state: dict, country_params: dict) -> float:
        """Train loss of the state."""

        global_params, confederation_params, final_stage_params = self._unflatten(state)
        team_params = self.team_params(global_params, confederation_params, country_params)
        matches_params = self._match_params(team_params, final_stage_params=final_stage_params)

        return self.calculate_loss(team_params, matches_params)

    def _country_penalty(self, offsets: dict) -> float:
        """L2 penalty of one country: its offsets to the confederation level."""

        return self.country_penalty * sum((offsets.get(name, 0.) / self.country_penalty_scales[name]) ** 2
                                          for name in self.country_fit_params)

    # fits

    def _next_step(self, step: float, delta: float, improved: bool) -> float:
        if improved:
            return round(min(step * 2, self.max_step_factor * delta), self.round_digits)
        return round(max(step / 2, delta / self.min_step_divisor), self.round_digits)

    def fit_global_params(self, number_iterations: int, fit_final_stage=False, fit_alpha=False):
        """Coordinate descent over the global params, init_mu of confederations and rd_growth
        (and host / beta if fit_final_stage, alpha if fit_alpha). Country params are fixed.

        alpha is identified only by final stages without host: alpha * (crowd_h - crowd_a) is 0 while crowd is global,
        so with fit_alpha=False alpha is fixed to 1 and host keeps the product alpha * host (the host crowd factor).
        Fit alpha after fit_country_params (country crowd): the support intensity of home matches transfers to a neutral venue.

        A shift of all init_mu changes the loss only through strength_overall != 1 and strength != 0,
        normalize_params compensates it, the level is set by normalize_params."""

        self._setup_train()

        state, country_params = self._initial_state()

        # alpha, host, beta replaced by gamma (28.10), fit_alpha is not used
        # # alpha fixed to 1 (fit_alpha=False) or outside its constraint (e.g. fitted before the constraint):
        # # alpha = 1, host keeps the product alpha * host (the host crowd factor)
        # if fit_final_stage and (not fit_alpha or not self.constraints['alpha'](state['final_stage.alpha'])):
        #     state['final_stage.host'] = round(state['final_stage.alpha'] * state['final_stage.host'], self.round_digits)
        #     state['final_stage.alpha'] = 1.
        #     self._save_state(state)
        #
        # if fit_final_stage and not self.constraints['beta'](state['final_stage.beta']):
        #     state['final_stage.beta'] = min(max(state['final_stage.beta'], 0.), 1.)
        #     self._save_state(state)

        keys = ([f'global.{name}' for name in self.global_fit_params]
                + (sorted(key for key in state if key.startswith('confederation.')) if self.fit_confederation_levels else [])
                + ['rd_growth'])

        if fit_final_stage:
            # keys += ['final_stage.host', 'final_stage.beta'] + (['final_stage.alpha'] if fit_alpha else [])
            keys += ['final_stage.gamma']

        current_loss = self._state_loss(state, country_params)
        print(f'Start loss: {current_loss:.2f}', flush=True)

        deltas = {key: self.deltas.get(self._param_name(key), self.deltas['default']) for key in keys}
        steps = deltas.copy()

        for i in range(number_iterations):
            any_improved = False

            for key in keys:
                name = self._param_name(key)
                step = steps[key]

                improved = False
                for candidate in (round(state[key] - step, self.round_digits), round(state[key] + step, self.round_digits)):
                    if name in self.constraints and not self.constraints[name](candidate):
                        continue

                    candidate_state = {**state, key: candidate}
                    loss = self._state_loss(candidate_state, country_params)

                    if loss < current_loss - self.loss_epsilon:
                        print(f'Loss down by: {current_loss - loss:.2f}, {key}: {state[key]} -> {candidate}, '
                              f'step={step}, iteration: {i}', flush=True)

                        state = candidate_state
                        current_loss = loss
                        self._save_state(state)

                        improved = True
                        break

                steps[key] = self._next_step(step, deltas[key], improved)
                any_improved |= improved

            if not any_improved and all(steps[key] <= deltas[key] / self.min_step_divisor for key in keys):
                print(f'Converged, iteration: {i}')
                break

        # _unflatten changed rd_growth to the last candidate: restore the accepted state
        self._unflatten(state)
        print(f'Final loss: {current_loss:.2f}')

        return state

    def fit_country_params(self, number_iterations: int, reset=False):
        """Country offsets (country_fit_params) to the confederation level with L2 shrinkage to 0.
        Countries with at least min_country_matches train matches, the most active first.
        Constraints (crowd >= 0, ...) are checked on the effective value (confederation level + offset).
        Only matches of the changed country are rebuilt (incremental _match_params).
        reset=True starts from zero offsets."""

        self._setup_train()

        state, country_params = self._initial_state()
        global_params, confederation_params, final_stage_params = self._unflatten(state)

        if reset:
            country_params = dict()

        n_matches = pd.Series({team: len(match_ids) for team, match_ids in self.country_match_ids.items()})
        teams = n_matches[n_matches >= self.min_country_matches].sort_values(ascending=False).index.to_list()
        print(f'countries to fit: {len(teams)}')

        team_params = self.team_params(global_params, confederation_params, country_params)
        matches_params = self._match_params(team_params, final_stage_params=final_stage_params)

        penalty = sum(self._country_penalty(offsets) for offsets in country_params.values())
        current_loss = self.calculate_loss(team_params, matches_params) + penalty
        print(f'Start loss (with penalty): {current_loss:.2f}', flush=True)

        for i in range(number_iterations):
            n_updated = 0

            for team in teams:
                parent = {**global_params, **confederation_params.get(self.team_confederation[team], dict())}

                for name in self.country_fit_params:
                    delta = self.deltas.get(name, self.deltas['default'])

                    for direction in (-1, 1):
                        moved = False

                        # repeat the move while the loss decreases
                        while True:
                            offsets = country_params.get(team, dict())
                            candidate_offset = round(offsets.get(name, 0.) + direction * delta, self.round_digits)

                            if name in self.constraints and not self.constraints[name](parent[name] + candidate_offset):
                                break

                            candidate_offsets = {**offsets, name: candidate_offset}
                            candidate_params = {**team_params[team], name: parent[name] + candidate_offset}

                            candidate_team_params = {**team_params, team: candidate_params}
                            candidate_matches_params = self._match_params(candidate_team_params, final_stage_params=final_stage_params,
                                                                          matches_params=matches_params, changed_teams={team})

                            candidate_penalty = penalty - self._country_penalty(offsets) + self._country_penalty(candidate_offsets)

                            loss = self.calculate_loss(candidate_team_params, candidate_matches_params) + candidate_penalty

                            if loss >= current_loss - self.loss_epsilon:
                                break

                            print(f'Loss down by: {current_loss - loss:.2f}, {team}, {name} offset: '
                                  f'{offsets.get(name, 0.)} -> {candidate_offset}, iteration: {i}', flush=True)

                            team_params = candidate_team_params
                            matches_params = candidate_matches_params
                            penalty = candidate_penalty
                            current_loss = loss
                            moved = True

                            country_params = {**country_params, team: candidate_offsets}
                            save_pickle(country_params, self.national_params_paths['country'])

                        if moved:
                            n_updated += 1
                            break

            print(f'Iteration {i}: loss {current_loss:.2f} (penalty {penalty:.2f}), updates {n_updated}', flush=True)
            if n_updated == 0:
                print(f'Converged, iteration: {i}')
                break

        return country_params

    def _shift_compensation(self, state: dict, country_params: dict, delta: float) -> tuple[dict, dict]:
        """Params that keep the probabilities when all ratings are shifted by -delta (rating points).
        In SoccerGlicko.probabilities the level of ratings enters only through
            draw: draw_frequency + strength_overall * kappa * avg_mu vs win / loss: kappa * mu
            advantage: crowd + ... + strength * avg_mu
        so a shift by d (Glicko-2 scale) is compensated by
            draw_frequency -= (strength_overall - 1) * kappa * d   (exact)
            crowd -= strength * d                                   (exact for full crowd, approximate for reduced crowd statuses)
        Only the global level and init_mu of confederations change: country offsets follow the confederation level
        (NationalRatings.team_params), so country_params are returned unchanged."""

        d = -delta * self.q

        global_params, _, _ = self._unflatten(state)

        new_state = dict(state)
        new_state['global.draw_frequency'] = round(state['global.draw_frequency'] - (global_params['strength_overall'] - 1) * d,
                                                   self.round_digits)
        new_state['global.crowd'] = round(state['global.crowd'] - global_params['strength'] * d, self.round_digits)

        for key, value in state.items():
            if key.startswith('confederation.') and key.endswith('.init_mu'):
                new_state[key] = value - delta

        # extra time: kappa scales the whole rating scale
        et = self.extra_time_params
        self.extra_time_params = {**et, 'draw_frequency': round(et['draw_frequency'] - (et['strength_overall'] - 1) * et['kappa'] * d,
                                                               self.round_digits)}

        return new_state, country_params

    def normalize_params(self):
        """Shift init_mu of all confederations and countries, so the mean rating of the active teams
        (ratings_to_df) is init_mu. The shift is compensated in draw_frequency and crowd (_shift_compensation),
        so the probabilities and the loss stay the same."""

        matches = pd.read_feather(config.soccer['national_glicko_matches'])
        self._precalculate_info(matches)

        state, country_params = self._initial_state()

        while True:
            global_params, confederation_params, final_stage_params = self._unflatten(state)
            team_params = self.team_params(global_params, confederation_params, country_params)
            matches_params = self._match_params(team_params, final_stage_params=final_stage_params)

            ratings = self.rate_teams(team_params, matches_params)
            avg_mu = self.ratings_to_df(ratings)['rating'].mean()

            delta = avg_mu - self.init_mu
            print(f'avg_mu: {avg_mu:.2f}, delta: {delta:.2f}')

            if abs(delta) < 1:
                break

            state, country_params = self._shift_compensation(state, country_params, delta)

        self._save_state(state, country_params)
        save_pickle(self.extra_time_params, self.national_params_paths['extra_time'])

    def compare_losses(self, final_stage_variants: dict = None):
        """Train and test loss per match of the saved params and of final stage variants,
        e.g. {'no_final_stage_advantage': {'alpha': 0, 'host': 0, 'beta': 0}}, by match subsets (loss_subsets).
        Unweighted: no match type weights, no discount, no burn-in (comparable between variants and subsets).
        Ratings are updated by all matches, the loss is over the subset only."""

        match_type_loss_weights = self.match_type_loss_weights
        self.match_type_loss_weights = {match_type: 1 for match_type in match_type_loss_weights}

        variants = {'saved': None}
        variants.update(final_stage_variants or dict())

        rows = []
        for variant_name, variant_final_stage_params in variants.items():
            self._setup_train()

            state, country_params = self._initial_state()
            global_params, confederation_params, saved_final_stage_params = self._unflatten(state)
            final_stage_params = saved_final_stage_params if variant_final_stage_params is None else variant_final_stage_params

            train_team_params = self.team_params(global_params, confederation_params, country_params)
            train_matches_params = self._match_params(train_team_params, final_stage_params=final_stage_params)

            self.loss_filter = None
            train_ratings = self.rate_teams(train_team_params, train_matches_params)

            for subset_name, subset in self.loss_subsets.items():
                self.loss_filter = subset

                # train: the same pass as rate_teams, unweighted
                self.period_loss_weight = {period: 1. for period in self.period_loss_weight}
                n_train = self._n_loss_matches(self.home_away_match_info.keys())
                train_loss = self.calculate_loss(train_team_params, train_matches_params) / n_train if n_train else np.nan

                rows.append({'variant': variant_name, 'subset': subset_name, 'split': 'train',
                             'n_matches': n_train, 'loss': train_loss})

            self._setup_eval()
            team_params = self.team_params(global_params, confederation_params, country_params)
            matches_params = self._match_params(team_params, final_stage_params=final_stage_params)

            for subset_name, subset in self.loss_subsets.items():
                self.loss_filter = subset

                n_test = self._n_loss_matches(self.home_away_match_info.keys())
                test_loss = self.calculate_test_loss(team_params, train_ratings, matches_params) if n_test else np.nan

                rows.append({'variant': variant_name, 'subset': subset_name, 'split': 'test',
                             'n_matches': n_test, 'loss': test_loss})

        self.loss_filter = None
        self.match_type_loss_weights = match_type_loss_weights

        report = (pd.DataFrame(rows)
                  .pivot_table(index=['split', 'subset', 'n_matches'], columns='variant', values='loss')
                  .round(6))

        print(report.to_string())

        return report


def _backup(paths: dict, suffix: str) -> dict:
    """Copy the params pickles to <name>_<suffix>.pkl, returns {path: backup path}."""
    backups = dict()
    for path in paths.values():
        path = Path(path)
        if path.exists():
            backup = path.with_name(f'{path.stem}_{suffix}{path.suffix}')
            shutil.copy(path, backup)
            backups[path] = backup
    return backups


def _restore(backups: dict):
    for path, backup in backups.items():
        shutil.copy(backup, path)


if __name__ == '__main__':
    train = TrainNationalGlicko()
    paths = config.soccer['national_params']

    # every variant starts from the same global params: fitted without countries (alpha = 1, not identified without them)
    save_pickle(dict(), paths['country'])
    train.fit_global_params(number_iterations=100, fit_final_stage=True)
    baseline = _backup(paths, 'baseline')

    print('\n=== no countries ===')
    train.compare_losses()

    for penalty in [1., 5., 20.]:
        print(f'\n=== countries {train.country_fit_params}, penalty {penalty} ===')
        _restore(baseline)
        train = TrainNationalGlicko()
        train.country_penalty = penalty

        train.fit_country_params(number_iterations=20, reset=True)
        train.fit_global_params(number_iterations=100, fit_final_stage=True, fit_alpha=True)
        train.compare_losses()

        _backup(paths, f'penalty_{int(penalty)}')

    # the best variant: _restore to its backup (e.g. country_params_penalty_5.pkl -> country_params.pkl, ...),
    # then train.normalize_params()
    _restore(baseline)