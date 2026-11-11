import gc
import warnings
from collections import namedtuple
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from config import config
from three_outcome_models.soccer_glicko import SoccerGlicko, Rating
from utils.functions import load_pickle, save_pickle
from utils.metrics import three_outcomes_log_loss

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

MatchParams = namedtuple('MatchParams', ['home', 'home_league', 'away_league', 'home_extra_time'])


class Ratings(SoccerGlicko):
    """Parent class for ratings training and prediction."""

    def __init__(self, init_mu=1500, init_rd=250, update_rd=22, rd_threshold=120, min_active_teams_share=0.75,
                 recent_seasons=3, current_season_days=60, min_season_progress=0.5):
        super().__init__()
        self.seasons = None
        self.rating_periods = None
        self.match_info = None
        self.init_mu = init_mu
        self.init_rd = init_rd

        # hyperparams
        self.update_rd_squared = update_rd ** 2
        self.rd_threshold = rd_threshold

        # speed of the ratings, three global params (fitted by TrainGlicko.fit_global_params):
        #   update_rd           - RD growth before every rating period (sqrt(rd^2 + update_rd^2))
        #   ms_update_rd_first  - RD growth at the season transition of the teams of the first leagues (rd + value)
        #   ms_update_rd_second - the same for the second leagues
        # constraint: ms_update_rd_first, ms_update_rd_second >= update_rd
        self.update_rd = update_rd
        self.ms_update_rd_first = update_rd
        self.ms_update_rd_second = update_rd

        # latest season of a league is used for the ratings table only if it is completed or has enough teams
        # (a new season may have just started), see _latest_championship_seasons
        self.min_active_teams_share = min_active_teams_share
        self.recent_seasons = recent_seasons
        self.current_season_days = current_season_days
        self.min_season_progress = min_season_progress

        # original glicko params
        self.original_glicko_league_params = {'init_mu': self.init_mu,

                                              'ms_lift_update_mu': 0,
                                              'ms_new_team_update_mu': 0,

                                              'crowd': 0.127,
                                              'logistic': 0.127,
                                              'strength': 0.001,
                                              'scoring': 0,

                                              'draw_frequency': -0.277,
                                              'strength_overall': 0.995,
                                              'scoring_overall': 0}

        self.init_params = {'init_mu': self.init_mu,

                            'ms_lift_update_mu': 30,

                            'crowd': 0.127,
                            'logistic': 0.127,
                            'strength': 0.001,
                            'scoring': 0,

                            'draw_frequency': -0.277,
                            'strength_overall': 0.995,
                            'scoring_overall': 0
                            }

        # for neutral matches these params are averaged over both leagues
        self.neutral_averaged_params = ['draw_frequency', 'strength_overall', 'scoring_overall']

        # extra time part of the loss, accumulated in _loss_inside_period (without discount), see compare_losses
        self.extra_time_loss_sum = 0.

        # if a list, _loss_inside_period appends (match_id, p_win, p_draw, p_lose) of 90 minutes:
        # the predictions of the sequential scheme of the loss (EvaluateGlicko.save_test_predictions)
        self.prediction_records = None

        # weights of the matches in the loss: the objective of the fit (the same in TrainGlicko),
        # compare_losses and EvaluateGlicko set them to 1 for the honest comparison with the boostings
        self.world_cup_loss_weight = 10
        self.international_cup_loss_weight = 4

        self.leagues_with_low_international_history = None
        self.min_international_matches_for_country = 30
        self.low_international_history_init_mu = 1400

        # ms_lift_update_mu is shared by the first and second leagues of a country:
        # a team promoted to the first league gets +lift, a team relegated to the second league gets -lift,
        # lift is stored only in the first league params (see _tie_lift_params)
        self.first_league_of_country = None
        # {first league: [second leagues of its country]}
        self.second_leagues_dict = None

        # a newcomer (a team without the previous season in the data, e.g. promoted from a division we do not have) starts at
        # anchor + ms_new_team_update_mu, where the anchor is the first available of:
        #   1. the mean rating of the teams that left the league after the previous season (relegated or gone),
        #      if there are at least min_anchor_leavers of them: the newcomers take their places;
        #   2. the newcomer_quantile of the ratings of the remaining teams (the league expanded),
        #      if there are at least min_anchor_teams of them;
        #   3. init_mu (the first season of the league).
        # so a newcomer starts near the bottom of its league as it is now, not at a fixed value.
        self.min_anchor_leavers = 1
        self.newcomer_quantile = 0.1
        self.min_anchor_teams = 4

        # the leavers' ratings are too low at the moment of leaving (a bad season), so a newcomer starts above them.
        # one fixed value for all leagues, not fitted
        self.ms_new_team_update_mu = 90

        # a league is comparable in the league table if over the last connectivity_seasons seasons it is linked to the other
        # leagues: by matches against them (cups, international) or by the transitions of teams between the divisions
        self.connectivity_seasons = 3

        # (league, season) -> anchor, computed once per run (reset in _rating_initialization)
        self._anchor_cache = dict()

        # extra time params: one set for all leagues (nested model, extra time stage),
        # kappa scales the strength difference in extra time,
        # home advantage in extra time = home_advantage_scale * home team league advantage (crowd, logistic, strength, scoring)
        self.init_extra_time_params = {'kappa': 0.75,
                                       'home_advantage_scale': 0.4,

                                       'draw_frequency': 0.25,
                                       'strength_overall': 0.995,
                                       'scoring_overall': 0}

        extra_time_params_path = Path(config.soccer['params']['extra_time'])
        if extra_time_params_path.exists():
            self.extra_time_params = load_pickle(extra_time_params_path)
        else:
            self.extra_time_params = self.init_extra_time_params.copy()

        # the global params of the baselines (original Glicko, init params), fitted ones are in league_params.pkl
        self.default_global_params = self.global_params()

    @staticmethod
    def _country(league: str) -> str:
        """'Spain. Second League' -> 'Spain'"""
        return league.split('.', 1)[0].strip()

    @staticmethod
    def _drop_teams_without_championship(matches: pd.DataFrame) -> pd.DataFrame:
        """Drop matches of teams without championship matches in the given data.
        ClubPreprocessing drops such teams on the full data, but after a season split (train / eval)
        a team may lose all its championship matches (e.g. its only championship season is excluded as short).
        Same criterion as the assert in _precalculate_info: is_championship == True.

        One pass is enough: dropped matches are not championship matches
        (otherwise the team would have a championship match), so other teams keep all their championship matches."""

        teams_with_championship = set(matches.loc[matches['is_championship'] == True, 'team'])

        bad_match_ids = set(matches.loc[~matches['team'].isin(teams_with_championship), 'match_id'])

        if bad_match_ids:
            bad_teams = sorted(set(matches['team']) - teams_with_championship)
            print(f'#dropped matches with teams without championship after the season split: {len(bad_match_ids)}, '
                  f'teams: {bad_teams}')

        # by match_id: both rows of a match (team and opp_team) are dropped
        return matches[~matches['match_id'].isin(bad_match_ids)]

    def _precalculate_info(self, matches: pd.DataFrame):
        """Cache match-level attributes for fast lookup during ratings computation.
        Uses namedtuple for immutability and faster attribute access."""

        self.seasons = sorted(list(set(matches['season'])))

        self.rating_periods = matches['rating_period_id'].max()

        self.first_leagues = set(matches.loc[matches['tournament_type'] == 'first_league', 'tournament'])
        self.second_leagues = set(matches.loc[matches['tournament_type'] == 'second_league', 'tournament'])

        # pairs first league <-> second leagues of the same country (shared ms_lift_update_mu)
        self.first_league_of_country = {self._country(league): league for league in self.first_leagues}

        self.second_leagues_dict = dict()
        for league in self.second_leagues:
            first = self.first_league_of_country.get(self._country(league))
            if first is not None:
                self.second_leagues_dict.setdefault(first, []).append(league)

        # home team row of each match: is_home, not home_indicator (it is 0 for both teams in neutral matches)
        home_matches = matches.loc[matches['is_home'] == True]
        assert home_matches['match_id'].is_unique, 'each match must have exactly one home row'

        league_match_counts = pd.concat([home_matches[['match_id', 'home_team_league']]
                                        .rename(columns={'home_team_league': 'league'}),

                                         home_matches[['match_id', 'away_team_league']]
                                        .rename(columns={'away_team_league': 'league'})])

        self.leagues_sorted_by_matches = (league_match_counts
                                          .groupby('league', sort=False)
                                          ['match_id']
                                          .nunique()
                                          .sort_values(ascending=False)
                                          .index
                                          .to_list())

        # countries with low international history.
        # if a country has too few international/non-domestic matches,
        # its league rating is poorly connected to the global rating graph.
        international_matches = home_matches[home_matches['is_domestic'] == False]

        international_matches['home_team_country'] = international_matches['home_team_league'].map(lambda x: x.split('.')[0])
        international_matches['away_team_country'] = international_matches['away_team_league'].map(lambda x: x.split('.')[0])

        international_countries = pd.concat([
            international_matches[['match_id', 'home_team_country']]
            .rename(columns={'home_team_country': 'country'}),

            international_matches[['match_id', 'away_team_country']]
            .rename(columns={'away_team_country': 'country'}),
        ])

        country_international_matches = (international_countries
                                         .groupby('country', sort=False)
                                         ['match_id']
                                         .nunique())

        countries_with_low_international_history = set(country_international_matches
                                                       .loc[country_international_matches < self.min_international_matches_for_country]
                                                       .index)

        self.leagues_with_low_international_history = {league
                                                       for league in set(home_matches['home_team_league']) | set(home_matches['away_team_league'])
                                                       if league.split('.', 1)[0].strip() in countries_with_low_international_history
                                                       }

        self.season_team_leagues = {season: self._team_leagues(matches, season) for season in self.seasons}

        # teams without championship matches are dropped in ClubPreprocessing (weak teams inflate ratings)
        # and after a season split in _drop_teams_without_championship
        teams_with_championship = {team for team_leagues in self.season_team_leagues.values() for team in team_leagues}
        teams_without_championship = set(matches['team']) - teams_with_championship
        assert not teams_without_championship, f'teams without championship matches: {sorted(teams_without_championship)}'

        # continuing teams of (league, season): in the league in this season and in its previous season
        league_season_teams = dict()
        for season, team_leagues in self.season_team_leagues.items():
            for team, league in team_leagues.items():
                league_season_teams.setdefault(league, dict()).setdefault(season, set()).add(team)

        # leavers of (league, season): in the league in its previous season, not in the league in this season,
        # and not moved up to a first league. Newcomers come from below (lower tier or outside the data),
        # so they replace the teams that went down or left the data, not the promoted ones
        # (e.g. a second league: promoted teams are the best teams of the league and would bias the anchor up)
        self.league_season_continuing = dict()
        self.league_season_leavers = dict()
        for league, season_teams in league_season_teams.items():
            league_seasons = sorted(season_teams)
            for previous, current in zip(league_seasons[:-1], league_seasons[1:]):
                current_team_leagues = self.season_team_leagues.get(current, dict())

                self.league_season_continuing[(league, current)] = tuple(sorted(season_teams[current] & season_teams[previous]))
                self.league_season_leavers[(league, current)] = tuple(sorted(team for team in season_teams[previous] - season_teams[current]
                                                                             if current_team_leagues.get(team) not in self.first_leagues))

        self.match_season = dict(zip(home_matches['match_id'].values, home_matches['season'].values))

        MatchInfo = namedtuple('MatchInfo', ['opp_team', 'season_transition', 'opp_season_transition',
                                             'is_home', 'home_indicator', 'goals_expectations', 'goals_expectations_et',
                                             'outcome', 'is_extra_time', 'extra_time_outcome',
                                             'status', 'home_team_league', 'away_team_league'])

        self.match_info = {
            (match_id, team): MatchInfo(
                opp_team, season_transition, opp_season_transition,
                is_home, home_indicator, goals_expectations, goals_expectations_et,
                outcome, is_extra_time, extra_time_outcome,
                status, home_team_league, away_team_league
            )
            for (match_id, team, opp_team, season_transition, opp_season_transition,
                 is_home, home_indicator, goals_expectations, goals_expectations_et,
                 outcome, is_extra_time, extra_time_outcome,
                 status, home_team_league, away_team_league) in zip(
                matches['match_id'].values,
                matches['team'].values,
                matches['opp_team'].values,
                matches['season_transition'].values,
                matches['opp_season_transition'].values,
                matches['is_home'].values,
                matches['home_indicator'].values,
                matches['goals_expectations'].values,
                matches['goals_expectations_et'].values,
                matches['outcome'].values,
                matches['is_extra_time'].values,
                matches['extra_time_outcome'].values,
                matches['status'].values,
                matches['home_team_league'].values,
                matches['away_team_league'].values
            )
        }

        HomeAwayMatchInfo = namedtuple('HomeAwayMatchInfo', ['home_team', 'away_team',
                                                             'home_season_transition', 'away_season_transition',
                                                             'home_indicator',
                                                             'is_domestic', 'is_world_cup', 'goals_expectations', 'goals_expectations_et',
                                                             'outcome', 'is_extra_time', 'extra_time_outcome',
                                                             'status', 'home_team_league', 'away_team_league'])

        is_world_cup = home_matches['tournament'].isin({'World FIFA Club World Cup', 'World FIFA Intercontinental Cup'})

        self.home_away_match_info = {
            match_id: HomeAwayMatchInfo(
                home_team, away_team,
                home_season_transition, away_season_transition,
                home_indicator,
                is_domestic, is_world_cup, goals_expectations, goals_expectations_et,
                outcome, is_extra_time, extra_time_outcome,
                status, home_team_league, away_team_league
            )
            for (match_id, home_team, away_team,
                 home_season_transition, away_season_transition,
                 home_indicator,
                 is_domestic, is_world_cup, goals_expectations, goals_expectations_et,
                 outcome, is_extra_time, extra_time_outcome,
                 status, home_team_league, away_team_league) in zip(
                home_matches['match_id'].values,
                home_matches['team'].values,
                home_matches['opp_team'].values,
                home_matches['season_transition'].values,
                home_matches['opp_season_transition'].values,
                home_matches['home_indicator'].values,
                home_matches['is_domestic'].values,
                is_world_cup.values,
                home_matches['goals_expectations'].values,
                home_matches['goals_expectations_et'].values,
                home_matches['outcome'].values,
                home_matches['is_extra_time'].values,
                home_matches['extra_time_outcome'].values,
                home_matches['status'].values,
                home_matches['home_team_league'].values,
                home_matches['away_team_league'].values,
            )
        }

        # matches of each league (home or away team league), for incremental update of match params
        self.league_match_ids = dict()
        for match_id, info in self.home_away_match_info.items():
            self.league_match_ids.setdefault(info.home_team_league, []).append(match_id)
            if info.away_team_league != info.home_team_league:
                self.league_match_ids.setdefault(info.away_team_league, []).append(match_id)

        self.period_team_matches = {
            rating_period_id: {
                team: group['match_id'].values
                for team, group in period_df.groupby('team', sort=False)
            }
            for rating_period_id, period_df in (matches
                                                .loc[:, ['date', 'rating_period_id', 'team', 'match_id']]
                                                .sort_values('date')
                                                .groupby('rating_period_id', sort=False))
        }

        self.period_matches = {
            rating_period_id: period_df['match_id'].to_list()
            for rating_period_id, period_df in (matches
                                                .loc[:, ['date', 'rating_period_id', 'match_id']]
                                                .sort_values('date')
                                                .drop_duplicates('match_id')
                                                .groupby('rating_period_id', sort=False))
        }

        # for save memory
        del matches, home_matches, is_world_cup
        gc.collect()

    def load_params(self, path) -> dict:
        """Params of the model (league_params.pkl, its backups): {'global': global params, 'leagues': league params}.
        The global params are set to the model, the league params are returned.
        The intermediate files (original Glicko, init params) are league params only."""

        params = load_pickle(path)
        self.set_global_params(params['global'])

        return params['leagues']

    def save_params(self, league_params: dict, path):
        """"""

        save_pickle({'global': self.global_params(), 'leagues': league_params}, path)

    def set_global_params(self, global_params: dict):
        """"""

        self.update_rd = global_params['update_rd']
        self.update_rd_squared = self.update_rd ** 2
        self.ms_update_rd_first = global_params['ms_update_rd_first']
        self.ms_update_rd_second = global_params['ms_update_rd_second']

    def global_params(self) -> dict:
        """"""

        return {'update_rd': self.update_rd,
                'ms_update_rd_first': self.ms_update_rd_first,
                'ms_update_rd_second': self.ms_update_rd_second}

    def _tie_lift_params(self, league_params: dict) -> dict:
        """Effective params: ms_lift_update_mu of a second league = -ms_lift_update_mu of the first league of the country.
        Promoted team: +lift, relegated team: -lift (one value per country).
        A second league without a first league in the data gets lift 0.
        Returns a new dict, league_params is not mutated."""

        tied = dict(league_params)

        for league in self.second_leagues:
            if league not in league_params:
                continue

            first = self.first_league_of_country.get(self._country(league))
            lift = league_params[first].get('ms_lift_update_mu', 0) if first in league_params else 0

            tied[league] = {**league_params[league], 'ms_lift_update_mu': -lift}

        return tied

    def _fill_missing_leagues(self, league_params: dict) -> dict:
        """Leagues without fitted params (e.g. appeared only in test seasons) get group params:
        median over fitted first / second leagues. init_mu is capped for leagues with low international history.
        Returns a copy, the saved params are not changed."""

        league_params = {k: v.copy() for k, v in league_params.items()}

        missing_leagues = (self.first_leagues | self.second_leagues) - set(league_params)
        if not missing_leagues:
            return league_params

        for group in [self.first_leagues, self.second_leagues]:
            fitted = [params for league, params in league_params.items() if league in group]
            if not fitted:
                continue

            param_names = {name for params in fitted for name in params}
            group_params = {name: float(np.median([params[name] for params in fitted if name in params])) for name in param_names}

            for league in missing_leagues & group:
                params = group_params.copy()
                if league in self.leagues_with_low_international_history:
                    params['init_mu'] = min(params['init_mu'], self.low_international_history_init_mu)
                league_params[league] = params

        still_missing = (self.first_leagues | self.second_leagues) - set(league_params)
        assert not still_missing, f'leagues without params: {sorted(still_missing)}'

        print(f'{len(missing_leagues)} leagues without fitted params, group params are used')

        return league_params

    def _original_glicko_league_params_initialization(self) -> dict:
        """Each league (first and second) has a specific set of parameters."""

        league_params = dict()
        for league in self.first_leagues | self.second_leagues:
            league_params[league] = self.original_glicko_league_params.copy()

        return league_params

    def _league_params_initialization(self) -> dict:
        """Each league (first and second) has a specific set of parameters.
        Second leagues have no ms_lift_update_mu: it is shared with the first league of the country."""

        league_params = dict()
        for league in self.first_leagues:
            league_params[league] = self.init_params.copy()

        for league in self.second_leagues:
            params = self.init_params.copy()
            params.pop('ms_lift_update_mu')
            league_params[league] = params

        return league_params

    @staticmethod
    def _team_leagues(matches: pd.DataFrame, season: int) -> dict:
        """Returns {team: league} for a season: the league with the most championship matches of the team
        (deterministic, does not depend on the rows order)."""

        season_championship_matches = matches.loc[(matches['is_championship'] == True) & (matches['season'] == season)]

        team_league_counts = (season_championship_matches
                              .groupby(['team', 'tournament'])['match_id']
                              .nunique()
                              .rename('n_matches')
                              .reset_index()
                              .sort_values(['team', 'n_matches', 'tournament'], ascending=[True, False, True])
                              .drop_duplicates('team', keep='first'))

        return dict(zip(team_league_counts['team'], team_league_counts['tournament']))

    @staticmethod
    def _status_adjusted_params(params: dict, status: str) -> dict:
        """Adjust home field advantage params for crowd and logistic status."""

        if status == 'FULL_CROWD_FULL_LOGISTIC':
            return params
        elif status == 'NEUTRAL':
            return params
        elif status == 'HALF_CROWD_FULL_LOGISTIC':
            return {**params, 'crowd': params['crowd'] / 2}
        elif status == 'ZERO_CROWD_FULL_LOGISTIC':
            return {**params, 'crowd': 0}
        elif status == 'FULL_CROWD_ZERO_LOGISTIC':
            return {**params, 'logistic': 0}
        elif status == 'HALF_CROWD_ZERO_LOGISTIC':
            return {**params, 'crowd': params['crowd'] / 2, 'logistic': 0}
        elif status == 'ZERO_CROWD_ZERO_LOGISTIC':
            return {**params, 'crowd': 0, 'logistic': 0}

        raise KeyError(f'Wrong status: {status}')

    def _single_match_params(self, home_params: dict, away_params: dict, status: str, is_extra_time: bool, extra_time_params: dict):
        """Params of one match (see _match_params)."""

        adjusted_home = self._status_adjusted_params(home_params, status)

        if status == 'NEUTRAL':
            adjusted_home = {**adjusted_home,
                             **{param: (home_params[param] + away_params[param]) / 2
                                for param in self.neutral_averaged_params}}

        if is_extra_time:
            home_advantage_scale = extra_time_params['home_advantage_scale']

            home_extra_time = {'kappa': extra_time_params['kappa'],

                               'crowd': home_advantage_scale * adjusted_home['crowd'],
                               'logistic': home_advantage_scale * adjusted_home['logistic'],
                               'strength': home_advantage_scale * adjusted_home['strength'],
                               'scoring': home_advantage_scale * adjusted_home['scoring'],

                               'draw_frequency': extra_time_params['draw_frequency'],
                               'strength_overall': extra_time_params['strength_overall'],
                               'scoring_overall': extra_time_params['scoring_overall']}
        else:
            home_extra_time = None

        return MatchParams(home=adjusted_home,
                           home_league=home_params,
                           away_league=away_params,
                           home_extra_time=home_extra_time)

    def _match_params(self, league_params: dict, extra_time_params: dict = None,
                      matches_params: dict = None, changed_leagues: set = None) -> dict:
        """Map each match to its league params.
        home: home team league params adjusted for crowd and logistic status (used for probabilities),
              for neutral matches draw and overall params are averaged over both leagues
              (home advantage is zeroed by home_indicator, the listed home team is arbitrary)
        home_league: unadjusted home team league params (used for season transition update)
        away_league: away team league params (used for season transition update)
        home_extra_time: extra time params, only for extra time matches (None otherwise):
                         home advantage of the home team league (adjusted for status) scaled by home_advantage_scale,
                         kappa and draw params are common for all leagues

        ms_lift_update_mu of second leagues is tied to the first league of the country (_tie_lift_params).

        Matches with the same (home league, away league, status, is_extra_time) share one MatchParams object,
        params dicts must not be mutated afterwards.

        Incremental mode: if matches_params and changed_leagues are passed,
        only matches of changed_leagues are rebuilt (the other params must be the same as in matches_params).
        A changed first league also rebuilds the matches of its second leagues (shared ms_lift_update_mu).
        """

        if extra_time_params is None:
            extra_time_params = self.extra_time_params

        league_params = self._tie_lift_params(league_params)

        if matches_params is None or changed_leagues is None:
            match_ids = self.home_away_match_info.keys()
            new_matches_params = dict()
        else:
            changed_leagues = set(changed_leagues) | {second
                                                      for league in changed_leagues
                                                      for second in self.second_leagues_dict.get(league, ())}

            match_ids = {match_id for league in changed_leagues for match_id in self.league_match_ids.get(league, ())}
            new_matches_params = matches_params.copy()

        cache = dict()
        for match_id in match_ids:
            info = self.home_away_match_info[match_id]

            key = (info.home_team_league, info.away_team_league, info.status, info.is_extra_time)

            params = cache.get(key)
            if params is None:
                params = self._single_match_params(league_params[info.home_team_league], league_params[info.away_team_league],
                                                   info.status, info.is_extra_time, extra_time_params)
                cache[key] = params

            new_matches_params[match_id] = params

        return new_matches_params

    def _rating_initialization(self, league_params: dict) -> dict:
        """Initializes teams ratings with the init_mu of the first league of each team.
        The same init_mu is used for teams that appear later (missed_previous in _season_update_rating),
        so all teams start on one scale. The level of the scale is set by normalize_params (shift of init_mu):
        Glicko is invariant to adding a constant to all ratings."""

        # new run: newcomer anchors are computed from the ratings of this run
        self._anchor_cache = dict()

        # seasons should be sorted to accurately catch the first match of each team
        ratings = dict()
        for season in self.seasons:
            team_leagues = self.season_team_leagues[season]

            for team, league in team_leagues.items():
                if team not in ratings:
                    ratings[team] = Rating(mu=league_params[league]['init_mu'], rd=self.init_rd)

        return ratings

    def _transition_value(self, ratings: dict, league: str, season, params: dict, season_transition: str):
        """Value for _season_update_rating:
            missed_previous: anchor of the newcomer (absolute mu, see below)
            other transitions: None

        Anchor: mean current rating of the leavers of the league, else newcomer_quantile of the continuing teams,
        else init_mu (see __init__). Computed once per (league, season) and cached for the run: the first request
        uses the ratings at the start of the rating period (the same in the loss and in the update).
        Leavers that already played in another league this season have their new mu (e.g. with -lift),
        the others keep the mu of the end of the previous season."""

        if season_transition != 'missed_previous':
            return None

        key = (league, season)
        anchor = self._anchor_cache.get(key)
        if anchor is None:
            leavers = [ratings[team].mu for team in self.league_season_leavers.get(key, ()) if team in ratings]
            continuing = [ratings[team].mu for team in self.league_season_continuing.get(key, ()) if team in ratings]

            if len(leavers) >= self.min_anchor_leavers:
                anchor = float(np.mean(leavers))
            elif len(continuing) >= self.min_anchor_teams:
                anchor = float(np.quantile(continuing, self.newcomer_quantile))
            else:
                anchor = params['init_mu']

            self._anchor_cache[key] = anchor

        return anchor

    def _ms_update_rd(self, league: str) -> float:
        """RD growth at the season transition: ms_update_rd_first or ms_update_rd_second by the level of the league."""
        return self.ms_update_rd_second if league in self.second_leagues else self.ms_update_rd_first

    def _season_update_rating(self, rating: Rating, params: dict, league: str, season_transition: str, value: float = None) -> Rating:
        """Update rating's mu and RD after each season.
        'same_league', 'changed_league', 'missed_previous'
        value (see _transition_value): missed_previous: anchor + ms_new_team_update_mu, init_mu + ms_new_team_update_mu if None"""

        if season_transition == 'same_season':
            return rating

        if season_transition == 'same_league':
            return Rating(mu=rating.mu, rd=rating.rd + self._ms_update_rd(league))

        if season_transition == 'changed_league':
            return Rating(mu=rating.mu + params['ms_lift_update_mu'],
                          rd=rating.rd + self._ms_update_rd(league))

        if season_transition == 'missed_previous':
            base = params['init_mu'] if value is None else value
            return Rating(mu=base + self.ms_new_team_update_mu,
                          rd=self.init_rd)

        raise KeyError('Wrong season_transition')

    def _inflate_rd(self, ratings: dict, rating_period_id=None):
        """RD growth before a rating period (in place): constant update_rd while rd < rd_threshold.
        rating_period_id is not used for clubs (constant growth per period), national teams grow RD by days."""

        for team, rating in ratings.items():
            if rating.rd < self.rd_threshold:
                ratings[team] = Rating(rating.mu, sqrt(rating.rd ** 2 + self.update_rd_squared))

    def _rate_teams_inside_period(self, ratings: dict, team_matches: dict, matches_params: dict) -> dict:
        """"""

        new_ratings = ratings.copy()

        for team, match_ids in team_matches.items():

            rating = ratings[team]
            rating = self._convert_into_glicko2(rating)

            d1_sum = 0
            d2_sum = 0
            for match_id in match_ids:

                info = self.match_info[(match_id, team)]

                params = matches_params[match_id]

                team_params = params.home_league if info.is_home else params.away_league
                opp_params = params.away_league if info.is_home else params.home_league

                team_league = info.home_team_league if info.is_home else info.away_team_league
                opp_league = info.away_team_league if info.is_home else info.home_team_league
                season = self.match_season[match_id]

                # double converting here but this case should work rarely
                if info.season_transition != 'same_season':
                    value = self._transition_value(ratings, team_league, season, team_params, info.season_transition)
                    rating = self._season_update_rating(self._convert_from_glicko2(rating), team_params, team_league,
                                                        info.season_transition, value)
                    rating = self._convert_into_glicko2(rating)

                opp_value = self._transition_value(ratings, opp_league, season, opp_params, info.opp_season_transition)
                opp_rating = self._season_update_rating(ratings[info.opp_team], opp_params, opp_league, info.opp_season_transition, opp_value)
                opp_rating = self._convert_into_glicko2(opp_rating)

                d1, d2 = self._d1_d2(rating, opp_rating, info.home_indicator, info.goals_expectations, params.home, info.outcome)

                d1_sum += d1
                d2_sum += d2

                # extra time is taken from data (AET / Pen), log-likelihoods of 90 minutes and extra time add up
                if info.is_extra_time:
                    d1, d2 = self._d1_d2(rating, opp_rating, info.home_indicator, info.goals_expectations_et,
                                         params.home_extra_time, info.extra_time_outcome)

                    d1_sum += d1
                    d2_sum += d2

            # step f
            new_rd = 1 / sqrt(1 / rating.rd ** 2 - d2_sum)

            new_mu = rating.mu + new_rd ** 2 * d1_sum

            # step g
            new_ratings[team] = self._convert_from_glicko2(Rating(new_mu, new_rd))

        return new_ratings

    def _loss_weight(self, info) -> float:
        """Weight of a match in the loss."""

        if info.is_domestic:
            return 1
        if info.is_world_cup:
            return self.world_cup_loss_weight
        return self.international_cup_loss_weight

    def _loss_inside_period(self, ratings: dict, match_ids: list, matches_params: dict) -> float:
        """Nested model: loss of 90 minutes outcome plus loss of extra time outcome if extra time was played."""

        new_ratings = ratings.copy()

        log_loss_value = 0
        for match_id in match_ids:
            info = self.home_away_match_info[match_id]
            params = matches_params[match_id]

            home_team = info.home_team
            away_team = info.away_team

            home_rating = new_ratings[home_team]
            away_rating = new_ratings[away_team]

            season = self.match_season[match_id]
            home_value = self._transition_value(ratings, info.home_team_league, season, params.home_league, info.home_season_transition)
            away_value = self._transition_value(ratings, info.away_team_league, season, params.away_league, info.away_season_transition)

            home_rating = self._season_update_rating(home_rating, params.home_league, info.home_team_league,
                                                     info.home_season_transition, home_value)
            away_rating = self._season_update_rating(away_rating, params.away_league, info.away_team_league,
                                                     info.away_season_transition, away_value)

            home_rating = self._convert_into_glicko2(home_rating)
            away_rating = self._convert_into_glicko2(away_rating)

            loss_weight = self._loss_weight(info)

            p_win, p_draw, p_lose = self.probabilities(home_rating.mu, away_rating.mu, info.home_indicator, info.goals_expectations, params.home)

            log_loss_value += loss_weight * three_outcomes_log_loss(p_win, p_draw, p_lose, info.outcome)

            if self.prediction_records is not None:
                self.prediction_records.append((match_id, p_win, p_draw, p_lose))

            if info.is_extra_time:
                p_win, p_draw, p_lose = self.probabilities(home_rating.mu, away_rating.mu, info.home_indicator, info.goals_expectations_et,
                                                           params.home_extra_time)

                extra_time_loss = loss_weight * three_outcomes_log_loss(p_win, p_draw, p_lose, info.extra_time_outcome)

                log_loss_value += extra_time_loss

                # accumulated separately to report the 90 minutes loss (comparable with models without extra time),
                # divided by 3 as the returned loss
                self.extra_time_loss_sum += extra_time_loss / 3

            # update home team
            d1, d2 = self._d1_d2(home_rating, away_rating, info.home_indicator, info.goals_expectations, params.home, info.outcome)

            if info.is_extra_time:
                d1_et, d2_et = self._d1_d2(home_rating, away_rating, info.home_indicator, info.goals_expectations_et,
                                           params.home_extra_time, info.extra_time_outcome)
                d1 += d1_et
                d2 += d2_et

            new_rd = 1 / sqrt(1 / home_rating.rd ** 2 - d2)
            new_mu = home_rating.mu + new_rd ** 2 * d1
            new_ratings[home_team] = self._convert_from_glicko2(Rating(new_mu, new_rd))

            # update away team
            d1, d2 = self._d1_d2(away_rating, home_rating, -info.home_indicator, info.goals_expectations, params.home, 1 - info.outcome)

            if info.is_extra_time:
                d1_et, d2_et = self._d1_d2(away_rating, home_rating, -info.home_indicator, info.goals_expectations_et,
                                           params.home_extra_time, 1 - info.extra_time_outcome)
                d1 += d1_et
                d2 += d2_et

            new_rd = 1 / sqrt(1 / away_rating.rd ** 2 - d2)
            new_mu = away_rating.mu + new_rd ** 2 * d1
            new_ratings[away_team] = self._convert_from_glicko2(Rating(new_mu, new_rd))

        return log_loss_value / 3

    def rate_teams(self, league_params: dict, matches_params: dict) -> dict:
        """Calculate team ratings."""

        ratings = self._rating_initialization(league_params)

        for rating_period_id, team_matches in self.period_team_matches.items():
            # update RD
            self._inflate_rd(ratings, rating_period_id)

            ratings = self._rate_teams_inside_period(ratings, team_matches, matches_params)

        return ratings

    def rate_teams_with_history(self, league_params: dict, matches_params: dict) -> tuple[dict, pd.DataFrame]:
        """Team ratings and their history: mu and rd of every team after every rating period it played in."""

        ratings = self._rating_initialization(league_params)

        history = []
        for rating_period_id, team_matches in self.period_team_matches.items():
            # update RD
            self._inflate_rd(ratings, rating_period_id)

            ratings = self._rate_teams_inside_period(ratings, team_matches, matches_params)

            history += [(rating_period_id, team, ratings[team].mu, ratings[team].rd) for team in team_matches]

        return ratings, pd.DataFrame(history, columns=['rating_period_id', 'team', 'rating', 'rd'])

    def _latest_championship_seasons(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Latest championship season of every league used for the ratings table.

        A season is accepted if any of the conditions holds:
            completed: last match is older than current_season_days before the end of the data
            first season of the league
            enough teams: n_teams >= min_active_teams_share * median over the previous recent_seasons seasons
            enough progress: median matches per team >= min_season_progress * median over the previous recent_seasons seasons
                             (a league that changed its size and has almost finished the season, e.g. Benin 2025)
        So a just started season (e.g. one match played) is skipped in favour of the previous one.

        Assumes season is already normalized, e.g. 2025, 2026.
        """

        championship_matches = matches.loc[matches['is_championship'] == True]

        team_season_matches = (championship_matches
                               .groupby(['tournament', 'season', 'team'])['match_id']
                               .nunique()
                               .rename('n_matches')
                               .reset_index())

        season_stats = (team_season_matches
                        .groupby(['tournament', 'season'])
                        .agg(n_teams=('team', 'nunique'), median_team_matches=('n_matches', 'median'))
                        .join(championship_matches.groupby(['tournament', 'season'])['date'].max().rename('last_date'))
                        .reset_index()
                        .sort_values(['tournament', 'season']))

        def previous_seasons_median(values: pd.Series) -> pd.Series:
            """Median over the previous recent_seasons seasons, the season itself is not included."""
            return values.shift().rolling(self.recent_seasons, min_periods=1).median()

        season_stats['typical_teams'] = season_stats.groupby('tournament')['n_teams'].transform(previous_seasons_median)
        season_stats['typical_team_matches'] = season_stats.groupby('tournament')['median_team_matches'].transform(previous_seasons_median)

        data_end_date = matches['date'].max()
        is_completed = season_stats['last_date'] < data_end_date - pd.Timedelta(days=self.current_season_days)

        is_first_season = season_stats['typical_teams'].isna()

        has_enough_teams = season_stats['n_teams'] >= self.min_active_teams_share * season_stats['typical_teams']
        has_enough_progress = (season_stats['median_team_matches']
                               >= self.min_season_progress * season_stats['typical_team_matches'])

        latest_league_seasons = (season_stats
                                 .loc[is_completed | is_first_season | has_enough_teams | has_enough_progress]
                                 .groupby('tournament', as_index=False)['season']
                                 .max())

        return latest_league_seasons

    def _active_team_leagues(self, matches: pd.DataFrame) -> dict:
        """Return {team: league} only for teams active in the latest season of each league."""

        latest_league_seasons = self._latest_championship_seasons(matches)

        active_championship_matches = (matches
                                       .loc[matches['is_championship'] == True]
                                       .merge(latest_league_seasons, on=['tournament', 'season'], how='inner', validate='m:1'))

        team_leagues = (active_championship_matches
                        .loc[:, ['team', 'tournament']]
                        .rename(columns={'tournament': 'league'})
                        .value_counts(['team', 'league'])
                        .reset_index(name='n_matches')
                        .sort_values(['team', 'n_matches'], ascending=[True, False])
                        .drop_duplicates('team', keep='first'))

        return dict(zip(team_leagues['team'], team_leagues['league']))

    def _last_known_team_leagues(self, matches: pd.DataFrame) -> dict:
        """Return {team: league} using the last known championship league."""

        team_leagues = dict()

        for season in self.seasons:
            team_leagues.update(self._team_leagues(matches, season))

        return team_leagues

    def ratings_to_df(self, ratings: dict, matches: pd.DataFrame, active_only: bool = True) -> pd.DataFrame:
        """Convert club ratings into a human-readable table."""

        if active_only:
            team_leagues = self._active_team_leagues(matches)
        else:
            team_leagues = self._last_known_team_leagues(matches)

        ratings_df = pd.DataFrame({'team': list(ratings),
                                   'rating': [rating.mu for rating in ratings.values()],
                                   'rd': [rating.rd for rating in ratings.values()]})

        ratings_df['league'] = ratings_df['team'].map(team_leagues)

        if active_only:
            ratings_df = ratings_df[ratings_df['league'].notna()]

        # Russia is not participating in international tournaments
        ratings_df = (ratings_df
                      .loc[~ratings_df['league'].str.contains(r'\bRussia\b', regex=True, na=False)]
                      .sort_values('rating', ascending=False)
                      .reset_index(drop=True))

        ratings_df.index = ratings_df.index + 1

        ratings_df = (ratings_df
                      .reset_index()
                      .rename(columns={'index': '#'}))

        return ratings_df

    def league_connectivity(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Connectivity of every league to the global rating graph over the last connectivity_seasons seasons:
            n_international: matches of the clubs of the league against foreign clubs
            n_cross_league: matches of the league against other leagues (domestic cups included)
            n_transitions: teams that came to the league from the other division of the country (promotion, relegation)
            comparable: the level of the league is identified by the data, it is linked to the other leagues
        The level of an isolated league is set only by its init_mu, so its place in the table is not comparable."""

        home = matches.loc[matches['is_home'] == True]

        rows = pd.concat([
            home[['match_id', 'season', 'home_team_league', 'away_team_league']]
            .rename(columns={'home_team_league': 'league', 'away_team_league': 'opp_league'}),
            home[['match_id', 'season', 'away_team_league', 'home_team_league']]
            .rename(columns={'away_team_league': 'league', 'home_team_league': 'opp_league'}),
        ])

        seasons = sorted(rows['season'].unique())[-self.connectivity_seasons:]
        rows = rows.loc[rows['season'].isin(seasons)]

        cross = rows.loc[rows['league'] != rows['opp_league']]
        foreign = cross.loc[cross['league'].map(self._country) != cross['opp_league'].map(self._country)]

        # a team in the league in a season and in the other division of the same country in the previous season
        transitions = []
        for season in seasons:
            previous_team_leagues = self.season_team_leagues.get(season - 1, dict())
            for team, league in self.season_team_leagues.get(season, dict()).items():
                previous_league = previous_team_leagues.get(team)
                if previous_league is not None and previous_league != league and self._country(previous_league) == self._country(league):
                    transitions.append(league)

        df = pd.DataFrame(index=pd.Index(sorted(set(matches['home_team_league']) | set(matches['away_team_league'])), name='league'))
        df['n_international'] = foreign.groupby('league')['match_id'].nunique().reindex(df.index).fillna(0).astype(int)
        df['n_cross_league'] = cross.groupby('league')['match_id'].nunique().reindex(df.index).fillna(0).astype(int)
        df['n_transitions'] = pd.Series(transitions, dtype=object).value_counts().reindex(df.index).fillna(0).astype(int)

        df['comparable'] = (df['n_cross_league'] > 0) | (df['n_transitions'] > 0)

        return df

    @staticmethod
    def league_ratings(ratings_df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
        """League rating: weighted mean of the top_n clubs (weights 1, 0.9, ..., 0.3) with an excessive-dispersion penalty.
        top_n = 8 is the size of the smallest league, so every league is measured by the same number of clubs;
        the decreasing weights: the level of a league is set first of all by its strongest clubs (the international cups).
        Leagues with less than top_n teams are kept (the mean over all teams), a warning is printed.
        top_5, top_7, top_10 and the ratings of the places (team_1, ..., team_10) are kept for comparison."""

        df = ratings_df.copy()
        df = df.sort_values(['league', 'rating'], ascending=[True, False])
        df['league_rank'] = df.groupby('league').cumcount() + 1

        grouped = df.groupby('league')

        top_5 = (df
                 .loc[df['league_rank'] <= 5]
                 .groupby('league')['rating']
                 .mean()
                 .rename('top_5'))

        top_7 = (df
                 .loc[df['league_rank'] <= 7]
                 .groupby('league')['rating']
                 .mean()
                 .rename('top_7'))

        top_10 = (df
                  .loc[df['league_rank'] <= 10]
                  .groupby('league')['rating']
                  .mean()
                  .rename('top_10'))

        std = grouped['rating'].std().rename('std')
        n_teams = grouped['rating'].size().rename('n_teams')

        # rating of the 1st, 2nd, ..., 10th team of the league (NaN if the league is smaller)
        team_places = (df
                       .loc[df['league_rank'] <= 10]
                       .pivot(index='league', columns='league_rank', values='rating'))
        team_places.columns = [f'team_{place}' for place in team_places.columns]

        weights = pd.Series([1 - 0.1 * place for place in range(top_n)], index=range(1, top_n + 1))

        top_weighted = (df
                        .loc[df['league_rank'] <= top_n]
                        .assign(weight=lambda d: d['league_rank'].map(weights))
                        .assign(weighted_rating=lambda d: d['rating'] * d['weight'])
                        .groupby('league')[['weighted_rating', 'weight']]
                        .sum()
                        .pipe(lambda d: d['weighted_rating'] / d['weight'])
                        .rename(f'top_{top_n}_weighted'))

        leagues_ratings = pd.concat([top_weighted, top_5, top_7, top_10, std, n_teams, team_places], axis=1).reset_index()

        small_leagues = leagues_ratings.loc[leagues_ratings['n_teams'] < top_n, ['league', 'n_teams']]
        if not small_leagues.empty:
            print(f'Leagues with less than {top_n} teams (the mean over all teams):')
            print(small_leagues.to_string(index=False))

        std_penalty = np.maximum(leagues_ratings['std'].fillna(0) - 120, 0)

        leagues_ratings['league_score'] = leagues_ratings[f'top_{top_n}_weighted'] - 0.15 * std_penalty

        # league_score (with the dispersion penalty) is kept for comparison, the order is by top_n_weighted
        leagues_ratings = (leagues_ratings
                           .sort_values([f'top_{top_n}_weighted', 'top_5'], ascending=False)
                           .reset_index(drop=True))

        leagues_ratings.insert(0, '#', np.arange(1, len(leagues_ratings) + 1))

        rating_cols = ['league_score', f'top_{top_n}_weighted', 'top_5', 'top_7', 'top_10', 'std']
        leagues_ratings[rating_cols] = leagues_ratings[rating_cols].round().astype(int)

        place_cols = list(team_places.columns)
        leagues_ratings[place_cols] = leagues_ratings[place_cols].round().astype('Int64')

        return leagues_ratings

    def add_predictions(self, matches: pd.DataFrame, league_params: dict, matches_params: dict) -> pd.DataFrame:
        """Add pre-match ratings and outcome probabilities to every row of matches (team / opp_team orientation).

        Ratings are taken before the rating period of the match (same batch scheme as rate_teams),
        after the season transition update, so the outcome of the match (and of other matches of the period)
        does not leak into its prediction.

        Home-oriented probabilities are computed once per match and mirrored for the away row:
            p_team_win / p_draw / p_team_lose - 90 minutes
            p_et_team_win / p_et_draw / p_et_team_lose - extra time (NaN if extra time was not played)

        matches must be the same data that was passed to _precalculate_info."""

        ratings = self._rating_initialization(league_params)

        records = []
        for rating_period_id, team_matches in self.period_team_matches.items():

            # update RD
            self._inflate_rd(ratings, rating_period_id)

            # predictions with ratings before the period
            for match_id in self.period_matches[rating_period_id]:
                info = self.home_away_match_info[match_id]
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

                home_g2 = self._convert_into_glicko2(home_rating)
                away_g2 = self._convert_into_glicko2(away_rating)

                p_win, p_draw, p_lose = self.probabilities(home_g2.mu, away_g2.mu, info.home_indicator,
                                                           info.goals_expectations, params.home)

                if info.is_extra_time:
                    p_et_win, p_et_draw, p_et_lose = self.probabilities(home_g2.mu, away_g2.mu, info.home_indicator,
                                                                        info.goals_expectations_et, params.home_extra_time)
                else:
                    p_et_win = p_et_draw = p_et_lose = np.nan

                records.append((match_id,
                                home_rating.mu, home_rating.rd, away_rating.mu, away_rating.rd,
                                p_win, p_draw, p_lose,
                                p_et_win, p_et_draw, p_et_lose))

            ratings = self._rate_teams_inside_period(ratings, team_matches, matches_params)

        predictions = pd.DataFrame(records, columns=['match_id',
                                                     'home_rating', 'home_rd', 'away_rating', 'away_rd',
                                                     'p_home_win', 'p_draw', 'p_away_win',
                                                     'p_et_home_win', 'p_et_draw', 'p_et_away_win'])

        result = matches.merge(predictions, on='match_id', how='left', validate='many_to_one')

        # orientation of the row: team / opp_team
        is_home = result['is_home'].values == True

        result['team_rating'] = np.where(is_home, result['home_rating'], result['away_rating'])
        result['team_rd'] = np.where(is_home, result['home_rd'], result['away_rd'])
        result['opp_rating'] = np.where(is_home, result['away_rating'], result['home_rating'])
        result['opp_rd'] = np.where(is_home, result['away_rd'], result['home_rd'])

        result['p_team_win'] = np.where(is_home, result['p_home_win'], result['p_away_win'])
        result['p_team_lose'] = np.where(is_home, result['p_away_win'], result['p_home_win'])

        result['p_et_team_win'] = np.where(is_home, result['p_et_home_win'], result['p_et_away_win'])
        result['p_et_team_lose'] = np.where(is_home, result['p_et_away_win'], result['p_et_home_win'])

        result = result.drop(columns=['home_rating', 'home_rd', 'away_rating', 'away_rd',
                                      'p_home_win', 'p_away_win', 'p_et_home_win', 'p_et_away_win'])

        # final ratings must be the same as in rate_teams
        return result

    def transform(self):
        """"""

        matches = pd.read_feather(config.soccer['glicko_matches'])

        self._precalculate_info(matches)

        league_params = self.load_params(config.soccer['params']['league_params'])

        # leagues that appeared after the train seasons
        league_params = self._fill_missing_leagues(league_params)

        # extra time params are loaded in __init__ (defaults if not fitted yet)
        # ms_lift_update_mu of second leagues is tied to the first league inside _match_params
        matches_params = self._match_params(league_params)

        ratings, history = self.rate_teams_with_history(league_params, matches_params)
        history.to_feather(config.soccer['rating_history'])

        ratings_df = self.ratings_to_df(ratings, matches)

        print(ratings_df.head(100))

        ratings_df.to_csv('club_ratings.csv', index=False)

        leagues_ratings = self.league_ratings(ratings_df)

        leagues_ratings = (leagues_ratings
                           .sort_values('top_8_weighted', ascending=False)
                           .reset_index(drop=True))

        leagues_ratings['#'] = np.arange(1, len(leagues_ratings) + 1)

        connectivity = self.league_connectivity(matches)
        leagues_ratings = leagues_ratings.merge(connectivity, left_on='league', right_index=True, how='left')

        place_cols = [f'team_{place}' for place in range(1, 11)]
        print(leagues_ratings[['#', 'league', 'league_score', 'top_8_weighted', 'top_7', 'n_teams', 'comparable'] + place_cols]
              .to_string(index=False))
        leagues_ratings.to_csv('league_ratings.csv', index=False)

        not_comparable = leagues_ratings.loc[leagues_ratings['comparable'] != True]
        print(f'\nLeagues with a level not identified by the data: {len(not_comparable)}')
        print(not_comparable[['#', 'league', 'league_score', 'n_international', 'n_cross_league', 'n_transitions']].to_string(index=False))


if __name__ == '__main__':
    Ratings().transform()
