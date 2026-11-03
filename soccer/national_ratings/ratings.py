from collections import namedtuple
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from config import config
from soccer.club_ratings.ratings import Ratings, MatchParams
from three_outcome_models.soccer_glicko import Rating
from utils.functions import load_pickle


class NationalRatings(Ratings):
    """Ratings of national teams. The club core (_rate_teams_inside_period, _loss_inside_period, add_predictions,
    nested extra time model) is reused: in national_glicko_matches the "league" of a team is the team itself
    and all season transitions are 'same_season' (see national_ratings/preprocessing.py).

    Params hierarchy: global -> confederation -> country, effective params of a team are
        {**global_params, **confederation_params[confederation], **country_params[team]}
    Only culture-dependent params are per country (crowd, logistic, init_mu, maybe draw_frequency),
    the rest are global or per confederation. L2 shrinkage to the upper level is in the training.

    Home advantage (see _single_match_params):
        regular stage: crowd + logistic + strength * avg_mu + scoring * ge of the team playing at home
        final stage without host: 0 (neutral venue)
        final stage with host: A_h - A_a, A_x = H_x * (gamma * (crowd_x + logistic_x) + strength_x * avg_mu + scoring_x * ge)
    gamma is global (final_stage_params): the share of the usual home advantage that the host of a final stage gets.
    Before 28.10: alpha * (crowd_h - crowd_a) without host and alpha * host * crowd + beta * logistic with host,
    the three params were not identified (crowd is almost the same for all countries, host and beta replace each other).

    RD grows with time: rd^2 + rd_growth^2 * period_days / 30 before each period, while rd < rd_threshold.
    """

    def __init__(self, init_mu=1500, init_rd=250, rd_growth=22, rd_threshold=200, max_inactive_days=3 * 365):
        super().__init__(init_mu=init_mu, init_rd=init_rd, update_rd=rd_growth, rd_threshold=rd_threshold)

        # RD growth per 30 days (rating points), rd_growth ** 2 * days / 30 is added to rd ** 2
        self.rd_growth_squared = rd_growth ** 2

        # teams without matches for longer are not in the ratings table (e.g. banned or inactive teams)
        self.max_inactive_days = max_inactive_days

        # params of all teams unless overridden by the confederation or the country
        self.init_global_params = {'init_mu': self.init_mu,

                                   'crowd': 0.127,
                                   'logistic': 0.127,
                                   'strength': 0.001,
                                   'scoring': 0,

                                   'draw_frequency': -0.277,
                                   'strength_overall': 0.995,
                                   'scoring_overall': 0}

        # final stage home advantage, global
        # self.init_final_stage_params = {'alpha': 0.5,
        #                                 'host': 1.,
        #                                 'beta': 0.5}
        self.init_final_stage_params = {'gamma': 1.}

        # loss weights by match_type (see preprocessing)
        self.match_type_loss_weights = {'world_cup_final_stage': 3,
                                        'final_stage': 2,
                                        'nations_league': 1,
                                        'qualification': 1}

        self.national_params_paths = config.soccer['national_params']

        # own extra time params: the club ones are loaded in Ratings.__init__
        extra_time_params_path = Path(self.national_params_paths['extra_time'])
        if extra_time_params_path.exists():
            self.extra_time_params = load_pickle(extra_time_params_path)
        else:
            self.extra_time_params = self.init_extra_time_params.copy()

        final_stage_params_path = Path(self.national_params_paths['final_stage'])
        if final_stage_params_path.exists():
            self.final_stage_params = load_pickle(final_stage_params_path)
        else:
            self.final_stage_params = self.init_final_stage_params.copy()

        # the old format (alpha, host, beta) before 28.10: gamma starts from the default
        if 'gamma' not in self.final_stage_params:
            self.final_stage_params = self.init_final_stage_params.copy()

        self.team_confederation = None
        self.period_days = None
        self.team_last_date = None
        self.country_match_ids = None

    def load_params(self) -> tuple[dict, dict, dict]:
        """(global_params, confederation_params, country_params), defaults if not fitted yet."""

        def load(name: str, default):
            path = Path(self.national_params_paths[name])
            return load_pickle(path) if path.exists() else default

        global_params = dict(load('global', self.init_global_params.copy()))

        # rd_growth is a hyperparam of the RD growth, stored with the global params (fitted in TrainNationalGlicko)
        if 'rd_growth' in global_params:
            self.rd_growth_squared = global_params.pop('rd_growth') ** 2

        confederation_params = load('confederation', dict())
        country_params = load('country', dict())

        # country params are offsets to the confederation level, an absolute init_mu means the old format
        old_format = [team for team, params in country_params.items() if abs(params.get('init_mu', 0.)) > 1000]
        assert not old_format, f'country_params contain absolute values (old format), delete ' \
                               f'{self.national_params_paths["country"]}: {old_format[:5]}'

        return global_params, confederation_params, country_params

    def team_params(self, global_params: dict, confederation_params: dict, country_params: dict) -> dict:
        """{team: effective params}. The confederation overrides the global params (absolute values),
        a country adds its offsets to the confederation level: country_params = {team: {param: offset}},
        a missing offset is 0 (a team without fitted country params gets the confederation params).
        So a shift of the confederation level shifts all its countries, a country keeps only its difference."""

        team_params = dict()
        for team, confederation in self.team_confederation.items():
            parent = {**global_params, **confederation_params.get(confederation, dict())}
            offsets = country_params.get(team, dict())
            team_params[team] = {name: value + offsets.get(name, 0.) for name, value in parent.items()}

        return team_params

    def _loss_weight(self, info) -> float:
        """Weight of a match in the loss by match_type."""

        return self.match_type_loss_weights[info.match_type]

    def _precalculate_info(self, matches: pd.DataFrame):
        """Cache match-level attributes (national_glicko_matches) for fast lookup during ratings computation.
        Field names of MatchInfo / HomeAwayMatchInfo are the ones the club core uses."""

        self.rating_periods = matches['rating_period_id'].max()

        # club season logic is not used: no leagues, no newcomers, no lift
        self.seasons = sorted(set(matches['season']))
        self.first_leagues = set()
        self.second_leagues = set()
        self.first_league_of_country = dict()
        self.second_leagues_dict = dict()

        home_matches = matches.loc[matches['is_home'] == True]
        assert home_matches['match_id'].is_unique, 'each match must have exactly one home row'

        self.team_confederation = dict(zip(matches['team'], matches['team_confederation']))

        self.team_last_date = matches.groupby('team')['date'].max().to_dict()

        self.match_season = dict(zip(home_matches['match_id'].values, home_matches['season'].values))

        self.period_days = (matches
                            .drop_duplicates('rating_period_id')
                            .set_index('rating_period_id')['period_days']
                            .to_dict())

        MatchInfo = namedtuple('MatchInfo', ['opp_team', 'season_transition', 'opp_season_transition',
                                             'is_home', 'home_indicator', 'goals_expectations', 'goals_expectations_et',
                                             'outcome', 'is_extra_time', 'extra_time_outcome',
                                             'home_team_league', 'away_team_league'])

        self.match_info = {
            (match_id, team): MatchInfo(
                opp_team, season_transition, opp_season_transition,
                is_home, home_indicator, goals_expectations, goals_expectations_et,
                outcome, is_extra_time, extra_time_outcome,
                home_team_league, away_team_league
            )
            for (match_id, team, opp_team, season_transition, opp_season_transition,
                 is_home, home_indicator, goals_expectations, goals_expectations_et,
                 outcome, is_extra_time, extra_time_outcome,
                 home_team_league, away_team_league) in zip(
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
                matches['home_team_league'].values,
                matches['away_team_league'].values
            )
        }

        HomeAwayMatchInfo = namedtuple('HomeAwayMatchInfo', ['home_team', 'away_team',
                                                             'home_season_transition', 'away_season_transition',
                                                             'home_indicator', 'match_type',
                                                             'stage', 'home_is_host', 'away_is_host',
                                                             'goals_expectations', 'goals_expectations_et',
                                                             'outcome', 'is_extra_time', 'extra_time_outcome',
                                                             'status', 'home_team_league', 'away_team_league'])

        self.home_away_match_info = {
            match_id: HomeAwayMatchInfo(
                home_team, away_team,
                home_season_transition, away_season_transition,
                home_indicator, match_type,
                stage, home_is_host, away_is_host,
                goals_expectations, goals_expectations_et,
                outcome, is_extra_time, extra_time_outcome,
                status, home_team_league, away_team_league
            )
            for (match_id, home_team, away_team,
                 home_season_transition, away_season_transition,
                 home_indicator, match_type,
                 stage, home_is_host, away_is_host,
                 goals_expectations, goals_expectations_et,
                 outcome, is_extra_time, extra_time_outcome,
                 status, home_team_league, away_team_league) in zip(
                home_matches['match_id'].values,
                home_matches['team'].values,
                home_matches['opp_team'].values,
                home_matches['season_transition'].values,
                home_matches['opp_season_transition'].values,
                home_matches['home_indicator'].values,
                home_matches['match_type'].values,
                home_matches['stage'].values,
                home_matches['home_is_host'].values,
                home_matches['away_is_host'].values,
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

        # matches of each country, for incremental update of match params
        self.country_match_ids = dict()
        for match_id, info in self.home_away_match_info.items():
            self.country_match_ids.setdefault(info.home_team, []).append(match_id)
            self.country_match_ids.setdefault(info.away_team, []).append(match_id)

        self.period_team_matches = {
            rating_period_id: {
                team: group['match_id'].values
                for team, group in period_df.groupby('team', sort=False)
            }
            for rating_period_id, period_df in (matches
                                                .loc[:, ['date', 'rating_period_id', 'team', 'match_id']]
                                                .sort_values(['rating_period_id', 'date'])
                                                .groupby('rating_period_id', sort=True))
        }

        self.period_matches = {
            rating_period_id: period_df['match_id'].to_list()
            for rating_period_id, period_df in (matches
                                                .loc[:, ['date', 'rating_period_id', 'match_id']]
                                                .sort_values(['rating_period_id', 'date'])
                                                .drop_duplicates('match_id')
                                                .groupby('rating_period_id', sort=True))
        }

    def _single_match_params(self, home_params: dict, away_params: dict, status: str, is_extra_time: bool,
                             extra_time_params: dict, stage: str = 'regular', home_indicator: int = 1,
                             home_is_host: bool = False, away_is_host: bool = False,
                             final_stage_params: dict = None):
        """Params of one match.
        stage:
            'regular': qualifiers, Nations League. Advantage params of the team that plays at home
                       (away params if the host is listed away, home_indicator = -1 turns the sign)
            'final': final stage, home_indicator = 1 (preprocessing), the signed advantage is inside the params:
                     without host: 0
                     with host: A_h - A_a, A_x = H_x * (gamma * (crowd_x + logistic_x) + strength_x * avg_mu + scoring_x * ge)
        advantage = home_indicator * (crowd + logistic + strength * avg_mu + scoring * ge) is linear in the params,
        so SoccerGlicko.probabilities and _d1_d2 compute the difference exactly.
        Draw params: of the home team for regular non-neutral matches, averaged over both countries otherwise."""

        if final_stage_params is None:
            final_stage_params = self.final_stage_params

        adjusted_home = self._status_adjusted_params(home_params, status)
        adjusted_away = self._status_adjusted_params(away_params, status)

        is_regular_home_away = stage == 'regular' and status != 'NEUTRAL'

        if is_regular_home_away:
            draw_params = home_params if home_indicator >= 0 else away_params
        else:
            draw_params = {param: (home_params[param] + away_params[param]) / 2 for param in self.neutral_averaged_params}

        if stage == 'regular':
            if is_regular_home_away:
                advantage_params = adjusted_home if home_indicator >= 0 else adjusted_away
                advantage = {param: advantage_params[param] for param in ('crowd', 'logistic', 'strength', 'scoring')}
            else:
                advantage = {'crowd': 0, 'logistic': 0, 'strength': 0, 'scoring': 0}

        elif not (home_is_host or away_is_host):
            # a neutral venue: crowd is almost the same for all countries, alpha * (crowd_h - crowd_a) was ~0 (28.10)
            advantage = {'crowd': 0, 'logistic': 0, 'strength': 0, 'scoring': 0}

        else:
            gamma = final_stage_params['gamma']
            h_home, h_away = float(home_is_host), float(away_is_host)

            advantage = {'crowd': gamma * (h_home * adjusted_home['crowd'] - h_away * adjusted_away['crowd']),
                         'logistic': gamma * (h_home * adjusted_home['logistic'] - h_away * adjusted_away['logistic']),
                         'strength': h_home * adjusted_home['strength'] - h_away * adjusted_away['strength'],
                         'scoring': h_home * adjusted_home['scoring'] - h_away * adjusted_away['scoring']}

        match_home = {**advantage, **{param: draw_params[param] for param in self.neutral_averaged_params}}

        if is_extra_time:
            home_advantage_scale = extra_time_params['home_advantage_scale']
            home_extra_time = {'kappa': extra_time_params['kappa'],
                               **{param: home_advantage_scale * value for param, value in advantage.items()},
                               'draw_frequency': extra_time_params['draw_frequency'],
                               'strength_overall': extra_time_params['strength_overall'],
                               'scoring_overall': extra_time_params['scoring_overall']}
        else:
            home_extra_time = None

        return MatchParams(home=match_home, home_league=home_params, away_league=away_params,
                           home_extra_time=home_extra_time)

    def _match_params(self, team_params: dict, extra_time_params: dict = None, final_stage_params: dict = None,
                      matches_params: dict = None, changed_teams: set = None) -> dict:
        """Map each match to its params (see _single_match_params). team_params: {team: effective params}.
        Incremental mode: if matches_params and changed_teams are passed, only matches of changed_teams are rebuilt.
        Matches with the same key share one MatchParams object, params dicts must not be mutated afterwards."""

        if extra_time_params is None:
            extra_time_params = self.extra_time_params

        if final_stage_params is None:
            final_stage_params = self.final_stage_params

        if matches_params is None or changed_teams is None:
            match_ids = self.home_away_match_info.keys()
            new_matches_params = dict()
        else:
            match_ids = {match_id for team in changed_teams for match_id in self.country_match_ids.get(team, ())}
            new_matches_params = matches_params.copy()

        cache = dict()
        for match_id in match_ids:
            info = self.home_away_match_info[match_id]

            key = (info.home_team, info.away_team, info.status, info.is_extra_time,
                   info.stage, info.home_indicator, info.home_is_host, info.away_is_host)

            params = cache.get(key)
            if params is None:
                params = self._single_match_params(team_params[info.home_team], team_params[info.away_team],
                                                   info.status, info.is_extra_time, extra_time_params,
                                                   info.stage, info.home_indicator, info.home_is_host, info.away_is_host,
                                                   final_stage_params)
                cache[key] = params

            new_matches_params[match_id] = params

        return new_matches_params

    def _rating_initialization(self, team_params: dict) -> dict:
        """Every team starts with init_mu of its effective params (confederation level, country if fitted)."""

        self._anchor_cache = dict()

        return {team: Rating(mu=params['init_mu'], rd=self.init_rd) for team, params in team_params.items()}

    def _inflate_rd(self, ratings: dict, rating_period_id=None):
        """RD growth before a rating period (in place), proportional to the days since the previous period:
        rd^2 + rd_growth^2 * period_days / 30 while rd < rd_threshold."""

        # the base loops must pass the period (Ratings.rate_teams, add_predictions, TrainGlicko losses)
        assert rating_period_id is not None, '_inflate_rd needs rating_period_id for national teams'

        days = self.period_days[rating_period_id]
        if days <= 0:
            return

        growth_squared = self.rd_growth_squared * days / 30
        for team, rating in ratings.items():
            if rating.rd < self.rd_threshold:
                ratings[team] = Rating(rating.mu, sqrt(rating.rd ** 2 + growth_squared))

    def ratings_to_df(self, ratings: dict, matches: pd.DataFrame = None, active_only: bool = True) -> pd.DataFrame:
        """National team ratings table: active teams (a match in the last max_inactive_days days of the data)."""

        data_end_date = max(self.team_last_date.values())

        df = pd.DataFrame([(team, rating.mu, rating.rd, self.team_confederation.get(team), self.team_last_date.get(team))
                           for team, rating in ratings.items()],
                          columns=['team', 'rating', 'rd', 'confederation', 'last_match'])

        if active_only:
            df = df[df['last_match'] >= data_end_date - pd.Timedelta(days=self.max_inactive_days)]

        df = df.sort_values('rating', ascending=False).reset_index(drop=True)
        df.insert(0, '#', np.arange(1, len(df) + 1))

        df['rating'] = df['rating'].round().astype(int)
        df['rd'] = df['rd'].round().astype(int)

        return df

    def transform(self):
        """"""

        matches = pd.read_feather(config.soccer['national_glicko_matches'])

        self._precalculate_info(matches)

        global_params, confederation_params, country_params = self.load_params()
        team_params = self.team_params(global_params, confederation_params, country_params)

        matches_params = self._match_params(team_params)

        ratings = self.rate_teams(team_params, matches_params)

        ratings_df = self.ratings_to_df(ratings)

        print(ratings_df.head(100).to_string(index=False))

        ratings_df.to_csv('national_ratings.csv', index=False)


if __name__ == '__main__':
    NationalRatings().transform()