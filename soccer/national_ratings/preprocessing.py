import numpy as np
import pandas as pd

from config import config
from utils.functions import save_feather


class NationalPreprocessing(object):
    """Setup the national teams dataset for ratings calculation.

    The output has the same columns as the club glicko_matches, so the club ratings core
    (_rate_teams_inside_period, _loss_inside_period, add_predictions) works without changes:
        home_team_league / away_team_league = the team itself (params are per country)
        season_transition / opp_season_transition = 'same_season' (no seasons for national teams)
    plus national columns: stage, match_type, home_is_host, away_is_host, confederations, period_days.

    home_indicator:
        regular stage: 1 / 0 / -1 as in the data (who plays at home)
        final stage: always 1, the signed home advantage is inside the match params (see NationalRatings)
    """

    def __init__(self):
        # goals expectations are clipped at this quantile of the match total (see _add_goals_expectations)
        self.goals_expectations_clip_quantile = 0.999

        # extra time goals relative to regular time, the same value as Features.extra_time_goals_ratio
        self.extra_time_goals_ratio = 0.25

        self.input_goals_expectations_columns = ['match_id', 'home_goals_expectations', 'away_goals_expectations']

        self.input_match_columns = ['match_id', 'source', 'country', 'tournament', 'season', 'date', 'home_team', 'away_team',
                                    'home_score', 'away_score',
                                    'outcome_regular_time', 'outcome_5', 'is_extra_time',
                                    'is_no_spectators', 'is_different_stadium', 'is_limited_spectators',
                                    'home_indicator', 'is_final_stage', 'home_team_is_host', 'away_team_is_host']

        self.output_match_columns = ['match_id', 'team', 'outcome', 'opp_team', 'tournament', 'season', 'date',
                                     'stage', 'match_type',
                                     'goals_expectations', 'rating_period_id', 'period_days',
                                     'season_transition', 'opp_season_transition',
                                     'status', 'home_indicator', 'home_is_host', 'away_is_host',
                                     'home_team_league', 'away_team_league',
                                     'team_confederation', 'opp_confederation',
                                     'is_extra_time', 'extra_time_outcome', 'goals_expectations_et', 'is_home']

    @staticmethod
    def _team_confederations(matches: pd.DataFrame) -> dict:
        """{team: confederation}: the most frequent region (country column) of the team matches except 'World'
        (World Cup final stages). Ties: alphabetical, deterministic."""

        regional = matches[matches['country'] != 'World']

        team_regions = pd.concat([regional[['home_team', 'country']].rename(columns={'home_team': 'team'}),
                                  regional[['away_team', 'country']].rename(columns={'away_team': 'team'})],
                                 ignore_index=True)

        confederations = (team_regions
                          .value_counts(['team', 'country'])
                          .rename('n_matches')
                          .reset_index()
                          .sort_values(['team', 'n_matches', 'country'], ascending=[True, False, True])
                          .drop_duplicates('team', keep='first'))

        confederations = dict(zip(confederations['team'], confederations['country']))

        teams = set(matches['home_team']) | set(matches['away_team'])
        without_confederation = sorted(teams - set(confederations))
        assert not without_confederation, f'teams only in World matches: {without_confederation}'

        return confederations

    @staticmethod
    def _nations_league_finals(matches: pd.DataFrame) -> pd.DataFrame:
        """Finals of the Nations Leagues (UEFA, CONCACAF) are not final stages: a mini tournament of 4 teams,
        half friendly, the host has no tournament advantage. They are regular matches: the host plays at home
        (home_indicator 1 / -1 as in the data), the others at a neutral venue (0).
        The relegation playoffs of the lower leagues (Gibraltar - Latvia, two legs) are marked as the final stage
        with home_indicator 0 in the data: a pair that plays twice in an edition is home and away (1 for the listed home)."""

        is_nations_league_final = (matches['is_final_stage'] == True) & matches['tournament'].str.contains('Nations League', na=False)

        # the pair regardless of the order of the teams
        pair = np.minimum(matches['home_team'], matches['away_team']) + '|' + np.maximum(matches['home_team'], matches['away_team'])
        # the matches of the pair in the finals of the edition (the league phase is not counted)
        n_pair_matches = (matches
                          .loc[is_nations_league_final]
                          .groupby(['tournament', 'season', pair])['match_id']
                          .transform('size')
                          .reindex(matches.index, fill_value=0))

        is_two_legs = n_pair_matches > 1

        matches.loc[is_two_legs & (matches['home_indicator'] == 0), 'home_indicator'] = 1
        matches.loc[is_nations_league_final, 'is_final_stage'] = False

        print(f'#Nations League finals as regular matches: {is_nations_league_final.sum()}, two legs: {is_two_legs.sum()}')

        return matches

    @staticmethod
    def _stages(matches: pd.DataFrame) -> pd.DataFrame:
        """stage, host by stadium, match_type.

        Host by stadium: home_team_is_host means the team is one of the tournament hosts,
        home_indicator knows the stadium (FINAL_STAGE_HOME_INDICATOR_OVERRIDES for multi-host tournaments),
        so a host playing in another host country (Euro 2020) is not a host of the match.

        match_type (loss weights): world_cup_final_stage, final_stage, nations_league, qualification.
        """

        is_final = matches['is_final_stage'] == True

        matches['stage'] = np.where(is_final, 'final', 'regular')

        matches['home_is_host'] = is_final & (matches['home_team_is_host'] == True) & (matches['home_indicator'] == 1)
        matches['away_is_host'] = is_final & (matches['away_team_is_host'] == True) & (matches['home_indicator'] == -1)

        masks = [is_final & (matches['country'] == 'World'),
                 is_final,
                 matches['tournament'].str.contains('Nations League', na=False)]

        matches['match_type'] = np.select(masks, ['world_cup_final_stage', 'final_stage', 'nations_league'], default='qualification')

        return matches

    def _add_goals_expectations(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Goals expectations of the lgbm trained on clubs and national teams (Features, one match_id).
        Clipped and centered over national team matches only: the level of scoring differs from clubs,
        a club mean would leave a shift that scoring_overall takes as a second intercept."""

        goals_expectations = pd.read_feather(config.soccer['outcomes']['lgbm_predictions'], columns=self.input_goals_expectations_columns)

        goals_expectations = goals_expectations[goals_expectations['match_id'].isin(matches['match_id'])]

        goals_expectations['goals_expectations'] = goals_expectations['home_goals_expectations'] + goals_expectations['away_goals_expectations']

        # for numerical stability
        upper = goals_expectations['goals_expectations'].quantile(self.goals_expectations_clip_quantile)
        goals_expectations['goals_expectations'] = goals_expectations['goals_expectations'].clip(upper=upper)

        goals_expectations['goals_expectations'] -= goals_expectations['goals_expectations'].mean()

        goals_expectations = dict(zip(goals_expectations['match_id'], goals_expectations['goals_expectations']))

        matches['goals_expectations'] = matches['match_id'].map(goals_expectations)

        # extra time: the same scale shrunk by the duration, centered over extra time matches (see ClubPreprocessing)
        matches['goals_expectations_et'] = self.extra_time_goals_ratio * matches['goals_expectations']
        extra_time_mean = matches.loc[matches['is_extra_time'], 'goals_expectations_et'].mean()
        matches['goals_expectations_et'] -= extra_time_mean

        n_missing = matches['goals_expectations'].isna().sum()
        assert n_missing == 0, f'{n_missing} matches without goals expectations'

        return matches

    def _rating_periods(self, matches: pd.DataFrame) -> pd.DataFrame:
        """rating_period_id and period_days. Rating periods are the calendar quarters, as for the clubs:
        a final stage is inside one or two periods.

        period_days: days from the start of the previous period, RD grows by c^2 * period_days / 30 before the period
        (the sum over periods is the elapsed time: national teams do not play in every quarter)."""

        match_day = matches['date'].dt.normalize()
        matches['period_key'] = match_day.dt.year.astype(str) + '|' + match_day.dt.quarter.astype(str)

        return self._number_periods(matches, match_day)

    @staticmethod
    def _number_periods(matches: pd.DataFrame, match_day: pd.Series) -> pd.DataFrame:
        """rating_period_id and period_days by period_key (a label of the period: "2024|3" for a quarter), sorted by the start date."""

        periods = (matches
                   .assign(match_day=match_day)
                   .groupby('period_key')['match_day']
                   .min()
                   .rename('start_date')
                   .reset_index()
                   .sort_values(['start_date', 'period_key'])
                   .reset_index(drop=True))

        periods['rating_period_id'] = np.arange(len(periods))
        periods['period_days'] = periods['start_date'].diff().dt.days.fillna(0).astype(int)

        matches = matches.merge(periods[['period_key', 'rating_period_id', 'period_days']], on='period_key', how='left', validate='m:1')

        return matches.drop(columns='period_key')

    @staticmethod
    def _status(matches: pd.DataFrame) -> pd.DataFrame:
        """Crowd and logistic status as in ClubPreprocessing._home_field_advantage.
        Regular stage: neutral venue (home_indicator = 0) -> NEUTRAL.
        Final stage: the status is taken from the spectators flags even at a neutral venue,
        the crowd of the host enters the home advantage (gamma * (crowd + logistic)),
        so limited spectators (Euro 2020) must halve it. Then home_indicator = 1 for all final stage matches."""

        is_regular_neutral = (matches['stage'] == 'regular') & (matches['home_indicator'] == 0)

        is_full_crowd = ((matches['is_no_spectators'] == False)
                         & (matches['is_limited_spectators'] == False)
                         & (matches['is_different_stadium'] == False))
        is_half_crowd = (matches['is_limited_spectators'] == True) | (matches['is_different_stadium'] == True)
        is_zero_crowd = matches['is_no_spectators'] == True

        is_full_logistic = matches['is_different_stadium'] == False
        is_zero_logistic = matches['is_different_stadium'] == True

        masks = [is_regular_neutral,

                 is_full_crowd & is_full_logistic,
                 is_half_crowd & is_full_logistic,
                 is_zero_crowd & is_full_logistic,

                 is_full_crowd & is_zero_logistic,
                 is_half_crowd & is_zero_logistic,
                 is_zero_crowd & is_zero_logistic]

        values = ['NEUTRAL',

                  'FULL_CROWD_FULL_LOGISTIC',
                  'HALF_CROWD_FULL_LOGISTIC',
                  'ZERO_CROWD_FULL_LOGISTIC',

                  'FULL_CROWD_ZERO_LOGISTIC',
                  'HALF_CROWD_ZERO_LOGISTIC',
                  'ZERO_CROWD_ZERO_LOGISTIC']

        matches['status'] = np.select(masks, values, default='bug')
        assert (matches['status'] == 'bug').sum() == 0, 'bugs in match status'

        # final stage: the sign of the advantage is inside the match params
        matches['home_indicator'] = np.where(matches['stage'] == 'final', 1, matches['home_indicator']).astype('int8')

        matches = matches.drop(columns=['is_no_spectators', 'is_limited_spectators', 'is_different_stadium'])

        return matches

    @staticmethod
    def _transform_data(matches: pd.DataFrame, confederations: dict) -> pd.DataFrame:
        """Two rows per match (team / opp_team), as in ClubPreprocessing._transform_data.
        home_indicator is the home team orientation: the away row gets -home_indicator in the ratings (as for clubs)."""

        matches['home_team_league'] = matches['home_team']
        matches['away_team_league'] = matches['away_team']

        matches_info = matches.loc[:, ['match_id', 'tournament', 'season', 'date', 'stage', 'match_type',
                                       'goals_expectations', 'goals_expectations_et', 'rating_period_id', 'period_days',
                                       'status', 'home_indicator', 'home_is_host', 'away_is_host',
                                       'home_team_league', 'away_team_league', 'is_extra_time']]

        home_team = (matches
                     .loc[:, ['match_id', 'home_team', 'outcome_regular_time', 'outcome_5', 'away_team']]
                     .rename(columns={'home_team': 'team', 'away_team': 'opp_team'}))

        away_team = (matches
                     .loc[:, ['match_id', 'away_team', 'outcome_regular_time', 'outcome_5', 'home_team']]
                     .rename(columns={'away_team': 'team', 'home_team': 'opp_team'}))

        home_team['is_home'] = True
        away_team['is_home'] = False

        home_masks = [home_team['outcome_regular_time'] == 'H', home_team['outcome_regular_time'] == 'A']
        home_team['outcome'] = np.select(home_masks, [1, 0], default=0.5)

        away_masks = [away_team['outcome_regular_time'] == 'A', away_team['outcome_regular_time'] == 'H']
        away_team['outcome'] = np.select(away_masks, [1, 0], default=0.5)

        home_team['extra_time_outcome'] = home_team['outcome_5'].map({'HOME_EXTRA_TIME': 1.0, 'DRAW_EXTRA_TIME': 0.5, 'AWAY_EXTRA_TIME': 0.0})
        away_team['extra_time_outcome'] = away_team['outcome_5'].map({'AWAY_EXTRA_TIME': 1.0, 'DRAW_EXTRA_TIME': 0.5, 'HOME_EXTRA_TIME': 0.0})

        home_team = home_team.drop(columns=['outcome_regular_time', 'outcome_5'])
        away_team = away_team.drop(columns=['outcome_regular_time', 'outcome_5'])

        matches = pd.concat([home_team, away_team], ignore_index=True)

        matches = (matches
                   .merge(matches_info, how='inner', on='match_id', validate='m:1')
                   .sort_values(['date', 'match_id']))

        assert (matches['extra_time_outcome'].notna() == matches['is_extra_time']).all(), \
            'extra_time_outcome must be known exactly for extra time matches'

        matches['team_confederation'] = matches['team'].map(confederations)
        matches['opp_confederation'] = matches['opp_team'].map(confederations)

        # no seasons for national teams: the season transition update is never applied
        matches['season_transition'] = 'same_season'
        matches['opp_season_transition'] = 'same_season'

        return matches

    def transform(self):
        """national_glicko_matches.feather from the national matches of matches.feather and the goals expectations of the lgbm."""

        matches = pd.read_feather(config.soccer['matches'], columns=self.input_match_columns)
        matches = matches[matches['source'] == 'national'].drop(columns='source').reset_index(drop=True)

        confederations = self._team_confederations(matches)

        matches = self._nations_league_finals(matches)
        matches = self._stages(matches)

        matches = self._add_goals_expectations(matches)

        matches = self._rating_periods(matches)

        matches = self._status(matches)

        matches = self._transform_data(matches, confederations)

        matches = matches[self.output_match_columns].reset_index(drop=True)

        save_feather(matches, config.soccer['national_glicko_matches'])

        return matches
