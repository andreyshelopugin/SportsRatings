import numpy as np
import pandas as pd

from config import config
from utils.functions import save_feather


class ClubPreprocessing(object):
    """Setup the dataset for ratings calculation."""

    def __init__(self):
        self.number_rating_periods_in_seasons = 4

        # the boundaries of the rating periods are shifted by up to max_shift_days, so fewer teams start a new season
        # inside a period: the season starts within ±boundary_window_days of a boundary are counted (see _rating_periods)
        self.max_shift_days = 5
        self.boundary_window_days = 10

        # goals expectations are clipped at this quantile of the match total (see _add_goals_expectations)
        self.goals_expectations_clip_quantile = 0.999

        # extra time goals relative to regular time, the same value as Features.extra_time_goals_ratio
        self.extra_time_goals_ratio = 0.25

        # short seasons (see _exclude_short_seasons): excluded from championships if the median matches per team <= 2,
        # printed if < 0.5 of the typical median of the tournament; the seasons of the last 60 days are current
        self.max_short_season_median = 2
        self.short_season_share = 0.5
        self.current_season_days = 60
        self.excluded_short_seasons = set()

        self.input_goals_expectations_columns = ['match_id', 'home_goals_expectations', 'away_goals_expectations']

        self.input_match_columns = ['match_id', 'source', 'country', 'tournament', 'tournament_type', 'is_domestic', 'is_championship',
                                    'season', 'date', 'home_team', 'away_team',
                                    'outcome_regular_time', 'outcome_5', 'is_extra_time',
                                    'is_neutral', 'is_no_spectators', 'is_different_stadium', 'is_limited_spectators',
                                    'home_team_league', 'away_team_league']

        self.output_match_columns = ['match_id', 'team', 'outcome', 'opp_team', 'country', 'tournament',
                                     'tournament_type', 'is_domestic', 'is_championship', 'season', 'date',
                                     'goals_expectations', 'rating_period_id', 'season_transition',
                                     'opp_season_transition', 'status', 'home_indicator', 'home_team_league', 'away_team_league',
                                     'is_extra_time', 'extra_time_outcome', 'goals_expectations_et', 'is_home']

    @staticmethod
    def _team_rows(championship_matches: pd.DataFrame) -> pd.DataFrame:
        """One row per team and match (match-level data)."""

        columns = ['match_id', 'tournament', 'season', 'date']

        return pd.concat([
            championship_matches[columns + ['home_team']].rename(columns={'home_team': 'team'}),
            championship_matches[columns + ['away_team']].rename(columns={'away_team': 'team'}),
        ], ignore_index=True)

    def _exclude_short_seasons(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Seasons with median matches per team <= max_short_season_median (cancelled seasons, data gaps)
        are excluded from championships, so they do not create extra season transitions.
        Matches stay in the ratings. The current season is kept.
        Other short seasons are printed for manual check.
        """

        season_keys = ['tournament', 'season']

        championship_matches = matches[matches['is_championship'] == True]

        team_season_counts = (self._team_rows(championship_matches)
                              .groupby(season_keys + ['team'])['match_id']
                              .nunique()
                              .rename('n_matches')
                              .reset_index())

        season_stats = (team_season_counts
                        .groupby(season_keys)
                        .agg(median_matches=('n_matches', 'median'), n_teams=('team', 'nunique'))
                        .join(championship_matches
                              .groupby(season_keys)
                              .agg(first_date=('date', 'min'), last_date=('date', 'max')))
                        .reset_index())

        season_stats['typical_median'] = season_stats.groupby('tournament')['median_matches'].transform('median')
        season_stats['span_days'] = (season_stats['last_date'] - season_stats['first_date']).dt.days

        data_end_date = matches['date'].max()
        is_last_season = season_stats['season'] == season_stats.groupby('tournament')['season'].transform('max')
        is_current = is_last_season & (season_stats['last_date'] >= data_end_date - pd.Timedelta(days=self.current_season_days))

        is_excluded = (season_stats['median_matches'] <= self.max_short_season_median) & ~is_current
        is_short = (season_stats['median_matches'] < self.short_season_share * season_stats['typical_median']) & ~is_current & ~is_excluded

        report_columns = season_keys + ['median_matches', 'typical_median', 'n_teams', 'first_date', 'last_date', 'span_days']

        excluded_seasons = season_stats.loc[is_excluded]
        self.excluded_short_seasons = set(excluded_seasons[season_keys].itertuples(index=False, name=None))

        if not excluded_seasons.empty:
            print('Short seasons excluded from championships:')
            print(excluded_seasons[report_columns].to_string(index=False))

        short_seasons = season_stats.loc[is_short]
        if not short_seasons.empty:
            print('Short seasons (kept, check manually):')
            print(short_seasons[report_columns].sort_values('median_matches').to_string(index=False))

        is_excluded_season = pd.MultiIndex.from_frame(matches[season_keys]).isin(list(self.excluded_short_seasons))
        matches.loc[is_excluded_season, 'is_championship'] = False

        return matches

    @staticmethod
    def _drop_teams_without_championship(matches: pd.DataFrame) -> pd.DataFrame:
        """Drop matches of teams without championship matches (e.g. only in excluded short seasons):
        such teams are weak, but would be initialized with a league rating and inflate the ratings of opponents."""

        championships = matches[matches['is_championship'] == True]
        teams_with_championships = set(championships['home_team']) | set(championships['away_team'])

        is_known = matches['home_team'].isin(teams_with_championships) & matches['away_team'].isin(teams_with_championships)

        print('#dropped matches with teams without championship:', (~is_known).sum())

        return matches[is_known].reset_index(drop=True)

    def _add_goals_expectations(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Goals expectations are predicted for club and national team matches, both use match_id from the union (Features)."""
        goals_expectations = pd.read_feather(config.soccer['outcomes']['lgbm_predictions'], columns=self.input_goals_expectations_columns)

        # club matches only, also for normalization
        goals_expectations = goals_expectations[goals_expectations['match_id'].isin(matches['match_id'])]

        goals_expectations['goals_expectations'] = goals_expectations['home_goals_expectations'] + goals_expectations['away_goals_expectations']

        # for numerical stability
        upper = goals_expectations['goals_expectations'].quantile(self.goals_expectations_clip_quantile)
        goals_expectations['goals_expectations'] = goals_expectations['goals_expectations'].clip(upper=upper)

        # normalization (over club matches only)
        # to do: think about other techniques
        goals_expectations['goals_expectations'] -= goals_expectations['goals_expectations'].mean()

        goals_expectations = dict(zip(goals_expectations['match_id'], goals_expectations['goals_expectations']))

        matches['goals_expectations'] = matches['match_id'].map(goals_expectations)

        # extra time goals expectations: the same scale as regular time, shrunk by extra time duration
        matches['goals_expectations_et'] = self.extra_time_goals_ratio * matches['goals_expectations']

        # centered over extra time matches: they are mostly knockout ties with lower goals expectations,
        # without centering scoring_overall in extra time acts as a second intercept (collinear with draw_frequency)
        extra_time_mean = matches.loc[matches['is_extra_time'], 'goals_expectations_et'].mean()
        matches['goals_expectations_et'] -= extra_time_mean

        n_missing = matches['goals_expectations'].isna().sum()
        assert n_missing == 0, f'{n_missing} matches without goals expectations'

        return matches

    @staticmethod
    def _count_season_transitions_at_boundary(league_matches: pd.DataFrame, boundary_day: int, window: int) -> int:
        """Count how many teams start new seasons near this boundary.

        Args:
            league_matches: DataFrame of league matches with 'day_of_year', 'season',
                            'home_team', and 'away_team' columns.
            boundary_day: The day of year for the candidate boundary.
            window: Number of days around the boundary to check for season starts.

        Returns:
            Number of team-season starts falling within the window around the boundary.
        """
        # get first match of each team's season
        season_starts = []
        for team_col in ['home_team', 'away_team']:
            starts = (league_matches
                      .groupby([team_col, 'season'])['day_of_year']
                      .min()
                      .reset_index()
                      .rename(columns={team_col: 'team', 'day_of_year': 'season_start_day'}))
            season_starts.append(starts)

        season_starts = pd.concat(season_starts).drop_duplicates(subset=['team', 'season'])

        # count season starts within ±window days of boundary
        near_boundary = season_starts[(season_starts['season_start_day'] >= boundary_day - window)
                                      & (season_starts['season_start_day'] <= boundary_day + window)]

        return len(near_boundary)

    def _rating_periods(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Add rating period identifier to each match with optimized boundaries.

        Shifts period boundaries by up to max_shift_days to reduce the number of
        teams starting new seasons within the same rating period.
        """

        # get basic period info
        matches['day_of_year'] = matches['date'].dt.dayofyear
        matches['days_in_year'] = matches['date'].dt.is_leap_year.map({True: 366, False: 365})
        matches['year'] = matches['date'].dt.year

        # filter league matches AFTER adding columns
        league_matches = matches[matches['is_domestic'] == True]

        # calculate base period boundaries (days of year)
        base_period_length = 365 / self.number_rating_periods_in_seasons
        base_boundaries = [int(base_period_length * i) for i in range(self.number_rating_periods_in_seasons + 1)]
        base_boundaries[0] = 1
        base_boundaries[-1] = 367  # exclusive upper bound, handles leap year day 366

        # for each boundary (except first and last), try shifts
        optimized_boundaries = base_boundaries.copy()

        for i in range(1, len(base_boundaries) - 1):
            base_day = base_boundaries[i]
            best_day = base_day
            min_transitions = float('inf')

            # determine valid range to prevent overlap with neighbors
            lower_limit = optimized_boundaries[i - 1] + 1
            upper_limit = (base_boundaries[i + 1] if i + 1 < len(base_boundaries) - 1
                           else optimized_boundaries[i + 1]) - 1

            # try shifts from -max_shift_days to +max_shift_days
            for shift in range(-self.max_shift_days, self.max_shift_days + 1):
                candidate_day = base_day + shift

                # ensure boundaries don't overlap or cross neighbors
                if candidate_day <= lower_limit or candidate_day >= upper_limit:
                    continue

                # count season transitions if we use this boundary
                transitions = self._count_season_transitions_at_boundary(league_matches, candidate_day, self.boundary_window_days)

                if transitions < min_transitions:
                    min_transitions = transitions
                    best_day = candidate_day

            optimized_boundaries[i] = best_day

        # assign periods based on optimized boundaries
        matches['rating_period_in_year'] = 0
        for i in range(len(optimized_boundaries) - 1):
            start = optimized_boundaries[i]
            end = optimized_boundaries[i + 1]
            mask = (matches['day_of_year'] >= start) & (matches['day_of_year'] < end)
            matches.loc[mask, 'rating_period_in_year'] = i

        # create global period id
        min_year = matches['year'].min()
        matches['rating_period_id'] = ((matches['year'] - min_year) * self.number_rating_periods_in_seasons + matches['rating_period_in_year'])

        # clean up
        matches = matches.drop(columns=['day_of_year', 'days_in_year', 'year', 'rating_period_in_year'])

        return matches

    @staticmethod
    def _transform_data(matches: pd.DataFrame) -> pd.DataFrame:
        """Transform to glicko approach: each match has two rows,
        for convenient work with a team and its opponent.

        outcome: regular time (90') outcome, 1 / 0.5 / 0 for the team
        extra_time_outcome: extra time outcome, 1 / 0.5 (penalties) / 0 for the team, NaN if extra time was not played
        """

        # is_extra_time_possible is not used for now: in such matches (e.g. a second leg with a tied aggregate)
        # the probability of a 90' draw is likely slightly higher, can be added later as a draw_frequency shift
        matches_info = matches.loc[:, ['match_id', 'country', 'tournament', 'tournament_type', 'is_domestic', 'is_championship',
                                       'season', 'date',
                                       'is_neutral', 'is_no_spectators', 'is_different_stadium', 'is_limited_spectators',
                                       'goals_expectations', 'goals_expectations_et', 'home_team_league', 'away_team_league', 'rating_period_id',
                                       'is_extra_time']]

        home_team = (matches
                     .loc[:, ['match_id', 'home_team', 'outcome_regular_time', 'outcome_5', 'away_team']]
                     .rename(columns={'home_team': 'team', 'away_team': 'opp_team'}))

        away_team = (matches
                     .loc[:, ['match_id', 'away_team', 'outcome_regular_time', 'outcome_5', 'home_team']]
                     .rename(columns={'away_team': 'team', 'home_team': 'opp_team'}))

        home_team['is_home'] = True
        away_team['is_home'] = False

        # regular time (90') outcome to glicko logic
        home_masks = [home_team['outcome_regular_time'] == 'H', home_team['outcome_regular_time'] == 'A']
        home_team['outcome'] = np.select(home_masks, [1, 0], default=0.5)

        away_masks = [away_team['outcome_regular_time'] == 'A', away_team['outcome_regular_time'] == 'H']
        away_team['outcome'] = np.select(away_masks, [1, 0], default=0.5)

        # extra time outcome to glicko logic, NaN if extra time was not played
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

        return matches

    def _season_transition_tags(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Tag matches with season transition type for home and away teams.

        Adds columns:
            home_season_transition: 'same_league', 'changed_league', 'missed_previous', or None
            away_season_transition: 'same_league', 'changed_league', 'missed_previous', or None

        A non-None value means this is the first championship match of a new season for that team.
        A gap of excluded short seasons of the previous league is not a missed season.

        Note: if a team begins a season in a cup or international tournament,
        season_transition logic may be incorrect; to improve later.
        """

        # first championship match per team per season
        championships = (matches
                         .loc[matches['is_championship'] == True, ['date', 'match_id', 'team', 'country', 'tournament_type', 'season',
                                                                   'tournament', 'is_home']]
                         .sort_values(['team', 'date'])
                         .drop_duplicates(['team', 'season'], keep='first'))

        # get first season per team in each tournament
        team_tournament_first = championships.groupby(['team', 'tournament'])['season'].min().reset_index()
        team_tournament_first.columns = ['team', 'tournament', 'team_first_season_in_tournament']

        # get first season per tournament (overall)
        tournament_first_seasons = championships.groupby('tournament')['season'].min()

        # merge to get both first seasons
        championships = championships.merge(team_tournament_first, on=['team', 'tournament'], how='left', validate='m:1')
        championships['tournament_first_season'] = championships['tournament'].map(tournament_first_seasons)

        # team is new to an existing tournament if its first season in the tournament is later than the tournament's first season
        is_new_to_existing_tournament = ((championships['season'] == championships['team_first_season_in_tournament'])
                                         & (championships['team_first_season_in_tournament'] > championships['tournament_first_season']))

        # classify transition type
        is_same_team = (championships['team'] == championships['team'].shift())

        # for example, Argentina has the transfer from season 2019/2020 to 2021
        # TODO: replace by a rule: consecutive = the previous season of the tournament in the data (not season - 1),
        # for changed_league of the old or the new tournament. It covers the transfer seasons and the gaps
        # of excluded short seasons (is_gap_of_excluded_seasons)
        tournaments_with_transfer_seasons = {
            ('Argentina', 'first_league', 2019),
            ('Aruba', 'first_league', 2019),
            ('Bangladesh', 'first_league', 2014),
            ('Bangladesh', 'first_league', 2017),
            ('Barbados', 'first_league', 2018),
            ('Bolivia', 'first_league', 2016),
            ('Cameroon', 'first_league', 2010),
            ('Croatia', 'second_league', 2002),
            ('Cyprus', 'second_league', 2012),
            ('Gabon', 'first_league', 2013),
            ('Gabon', 'first_league', 2017),
            ('Ghana', 'first_league', 2015),
            ('Indonesia', 'first_league', 2011),
            ('Iraq', 'first_league', 2002),
            ('Jordan', 'first_league', 2018),
            ('Macao', 'first_league', 2007),
            ('Malaysia', 'first_league', 2007),
            ('Mali', 'first_league', 2014),
            ('Mali', 'first_league', 2017),
            ('Mauritania', 'first_league', 2007),
            ('Somalia', 'first_league', 2016),
            ('Somalia', 'first_league', 2020),
            ('Sudan', 'first_league', 2022),
            ('Thailand', 'first_league', 2004),
            ('Togo', 'first_league', 2012),
            ('Vietnam', 'first_league', 2001),
            ('Zimbabwe', 'first_league', 2021),
        }

        is_transfer_season = pd.Series(False, index=championships.index)
        for country, tournament, season in tournaments_with_transfer_seasons:
            # +2 because the previous code logic looked at the next season, here it looks at the previous one
            is_transfer_season |= ((championships['country'] == country)
                                   & (championships['tournament_type'] == tournament)
                                   & (championships['season'] == season + 2))

        previous_season = championships['season'].shift()
        previous_tournament = championships['tournament'].shift()

        # e.g. Congo 2024 -> [2025 excluded] -> 2026 is consecutive, not missed_previous with rating reset
        is_gap_of_excluded_seasons = pd.Series([
            pd.notna(prev_season) and season > prev_season + 1
            and all((prev_tournament, gap_season) in self.excluded_short_seasons
                    for gap_season in range(int(prev_season) + 1, int(season)))
            for season, prev_season, prev_tournament in zip(championships['season'], previous_season, previous_tournament)
        ], index=championships.index, dtype=bool)

        is_consecutive = (championships['season'] == previous_season + 1) | is_transfer_season | is_gap_of_excluded_seasons

        is_same_tournament = (championships['tournament'] == championships['tournament'].shift())

        masks = [is_same_team & is_consecutive & is_same_tournament,
                 is_same_team & is_consecutive & ~is_same_tournament,
                 is_same_team & ~is_consecutive,
                 is_new_to_existing_tournament]

        values = ['same_league', 'changed_league', 'missed_previous', 'missed_previous']

        championships['season_transition'] = np.select(masks, values, default='same_season')

        # clean up temporary columns
        transitions = championships.loc[championships['season_transition'] != 'same_season', ['match_id', 'is_home', 'season_transition']]

        opp_transitions = transitions.copy().rename(columns={'season_transition': 'opp_season_transition'})
        opp_transitions['is_home'] = ~opp_transitions['is_home']

        matches = (matches
                   .merge(transitions, on=['match_id', 'is_home'], how='left', validate='1:1')
                   .merge(opp_transitions, on=['match_id', 'is_home'], how='left', validate='1:1'))

        matches['season_transition'] = matches['season_transition'].fillna('same_season')
        matches['opp_season_transition'] = matches['opp_season_transition'].fillna('same_season')

        return matches

    @staticmethod
    def _home_field_advantage(matches: pd.DataFrame):
        """In our logic home field advantage based on crowd and logistic effects.
            is_neutral: logistic = 0, crowd = 0

            is_no_spectators: crowd = 0
            is_limited_spectators: crowd /= 2
            is_different_stadium: logistic = 0, crowd /= 2
        """

        is_neutral = (matches['is_neutral'] == True)
        is_home = (matches['is_home'] == True)
        is_away = (matches['is_home'] == False)

        # crowd effect
        is_full_crowd = ((matches['is_no_spectators'] == False)
                         & (matches['is_limited_spectators'] == False)
                         & (matches['is_different_stadium'] == False))

        is_half_crowd = (matches['is_limited_spectators'] == True) | (matches['is_different_stadium'] == True)

        is_zero_crowd = (matches['is_no_spectators'] == True)

        # logistic effect
        is_full_logistic = (matches['is_different_stadium'] == False)

        is_zero_logistic = (matches['is_different_stadium'] == True)

        masks = [
            is_neutral,

            is_full_crowd & is_full_logistic,
            is_half_crowd & is_full_logistic,
            is_zero_crowd & is_full_logistic,

            is_full_crowd & is_zero_logistic,
            is_half_crowd & is_zero_logistic,
            is_zero_crowd & is_zero_logistic
        ]

        values = [
            'NEUTRAL',

            'FULL_CROWD_FULL_LOGISTIC',
            'HALF_CROWD_FULL_LOGISTIC',
            'ZERO_CROWD_FULL_LOGISTIC',

            'FULL_CROWD_ZERO_LOGISTIC',
            'HALF_CROWD_ZERO_LOGISTIC',
            'ZERO_CROWD_ZERO_LOGISTIC']

        matches['status'] = np.select(masks, values, default='bug')

        assert (matches['status'] == 'bug').sum() == 0, 'bugs in match status'

        masks = [is_neutral, is_home, is_away]

        values = [0, 1, -1]

        matches['home_indicator'] = np.select(masks, values)

        # is_home is kept: for neutral matches home_indicator is 0 for both teams,
        # is_home defines which team is listed home (home_team_league / away_team_league)
        matches = matches.drop(columns=['is_neutral', 'is_no_spectators', 'is_limited_spectators', 'is_different_stadium'])

        return matches

    def transform(self):
        """"""

        # union of club and national team matches with one match_id (Features), ratings for clubs
        matches = pd.read_feather(config.soccer['matches'], columns=self.input_match_columns)
        matches = matches[matches['source'] == 'clubs'].drop(columns='source').reset_index(drop=True)

        # before _season_transition_tags: short seasons must not create season transitions
        matches = self._exclude_short_seasons(matches)
        matches = self._drop_teams_without_championship(matches)

        matches = self._add_goals_expectations(matches)

        matches = self._rating_periods(matches)

        matches = self._transform_data(matches)

        matches = self._season_transition_tags(matches)

        matches = self._home_field_advantage(matches)

        save_feather(matches, config.soccer['glicko_matches'])
