import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import config
from utils.functions import save_feather


class Features(object):
    """Calculate features for boosting models."""

    def __init__(self, eval_set_size=0.1, random_state=7, train_test_split_season=2021):
        self.number_matches = None
        self.eval_set_size = eval_set_size
        self.random_state = random_state
        self.train_test_split_season = train_test_split_season

        self.windows = [(5, 3), (10, 5), (20, 10), (30, 20)]

        # goals of the extra time relative to the regular time (30 of 90 minutes, without a fit)
        self.extra_time_goals_ratio = 0.25

        # union of club and national team matches (MatchesPreprocessor), match_id is common for the whole project
        self.input_columns = ['match_id', 'source', 'date', 'home_team', 'away_team', 'home_score', 'away_score',
                              'season', 'tournament', 'tournament_type',
                              'is_championship', 'is_neutral',
                              'is_no_spectators', 'is_different_stadium', 'is_limited_spectators', 'is_pandemic',
                              'home_indicator', 'is_final_stage',
                              'is_extra_time_possible', 'is_extra_time', 'outcome_regular_time']

        # is_extra_time, exposure and outcome_regular_time are not features (leakage):
        # exposure is used as offset, outcome_regular_time for outcome metrics
        self.output_columns = [
            'match_id', 'date', 'team', 'opp_team', 'season', 'tournament_type',

            'score', 'opp_score',
            'is_home', 'is_no_spectators', 'is_different_stadium', 'is_limited_spectators', 'is_neutral', 'is_pandemic',
            'final_stage', 'home_indicator', 'venue', 'is_extra_time_possible', 'is_extra_time', 'exposure', 'outcome_regular_time',
            'days_since_last_game', 'games_in_season',
            'opp_days_since_last_game', 'opp_games_in_season',

            'avg_scoring_5', 'avg_scoring_10', 'avg_scoring_20', 'avg_scoring_30',
            'avg_scoring_tournament_5', 'avg_scoring_tournament_10', 'avg_scoring_tournament_20', 'avg_scoring_tournament_30',
            'avg_conceded_5', 'avg_conceded_10', 'avg_conceded_20', 'avg_conceded_30',
            'opp_avg_conceded_5', 'opp_avg_conceded_10', 'opp_avg_conceded_20', 'opp_avg_conceded_30',
            'opp_avg_scoring_5', 'opp_avg_scoring_10', 'opp_avg_scoring_20', 'opp_avg_scoring_30',

            'location_mean_score_5', 'location_mean_score_10', 'location_mean_score_20', 'location_mean_score_30',
            'location_median_score_5', 'location_median_score_10', 'location_median_score_20', 'location_median_score_30',

            'location_mean_score_5_conceded', 'location_mean_score_10_conceded', 'location_mean_score_20_conceded', 'location_mean_score_30_conceded',
            'location_median_score_5_conceded', 'location_median_score_10_conceded', 'location_median_score_20_conceded',
            'location_median_score_30_conceded',

            'location_max_score_10',

            'league', 'opp_league', 'tournament',
            'leagues_difference'
        ]

    @staticmethod
    def _prepare_matches(matches: pd.DataFrame) -> pd.DataFrame:
        """Unify home advantage and tournament columns of club and national team matches.
        Columns specific to one source are NaN for the other one in the union."""

        is_clubs = matches['source'] == 'clubs'

        # clubs: is_neutral already exists, national: derive is_neutral from home_indicator
        home_indicator = np.where(matches['is_neutral'] == True, 0, 1)
        matches['home_indicator'] = np.where(is_clubs, home_indicator, matches['home_indicator']).astype('int8')

        matches['is_neutral'] = np.where(is_clubs, matches['is_neutral'] == True, matches['home_indicator'] == 0)

        matches['is_championship'] = np.where(is_clubs, matches['is_championship'] == True, False)

        matches['final_stage'] = np.where(is_clubs, 'clubs', (matches['is_final_stage'] == True).astype(str))

        matches = matches.drop(columns=['source', 'is_final_stage'])

        return matches

    @staticmethod
    def rolling_stats(matches: pd.DataFrame, team_col: str, score_col: str, stats_type: str,
                      window: int, min_periods: int) -> dict:
        """Calculate rolling statistics for specified team and score columns.
        Ties by date (a team played twice on the same date) are broken by match_id,
        so team and opponent rolling windows see matches in the same order."""

        sorted_matches = matches.sort_values([team_col, 'date', 'match_id'], kind='stable').copy()

        sorted_matches['shifted_score'] = (sorted_matches
                                           .groupby(team_col, sort=False)
                                           [score_col]
                                           .shift())

        # apply rolling within each team group
        rolling = (sorted_matches
                   .groupby(team_col, sort=False)
                   ['shifted_score']
                   .rolling(window=window, min_periods=min_periods))

        # mean, median or max
        result = getattr(rolling, stats_type)()

        # reset index to match original dataframe
        result.index = result.index.droplevel(0)

        return result.to_dict()

    @staticmethod
    def rolling_mean_by_tournament(matches: pd.DataFrame, team_col: str, score_col: str, window: int, min_periods: int) -> dict:
        """Rolling mean within each team-tournament group."""

        group_cols = [team_col, 'tournament_type']
        sorted_matches = matches.sort_values(group_cols + ['date', 'match_id'], kind='stable').copy()

        sorted_matches['shifted_score'] = (sorted_matches
                                           .groupby(group_cols, sort=False)
                                           [score_col]
                                           .shift())

        rolling = (sorted_matches
                   .groupby(group_cols, sort=False)
                   ['shifted_score']
                   .rolling(window=window, min_periods=min_periods))

        result = rolling.mean()
        result.index = result.index.droplevel([0, 1])

        return result.to_dict()

    @staticmethod
    def _calculate_games_in_season(matches: pd.DataFrame, team_col: str) -> dict:
        """Calculate number of games played in current season before this match.
        Helps understand whether features are stale, as games may have been played in the previous season."""
        return (matches
                .sort_values([team_col, 'season', 'date', 'match_id'], kind='stable')
                .groupby([team_col, 'season'], sort=False)
                .cumcount()
                .to_dict())

    @staticmethod
    def _calculate_days_since_last_game(matches: pd.DataFrame, team_col: str) -> dict:
        """Calculate days since previous game for each team."""
        return (matches
                .sort_values([team_col, 'date', 'match_id'], kind='stable')
                .groupby(team_col, sort=False)['date']
                .diff()
                .dt
                .days
                .fillna(9999)  # first game for team
                .to_dict())

    @staticmethod
    def _team_leagues(matches: pd.DataFrame) -> pd.DataFrame:
        """The league of every team in every season."""

        team_leagues = (matches
                        .loc[matches['is_championship'] == True, ['home_team', 'tournament', 'season', 'tournament_type']]
                        .drop_duplicates(['home_team', 'season'], keep='first')
                        .rename(columns={'home_team': 'team', 'tournament': 'league'}))

        team_leagues['is_first_league'] = (team_leagues['tournament_type'] == 'first_league')

        team_leagues = team_leagues.loc[:, ['season', 'team', 'league', 'is_first_league']]

        home_team_leagues = team_leagues.rename(columns={'league': 'home_league', 'team': 'home_team', 'is_first_league': 'is_home_first_league'})
        away_team_leagues = team_leagues.rename(columns={'league': 'away_league', 'team': 'away_team', 'is_first_league': 'is_away_first_league'})

        matches = (matches
                   .merge(home_team_leagues, how='left', on=['home_team', 'season'], validate='m:1')
                   .merge(away_team_leagues, how='left', on=['away_team', 'season'], validate='m:1'))

        masks = [(matches['is_home_first_league'] == True) & (matches['is_away_first_league'] == False),
                 (matches['is_home_first_league'] == False) & (matches['is_away_first_league'] == True)]

        values = [1, -1]

        matches['leagues_difference'] = np.select(masks, values, default=0)

        return matches

    def _features(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Calculate window functions based on outcomes scored in the previous matches."""

        # gets "index" column
        matches = matches.sort_values(['date', 'match_id'], kind='stable').reset_index(drop=True).reset_index()

        index = matches['index']
        m = matches.loc[:, ['match_id', 'date', 'home_team', 'away_team', 'home_score', 'away_score']].copy()

        for window, periods in self.windows:
            w = str(window)

            # home team scoring when played at home
            matches['home_mean_score_' + w] = index.map(self.rolling_stats(m, 'home_team', 'home_score', 'mean', window, periods))
            matches['home_median_score_' + w] = index.map(self.rolling_stats(m, 'home_team', 'home_score', 'median', window, periods))

            # away team scoring when played away
            matches['away_mean_score_' + w] = index.map(self.rolling_stats(m, 'away_team', 'away_score', 'mean', window, periods))
            matches['away_median_score_' + w] = index.map(self.rolling_stats(m, 'away_team', 'away_score', 'median', window, periods))

            # home team conceding when played at home
            matches['home_mean_score_' + w + '_conceded'] = index.map(self.rolling_stats(m, 'home_team', 'away_score', 'mean', window, periods))
            matches['home_median_score_' + w + '_conceded'] = index.map(self.rolling_stats(m, 'home_team', 'away_score', 'median', window, periods))

            # away team conceding when played away
            matches['away_mean_score_' + w + '_conceded'] = index.map(self.rolling_stats(m, 'away_team', 'home_score', 'mean', window, periods))
            matches['away_median_score_' + w + '_conceded'] = index.map(self.rolling_stats(m, 'away_team', 'home_score', 'median', window, periods))

        # max scoring
        matches['home_max_score_10'] = index.map(self.rolling_stats(m, 'home_team', 'home_score', 'max', 10, 5))
        matches['away_max_score_10'] = index.map(self.rolling_stats(m, 'away_team', 'away_score', 'max', 10, 5))

        matches = matches.drop(columns='index')

        assert self.number_matches == matches.shape[0], 'number of matches mismatched'

        return matches

    def scoring_features(self, scoring: pd.DataFrame) -> pd.DataFrame:
        """Rolling scoring and conceding of the team and the opponent, a row per team and match."""
        scoring = scoring.sort_values(['date', 'match_id', 'is_home'], kind='stable').reset_index(drop=True).reset_index()

        for window, min_periods in self.windows:
            # team scoring
            avg_scoring = self.rolling_stats(scoring, 'team', 'score', 'mean', window, min_periods)
            scoring[f'avg_scoring_{window}'] = scoring['index'].map(avg_scoring)

            # team conceding
            avg_conceded = self.rolling_stats(scoring, 'team', 'opp_score', 'mean', window, min_periods)
            scoring[f'avg_conceded_{window}'] = scoring['index'].map(avg_conceded)

            # opponent conceding (their defensive weakness)
            opp_avg_conceded = self.rolling_stats(scoring, 'opp_team', 'score', 'mean', window, min_periods)
            scoring[f'opp_avg_conceded_{window}'] = scoring['index'].map(opp_avg_conceded)

            # opponent scoring (their offensive strength)
            opp_avg_scoring = self.rolling_stats(scoring, 'opp_team', 'opp_score', 'mean', window, min_periods)
            scoring[f'opp_avg_scoring_{window}'] = scoring['index'].map(opp_avg_scoring)

            # team scoring in the given tournament
            avg_scoring_tournament = self.rolling_mean_by_tournament(scoring, 'team', 'score', window, min_periods)
            scoring[f'avg_scoring_tournament_{window}'] = scoring['index'].map(avg_scoring_tournament)

        scoring['days_since_last_game'] = scoring['index'].map(self._calculate_days_since_last_game(scoring, 'team'))
        scoring['games_in_season'] = scoring['index'].map(self._calculate_games_in_season(scoring, 'team'))

        scoring['opp_days_since_last_game'] = scoring['index'].map(self._calculate_days_since_last_game(scoring, 'opp_team'))
        scoring['opp_games_in_season'] = scoring['index'].map(self._calculate_games_in_season(scoring, 'opp_team'))

        scoring = scoring.drop(columns='index')

        assert self.number_matches == scoring.shape[0] // 2, 'number of matches mismatched'
        assert self.number_matches == scoring['match_id'].nunique(), 'number of matches mismatched'

        return scoring

    def _transform_data(self, matches: pd.DataFrame) -> pd.DataFrame:
        """From a row per match (home, away) to a row per team and match (team, opponent)."""

        home_scoring = (matches
                        .loc[:, ['date', 'home_team', 'away_team', 'home_score', 'away_score', 'match_id', 'tournament_type', 'season']]
                        .rename(columns={'home_team': 'team', 'away_team': 'opp_team', 'home_score': 'score', 'away_score': 'opp_score'}))

        away_scoring = (matches
                        .loc[:, ['date', 'away_team', 'home_team', 'away_score', 'home_score', 'match_id', 'tournament_type', 'season']]
                        .rename(columns={'away_team': 'team', 'home_team': 'opp_team', 'away_score': 'score', 'home_score': 'opp_score'}))

        home_scoring['is_home'] = True
        away_scoring['is_home'] = False

        scoring = pd.concat([home_scoring, away_scoring], ignore_index=True)

        assert self.number_matches == scoring.shape[0] // 2, 'number of matches mismatched'
        assert self.number_matches == scoring['match_id'].nunique(), 'number of matches mismatched'

        scoring = self.scoring_features(scoring)

        # columns to merge from matches
        matches_columns = ['match_id',
                           'tournament',
                           'home_league', 'away_league',
                           'is_pandemic', 'is_no_spectators', 'is_different_stadium', 'is_limited_spectators', 'is_neutral',
                           'final_stage', 'home_indicator', 'is_extra_time_possible', 'is_extra_time', 'outcome_regular_time',
                           'home_mean_score_5', 'away_mean_score_5',
                           'home_mean_score_10', 'away_mean_score_10',
                           'home_mean_score_20', 'away_mean_score_20',
                           'home_mean_score_30', 'away_mean_score_30',
                           'home_median_score_5', 'away_median_score_5',
                           'home_median_score_10', 'away_median_score_10',
                           'home_median_score_20', 'away_median_score_20',
                           'home_median_score_30', 'away_median_score_30',
                           'home_mean_score_5_conceded', 'away_mean_score_5_conceded',
                           'home_mean_score_10_conceded', 'away_mean_score_10_conceded',
                           'home_mean_score_20_conceded', 'away_mean_score_20_conceded',
                           'home_mean_score_30_conceded', 'away_mean_score_30_conceded',
                           'home_median_score_5_conceded', 'away_median_score_5_conceded',
                           'home_median_score_10_conceded', 'away_median_score_10_conceded',
                           'home_median_score_20_conceded', 'away_median_score_20_conceded',
                           'home_median_score_30_conceded', 'away_median_score_30_conceded',
                           'home_max_score_10', 'away_max_score_10',
                           'leagues_difference']

        scoring = scoring.merge(matches.loc[:, matches_columns], how='left', on='match_id', validate='m:1')

        is_home = (scoring['is_home'] == True)

        scoring['league'] = np.where(is_home, scoring['home_league'], scoring['away_league'])
        scoring['opp_league'] = np.where(is_home, scoring['away_league'], scoring['home_league'])
        scoring['leagues_difference'] = np.where(is_home, scoring['leagues_difference'], -scoring['leagues_difference'])

        # venue of the participant: +1 at home, 0 neutral, -1 away (home_indicator is of the match, the same in both rows;
        # -1 of national teams: the listed home team plays in the country of the opponent). Replaces is_home, is_neutral
        # and home_indicator in the models
        scoring['venue'] = np.where(is_home, scoring['home_indicator'], -scoring['home_indicator']).astype('int8')

        for window, _ in self.windows:
            w = str(window)

            # location-based scoring
            scoring['location_mean_score_' + w] = np.where(is_home, scoring['home_mean_score_' + w], scoring['away_mean_score_' + w])
            scoring['location_median_score_' + w] = np.where(is_home, scoring['home_median_score_' + w], scoring['away_median_score_' + w])

            # location-based conceding
            scoring['location_mean_score_' + w + '_conceded'] = np.where(is_home,
                                                                         scoring['home_mean_score_' + w + '_conceded'],
                                                                         scoring['away_mean_score_' + w + '_conceded'])

            scoring['location_median_score_' + w + '_conceded'] = np.where(is_home,
                                                                           scoring['home_median_score_' + w + '_conceded'],
                                                                           scoring['away_median_score_' + w + '_conceded'])

        scoring['location_max_score_10'] = np.where(is_home, scoring['home_max_score_10'], scoring['away_max_score_10'])

        assert self.number_matches == scoring.shape[0] // 2, 'number of matches mismatched'
        assert self.number_matches == scoring['match_id'].nunique(), 'number of matches mismatched'

        return scoring

    def _postprocessing(self, scoring: pd.DataFrame) -> pd.DataFrame:
        """Exposure, the output columns and their types."""
        # 120' score for extra time matches (AET and Pen): regular time equivalent duration
        scoring['exposure'] = np.where(scoring['is_extra_time'] == True, 1 + self.extra_time_goals_ratio, 1.0)

        scoring = scoring.loc[:, self.output_columns]

        # the final tournament of the national teams as a separate tournament type (the World Cup, not its qualification).
        # after the rolling stats: there the qualification and the final tournament are one group
        scoring['tournament_type'] = np.where(scoring['final_stage'] == 'True', 'national_final_stage', scoring['tournament_type'])

        cat_features = ['tournament_type', 'tournament', 'league', 'opp_league', 'final_stage']

        scoring[cat_features] = scoring[cat_features].fillna('NaN').astype('category')

        to_int = ['days_since_last_game', 'opp_days_since_last_game']
        scoring[to_int] = scoring[to_int].to_numpy('int')

        return scoring

    @staticmethod
    def _test_no_same_day_duplicates(scoring: pd.DataFrame):
        """Print the teams playing twice on the same day."""

        dupes = scoring.groupby(['team', 'date']).size()
        dupes = dupes[dupes > 1]

        if len(dupes) > 0:
            print(f"teams playing twice on the same day ({len(dupes)} cases)")

    @staticmethod
    def _test_days_since_last_game(scoring: pd.DataFrame):
        """Validate days_since_last_game is calculated per team, not per home/away (9999 for the first game)."""

        sorted_scoring = scoring.sort_values(['team', 'date', 'match_id'], kind='stable')

        expected_days = (sorted_scoring
                         .groupby('team')['date']
                         .diff()
                         .dt.days
                         .fillna(9999)
                         .astype(int))

        assert (sorted_scoring['days_since_last_game'].values == expected_days.values).all(), \
            'days_since_last_game does not match actual calendar days between games'

        assert (expected_days[expected_days != 9999] <= 365).mean() > 0.90, 'too many games with >365 days gap'

    @staticmethod
    def _test_games_in_season(scoring: pd.DataFrame):
        """Validate games_in_season increments correctly per team."""

        sorted_scoring = scoring.sort_values(['team', 'season', 'date', 'match_id'], kind='stable')

        expected_counts = (sorted_scoring
                           .groupby(['team', 'season'])
                           .cumcount())

        assert (sorted_scoring['games_in_season'].values == expected_counts.values).all(), \
            'games_in_season does not increment sequentially per team per season'

    @staticmethod
    def _test_rolling_stats(scoring: pd.DataFrame):
        """Validate rolling statistics have expected NaN patterns."""

        # early season games should have more NaN than late season games
        early_nan_rate = scoring.loc[scoring['games_in_season'] < 3, 'avg_scoring_5'].isna().mean()
        late_nan_rate = scoring.loc[scoring['games_in_season'] >= 10, 'avg_scoring_5'].isna().mean()

        assert early_nan_rate > late_nan_rate, \
            f"early games should have more NaN than late games. early: {early_nan_rate:.1%}, late: {late_nan_rate:.1%}"

        # first games should have higher NaN rate
        first_nan_rate = scoring.loc[scoring['days_since_last_game'] == 9999, 'avg_scoring_5'].isna().mean()
        other_nan_rate = scoring.loc[scoring['days_since_last_game'] != 9999, 'avg_scoring_5'].isna().mean()

        assert first_nan_rate > other_nan_rate, \
            f"first games should have more NaN. first: {first_nan_rate:.1%}, others: {other_nan_rate:.1%}"

    @staticmethod
    def _test_no_negative_values(scoring: pd.DataFrame):
        """Validate no negative values in numeric columns (except leagues_difference)."""

        numeric = scoring.select_dtypes(include=[np.number]).drop(columns=['home_indicator', 'venue', 'leagues_difference'])
        min_vals = numeric.min()
        invalid = min_vals[min_vals < 0]

        assert len(invalid) == 0, f"negative values found in: {invalid.to_dict()}"

    @staticmethod
    def _test_match_symmetry(scoring: pd.DataFrame):
        """For every match, validate that the home row and the away row are mirror images of each other."""

        # exclude matches where a team played twice on the same date
        date_counts = scoring.groupby(['team', 'date'])['match_id'].transform('count')
        bad_match_ids = scoring.loc[date_counts > 1, 'match_id'].unique()

        features = scoring.loc[~scoring['match_id'].isin(bad_match_ids)]

        home = features.loc[features['is_home'] == True].set_index('match_id')
        away = features.loc[features['is_home'] == False].set_index('match_id')

        assert home.index.is_unique and away.index.is_unique, 'duplicate rows for the same match_id'
        assert set(home.index) == set(away.index), 'home and away match_ids do not align'

        away = away.loc[home.index]

        # (column, column of the opponent): the same value from both sides
        pairs = [('team', 'opp_team'), ('score', 'opp_score'), ('days_since_last_game', 'opp_days_since_last_game'),
                 ('games_in_season', 'opp_games_in_season'), ('league', 'opp_league')]

        for col, opp_col in pairs:
            for row, opp_row, side in [(home, away, 'home'), (away, home, 'away')]:
                is_mismatch = row[col] != opp_row[opp_col]
                assert not is_mismatch.any(), f'{side} {col} != opp {opp_col} ({is_mismatch.sum()} cases)'

        # the same value with the opposite sign
        for col in ['leagues_difference', 'venue']:
            assert (home[col] == -away[col]).all(), f'home {col} != -away {col}'

        # rolling stats, NaN-aware: printed, not asserted
        for window in [5, 10, 20, 30]:
            for col, opp_col in [(f'avg_scoring_{window}', f'opp_avg_scoring_{window}'),
                                 (f'avg_conceded_{window}', f'opp_avg_conceded_{window}')]:
                for row, opp_row, side in [(home, away, 'home'), (away, home, 'away')]:
                    is_mismatch = ~np.isclose(row[col], opp_row[opp_col], equal_nan=True)
                    if is_mismatch.any():
                        print(f'{side} {col} != opp {opp_col} ({is_mismatch.sum()} cases)')

    @staticmethod
    def _test_no_duplicate_columns(scoring: pd.DataFrame):
        """Check that no two numeric columns are identical."""
        numeric = scoring.select_dtypes(include=[np.number])
        duplicates = []
        cols = numeric.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                both_valid = numeric[cols[i]].notna() & numeric[cols[j]].notna()
                if both_valid.sum() == 0:
                    continue
                if np.allclose(numeric.loc[both_valid, cols[i]], numeric.loc[both_valid, cols[j]]):
                    duplicates.append((cols[i], cols[j]))
        if duplicates:
            print(f'duplicate column pairs: {duplicates}')
        assert not duplicates, f'found {len(duplicates)} duplicate column pairs'

    def test_features(self, scoring: pd.DataFrame):
        """Run all feature validation tests."""

        assert self.number_matches == scoring.shape[0] // 2, 'number of matches mismatched'
        assert self.number_matches == scoring['match_id'].nunique(), 'number of matches mismatched'

        self._test_no_same_day_duplicates(scoring)
        self._test_days_since_last_game(scoring)
        self._test_games_in_season(scoring)
        self._test_rolling_stats(scoring)
        self._test_no_negative_values(scoring)
        self._test_match_symmetry(scoring)
        self._test_no_duplicate_columns(scoring)

    def _train_validation_test(self, dataset: pd.DataFrame):
        """Split data ensuring each match_id appears in only one set."""

        unique_matches = dataset[['match_id', 'season']].drop_duplicates()

        # split by season for test set
        test_match_ids = unique_matches.loc[unique_matches['season'] >= self.train_test_split_season, 'match_id']
        train_val_match_ids = unique_matches.loc[unique_matches['season'] < self.train_test_split_season, 'match_id']

        # split train/validation on match_id level
        train_match_ids, val_match_ids = train_test_split(train_val_match_ids,
                                                          test_size=self.eval_set_size,
                                                          shuffle=True,
                                                          random_state=self.random_state)

        # create datasets using match_id filters
        train_set = dataset[dataset['match_id'].isin(train_match_ids)]
        eval_set = dataset[dataset['match_id'].isin(val_match_ids)]
        test_set = dataset[dataset['match_id'].isin(test_match_ids)]

        # verify no leakage
        assert len(set(train_set['match_id']) & set(eval_set['match_id'])) == 0, "Train/validation leak detected!"
        assert len(set(train_set['match_id']) & set(test_set['match_id'])) == 0, "Train/test leak detected!"
        assert len(set(eval_set['match_id']) & set(test_set['match_id'])) == 0, "Validation/test leak detected!"

        # verify all matches accounted for
        total_unique = dataset['match_id'].nunique()
        split_total = train_set['match_id'].nunique() + eval_set['match_id'].nunique() + test_set['match_id'].nunique()
        assert total_unique == split_total, f"Match count mismatch: {total_unique} != {split_total}"

        # save to feather
        save_feather(train_set, config.soccer['outcomes']['datasets']['train'])
        save_feather(eval_set, config.soccer['outcomes']['datasets']['eval'])
        save_feather(test_set, config.soccer['outcomes']['datasets']['test'])

    def transform(self) -> pd.DataFrame:
        """Complete feature engineering pipeline: dataset.feather and its train, eval, test split."""

        matches = pd.read_feather(config.soccer['matches'], columns=self.input_columns)

        matches = self._prepare_matches(matches)
        self.number_matches = matches.shape[0]

        features = self._features(matches)
        features = self._team_leagues(features)
        features = self._transform_data(features)
        features = self._postprocessing(features)

        save_feather(features, config.soccer['outcomes']['datasets']['dataset'])

        print('features saved')

        self.test_features(features)

        self._train_validation_test(features)

        return features
