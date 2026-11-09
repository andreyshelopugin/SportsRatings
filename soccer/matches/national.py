import numpy as np
import pandas as pd

from config import config
from soccer.constants import FINAL_STAGE_HOSTS, FINAL_STAGE_START, KNOCKOUT_ONLY_TOURNAMENTS, PANDEMIC_START, PANDEMIC_END, FINAL_STAGE_HOME_INDICATOR_OVERRIDES
from soccer.matches.base import DataPreprocessor
from utils.functions import save_feather


class NationalDataPreprocessor(DataPreprocessor):

    def __init__(self):
        super().__init__()
        self.matches_columns = ['country', 'tournament', 'season', 'full_date', 'home_team', 'away_team', 'home_suffix',
                                'away_suffix', 'home_score', 'away_score', 'date_raw', 'long_note', 'start_season_year', 'note', 'tournament_type',
                                'is_with_full_date', 'season_id'] + self.new_format_columns + self.notes_columns

        # final tournaments are neutral by default
        # host gets home advantage only when host team plays
        self.final_stage_hosts = FINAL_STAGE_HOSTS

        # finals date bounds per (tournament, season), see constants
        self.final_stage_start = FINAL_STAGE_START

        # final stages without group stage: every match is a knockout
        self.knockout_only_tournaments = KNOCKOUT_ONLY_TOURNAMENTS

        # fallback window for any host tournament-season missing from final_stage_start
        self.final_stage_window_days = 60

    def _get_data(self) -> pd.DataFrame:
        """Initial preprocessing of raw data with match results."""

        matches = pd.read_feather(config.soccer['national_matches'], self.matches_columns)

        duplicate_cols = [
            'tournament',
            'season',
            'full_date',
            'home_team',
            'away_team',
            'home_score',
            'away_score',
        ]

        # Some World Cup final-stage matches are duplicated under continents.
        # Keep the row with country == World.
        matches['is_world_row'] = matches['country'].eq('World')

        matches = (matches
                   .sort_values('is_world_row', ascending=False)
                   .drop_duplicates(duplicate_cols, keep='first')
                   .drop(columns='is_world_row')
                   .reset_index(drop=True))

        matches = self._handle_visit_stadium(matches)

        matches = (matches
                   .rename(columns={'season': 'season_name',
                                    'start_season_year': 'season',
                                    'full_date': 'date'})
                   .sort_values('date'))

        matches = self._duplicated_matches_with_different_seasons(matches)
        matches = self._drop_duplicate_matches(matches)

        # matches played during the pandemic
        matches['is_pandemic'] = (matches['date'] > PANDEMIC_START) & (matches['date'] < PANDEMIC_END)

        # season start, end
        # but be careful as cups can start too early
        matches['start_season'] = matches.groupby(['country', 'season'])['date'].transform('min')
        matches['end_season'] = matches.groupby(['country', 'season'])['date'].transform('max')

        # create match id
        matches = matches.reset_index(drop=True).reset_index().rename(columns={'index': 'match_id'})

        return matches

    @staticmethod
    def _drop_duplicate_matches(matches: pd.DataFrame) -> pd.DataFrame:
        """The same match (day, home team, away team) loaded twice with a different kick-off time
        (old rows were scraped in different timezones). _duplicated_matches_with_different_seasons catches only exact duplicates.
        Otherwise _stable_match_id fails on the same day and teams. Kept copy: the copy with the original note."""

        matches = matches.assign(match_day=matches['date'].dt.normalize(),
                                 has_note=matches['original_long_note'].notna())

        matches = matches.sort_values(['has_note', 'date'], ascending=[False, True], kind='stable')

        key_columns = ['match_day', 'home_team', 'away_team']
        is_duplicate = matches.duplicated(key_columns, keep='first')

        if is_duplicate.any():
            duplicate_keys = matches.loc[is_duplicate, key_columns].drop_duplicates()
            all_copies = matches.merge(duplicate_keys, on=key_columns, how='inner')
            print(f'#duplicate national matches dropped (same day and teams): {is_duplicate.sum()}')
            print(all_copies[['tournament', 'season', 'date', 'home_team', 'away_team', 'home_score', 'away_score']]
                  .sort_values(['date', 'home_team']).to_string(index=False))

        matches = (matches
                   .loc[~is_duplicate]
                   .drop(columns=['match_day', 'has_note'])
                   .sort_values('date', kind='stable'))

        return matches

    def _final_stage_bounds(self, tournament: str, season, is_tournament: pd.Series,
                            matches: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Resolve (start, end) dates bounding the final stage for one tournament-season.

        Uses the hardcoded final_stage_start entry when present:
          - a bare 'YYYY-MM-DD' string sets the start; end defaults to the season max.
          - a (start, end) pair sets both, guarding against stray late-dated rows.
        Falls back to (max_date - window_days, max_date) when no entry exists.
        """
        season_max = matches.loc[is_tournament, 'date'].max()

        entry = self.final_stage_start.get((tournament, season))

        if entry is None:
            print(f'no final stage dates for {tournament} {season}, fallback window is used')
            return season_max - pd.Timedelta(days=self.final_stage_window_days), season_max

        # the dates of the entries are days, the match dates are UTC times (28.10): the end day is included
        # (the final of the World Cup 2010 at 19:30 was after '2010-07-11 00:00'), an evening match in the Americas
        # can be the day before the start in UTC (Gold Cup 2003: 2003-07-11 23:00 for '2003-07-12')
        margin = pd.Timedelta(days=1)

        if isinstance(entry, (tuple, list)):
            start, end = entry
            # return pd.Timestamp(start), pd.Timestamp(end)
            return pd.Timestamp(start) - margin, pd.Timestamp(end) + margin

        # return pd.Timestamp(entry), season_max
        return pd.Timestamp(entry) - margin, season_max

    def _neutral_venue(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Create home advantage indicator for national team matches.

        home_indicator:
             1: home_team has real home advantage
             0: neutral venue
            -1: away_team has real home advantage
        """

        matches['is_neutral_venue'] = matches['is_neutral_venue'].fillna(False).astype(bool)

        # by default, national team matches are home-away
        matches['home_indicator'] = np.where(matches['is_neutral_venue'] == True, 0, 1)

        matches['is_final_stage'] = False
        matches['home_team_is_host'] = False
        matches['away_team_is_host'] = False

        for (tournament, season), hosts in self.final_stage_hosts.items():
            is_tournament = ((matches['tournament'] == tournament) & (matches['season'] == season))

            if not is_tournament.any():
                continue

            final_stage_start_date, final_stage_end_date = self._final_stage_bounds(tournament, season, is_tournament, matches)

            is_tournament_final_stage = (is_tournament
                                         & (matches['date'] >= final_stage_start_date)
                                         & (matches['date'] <= final_stage_end_date))

            is_home_host = is_tournament_final_stage & matches['home_team'].isin(hosts)
            is_away_host = is_tournament_final_stage & matches['away_team'].isin(hosts)

            matches.loc[is_tournament_final_stage, 'is_final_stage'] = True
            matches.loc[is_home_host, 'home_team_is_host'] = True
            matches.loc[is_away_host, 'away_team_is_host'] = True

            # final stage is neutral by default
            matches.loc[is_tournament_final_stage, 'is_neutral_venue'] = True
            matches.loc[is_tournament_final_stage, 'home_indicator'] = 0

            # host team gets home advantage
            # both teams are hosts (multi-host tournaments): the listed home team is assumed to play at home,
            # exceptions are in FINAL_STAGE_HOME_INDICATOR_OVERRIDES
            matches.loc[is_home_host, 'is_neutral_venue'] = False
            matches.loc[is_home_host, 'home_indicator'] = 1

            is_only_away_host = is_away_host & ~is_home_host
            matches.loc[is_only_away_host, 'is_neutral_venue'] = False
            matches.loc[is_only_away_host, 'home_indicator'] = -1

        # multi-host tournaments: hosts playing outside their country
        for (tournament, season, home_team, away_team), indicator in FINAL_STAGE_HOME_INDICATOR_OVERRIDES.items():
            is_match = ((matches['tournament'] == tournament)
                        & (matches['season'] == season)
                        & (matches['home_team'] == home_team)
                        & (matches['away_team'] == away_team))

            if not is_match.any():
                print('override match not found:', tournament, season, home_team, away_team)

            matches.loc[is_match, 'home_indicator'] = indicator
            matches.loc[is_match, 'is_neutral_venue'] = indicator == 0

        # explicit neutral note is stronger for non-final-stage matches
        matches.loc[matches['is_neutral_venue'] & ~matches['is_final_stage'], 'home_indicator'] = 0

        matches['home_indicator'] = matches['home_indicator'].astype('int8')

        matches = matches.rename(columns={'is_neutral_venue': 'is_neutral'})

        return matches

    def _extra_time_possibility(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Whether extra time would have been played in case of a draw (level aggregate for second legs).

        Only information known before the match: it is a feature of the goals model.

        Rules, in order of priority:
          - replay if draw                                      → False
          - first leg of a two-legged tie                       → False
          - second leg (leg == 2 or first leg result known)     → True
          - knockout by tournament_stage (new rows)             → True
          - final stage knockout                                → True
          - other matches (qualifiers groups, Nations League
            league phase, final stage group matches)            → False

        Final stage knockout:
          - tournaments without group stage (knockout_only_tournaments): every final stage match;
          - other tournaments: team's k-th match in the edition is after the group stage.
            Number of group matches in the edition = minimal number of final stage matches played by a team,
            as teams eliminated in groups play only group matches.
            Only for finished editions: exactly one match (the final) on the last date.

        Must be called after _neutral_venue (is_final_stage).
        """
        is_extra_time_played = matches['note'].isin(['AET', 'Pen'])
        is_replayed_if_draw = matches['is_replayed_if_draw'] == True
        is_first_leg = matches['leg'] == 1
        is_second_leg = (matches['leg'] == 2) | matches['first_leg_home'].notna()

        is_knockout_only_final_stage = ((matches['is_final_stage'] == True)
                                        & matches['tournament'].isin(self.knockout_only_tournaments))

        final_stage = matches.loc[(matches['is_final_stage'] == True) & ~is_knockout_only_final_stage,
        ['match_id', 'tournament', 'season', 'date', 'home_team', 'away_team']]

        # unfinished editions: several matches on the last date
        last_date = final_stage.groupby(['tournament', 'season'])['date'].transform('max')
        n_last_date_matches = (final_stage
                               .loc[final_stage['date'] == last_date]
                               .groupby(['tournament', 'season'])
                               .size())
        unfinished_editions = set(n_last_date_matches[n_last_date_matches > 1].index)

        # data ends during the edition (e.g. data scraped in the middle of the group stage)
        edition_last_date = final_stage.groupby(['tournament', 'season'])['date'].max()
        data_last_date = matches['date'].max()
        in_progress = edition_last_date[edition_last_date >= data_last_date - pd.Timedelta(days=3)]
        unfinished_editions |= set(in_progress.index)
        if unfinished_editions:
            print('final stage editions considered unfinished:', unfinished_editions)

        is_finished_edition = ~pd.Series(list(zip(final_stage['tournament'], final_stage['season'])),
                                         index=final_stage.index).isin(unfinished_editions)
        final_stage = final_stage.loc[is_finished_edition]

        # team's k-th match in the edition
        appearances = pd.concat([final_stage.drop(columns='away_team').rename(columns={'home_team': 'team'}),
                                 final_stage.drop(columns='home_team').rename(columns={'away_team': 'team'})],
                                ignore_index=True)

        appearances = appearances.sort_values(['date', 'match_id'])
        appearances['team_match_number'] = appearances.groupby(['tournament', 'season', 'team']).cumcount() + 1
        appearances['n_team_matches'] = appearances.groupby(['tournament', 'season', 'team'])['match_id'].transform('count')
        # number of group matches = the most common number of matches per team (min of modes),
        # plain min breaks when a team withdrew or a match was dropped as forfeited (AFCON 2010, Gold Cup 2017)
        per_team = appearances.drop_duplicates(['tournament', 'season', 'team'])
        n_group_matches = (per_team
                           .groupby(['tournament', 'season'])['n_team_matches']
                           .agg(lambda x: x.mode().min())
                           .rename('n_group_matches'))
        appearances = appearances.merge(n_group_matches, how='left', on=['tournament', 'season'], validate='m:1')

        knockout_ids = appearances.loc[appearances['team_match_number'] > appearances['n_group_matches'], 'match_id'].unique()
        is_final_stage_knockout = matches['match_id'].isin(knockout_ids) | is_knockout_only_final_stage

        _, is_knockout_stage = self._knockout_stage(matches)

        conditions = [
            is_replayed_if_draw,
            is_first_leg,
            is_second_leg,
            is_knockout_stage,
            is_final_stage_knockout,
        ]
        possibilities = [False, False, True, True, True]
        matches['is_extra_time_possible'] = np.select([condition.fillna(False).astype(bool) for condition in conditions],
                                                      possibilities, default=False).astype(bool)

        # sanity check: extra time was possible, the match (or aggregate) was level, but no AET/Pen note
        aggregate_difference = (matches['aggregate_home'] - matches['aggregate_away']).astype('Float64')
        score_difference = (matches['home_score'] - matches['away_score']).astype('Float64')
        level_difference = aggregate_difference.where(aggregate_difference.notna(), score_difference)
        is_suspicious = (matches['is_extra_time_possible']
                         & ~is_extra_time_played
                         & (level_difference == 0).fillna(False).astype(bool))
        if is_suspicious.any():
            print('#extra time possible, level result, but no AET/Pen note:', is_suspicious.sum())
            print(matches.loc[is_suspicious, ['tournament', 'season', 'date', 'home_team', 'away_team',
                                              'home_score', 'away_score', 'aggregate_home', 'aggregate_away']])

        return matches

    @staticmethod
    def _hosts_test(matches: pd.DataFrame):
        """Final stage matches where the listed away team is the host."""

        away_hosts = matches.loc[matches['home_indicator'] == -1, ['tournament', 'season', 'home_team', 'away_team']]
        print('#home_indicator == -1:', away_hosts.shape[0])

        # both teams are hosts, but the listed away team got home advantage: must be empty,
        # except manual FINAL_STAGE_HOME_INDICATOR_OVERRIDES (e.g. Euro 2020 final Italy - England at Wembley)
        is_override = pd.Series(list(zip(matches['tournament'], matches['season'], matches['home_team'], matches['away_team'])),
                                index=matches.index).isin(set(FINAL_STAGE_HOME_INDICATOR_OVERRIDES))

        both_hosts = matches.loc[(matches['home_team_is_host'] == True)
                                 & (matches['away_team_is_host'] == True)
                                 & (matches['home_indicator'] == -1)
                                 & ~is_override]

        assert both_hosts.empty, f'both hosts with home_indicator == -1: {both_hosts.shape[0]}'

    def transform(self):
        """"""

        matches = self._get_data()

        matches = matches[matches['tournament'] != 'CONCACAF Central American Cup']

        matches = self._neutral_venue(matches)
        matches = self._outcomes(matches)
        matches = self._extra_time_possibility(matches)
        matches = self._five_outcomes(matches)

        save_feather(matches, config.soccer['preprocessed_national_matches'])

        self._hosts_test(matches.copy())

        return matches
