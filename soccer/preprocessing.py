import numpy as np
import pandas as pd

from config import config
from soccer.constants import (
    FINAL_STAGE_HOSTS,
    FINAL_STAGE_START,
    KNOCKOUT_ONLY_TOURNAMENTS,
    PANDEMIC_START,
    PANDEMIC_END,
    FINAL_STAGE_HOME_INDICATOR_OVERRIDES,
    CLUBS_NAMED_AS_COUNTRIES, CLUB_WORLD_CUP_HOSTS,
)
from utils.functions import save_feather


class DataPreprocessor(object):
    """Parent class for preprocessing club and national team raw data."""

    def __init__(self):
        # columns of the new Flashscore format (match pages), empty for old rows
        self.new_format_columns = ['flashscore_id', 'round', 'tournament_stage', 'home_penalty', 'away_penalty',
                                   'home_team_id', 'away_team_id', 'venue', 'venue_city', 'capacity', 'attendance',
                                   'referee', 'referee_country']

        # columns of the notes of Flashscore, parsed in the Flashscore project (MatchNotes)
        self.notes_columns = ['original_long_note', 'is_on_visit_stadium', 'is_no_spectators', 'is_different_stadium',
                              'is_neutral_venue', 'is_limited_spectators', 'is_replayed_if_draw', 'leg',
                              'first_leg_home', 'first_leg_away', 'aggregate_home', 'aggregate_away']

        # knockout rounds in the 'round' column
        self.knockout_round_pattern = r'^(Final|Semi-finals|Quarter-finals|1/\d+-finals|3rd place|4th place|5th place)$'

        self.matches_columns = ['country', 'tournament', 'season', 'full_date', 'home_team', 'away_team', 'home_suffix',
                                'away_suffix', 'home_score', 'away_score', 'date_raw', 'long_note', 'start_season_year', 'note', 'tournament_type',
                                'is_with_full_date', 'season_id'] + self.new_format_columns + self.notes_columns

        self.domestic_tournaments = {'first_league', 'second_league', 'cup', 'super_cup'}
        self.international_tournaments = {'international_cup', 'international_super_cup'}
        self.club_tournaments = {'first_league', 'second_league', 'cup', 'super_cup',
                                 'international_cup', 'international_super_cup'}

    def _knockout_stage(self, matches: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """(has_stage, is_knockout) from tournament_stage and round of the new rows, both are known before the match.

        Knockout:
          - knockout round ('Final', '1/8-finals', ...) and tournament_stage ends with it ('Play Offs - Final').
            The round alone is not enough: World Cup qualification group matches have round 'Final';
          - or the structure of the stage: every team of the stage plays only one opponent (a pair).
        """
        stage = matches['tournament_stage'].fillna('')
        round_name = matches['round'].fillna('')

        has_stage = stage != ''

        is_knockout_round = (round_name.str.match(self.knockout_round_pattern)
                             & pd.Series([s.endswith(r) for s, r in zip(stage, round_name)], index=matches.index))

        # number of opponents of each team in the stage,
        # without the matchday: 'Group Stage - Round 3' is one round of the group stage, not a stage
        stage_keys = ['country', 'tournament', 'season', 'stage_without_round']
        staged = (matches
                  .assign(stage_without_round=stage.str.replace(r'\s*-\s*Round \d+$', '', regex=True))
                  .loc[has_stage, stage_keys + ['home_team', 'away_team']])
        appearances = pd.concat([staged[stage_keys + ['home_team', 'away_team']].set_axis(stage_keys + ['team', 'opponent'], axis=1),
                                 staged[stage_keys + ['away_team', 'home_team']].set_axis(stage_keys + ['team', 'opponent'], axis=1)])
        max_opponents = (appearances
                         .groupby(stage_keys + ['team'])['opponent'].nunique()
                         .groupby(stage_keys).max()
                         .rename('max_opponents'))

        max_opponents = (matches
                         .assign(stage_without_round=stage.str.replace(r'\s*-\s*Round \d+$', '', regex=True))
                         [stage_keys]
                         .join(max_opponents, on=stage_keys)['max_opponents'])
        is_pair_stage = has_stage & (max_opponents == 1)

        is_knockout = has_stage & (is_knockout_round | is_pair_stage)

        return has_stage, is_knockout

    @staticmethod
    def _handle_visit_stadium(matches: pd.DataFrame) -> pd.DataFrame:
        """Swap home/away teams (with their Flashscore ids), scores, suffixes, first leg result and aggregate
        for matches played on the visiting team's stadium."""

        is_on_visit_stadium = matches['is_on_visit_stadium'] == True

        home_team = matches['home_team'].copy()
        away_team = matches['away_team'].copy()
        home_score = matches['home_score'].copy()
        away_score = matches['away_score'].copy()
        home_suffix = matches['home_suffix'].copy()
        away_suffix = matches['away_suffix'].copy()
        home_team_id = matches['home_team_id'].copy()
        away_team_id = matches['away_team_id'].copy()
        aggregate_home = matches['aggregate_home'].copy()
        aggregate_away = matches['aggregate_away'].copy()
        first_leg_home = matches['first_leg_home'].copy()
        first_leg_away = matches['first_leg_away'].copy()

        matches['home_team'] = np.where(is_on_visit_stadium, away_team, home_team)
        matches['away_team'] = np.where(is_on_visit_stadium, home_team, away_team)
        matches['home_score'] = np.where(is_on_visit_stadium, away_score, home_score)
        matches['away_score'] = np.where(is_on_visit_stadium, home_score, away_score)
        matches['home_suffix'] = np.where(is_on_visit_stadium, away_suffix, home_suffix)
        matches['away_suffix'] = np.where(is_on_visit_stadium, home_suffix, away_suffix)

        # ids must follow the names, otherwise the id of a team is the id of its opponent (Raja - Bayern, 2013)
        matches['home_team_id'] = home_team_id.where(~is_on_visit_stadium, away_team_id)
        matches['away_team_id'] = away_team_id.where(~is_on_visit_stadium, home_team_id)
        matches['aggregate_home'] = aggregate_home.where(~is_on_visit_stadium, aggregate_away)
        matches['aggregate_away'] = aggregate_away.where(~is_on_visit_stadium, aggregate_home)

        # "1st leg result" is written in the current match orientation, so it is swapped as well
        matches['first_leg_home'] = first_leg_home.where(~is_on_visit_stadium, first_leg_away)
        matches['first_leg_away'] = first_leg_away.where(~is_on_visit_stadium, first_leg_home)

        matches = matches.drop(columns='is_on_visit_stadium')

        return matches

    @staticmethod
    def _duplicated_matches_with_different_seasons(matches: pd.DataFrame) -> pd.DataFrame:
        """Remove identical matches that appear under different seasons."""
        subset = ['date', 'country', 'home_team', 'away_team', 'home_score', 'away_score']

        # duplicated = matches.duplicated(subset=subset, keep=False)
        # if duplicated.any():
        #     print(f'dropping {duplicated.sum() // 2} duplicated matches')
        #     print(matches[duplicated].sort_values(['date', 'home_team']))

        n_matches = matches.shape[0]

        # sort so non-null original_long_note comes first
        matches = (matches
                   .sort_values('original_long_note', na_position='last')
                   .drop_duplicates(subset=subset, keep='first')
                   .sort_values('date'))

        print('#duplicated matches with different seasons:', n_matches - matches.shape[0])

        return matches

    @staticmethod
    def _same_date_matches_warnings(matches: pd.DataFrame):
        """Find matches where a team played two or more times on the same date.
        Some cannot be fixed automatically, so just log them."""

        df = matches.copy()

        # day without time: the same match can be loaded with different kick-off times
        df['date'] = df['date'].dt.normalize()

        # no team should play twice on the same date
        all_appearances = pd.concat([df[['date', 'home_team']].rename(columns={'home_team': 'team'}),
                                     df[['date', 'away_team']].rename(columns={'away_team': 'team'})])

        # after renaming
        duplicates = all_appearances.duplicated(subset=['date', 'team'], keep=False)
        if duplicates.any():
            dup_dates_teams = all_appearances[duplicates].drop_duplicates()
            home_match = df.set_index(['date', 'home_team']).index.isin(dup_dates_teams.set_index(['date', 'team']).index)
            away_match = df.set_index(['date', 'away_team']).index.isin(dup_dates_teams.set_index(['date', 'team']).index)
            wrong_rows = df[home_match | away_match].sort_values(['date', 'home_team'])
            print(f'# matches where a team played multiple times on same date: {len(wrong_rows)}')

    @staticmethod
    def _outcomes(matches: pd.DataFrame) -> pd.DataFrame:
        """"""
        conditions = [(matches['home_score'] > matches['away_score']),
                      (matches['home_score'] == matches['away_score']),
                      (matches['home_score'] < matches['away_score'])]

        outcomes = ['H', 'D', 'A']
        matches['outcome'] = np.select(conditions, outcomes, default='bug')

        assert (matches['outcome'] != 'bug').all(), 'Some matches have outcome = "bug", check your conditions'

        return matches

    @staticmethod
    def _five_outcomes(matches: pd.DataFrame) -> pd.DataFrame:
        """Five-outcome labels for matches where extra time is possible.

        'H' / 'A'                              — win in regular time (90')
        'D'                                    — draw in regular time, no extra time played
        'HOME_EXTRA_TIME' / 'AWAY_EXTRA_TIME'  — win in extra time (note == 'AET')
        'DRAW_EXTRA_TIME'                      — draw after extra time → penalties (note == 'Pen')

        The Flashscore score for AET/Pen matches is the score after extra time.
        For second legs the match score may be unequal even when extra time ended level
        (e.g. aggregate tied → extra time 0-0 → penalties), so:
          - Pen → always 'DRAW_EXTRA_TIME';
          - AET → the tie winner: by aggregate if it is known and unequal,
                  otherwise by the match score.

        Must be called after _extra_time_possibility.
        """
        is_after_extra_time = matches['note'] == 'AET'
        is_penalties = matches['note'] == 'Pen'
        is_regular_time = ~is_after_extra_time & ~is_penalties

        # tie winner for extra time: aggregate first (second legs), then match score
        aggregate_difference = (matches['aggregate_home'] - matches['aggregate_away']).astype('Float64')
        score_difference = (matches['home_score'] - matches['away_score']).astype('Float64')
        is_aggregate_decisive = aggregate_difference.notna() & (aggregate_difference != 0)
        extra_time_difference = aggregate_difference.where(is_aggregate_decisive, score_difference)

        conditions = [
            # penalties or draw after extra time (MLS 2000-2003 golden goal era, away goals rule after extra time)
            is_penalties | (is_after_extra_time & (extra_time_difference == 0)),
            is_after_extra_time & (extra_time_difference > 0),
            is_after_extra_time & (extra_time_difference < 0),
            is_regular_time & (matches['home_score'] > matches['away_score']),
            is_regular_time & (matches['home_score'] == matches['away_score']),
            is_regular_time & (matches['home_score'] < matches['away_score']),
        ]
        outcomes = ['DRAW_EXTRA_TIME', 'HOME_EXTRA_TIME', 'AWAY_EXTRA_TIME', 'HOME_REGULAR_TIME', 'DRAW_REGULAR_TIME', 'AWAY_REGULAR_TIME']
        matches['outcome_5'] = np.select([condition.fillna(False).astype(bool) for condition in conditions],
                                         outcomes, default='bug')

        is_bug = matches['outcome_5'] == 'bug'
        if is_bug.any():
            print('#five outcomes bugs:', is_bug.sum())
            print(matches.loc[is_bug, ['date', 'home_team', 'away_team', 'home_score', 'away_score',
                                       'note', 'aggregate_home', 'aggregate_away']])

        matches['is_extra_time'] = is_after_extra_time | is_penalties

        # regular time (90') outcome: single matches with extra time were draws,
        # second legs: 90' goal difference levelled the aggregate ("1st leg result" is in the current match orientation),
        # replays: first match was a draw, so the difference is 0 as well
        extra_time_regular_difference = (matches['first_leg_away'] - matches['first_leg_home']).astype('Float64').fillna(0)
        regular_time_difference = score_difference.where(~matches['is_extra_time'], extra_time_regular_difference)

        matches['outcome_regular_time'] = np.select([(regular_time_difference > 0).fillna(False).astype(bool),
                                                     (regular_time_difference == 0).fillna(False).astype(bool),
                                                     (regular_time_difference < 0).fillna(False).astype(bool)],
                                                    ['H', 'D', 'A'], default='bug')

        assert (matches['outcome_regular_time'] != 'bug').all(), 'Some matches have outcome_regular_time = "bug"'

        # is_extra_time_possible is known before the match, so a few matches with extra time can be outside of it
        # (stage labels of Flashscore), they keep their extra time outcomes
        is_unexpected_extra_time = matches['is_extra_time'] & ~matches['is_extra_time_possible']
        if is_unexpected_extra_time.any():
            print('#extra time played, but not possible by the stage:', is_unexpected_extra_time.sum())

        return matches


class ClubDataPreprocessor(DataPreprocessor):

    def __init__(self):
        super().__init__()
        self.matches_columns = ['country', 'tournament', 'season', 'full_date', 'home_team', 'away_team', 'home_suffix',
                                'away_suffix', 'home_score', 'away_score', 'date_raw', 'long_note', 'start_season_year', 'note', 'tournament_type',
                                'is_with_full_date', 'season_id'] + self.new_format_columns + self.notes_columns

        # Club World Cup: host country by edition (see _club_world_cup_hosts)
        self.club_world_cup_hosts = CLUB_WORLD_CUP_HOSTS

        self.domestic_tournaments = {'first_league', 'second_league', 'cup', 'super_cup'}
        self.international_tournaments = {'international_cup', 'international_super_cup'}
        self.club_tournaments = {'first_league', 'second_league', 'cup', 'super_cup',
                                 'international_cup', 'international_super_cup'}

        # promotion / relegation playoffs inside championships (see _tag_playoffs)
        self.playoff_core_share = 0.5
        self.playoff_tail_days = 14
        self.playoff_break_days = 45
        self.playoff_max_legs = 2
        self.playoff_max_pairs_per_season = 3
        self.playoff_max_non_core_share = 0.15

        # first leagues containing the relegation group with second league teams (see _tag_relegation_groups)
        self.relegation_group_leagues = {'Moldova. First League'}

    @staticmethod
    def _drop_cup_matches_with_non_championship_teams(matches: pd.DataFrame) -> pd.DataFrame:
        """Drop cup matches where either team never played in a domestic league.
        Avoids ratings inflation, as these teams are typically weak but would be
        initialized with an average rating."""

        championships = matches[matches['is_championship'] == True]

        teams_with_championships = set(championships['home_team']).union(set(championships['away_team']))

        n_matches = matches.shape[0]

        # skip international cups here because pre-renaming team names contain brackets
        matches = matches[(matches['tournament_type'] != 'cup')
                          | (matches['home_team'].isin(teams_with_championships)
                             & matches['away_team'].isin(teams_with_championships))]

        print('#dropped cup matches with non-championship team:', n_matches - matches.shape[0])

        return matches

    def _preprocessing(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Initial preprocessing of raw data with match results."""

        # leave club tournaments
        matches = matches[matches['tournament_type'].isin(self.club_tournaments)]

        matches['is_championship'] = matches['tournament_type'].isin({'first_league', 'second_league'})

        matches = self._drop_cup_matches_with_non_championship_teams(matches)

        # two teams of one country under the same name play each other (Hapoel Jerusalem, Tokushima):
        # they cannot be told apart by the name, the matches are removed
        is_same_name = matches['home_team'] == matches['away_team']
        print(f'matches of two teams with the same name: {is_same_name.sum()}, '
              f'{matches.loc[is_same_name, "home_team"].value_counts().to_dict()}')
        matches = matches[~is_same_name]

        matches = self._handle_visit_stadium(matches)

        matches = (matches
                   .rename(columns={'season': 'season_name',
                                    'start_season_year': 'season',
                                    'full_date': 'date'})
                   .sort_values('date'))

        matches = self._duplicated_matches_with_different_seasons(matches)

        matches['is_domestic'] = matches['tournament_type'].isin(self.domestic_tournaments)

        # matches played during the pandemic
        matches['is_pandemic'] = (matches['date'] > PANDEMIC_START) & (matches['date'] < PANDEMIC_END)

        # season start, end
        # but be careful as cups can start too early
        matches['start_season'] = matches.groupby(['country', 'season'])['date'].transform('min')
        matches['end_season'] = matches.groupby(['country', 'season'])['date'].transform('max')

        # create match id
        matches = matches.reset_index(drop=True).reset_index().rename(columns={'index': 'match_id'})

        return matches

    def _data_tests(self, matches: pd.DataFrame):
        """"""

        df = matches.copy()

        # no duplicate rows
        assert not df[['country', 'tournament', 'season_name', 'season', 'date', 'home_team',
                       'away_team', 'home_score', 'away_score']].duplicated().any(), "there are duplicates in raw data"

        # required columns must exist and contain no NaNs
        required_cols = ['match_id', 'country', 'tournament', 'season_name', 'season',
                         'date', 'home_team', 'away_team',
                         'home_score', 'away_score', 'tournament_type']

        missing = df[required_cols].isna().any()
        assert not missing.any(), f"NaNs in columns: {missing[missing].index.tolist()}"

        # scores must be non-negative
        assert (df[['home_score', 'away_score']] >= 0).all().all(), "Negative scores found"

        assert (df[['home_score', 'away_score']] < 50).all().all(), "Huge scores found"

        # teams must not play against themselves
        self_play = df[df['home_team'] == df['away_team']]
        assert self_play.empty, f"Teams playing against themselves: {self_play.to_string()}"

        assert set(df['tournament_type']) == self.club_tournaments, f"Invalid tournament_type values"

        assert df['match_id'].nunique() == df.shape[0], "there are duplicated match_id"

    @staticmethod
    def _team_rows_by_side(matches: pd.DataFrame) -> pd.DataFrame:
        """A row per team and match. International names carry the country in brackets ('Valencia (Mdv)'):
        the bracket goes to code. In domestic names a bracket is a part of the name ('Austria Vienna (Am)')."""

        rows = pd.concat([matches[['match_id', 'date', 'season', 'country', 'is_domestic', f'{side}_team', f'{side}_team_id']]
                          .set_axis(['match_id', 'date', 'season', 'country', 'is_domestic', 'name', 'team_id'], axis=1)
                          .assign(side=side)
                          for side in ['home', 'away']], ignore_index=True)

        rows['team_id'] = rows['team_id'].fillna('')

        is_international = rows['is_domestic'] != True
        rows['code'] = rows['name'].str.extract(r'\((\w+)\)\s*$')[0].where(is_international)
        rows.loc[is_international, 'name'] = rows.loc[is_international, 'name'].str.split(' (', regex=False).str[0].str.strip()

        return rows

    @staticmethod
    def _fill_team_ids(rows: pd.DataFrame) -> pd.DataFrame:
        """A row without id gets the id of its name, if the name maps to exactly one id:
        (name, country, season), else (name, country), else the id with the nearest matches in time
        (two clubs of one name in a country: Hapoel Jerusalem), else the only id of the name.
        The country of a row: domestic - the country of the tournament; international - the country of the bracket code
        (learned from the rows with ids), else the only country of the name in domestic matches.
        The ids are learned only from the rows with a sure country: domestic rows and bracket codes
        ('Valencia (Mdv)' of the Maldives is not the only domestic Valencia, of Spain)."""

        domestic = rows['is_domestic'] == True
        has_id = rows['team_id'] != ''

        id_country = rows.loc[domestic & has_id].groupby('team_id')['country'].agg(lambda s: s.mode().iat[0])
        code_country = (rows.loc[~domestic & has_id & rows['code'].notna()]
                        .assign(id_country=lambda df: df['team_id'].map(id_country))
                        .dropna(subset=['id_country'])
                        .groupby('code')['id_country'].agg(lambda s: s.mode().iat[0]))

        name_countries = rows.loc[domestic].groupby('name')['country'].agg(['nunique', 'first'])
        only_country = name_countries.loc[name_countries['nunique'] == 1, 'first']

        # a bracket code without a learned country (the league of the country is not in the data: 'Valencia (Mdv)')
        # is not replaced by the only country of the name (Spain), the team stays without a country
        by_name = rows['name'].map(only_country).where(rows['code'].isna())
        rows['team_country'] = rows['country'].where(domestic, rows['code'].map(code_country).fillna(by_name))

        is_sure = domestic | rows['code'].map(code_country).notna()
        rows['is_sure_country'] = is_sure
        has_country = rows['team_country'].notna()

        for keys in (['name', 'team_country', 'season'], ['name', 'team_country']):
            ids = rows.loc[has_id & is_sure].groupby(keys)['team_id'].agg(['nunique', 'first'])
            ids = ids.loc[ids['nunique'] == 1, 'first']

            missing = has_country & (rows['team_id'] == '')
            rows.loc[missing, 'team_id'] = ids.reindex(pd.MultiIndex.from_frame(rows.loc[missing, keys])).fillna('').to_numpy()

        missing = has_country & (rows['team_id'] == '')
        id_dates = rows.loc[has_id & is_sure].groupby(['name', 'team_country', 'team_id'])['date'].agg(['min', 'max']).reset_index()
        candidates = rows.loc[missing, ['name', 'team_country', 'date']].reset_index().merge(id_dates, on=['name', 'team_country'])
        candidates['distance'] = (np.maximum(candidates['min'] - candidates['date'], candidates['date'] - candidates['max'])
                                  .dt.days.clip(lower=0))
        nearest = candidates.sort_values('distance').drop_duplicates('index').set_index('index')['team_id']
        rows.loc[nearest.index, 'team_id'] = nearest

        name_ids = rows.loc[has_id].groupby('name')['team_id'].agg(['nunique', 'first'])
        missing = ~has_country & (rows['team_id'] == '')
        rows.loc[missing, 'team_id'] = rows.loc[missing, 'name'].map(name_ids.loc[name_ids['nunique'] == 1, 'first']).fillna('')

        return rows

    @staticmethod
    def _unique_team_names(rows: pd.DataFrame, min_owner_matches=3) -> pd.Series:
        """{team: name}. A team is its id, a row without id is the team (name, country).
        The name of a team: the most common name in domestic matches; a name of several teams stays with the team
        with the latest match (with >= min_owner_matches matches), the others get the country ('River Plate Uruguay'),
        then the id."""

        rows['team'] = np.where(rows['team_id'] != '', rows['team_id'], rows['name'] + '|' + rows['team_country'].fillna(''))

        def most_common(df: pd.DataFrame, column: str) -> pd.Series:
            return (df
                    .groupby(['team', column])
                    .size()
                    .rename('n')
                    .reset_index()
                    .sort_values('n', ascending=False)
                    .drop_duplicates('team')
                    .set_index('team')[column])

        domestic = rows.loc[rows['is_domestic'] == True]

        teams = (rows
                 .groupby('team')
                 .agg(last_date=('date', 'max'), n_matches=('date', 'size'))
                 .assign(name=most_common(domestic, 'name').combine_first(most_common(rows, 'name')),
                         country=most_common(domestic, 'country')
                         .combine_first(most_common(rows.loc[rows['is_sure_country']], 'team_country'))))

        teams['is_regular'] = teams['n_matches'] >= min_owner_matches
        teams = teams.sort_values(['name', 'is_regular', 'last_date'], ascending=[True, False, False])
        is_owner = ~teams['name'].duplicated()

        names = teams['name'].where(is_owner, (teams['name'] + ' ' + teams['country'].fillna('')).str.strip())

        # still the same name (two clubs of one country, a team without a country): the name and the id
        is_id = ~teams.index.str.contains('|', regex=False)
        is_collision = names.duplicated(keep=False) & ~is_owner
        names = names.where(~is_collision, teams['name'] + ' ' + pd.Series(np.where(is_id, teams.index, 'unknown'), index=teams.index))
        assert names.is_unique, f'teams with the same name: {names[names.duplicated()].tolist()[:10]}'

        print(f'#teams: {len(teams)}, without id: {(~is_id).sum()}')

        return names

    def _rename_teams(self, matches: pd.DataFrame) -> pd.DataFrame:
        """A team is its Flashscore team id: clubs of one name in several countries (River Plate), a club in the cup
        of another country (Kitchee), renamed clubs (Al-Nil / Al Nil) are told apart without name lists."""

        rows = self._fill_team_ids(self._team_rows_by_side(matches))
        names = self._unique_team_names(rows)
        rows['team_name'] = rows['team'].map(names)

        for side in ['home', 'away']:
            matches[f'{side}_team'] = matches['match_id'].map(rows.loc[rows['side'] == side].set_index('match_id')['team_name'])

        # clubs named as countries collide with national teams in the union of clubs and national teams
        matches['home_team'] = matches['home_team'].replace(CLUBS_NAMED_AS_COUNTRIES)
        matches['away_team'] = matches['away_team'].replace(CLUBS_NAMED_AS_COUNTRIES)

        return matches

    @staticmethod
    def _championship_league(matches: pd.DataFrame) -> pd.Series:
        """Unified championship name (as in _tournament_preprocessing): 'Moldova. First League', NaN for other tournaments.
        Raw tournament names collide between countries, so country + league type is used."""

        league_names = {'first_league': 'First League', 'second_league': 'Second League'}

        return matches['country'] + '. ' + matches['tournament_type'].map(league_names)

    @staticmethod
    def _team_rows(championship_matches: pd.DataFrame) -> pd.DataFrame:
        """One row per team and match (match-level data), championship_matches must have the league column."""

        columns = ['match_id', 'league', 'season', 'date']

        return pd.concat([
            championship_matches[columns + ['home_team']].rename(columns={'home_team': 'team'}),
            championship_matches[columns + ['away_team']].rename(columns={'away_team': 'team'}),
        ], ignore_index=True)

    def _core_teams(self, championship_matches: pd.DataFrame) -> set:
        """{(league, season, team)} of core teams:
        matches in the league-season >= playoff_core_share * median over teams."""

        team_season_counts = (self._team_rows(championship_matches)
                              .groupby(['league', 'season', 'team'])['match_id']
                              .nunique()
                              .rename('n_matches')
                              .reset_index())

        season_median = team_season_counts.groupby(['league', 'season'])['n_matches'].transform('median')
        is_core = team_season_counts['n_matches'] >= self.playoff_core_share * season_median

        return set(team_season_counts.loc[is_core, ['league', 'season', 'team']].itertuples(index=False, name=None))

    @staticmethod
    def _drop_duplicate_matches(matches: pd.DataFrame) -> pd.DataFrame:
        """The same match (day, home team, away team) loaded twice, e.g. with a different time or under different tournaments.
        _duplicated_matches_with_different_seasons catches only exact duplicates (date with time, country, score).
        Kept copy: championship first, then the copy with the original note."""

        matches = matches.assign(match_day=matches['date'].dt.normalize(),
                                 has_note=matches['original_long_note'].notna())

        matches = matches.sort_values(['is_championship', 'has_note', 'date'], ascending=[False, False, True], kind='stable')

        key_columns = ['match_day', 'home_team', 'away_team']
        is_duplicate = matches.duplicated(key_columns, keep='first')

        if is_duplicate.any():
            duplicate_keys = matches.loc[is_duplicate, key_columns].drop_duplicates()
            all_copies = matches.merge(duplicate_keys, on=key_columns, how='inner')
            print(f'#duplicate matches dropped (same day and teams): {is_duplicate.sum()}')
            print(all_copies[['country', 'tournament', 'season', 'date', 'home_team', 'away_team', 'home_score', 'away_score']]
                  .sort_values(['date', 'home_team']).to_string(index=False))

        matches = (matches
                   .loc[~is_duplicate]
                   .drop(columns=['match_day', 'has_note'])
                   .sort_values('date', kind='stable'))

        return matches

    def _tag_playoffs(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Tag promotion / relegation playoffs inside championships: is_playoff = True, is_championship = False.

        Core team: matches in the league-season >= playoff_core_share * median.
        Playoff pair: one core and one non-core team, played at the end of the season,
        followed by a break of at least playoff_break_days, at most playoff_max_legs matches.
        Earlier rounds (Netherlands, Scotland): non-core vs non-core matches at the end of seasons with playoff pairs.

        Not tagged (printed): pairs where the non-core team has regular matches in the league (special formats)
        and league-seasons with more than playoff_max_pairs_per_season pairs.
        """

        season_keys = ['league', 'season']
        team_keys = season_keys + ['team']
        pair_keys = season_keys + ['team_1', 'team_2']

        matches['is_playoff'] = False

        championship_matches = matches.loc[matches['is_championship'] == True, ['match_id', 'season', 'date', 'home_team', 'away_team']].copy()
        championship_matches['league'] = self._championship_league(matches.loc[championship_matches.index])

        team_matches = self._team_rows(championship_matches)

        # core
        team_season_counts = (team_matches
                              .groupby(team_keys)['match_id']
                              .nunique()
                              .rename('n_matches')
                              .reset_index())

        team_season_counts['season_median'] = team_season_counts.groupby(season_keys)['n_matches'].transform('median')
        team_season_counts['is_core'] = team_season_counts['n_matches'] >= self.playoff_core_share * team_season_counts['season_median']

        core_teams = set(team_season_counts
                         .loc[team_season_counts['is_core'], team_keys]
                         .itertuples(index=False, name=None))

        championship_matches['is_home_core'] = [(league, season, team) in core_teams
                                                for league, season, team in zip(championship_matches['league'],
                                                                                championship_matches['season'],
                                                                                championship_matches['home_team'])]

        championship_matches['is_away_core'] = [(league, season, team) in core_teams
                                                for league, season, team in zip(championship_matches['league'],
                                                                                championship_matches['season'],
                                                                                championship_matches['away_team'])]

        # end of season: last core vs core match
        core_season_end = (championship_matches
                           .loc[championship_matches['is_home_core'] & championship_matches['is_away_core']]
                           .groupby(season_keys)['date']
                           .max()
                           .rename('core_season_end'))

        championship_matches = championship_matches.join(core_season_end, on=season_keys)
        championship_matches['is_tail'] = (championship_matches['date']
                                           >= championship_matches['core_season_end'] - pd.Timedelta(days=self.playoff_tail_days))

        # pair of teams independent of home / away
        championship_matches['team_1'] = np.minimum(championship_matches['home_team'], championship_matches['away_team'])
        championship_matches['team_2'] = np.maximum(championship_matches['home_team'], championship_matches['away_team'])

        # 1. mixed level pairs
        mixed_matches = championship_matches.loc[championship_matches['is_home_core'] != championship_matches['is_away_core']]

        pairs = (mixed_matches
                 .groupby(pair_keys)
                 .agg(first_leg_date=('date', 'min'),
                      last_leg_date=('date', 'max'),
                      n_legs=('match_id', 'nunique'),
                      core_season_end=('core_season_end', 'first'))
                 .reset_index())

        # 3. break after the last leg: next championship match of each team in any league
        data_end_date = matches['date'].max()

        team_dates = (team_matches[['team', 'date']]
                      .drop_duplicates()
                      .sort_values(['team', 'date']))
        team_dates['next_date'] = team_dates.groupby('team')['date'].shift(-1)

        # no next match: the break is known only up to the end of the data
        team_dates['next_date'] = team_dates['next_date'].fillna(data_end_date)
        next_match_date = dict(zip(zip(team_dates['team'], team_dates['date']), team_dates['next_date']))

        pairs['break_days'] = [min((next_match_date[(team_1, last_leg_date)] - last_leg_date).days,
                                   (next_match_date[(team_2, last_leg_date)] - last_leg_date).days)
                               for team_1, team_2, last_leg_date in zip(pairs['team_1'], pairs['team_2'], pairs['last_leg_date'])]

        is_end_of_season = pairs['first_leg_date'] >= pairs['core_season_end'] - pd.Timedelta(days=self.playoff_tail_days)  # 2
        is_break_after = pairs['break_days'] >= self.playoff_break_days  # 3
        is_few_legs = pairs['n_legs'] <= self.playoff_max_legs  # 4

        playoff_pairs = pairs.loc[is_end_of_season & is_break_after & is_few_legs].copy()

        # earlier rounds candidates: non-core vs non-core tail matches in league-seasons with playoff pairs
        seasons_with_pairs = set(playoff_pairs[season_keys].itertuples(index=False, name=None))

        early_round_matches = championship_matches.loc[~championship_matches['is_home_core']
                                                       & ~championship_matches['is_away_core']
                                                       & championship_matches['is_tail']]

        early_round_matches = early_round_matches.loc[[(league, season) in seasons_with_pairs
                                                       for league, season in zip(early_round_matches['league'],
                                                                                 early_round_matches['season'])]]

        # regular matches share: matches in the league-season except playoff candidates
        candidate_matches = pd.concat([mixed_matches.merge(playoff_pairs[pair_keys], on=pair_keys, how='inner'),
                                       early_round_matches])

        candidate_team_matches = pd.concat([
            candidate_matches[['match_id', 'league', 'season', 'home_team']].rename(columns={'home_team': 'team'}),
            candidate_matches[['match_id', 'league', 'season', 'away_team']].rename(columns={'away_team': 'team'}),
        ]).drop_duplicates()

        candidate_counts = (candidate_team_matches
                            .groupby(team_keys)['match_id']
                            .nunique()
                            .rename('n_candidate_matches'))

        team_season_counts = team_season_counts.join(candidate_counts, on=team_keys)
        team_season_counts['n_candidate_matches'] = team_season_counts['n_candidate_matches'].fillna(0)
        team_season_counts['regular_share'] = ((team_season_counts['n_matches'] - team_season_counts['n_candidate_matches'])
                                               / team_season_counts['season_median'])

        regular_share = dict(zip(team_season_counts[team_keys].itertuples(index=False, name=None),
                                 team_season_counts['regular_share']))

        def is_outsider(league, season, team):
            """Non-core team without (almost) regular matches in the league-season."""
            return ((league, season, team) not in core_teams
                    and regular_share[(league, season, team)] < self.playoff_max_non_core_share)

        # special formats: non-core team of the pair has regular matches in the league-season
        is_outsider_pair = pd.Series([is_outsider(league, season, team_1) or is_outsider(league, season, team_2)
                                      for league, season, team_1, team_2 in playoff_pairs[pair_keys].itertuples(index=False, name=None)],
                                     index=playoff_pairs.index, dtype=bool)

        partial_membership_pairs = playoff_pairs.loc[~is_outsider_pair]
        playoff_pairs = playoff_pairs.loc[is_outsider_pair]

        # too many pairs in one league-season: special format
        pairs_per_season = playoff_pairs.groupby(season_keys)['team_1'].transform('size')
        too_many_pairs = playoff_pairs.loc[pairs_per_season > self.playoff_max_pairs_per_season]
        playoff_pairs = playoff_pairs.loc[pairs_per_season <= self.playoff_max_pairs_per_season]

        # earlier rounds only in league-seasons with remaining pairs, both teams must be outsiders
        seasons_with_pairs = set(playoff_pairs[season_keys].itertuples(index=False, name=None))

        is_early_round = [(league, season) in seasons_with_pairs
                          and is_outsider(league, season, home_team)
                          and is_outsider(league, season, away_team)
                          for league, season, home_team, away_team in zip(early_round_matches['league'],
                                                                          early_round_matches['season'],
                                                                          early_round_matches['home_team'],
                                                                          early_round_matches['away_team'])]

        early_round_match_ids = set(early_round_matches.loc[is_early_round, 'match_id'])

        pair_match_ids = set(mixed_matches
                             .merge(playoff_pairs[pair_keys], on=pair_keys, how='inner')
                             ['match_id'])

        playoff_match_ids = pair_match_ids | early_round_match_ids

        if not partial_membership_pairs.empty:
            print('Playoff candidates with partial league membership (not tagged, check manually):')
            print(partial_membership_pairs[pair_keys].to_string(index=False))

        if not too_many_pairs.empty:
            print('League-seasons with too many playoff candidates (not tagged, check manually):')
            print(too_many_pairs.groupby(season_keys).size().rename('n_pairs').reset_index().to_string(index=False))

        is_playoff = matches['match_id'].isin(playoff_match_ids)
        matches.loc[is_playoff, 'is_playoff'] = True
        matches.loc[is_playoff, 'is_championship'] = False

        print(f'#playoff pairs: {len(playoff_pairs)}, matches: {len(pair_match_ids)}, '
              f'earlier rounds matches: {len(early_round_match_ids)}')

        return matches

    def _tag_relegation_groups(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Tag relegation groups stored inside the first league (Moldova: bottom first league teams
        play a round robin with top second league teams): is_relegation_group = True, is_championship = False.

        Matches of relegation_group_leagues with at least one non-core team. Call after _tag_playoffs.
        """

        matches['is_relegation_group'] = False

        league = self._championship_league(matches)

        for relegation_group_league in self.relegation_group_leagues:
            is_league = (matches['is_championship'] == True) & (league == relegation_group_league)

            championship_matches = matches.loc[is_league, ['match_id', 'season', 'date', 'home_team', 'away_team']].copy()
            championship_matches['league'] = relegation_group_league

            core_teams = self._core_teams(championship_matches)

            is_home_core = np.array([(relegation_group_league, season, team) in core_teams
                                     for season, team in zip(championship_matches['season'], championship_matches['home_team'])],
                                    dtype=bool)
            is_away_core = np.array([(relegation_group_league, season, team) in core_teams
                                     for season, team in zip(championship_matches['season'], championship_matches['away_team'])],
                                    dtype=bool)

            relegation_group_index = championship_matches.index[~(is_home_core & is_away_core)]

            matches.loc[relegation_group_index, 'is_relegation_group'] = True
            matches.loc[relegation_group_index, 'is_championship'] = False

            print(f'{relegation_group_league}: {len(relegation_group_index)} relegation group matches')
            print(matches.loc[relegation_group_index]
                  .groupby('season').size().rename('n_matches').reset_index().to_string(index=False))

        return matches

    @staticmethod
    def _drop_matches_with_non_championship_teams(matches: pd.DataFrame) -> pd.DataFrame:
        """Drop cup (including international cups) matches where either team never played in a domestic championship.
        Call after renaming."""

        championships = matches[matches['is_championship'] == True]

        teams_with_championships = set(championships['home_team']).union(set(championships['away_team']))

        n_matches = matches.shape[0]

        matches = matches[(matches['home_team'].isin(teams_with_championships)
                           & matches['away_team'].isin(teams_with_championships))]

        print('#dropped matches with non-championship team:', n_matches - matches.shape[0])

        return matches

    @staticmethod
    def _remove_matches_with_unknown_team(matches: pd.DataFrame) -> pd.DataFrame:
        """Remove matches featuring teams from unknown leagues to prevent rating inflation.
        For instance, matches from the national cup involving third league teams will be removed.

        Some notes:
            - Flashscore has data about Serbian tournaments only since 2006
            - Flashscore has data about Kazakhstan tournaments only since 2003
            - Flashscore has data about New Zealand  tournaments only since 2022
            - Argentina has the transfer from season 2019/2020 to 2021
            - Bolivia has the transfer from season 2016/2017 to 2018
            - Antigua & Barbuda did not have chmpionships in 2020/2021 and 2021/2022
            - Aruba has the transfer from season 2019/2020 to 2021
            - Bangladesh 14/15 to 2016
            - Bangladesh 17/18 to 2019
            """

        # drop if team appeared firstly in cup
        number_matches = matches.shape[0]

        known_teams = set()
        for season in sorted(matches['season'].unique()):
            championships = (matches.loc[(matches['is_championship'] == True)
                                         & (matches['season'] == season), ['home_team', 'away_team']])

            known_teams = known_teams.union(set(championships['home_team']).union(championships['away_team']))

            is_both_teams_known = ((matches['season'] == season)
                                   & matches['home_team'].isin(known_teams)
                                   & matches['away_team'].isin(known_teams))

            matches = matches.loc[(matches['season'] != season) | is_both_teams_known]

        print(f'#dropped matches with unknown team, when team appeared in cup before league: {number_matches - matches.shape[0]}')

        # for instance, autumn/sprint to spring/autumn
        tournaments_with_transfer_seasons = [
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
        ]

        number_matches = matches.shape[0]
        dropped = []
        for season in sorted(matches['season'].unique()):
            championships = matches[(matches['is_championship'] == True) & (matches['season'] == season)]

            known_teams = set(championships['home_team']).union(championships['away_team'])

            for country, tournament_type, season_before_transfer in tournaments_with_transfer_seasons:
                if season_before_transfer + 1 == season:
                    prev_season_matches = matches[(matches['country'] == country)
                                                  & (matches['tournament_type'] == tournament_type)
                                                  & (matches['season'] == season_before_transfer)]

                    also_known_teams = set(prev_season_matches['home_team']).union(set(prev_season_matches['away_team']))

                    known_teams = known_teams.union(also_known_teams)

            is_both_teams_known = ((matches['season'] == season)
                                   & matches['home_team'].isin(known_teams)
                                   & matches['away_team'].isin(known_teams))

            dropped.append(matches[(matches['season'] == season) & ~is_both_teams_known])

            matches = matches.loc[(matches['season'] != season) | is_both_teams_known]

        print('#dropped matches with unknown team:', number_matches - matches.shape[0])

        return matches

    def _club_world_cup_hosts(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Club World Cup: a club of the host country plays at home (the tournament is neutral otherwise).
        The host club becomes the home team (swapped by _handle_visit_stadium if listed as away),
        is_club_world_cup_host marks such matches for _neutral_venue.
        Both clubs from the host country (e.g. the 2000 final): neutral.

        Call after renaming (final team names) and before _outcomes (outcomes use the home / away orientation)."""

        is_club_world_cup = matches['tournament'] == 'FIFA Club World Cup'

        # country of a club: its domestic tournaments
        domestic = matches.loc[matches['is_domestic'] == True]
        team_country = dict(zip(pd.concat([domestic['home_team'], domestic['away_team']]),
                                pd.concat([domestic['country'], domestic['country']])))

        host = matches['season'].map(self.club_world_cup_hosts)
        is_home_host = is_club_world_cup & (matches['home_team'].map(team_country) == host)
        is_away_host = is_club_world_cup & (matches['away_team'].map(team_country) == host)

        matches['is_club_world_cup_host'] = is_home_host ^ is_away_host

        # host listed as the away team: swap, so the host is the home team
        matches['is_on_visit_stadium'] = is_away_host & ~is_home_host
        matches = self._handle_visit_stadium(matches)

        missing_hosts = sorted(set(matches.loc[is_club_world_cup, 'season']) - set(self.club_world_cup_hosts))
        if missing_hosts:
            print('Club World Cup seasons without a host:', missing_hosts)

        print(f'Club World Cup matches with a host club: {matches["is_club_world_cup_host"].sum()}')

        return matches

    def _extra_time_possibility(self, matches: pd.DataFrame, is_season_finished=False) -> pd.DataFrame:
        """Whether extra time would have been played in case of a draw (level aggregate for second legs).
        Only information known before the match: it is a feature of the goals model.

        Rules, in order of priority:
          - replay if draw                                  → False
          - first leg of a two-legged tie                   → False
          - second leg, not a league match                  → True
          - new rows with tournament_stage: knockout        → True (playoffs inside championships too)
          - league matches (without playoffs)               → False
          - other second legs                               → True
          - new rows with tournament_stage, not knockout    → False
          - old rows: domestic cup / super cup / playoff,
            single match                                    → True
          - international cup, final (last match of the
            tournament in the season)                       → True
          - international cup, other single matches
            (group stage)                                   → False

        Promotion / relegation playoffs (is_playoff) follow the cup rules,
        relegation groups (is_relegation_group) are round robin, league rules.

        The last season is considered unfinished by default, so its last match is not a final.
        Extra time actually played is not used: it gives the 90' draw (target leakage).

        Must be called after _handle_visit_stadium, columns renaming (date, season) and _tag_playoffs.
        """
        is_extra_time_played = matches['note'].isin(['AET', 'Pen'])
        has_stage, is_knockout_stage = self._knockout_stage(matches)
        is_playoff = matches['is_playoff'] == True
        is_league = matches['tournament_type'].isin(['first_league', 'second_league']) & ~is_playoff
        is_replayed_if_draw = matches['is_replayed_if_draw'] == True
        is_first_leg = matches['leg'] == 1
        is_second_leg = (matches['leg'] == 2) | matches['first_leg_home'].notna()
        is_single_match = matches['leg'].isna() & matches['first_leg_home'].isna()
        is_domestic_cup = matches['tournament_type'].isin(['cup', 'super_cup'])
        is_international_cup = matches['tournament_type'].isin(['international_cup', 'international_super_cup'])

        # final = last match of the tournament in the season
        last_date = matches.groupby(['season', 'tournament'])['date'].transform('max')
        is_final = matches['date'] == last_date

        if not is_season_finished:
            is_final = is_final & (matches['season'] != matches['season'].max())

        # knockout by the stage is before the league rule: playoffs inside championships ('Série A - Play Offs - Final')
        masks = [
            is_replayed_if_draw,
            is_first_leg,
            is_second_leg & ~is_league,
            has_stage & is_knockout_stage,
            is_league,
            is_second_leg,
            has_stage,
            is_single_match & (is_domestic_cup | is_playoff),
            is_single_match & is_international_cup & is_final,
            is_single_match & is_international_cup,
        ]
        values = [False, False, True, True, False, True, False, True, True, False]

        # leg and first_leg_home are nullable, so masks can contain pd.NA
        matches['is_extra_time_possible'] = np.select([mask.fillna(False).astype(bool) for mask in masks],
                                                      values, default=False).astype(bool)

        # a few matches on the last date → not a single final (e.g. third place match, or last group matchday)
        n_last_date_matches = is_final.groupby([matches['season'], matches['tournament']]).transform('sum')
        is_ambiguous_final = is_final & is_international_cup & (n_last_date_matches > 1)
        if is_ambiguous_final.any():
            print('#international cup seasons with several matches on the last date:', is_ambiguous_final.sum())
            print(matches.loc[is_ambiguous_final, ['tournament', 'season']].drop_duplicates())

        # sanity check: extra time was possible, the match (or aggregate) was level, but no AET/Pen note
        aggregate_difference = (matches['aggregate_home'] - matches['aggregate_away']).astype('Float64')
        score_difference = (matches['home_score'] - matches['away_score']).astype('Float64')
        level_difference = aggregate_difference.where(aggregate_difference.notna(), score_difference)
        is_suspicious = (matches['is_extra_time_possible']
                         & ~is_extra_time_played
                         & (level_difference == 0).fillna(False).astype(bool))
        if is_suspicious.any():
            print('#extra time possible, level result, but no AET/Pen note:', is_suspicious.sum())

        return matches

    @staticmethod
    def _neutral_venue(matches: pd.DataFrame, is_season_finished=False) -> pd.DataFrame:
        """Get final matches for the detection of neutral field games.
        Raw cup names collide between countries ('League Cup', 'FA Cup'), so the final is searched by country.
        Club World Cup matches of a host club are not neutral (see _club_world_cup_hosts), finals included."""

        if is_season_finished:
            is_cup = matches['tournament_type'].isin({'cup', 'international_cup'})
        else:
            is_cup = ((matches['season'] != matches['season'].max())
                      & matches['tournament_type'].isin({'cup', 'international_cup'}))

        final_ids = (matches
                     .loc[is_cup]
                     .sort_values(['country', 'tournament', 'season', 'date'])
                     .drop_duplicates(['country', 'season', 'tournament'], keep='last')
                     ['match_id']
                     .unique())

        matches['is_neutral'] = ((matches['match_id'].isin(final_ids)
                                  | (matches['is_neutral_venue'] == True)
                                  | matches['tournament_type'].isin({'super_cup', 'international_super_cup'})
                                  | matches['tournament'].isin({'FIFA Club World Cup'}))
                                 & ~matches['is_club_world_cup_host'])

        matches = matches.drop(columns='is_neutral_venue')

        return matches

    @staticmethod
    def _neutral_venue_by_city(matches: pd.DataFrame, min_city_matches=5, min_city_share=0.9) -> pd.DataFrame:
        """International matches played outside the country of the home club are neutral:
        hubs of the AFC Champions League 2020-2021 (Doha, Riyadh), Iran - Saudi Arabia in Oman,
        Shakhtar in Hamburg. Only rows with venue_city (re-scraped seasons).

        country of a city: the votes of the home matches played in it, the domestic ones (country of the tournament)
        and the international ones (country of the home club); a city needs >= min_city_matches votes and
        >= min_city_share of them for one country, so hubs of other countries do not change the country of a city.
        country of a club: the most common country of its domestic matches.
        A city where the home club plays its domestic home matches is never abroad for it (cross-border cities).
        Matches in the country of the away club are probably listed in the wrong order, they are made neutral
        (not swapped): the venue may be wrong too, neutral is the smaller error."""

        venue_city = matches['venue_city'].replace('', np.nan)
        is_domestic = matches['is_domestic'] == True

        domestic = matches.loc[is_domestic]
        club_country = (pd.concat([domestic[['home_team', 'country']].set_axis(['team', 'country'], axis=1),
                                   domestic[['away_team', 'country']].set_axis(['team', 'country'], axis=1)])
                        .groupby('team')['country']
                        .agg(lambda x: x.mode().iloc[0]))

        home_country = matches['home_team'].map(club_country)
        away_country = matches['away_team'].map(club_country)

        # votes for the country of a city: home matches, not neutral
        vote_country = home_country.where(~is_domestic, matches['country'])
        votes = (pd.DataFrame({'city': venue_city, 'country': vote_country})
                 .loc[venue_city.notna() & vote_country.notna() & ~matches['is_neutral']])

        city_votes = votes.groupby(['city', 'country']).size().rename('n').reset_index()
        city_votes['share'] = city_votes['n'] / city_votes.groupby('city')['n'].transform('sum')
        city_votes['total'] = city_votes.groupby('city')['n'].transform('sum')

        city_country = (city_votes
                        .loc[(city_votes['total'] >= min_city_matches) & (city_votes['share'] >= min_city_share)]
                        .set_index('city')['country'])

        venue_country = venue_city.map(city_country)

        # own home cities of a club (>= 2 domestic home matches): Auckland City in Auckland is at home,
        # though Auckland is a city of Australia by the votes (Auckland FC in the A-League)
        own_cities = (pd.DataFrame({'team': matches['home_team'], 'city': venue_city})
                      .loc[is_domestic & venue_city.notna()]
                      .value_counts()
                      .loc[lambda x: x >= 2]
                      .index)
        is_own_city = pd.Series(pd.MultiIndex.from_arrays([matches['home_team'], venue_city]).isin(own_cities),
                                index=matches.index)

        is_abroad = (~is_domestic
                     & ~matches['is_neutral']
                     & venue_country.notna()
                     & home_country.notna()
                     & (venue_country != home_country)
                     & ~is_own_city)

        # played in the country of the away club: no home advantage for the listed home club either
        is_away_country = is_abroad & (venue_country == away_country)

        matches['is_neutral'] = matches['is_neutral'] | is_abroad

        print(f'#international matches outside the country of the home club -> neutral: {is_abroad.sum()} '
              f'(in the country of the away club: {is_away_country.sum()}), cities with a country: {len(city_country)}')

        return matches

    @staticmethod
    def _exclude_leagues_without_international_history(matches: pd.DataFrame) -> pd.DataFrame:
        """If a team plays only in domestic championships its ratings cannot be calculated correctly as it is isolated"""
        home = matches[['home_team', 'country', 'is_domestic']].rename(columns={'home_team': 'team'})
        away = matches[['away_team', 'country', 'is_domestic']].rename(columns={'away_team': 'team'})

        stacked_matches = pd.concat([home, away], ignore_index=True)

        domestic_teams = (stacked_matches
                          .loc[stacked_matches['is_domestic'] == True, ['team', 'country']]
                          .drop_duplicates(['team', 'country']))

        international_teams = (stacked_matches
                               .loc[stacked_matches['is_domestic'] == False, 'team']
                               .unique())

        # teams that play in championships and also have international matches
        qualified_countries = set(domestic_teams.loc[domestic_teams['team'].isin(international_teams), 'country'])

        excluded_countries = set(domestic_teams['country']) - qualified_countries
        print(f'excluded countries: {excluded_countries}')

        matches = matches[~matches['country'].isin(excluded_countries)].reset_index(drop=True)
        return matches

    @staticmethod
    def _tournament_preprocessing(matches: pd.DataFrame) -> pd.DataFrame:
        """Rename tournaments to a unified format: country + league/cup type"""

        matches = matches.rename(columns={'tournament': 'tournament_name'})

        masks = [matches['tournament_type'] == 'first_league',
                 matches['tournament_type'] == 'second_league',
                 matches['tournament_type'].isin({'cup', 'super_cup'}),
                 matches['tournament_type'].isin({'international_cup', 'international_super_cup'})]

        values = [matches['country'] + '. First League',
                  matches['country'] + '. Second League',
                  matches['country'] + '. Cup',
                  matches['country'] + ' ' + matches['tournament_name']]

        matches['tournament'] = np.select(masks, values, default='bug')

        assert (matches['tournament'] == 'bug').sum() == 0, 'bugs in tournament column'

        return matches

    @staticmethod
    def _add_team_leagues(matches: pd.DataFrame) -> pd.DataFrame:
        """In our logic we use params related to the league of the home team.
        In most matches, leagues of home and away teams are the same, of course.

        Some teams appear in two different championships in one season. There are two reasons for this:
         1) Relegation/promotion matches at the end of the season (mostly excluded from championships by _tag_playoffs).
         2) Transitional seasons, e.g. Cameroon Elite One 2019 -> Cameroon Elite One 2019/2020.
        To resolve duplicates, we keep the championship where the team played the most matches.
        """

        championship_matches = pd.concat([
            matches.loc[matches['is_championship'] == True, ['season', 'home_team', 'tournament']],
            matches.loc[matches['is_championship'] == True, ['season', 'away_team', 'tournament']]
            .rename(columns={'away_team': 'home_team'})
        ], ignore_index=True)

        championships = (championship_matches
                         .groupby(['season', 'home_team', 'tournament'])
                         .size()
                         .rename('match_count')
                         .reset_index())

        n_dupes = championships.duplicated(['season', 'home_team'], keep=False).sum()
        print(f"season-team pairs appearing in multiple championships: {n_dupes}")

        championships = (championships
                         .sort_values('match_count', ascending=False)
                         .drop_duplicates(['season', 'home_team'], keep='first')
                         .drop(columns='match_count'))

        home_leagues = championships.rename(columns={'home_team': 'home_team', 'tournament': 'home_team_league'})
        away_leagues = championships.rename(columns={'home_team': 'away_team', 'tournament': 'away_team_league'})

        n_matches = matches.shape[0]

        matches = matches.merge(home_leagues, how='inner', on=['season', 'home_team'], validate='m:1')
        matches = matches.merge(away_leagues, how='inner', on=['season', 'away_team'], validate='m:1')

        print('#dropped matches with a team without championship in the season:', n_matches - matches.shape[0])

        return matches

    def transform(self) -> pd.DataFrame:
        """"""

        matches = pd.read_feather(config.soccer['club_matches'], self.matches_columns)

        matches = self._preprocessing(matches)

        self._data_tests(matches)

        matches = self._rename_teams(matches)

        # after renaming (final team names), before the drops based on is_championship:
        # playoffs and relegation groups do not define team leagues
        matches = self._drop_duplicate_matches(matches)
        matches = self._tag_playoffs(matches)
        matches = self._tag_relegation_groups(matches)

        matches = self._drop_matches_with_non_championship_teams(matches)
        matches = self._remove_matches_with_unknown_team(matches)

        # before _outcomes: host clubs become home teams
        matches = self._club_world_cup_hosts(matches)

        matches = self._outcomes(matches)
        matches = self._extra_time_possibility(matches)
        matches = self._five_outcomes(matches)
        matches = self._neutral_venue(matches)
        matches = self._neutral_venue_by_city(matches)

        matches = self._exclude_leagues_without_international_history(matches)
        matches = self._tournament_preprocessing(matches)
        matches = self._add_team_leagues(matches)

        save_feather(matches, config.soccer['preprocessed_club_matches'])

        matches_copy = matches.copy()
        self._data_tests(matches_copy)
        self._same_date_matches_warnings(matches_copy)

        return matches


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
    def eda(matches: pd.DataFrame):
        """Use this function for testing data.
        Final stage matches where the listed away team is the host."""

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

        self.eda(matches.copy())

        return matches


class MatchesPreprocessor(object):
    """Preprocess club and national team matches and union them with one match_id for the whole project.

    Features, goals expectations and ratings refer to this match_id, ratings filter matches by source.
    Matches of one source have NaN in columns specific to the other one.
    """

    def __init__(self):
        self.club_preprocessor = ClubDataPreprocessor()
        self.national_preprocessor = NationalDataPreprocessor()

    @staticmethod
    def _stable_match_id(matches: pd.DataFrame) -> pd.Series:
        """match_id from the natural key (source, day, home team, away team): the same match keeps its id
        after any rebuild (new matches, dropped rows), so features / predictions / ratings built on an older
        matches.feather either join the right match or fail on missing ids, never silently join another match.
        Day without time: kick-off times are corrected between scrapes.
        63-bit non-negative: uint64 hash without the highest bit fits int64 without negative values."""

        key = pd.DataFrame({'source': matches['source'].astype(str),
                            'day': matches['date'].dt.strftime('%Y-%m-%d'),
                            'home_team': matches['home_team'].astype(str),
                            'away_team': matches['away_team'].astype(str)})

        hashes = pd.util.hash_pandas_object(key, index=False).to_numpy()

        return pd.Series((hashes >> np.uint64(1)).astype(np.int64), index=matches.index, name='match_id')

    def _union(self, club_matches: pd.DataFrame, national_matches: pd.DataFrame) -> pd.DataFrame:
        """"""

        # rolling features and ratings group by team name, so names must not collide between sources,
        # clubs named as countries are renamed in ClubDataPreprocessor (CLUBS_NAMED_AS_COUNTRIES)
        club_teams = set(club_matches['home_team']) | set(club_matches['away_team'])
        national_teams = set(national_matches['home_team']) | set(national_matches['away_team'])
        common_teams = club_teams & national_teams
        assert not common_teams, f'teams in both club and national matches: {common_teams}'

        club_matches['source'] = 'clubs'
        national_matches['source'] = 'national'

        # club-only flags: False for national teams, so the columns stay boolean after concat
        for column in ['is_playoff', 'is_relegation_group']:
            national_matches[column] = False

        # match_id of each preprocessing is local to its source, one match_id for both sources is assigned below
        matches = pd.concat([club_matches.drop(columns='match_id'), national_matches.drop(columns='match_id')], ignore_index=True)

        matches = matches.sort_values(['date', 'source', 'home_team', 'away_team'], kind='stable').reset_index(drop=True)
        matches.insert(0, 'match_id', self._stable_match_id(matches))

        # the natural key must be unique (clubs: _drop_duplicate_matches), otherwise two matches share an id
        is_duplicate = matches['match_id'].duplicated(keep=False)
        assert not is_duplicate.any(), \
            f'matches with the same source, day and teams:\n' \
            f'{matches.loc[is_duplicate, ["source", "tournament", "date", "home_team", "away_team"]].to_string(index=False)}'

        print('#matches:', matches.shape[0], '| clubs:', (matches['source'] == 'clubs').sum(),
              '| national:', (matches['source'] == 'national').sum())

        return matches

    def transform(self) -> pd.DataFrame:
        """"""

        club_matches = self.club_preprocessor.transform()
        national_matches = self.national_preprocessor.transform()

        matches = self._union(club_matches, national_matches)

        save_feather(matches, config.soccer['matches'])

        return matches
