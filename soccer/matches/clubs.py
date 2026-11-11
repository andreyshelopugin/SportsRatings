import numpy as np
import pandas as pd

from config import config
from soccer.constants import PANDEMIC_START, PANDEMIC_END, CLUBS_NAMED_AS_COUNTRIES, CLUB_WORLD_CUP_HOSTS, TRANSFER_SEASONS
from soccer.matches.base import DataPreprocessor
from utils.functions import save_feather


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

        assert set(df['tournament_type']) == self.club_tournaments, "Invalid tournament_type values"

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


        number_matches = matches.shape[0]
        for season in sorted(matches['season'].unique()):
            championships = matches[(matches['is_championship'] == True) & (matches['season'] == season)]

            known_teams = set(championships['home_team']).union(championships['away_team'])

            for country, tournament_type, season_before_transfer in TRANSFER_SEASONS:
                if season_before_transfer + 1 == season:
                    prev_season_matches = matches[(matches['country'] == country)
                                                  & (matches['tournament_type'] == tournament_type)
                                                  & (matches['season'] == season_before_transfer)]

                    also_known_teams = set(prev_season_matches['home_team']).union(set(prev_season_matches['away_team']))

                    known_teams = known_teams.union(also_known_teams)

            is_both_teams_known = ((matches['season'] == season)
                                   & matches['home_team'].isin(known_teams)
                                   & matches['away_team'].isin(known_teams))

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
    def _exclude_isolated_leagues(matches: pd.DataFrame) -> pd.DataFrame:
        """Leagues not linked to the other leagues in any season: their teams never played in another league
        (promotion, relegation), against a team of another league (cups) or against foreign clubs.
        The level of such a league is set only by its initial rating (the second league of Singapore, 2025)."""

        is_league = matches['tournament_type'].isin({'first_league', 'second_league'})
        league = matches['country'] + ' ' + matches['tournament_type']

        team_leagues = pd.concat([pd.DataFrame({'team': matches.loc[is_league, column], 'league': league[is_league]})
                                  for column in ['home_team', 'away_team']]).drop_duplicates()
        leagues_of_team = team_leagues.groupby('team')['league'].agg(frozenset)

        home_leagues = matches['home_team'].map(leagues_of_team)
        away_leagues = matches['away_team'].map(leagues_of_team)

        # a foreign opponent or an opponent from another league (cups) links the leagues of the teams
        is_linking = (matches['is_domestic'] == False) | (home_leagues != away_leagues)
        linking_teams = set(matches.loc[is_linking, 'home_team']) | set(matches.loc[is_linking, 'away_team'])

        # promotion, relegation: the team played in two leagues
        linking_teams |= set(leagues_of_team.index[leagues_of_team.map(len) > 1])

        linked_leagues = set(team_leagues.loc[team_leagues['team'].isin(linking_teams), 'league'])
        isolated_leagues = set(team_leagues['league']) - linked_leagues
        print(f'isolated leagues: {isolated_leagues}')

        # the teams of an isolated league play only each other
        is_isolated = home_leagues.map(lambda leagues: isinstance(leagues, frozenset) and leagues <= isolated_leagues)
        print(f'#dropped matches of the isolated leagues: {is_isolated.sum()}')

        return matches.loc[~is_isolated].reset_index(drop=True)

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
        matches = self._exclude_isolated_leagues(matches)
        matches = self._tournament_preprocessing(matches)
        matches = self._add_team_leagues(matches)

        save_feather(matches, config.soccer['preprocessed_club_matches'])

        matches_copy = matches.copy()
        self._data_tests(matches_copy)
        self._same_date_matches_warnings(matches_copy)

        return matches
