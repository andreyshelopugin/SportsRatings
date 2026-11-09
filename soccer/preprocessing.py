import numpy as np
import pandas as pd

from config import config
from soccer.matches.clubs import ClubDataPreprocessor
from soccer.matches.national import NationalDataPreprocessor
from utils.functions import save_feather


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
