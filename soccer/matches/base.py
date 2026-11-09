import numpy as np
import pandas as pd


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
        """Remove identical matches that appear under different seasons, the copy with the note is kept.

        After the swap of the matches on the field of the visiting team: before it, the copies with and without
        the note are the same match, and the team ids of namesakes depend on which copy is kept."""
        subset = ['date', 'country', 'home_team', 'away_team', 'home_score', 'away_score']

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
