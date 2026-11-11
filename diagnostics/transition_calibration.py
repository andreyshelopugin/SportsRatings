"""Checks of the season transitions: are the teams after a transition rated right in their first season?
actual - expected score per match (p_win + 0.5 * p_draw as in Glicko, ratings before the match), domestic matches of the first season:
    newcomers (missed_previous)        - checks the newcomer offset (ms_new_team_update_mu = 90 from the anchor)
    promoted (to a first league)       - checks +lift
    relegated (to a second league)     - checks -lift
    continuing (same_league)           - reference, must be ~0
    international matches by league    - checks the level of the leagues (Iran, Romania, 2023+)
A group with actual - expected < 0 is overrated after the transition, > 0 underrated.
The production params (all seasons), the transitions since FIRST_SEASON, the international matches since RECENT_SEASON.
Run from the root of the project."""
import warnings

import numpy as np
import pandas as pd

from config import config
from soccer.club_ratings.ratings import Ratings



FIRST_SEASON = 2010
RECENT_SEASON = 2023

INTERNATIONAL_LEAGUES = ['Iran. First League', 'Romania. First League', 'Uzbekistan. First League',
                         'Saudi Arabia. First League', 'Japan. First League', 'Qatar. First League',
                         'Croatia. First League', 'Czech Republic. First League',
                         # South America: the level vs Europe only by the Club World Cup
                         'Brazil. First League', 'Argentina. First League',
                         # the leagues with a doubtful place in the league table
                         'Greece. First League', 'Cyprus. First League', 'Scotland. First League',
                         'United Arab Emirates. First League', 'Turkey. First League', 'Algeria. First League',
                         'Mexico. First League', 'Norway. First League', 'Poland. First League']


def team_season_groups(matches: pd.DataFrame, model: Ratings) -> pd.Series:
    """Group of every (team, season) by the transition at its first match of the season."""

    first_matches = (matches
                     .sort_values(['date', 'match_id'])
                     .drop_duplicates(['team', 'season'], keep='first'))

    team_league = np.where(first_matches['is_home'], first_matches['home_team_league'], first_matches['away_team_league'])

    groups = pd.Series(first_matches['season_transition'].to_numpy(), index=pd.MultiIndex.from_arrays(
        [first_matches['team'], first_matches['season']]))

    is_changed = first_matches['season_transition'].to_numpy() == 'changed_league'
    groups[is_changed & np.isin(team_league, list(model.first_leagues))] = 'promoted'
    groups[is_changed & np.isin(team_league, list(model.second_leagues))] = 'relegated'

    return groups


def calibration() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Actual - expected score by the transition groups, by the countries of the newcomers and by the leagues."""

    model = Ratings()

    matches = pd.read_feather(config.soccer['glicko_matches'])
    model._precalculate_info(matches)

    league_params = model._fill_missing_leagues(model.load_params(config.soccer['params']['league_params']))
    matches_params = model._match_params(league_params)

    predictions = model.add_predictions(matches, league_params, matches_params)

    predictions['year'] = predictions['season'].astype(str).str[:4].astype(int)
    predictions = predictions.loc[predictions['year'] >= FIRST_SEASON]

    # actual - expected score in the terms of Glicko (win 1, draw 0.5, loss 0), not the 3-point system:
    # the 3-point system counts a draw as 1/3 of a win and mixes the level of a team with its draw tendency
    predictions['diff'] = predictions['outcome'] - (predictions['p_team_win'] + 0.5 * predictions['p_draw'])

    groups = team_season_groups(predictions, model)
    predictions['group'] = pd.MultiIndex.from_arrays([predictions['team'], predictions['season']]).map(groups)

    predictions['team_league'] = np.where(predictions['is_home'], predictions['home_team_league'], predictions['away_team_league'])
    predictions['opp_league'] = np.where(predictions['is_home'], predictions['away_team_league'], predictions['home_team_league'])

    # transitions: domestic matches of the first season after the transition
    domestic = predictions.loc[predictions['is_domestic'] == True]

    by_group = (domestic
                .groupby('group')['diff']
                .agg(n_matches='size', diff='mean')
                .join(domestic.drop_duplicates(['team', 'season']).groupby('group').size().rename('n_team_seasons')))

    # newcomers by country: is the offset the same everywhere
    newcomers = domestic.loc[domestic['group'] == 'missed_previous']
    by_country = (newcomers
                  .assign(country=newcomers['team_league'].str.split('.').str[0])
                  .groupby('country')['diff']
                  .agg(n_matches='size', diff='mean')
                  .loc[lambda df: df['n_matches'] >= 100]
                  .sort_values('diff'))

    # level of the leagues: international matches of the recent seasons
    international = predictions.loc[(predictions['is_domestic'] == False)
                                    & (predictions['team_league'] != predictions['opp_league'])
                                    & (predictions['year'] >= RECENT_SEASON)]
    by_league = international.groupby('team_league')['diff'].agg(n_matches='size', diff='mean')

    return by_group, by_country, by_league


def main():
    warnings.filterwarnings('ignore')
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', None)

    groups, countries, leagues = calibration()

    print(f'===== first season after the transition, domestic matches since {FIRST_SEASON}: actual - expected score =====')
    print(groups.round(3).to_string())

    print('\n===== newcomers by country (>= 100 matches) =====')
    print(countries.round(3).to_string())

    print(f'\n===== international matches since {RECENT_SEASON} by league =====')
    print(leagues.loc[[league for league in INTERNATIONAL_LEAGUES if league in leagues.index]].round(3).to_string())

    is_large = leagues['n_matches'] >= 20
    print(f'mean |diff| over {is_large.sum()} leagues with >= 20 matches: {leagues.loc[is_large, "diff"].abs().mean():.3f}')


if __name__ == '__main__':
    main()
