import pandas as pd

"""
Ratings tables before and after the split of the namesakes by team id (25.10).
Run after full_pipeline.py:
    .venv_core/bin/python compare_tables.py
"""

BEFORE_LEAGUES = 'league_ratings_before_ids_20261024.csv'
BEFORE_CLUBS = 'club_ratings_before_ids_20261024.csv'

if __name__ == '__main__':
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', None)

    before = pd.read_csv(BEFORE_LEAGUES)[['league', 'top_8_weighted']]
    after = pd.read_csv('league_ratings.csv')[['league', 'top_8_weighted', 'comparable']]

    leagues = before.merge(after, on='league', how='outer', suffixes=('_before', '_after'))
    leagues['diff'] = leagues['top_8_weighted_after'] - leagues['top_8_weighted_before']
    leagues['#_before'] = leagues['top_8_weighted_before'].rank(ascending=False, method='first')
    leagues['#_after'] = leagues['top_8_weighted_after'].rank(ascending=False, method='first')

    print(f'median shift of the leagues: {leagues["diff"].median():+.1f}')

    print('\nleagues moved by more than 10 points:')
    print(leagues.loc[leagues['diff'].abs() > 10].sort_values('diff').round(0).to_string(index=False))

    print('\nleagues only in one table:')
    print(leagues.loc[leagues['diff'].isna(), ['league', 'top_8_weighted_before', 'top_8_weighted_after']].to_string(index=False))

    clubs_before = pd.read_csv(BEFORE_CLUBS).set_axis(['#', 'team', 'rating', 'league'], axis=1)
    clubs_after = pd.read_csv('club_ratings.csv').set_axis(['#', 'team', 'rating', 'league'], axis=1)

    clubs = clubs_before.merge(clubs_after, on='team', how='outer', suffixes=('_before', '_after'))
    clubs['diff'] = clubs['rating_after'] - clubs['rating_before']

    print('\nclubs of the top-200 moved by more than 30 points:')
    is_top = (clubs['#_before'] <= 200) | (clubs['#_after'] <= 200)
    print(clubs.loc[is_top & (clubs['diff'].abs() > 30)]
          .sort_values('diff')[['team', 'league_after', '#_before', '#_after', 'rating_before', 'rating_after', 'diff']]
          .round(0).to_string(index=False))

    print('\nnew teams (split namesakes):')
    print(clubs.loc[clubs['rating_before'].isna(), ['team', 'league_after', 'rating_after']]
          .sort_values('rating_after', ascending=False).round(0).to_string(index=False))
