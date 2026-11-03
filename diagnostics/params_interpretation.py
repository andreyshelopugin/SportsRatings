"""The interpretation of the production params (Section Estimated Parameters of the paper):
home advantage (crowd + logistic) and draws by league, home advantage of the national teams with own offsets.
Results: data/soccer/tables/params_interpretation_leagues.csv, params_interpretation_national.csv.
Run from the root of the project."""
import pickle
import warnings
from pathlib import Path

import pandas as pd

from config import config
from utils.functions import stage

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)
pd.set_option('display.max_rows', None)

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / 'data/soccer/tables'

# the estimates of small leagues stay close to their start values
MIN_LEAGUE_MATCHES = 1000


def leagues():
    stage('params interpretation: leagues')

    league_params = pickle.load(open(config.soccer['params']['league_params'], 'rb'))['leagues']
    params = pd.DataFrame.from_dict(league_params, orient='index').rename_axis('league').reset_index()
    params['home_advantage'] = params['crowd'] + params['logistic']

    # the number of home matches in the league
    matches = pd.read_feather(config.soccer['glicko_matches'], columns=['match_id', 'is_home', 'home_team_league'])
    n_matches = matches.loc[matches['is_home'] == True].groupby('home_team_league').size().rename('n_matches')
    params = params.merge(n_matches, left_on='league', right_index=True, how='left')

    params = params.sort_values('home_advantage', ascending=False)
    params.to_csv(TABLES / 'params_interpretation_leagues.csv', index=False)

    large = params.loc[params['n_matches'] >= MIN_LEAGUE_MATCHES]
    columns = ['league', 'n_matches', 'crowd', 'logistic', 'home_advantage', 'draw_frequency', 'init_mu']
    print(f'home advantage, top 15 (>= {MIN_LEAGUE_MATCHES} matches)\n', large[columns].head(15).to_string(index=False))
    print('home advantage, bottom 15\n', large[columns].tail(15).to_string(index=False))
    print('draws, top 10\n', large.sort_values('draw_frequency', ascending=False)[columns].head(10).to_string(index=False))
    print('draws, bottom 10\n', large.sort_values('draw_frequency')[columns].head(10).to_string(index=False))
    print('first vs second divisions (median)\n',
          large.assign(level=large['league'].str.extract(r'\. (\w+) League')[0])
          .groupby('level')[['crowd', 'logistic', 'home_advantage', 'draw_frequency']].median().to_string())


def national_teams():
    stage('params interpretation: national teams')
    from soccer.national_ratings.ratings import NationalRatings

    global_params, confederation_params, country_params = NationalRatings().load_params()
    countries = pd.DataFrame([{'country': country, **{name: global_params[name] + offsets.get(name, 0.) for name in ('crowd', 'logistic')}}
                              for country, offsets in country_params.items()])
    countries['home_advantage'] = countries['crowd'] + countries['logistic']
    countries = countries.sort_values('home_advantage', ascending=False)
    countries.to_csv(TABLES / 'params_interpretation_national.csv', index=False)

    print('global', global_params, '\nconfederations', confederation_params)
    print('national teams with own home advantage\n', countries.to_string(index=False))


def main():
    leagues()
    national_teams()


if __name__ == '__main__':
    main()
