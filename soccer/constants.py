# FIFA Club World Cup hosts by edition (season in the data = edition year:
# the 2020-2022 editions were played in February of the next year)
CLUB_WORLD_CUP_HOSTS = {
    2000: 'Brazil',
    2005: 'Japan',
    2006: 'Japan',
    2007: 'Japan',
    2008: 'Japan',
    2009: 'United Arab Emirates',
    2010: 'United Arab Emirates',
    2011: 'Japan',
    2012: 'Japan',
    2013: 'Morocco',
    2014: 'Morocco',
    2015: 'Japan',
    2016: 'Japan',
    2017: 'United Arab Emirates',
    2018: 'United Arab Emirates',
    2019: 'Qatar',
    2020: 'Qatar',
    2021: 'United Arab Emirates',
    2022: 'Morocco',
    2023: 'Saudi Arabia',
    2025: 'Usa',
}

# matches played during the pandemic (strict bounds)
PANDEMIC_START = '2020-03-03'
PANDEMIC_END = '2021-06-06'

# Final tournaments are neutral by default, host gets home advantage only when host team plays.
# Caveat: in multi-host tournaments (Euro 2020, Gold Cup 2019) a host gets home advantage
# in every final stage match, even when played in another host country.
FINAL_STAGE_HOSTS = {
    # World Cup
    ('World Championship', 2002): {'Japan', 'South Korea'},
    ('World Championship', 2006): {'Germany'},
    ('World Championship', 2010): {'South Africa'},
    ('World Championship', 2014): {'Brazil'},
    ('World Championship', 2018): {'Russia'},
    ('World Championship', 2022): {'Qatar'},
    ('World Championship', 2026): {'USA', 'Mexico', 'Canada'},

    # Euro
    ('Euro', 2000): {'Belgium', 'Netherlands'},
    ('Euro', 2004): {'Portugal'},
    ('Euro', 2008): {'Austria', 'Switzerland'},
    ('Euro', 2012): {'Poland', 'Ukraine'},
    ('Euro', 2016): {'France'},
    ('Euro', 2020): {
        'England', 'Scotland', 'Germany', 'Italy', 'Spain', 'Netherlands',
        'Denmark', 'Russia', 'Hungary', 'Romania', 'Azerbaijan'
    },
    ('Euro', 2024): {'Germany'},

    # Copa America
    ('Copa América', 2001): {'Colombia'},
    ('Copa América', 2004): {'Peru'},
    ('Copa América', 2007): {'Venezuela'},
    ('Copa América', 2011): {'Argentina'},
    ('Copa América', 2015): {'Chile'},
    ('Copa América', 2016): {'USA'},
    ('Copa América', 2019): {'Brazil'},
    ('Copa América', 2021): {'Brazil'},
    ('Copa América', 2024): {'USA'},

    # Africa Cup of Nations
    ('Africa Cup of Nations', 2000): {'Ghana', 'Nigeria'},
    ('Africa Cup of Nations', 2002): {'Mali'},
    ('Africa Cup of Nations', 2004): {'Tunisia'},
    ('Africa Cup of Nations', 2006): {'Egypt'},
    ('Africa Cup of Nations', 2008): {'Ghana'},
    ('Africa Cup of Nations', 2010): {'Angola'},
    ('Africa Cup of Nations', 2012): {'Gabon', 'Equatorial Guinea'},
    ('Africa Cup of Nations', 2013): {'South Africa'},
    ('Africa Cup of Nations', 2015): {'Equatorial Guinea'},
    ('Africa Cup of Nations', 2017): {'Gabon'},
    ('Africa Cup of Nations', 2019): {'Egypt'},
    ('Africa Cup of Nations', 2021): {'Cameroon'},
    ('Africa Cup of Nations', 2023): {'Ivory Coast'},
    ('Africa Cup of Nations', 2025): {'Morocco'},
    ('Africa Cup of Nations', 2027): {'Kenya', 'Tanzania', 'Uganda'},

    # Asian Cup
    ('Asian Cup', 2000): {'Lebanon'},
    ('Asian Cup', 2004): {'China'},
    ('Asian Cup', 2007): {'Indonesia', 'Malaysia', 'Thailand', 'Vietnam'},
    ('Asian Cup', 2011): {'Qatar'},
    ('Asian Cup', 2015): {'Australia'},
    ('Asian Cup', 2019): {'United Arab Emirates'},
    ('Asian Cup', 2023): {'Qatar'},
    ('Asian Cup', 2027): {'Saudi Arabia'},

    # Gold Cup
    ('Gold Cup', 2000): {'USA'},
    ('Gold Cup', 2002): {'USA'},
    ('Gold Cup', 2003): {'USA', 'Mexico'},
    ('Gold Cup', 2005): {'USA'},
    ('Gold Cup', 2007): {'USA'},
    ('Gold Cup', 2009): {'USA'},
    ('Gold Cup', 2011): {'USA'},
    ('Gold Cup', 2013): {'USA'},
    ('Gold Cup', 2015): {'USA', 'Canada'},
    ('Gold Cup', 2017): {'USA'},
    ('Gold Cup', 2019): {'USA', 'Costa Rica', 'Jamaica'},
    ('Gold Cup', 2021): {'USA'},
    ('Gold Cup', 2023): {'USA', 'Canada'},
    ('Gold Cup', 2025): {'USA', 'Canada'},

    # OFC Nations Cup
    ('OFC Nations Cup', 2004): {'Australia'},
    ('OFC Nations Cup', 2012): {'Solomon Islands'},
    ('OFC Nations Cup', 2016): {'Papua New Guinea'},
    ('OFC Nations Cup', 2024): {'Vanuatu'},

    # FIFA Arab Cup
    ('FIFA Arab Cup', 2012): {'Saudi Arabia'},
    ('FIFA Arab Cup', 2021): {'Qatar'},
    ('FIFA Arab Cup', 2025): {'Qatar'},

    # Finalissima (Wembley, host team did not play)
    ('Finalissima', 2022): set(),

    # UEFA Nations League finals
    ('UEFA Nations League', 2018): {'Portugal'},
    ('UEFA Nations League', 2020): {'Italy'},
    ('UEFA Nations League', 2022): {'Netherlands'},
    ('UEFA Nations League', 2024): {'Germany'},

    # CONCACAF Nations League finals
    ('CONCACAF Nations League', 2019): {'USA'},
    ('CONCACAF Nations League', 2022): {'USA'},
    ('CONCACAF Nations League', 2023): {'USA'},
    ('CONCACAF Nations League', 2024): {'USA'},

    # CONCACAF Cup
    ('CONCACAF Cup', 2015): {'USA'},
}

# Finals date bounds per (tournament, season).
# Value is either:
#   - a 'YYYY-MM-DD' start string (end defaults to the tournament-season max date), or
#   - a ('start', 'end') pair, used where a stray late-dated row or a later home-away stage
#     in the same season would otherwise leak into the window
#     (WC 2010/2018, Euro 2008, AFCON 2021, UEFA NL 2022, OFC 2004 two-legged final).
FINAL_STAGE_START = {
    ('World Championship', 2002): '2002-05-31',
    ('World Championship', 2006): '2006-06-09',
    ('World Championship', 2010): ('2010-06-11', '2010-07-11'),
    ('World Championship', 2014): '2014-06-12',
    ('World Championship', 2018): ('2018-06-14', '2018-07-15'),
    ('World Championship', 2022): '2022-11-20',
    ('World Championship', 2026): '2026-06-11',

    ('Euro', 2000): '2000-06-10',
    ('Euro', 2004): '2004-06-12',
    ('Euro', 2008): ('2008-06-07', '2008-06-29'),
    ('Euro', 2012): '2012-06-08',
    ('Euro', 2016): '2016-06-10',
    ('Euro', 2020): '2021-06-11',
    ('Euro', 2024): '2024-06-14',

    ('Copa América', 2001): '2001-07-11',
    ('Copa América', 2004): '2004-07-06',
    ('Copa América', 2007): '2007-06-26',
    ('Copa América', 2011): '2011-07-01',
    ('Copa América', 2015): '2015-06-11',
    ('Copa América', 2016): '2016-06-03',
    ('Copa América', 2019): '2019-06-14',
    ('Copa América', 2021): '2021-06-13',
    ('Copa América', 2024): '2024-06-20',

    ('Africa Cup of Nations', 2000): '2000-01-22',
    ('Africa Cup of Nations', 2002): '2002-01-19',
    ('Africa Cup of Nations', 2004): '2004-01-24',
    ('Africa Cup of Nations', 2006): '2006-01-20',
    ('Africa Cup of Nations', 2008): '2008-01-20',
    ('Africa Cup of Nations', 2010): '2010-01-10',
    ('Africa Cup of Nations', 2012): '2012-01-21',
    ('Africa Cup of Nations', 2013): '2013-01-19',
    ('Africa Cup of Nations', 2015): '2015-01-17',
    ('Africa Cup of Nations', 2017): '2017-01-14',
    ('Africa Cup of Nations', 2019): '2019-06-21',
    ('Africa Cup of Nations', 2021): ('2022-01-09', '2022-02-06'),
    ('Africa Cup of Nations', 2023): '2024-01-13',
    ('Africa Cup of Nations', 2025): '2025-12-21',
    ('Africa Cup of Nations', 2027): '2027-06-19',  # planned, recheck

    ('Asian Cup', 2000): '2000-10-12',
    ('Asian Cup', 2004): '2004-07-17',
    ('Asian Cup', 2007): '2007-07-07',
    ('Asian Cup', 2011): '2011-01-07',
    ('Asian Cup', 2015): '2015-01-09',
    ('Asian Cup', 2019): '2019-01-05',
    ('Asian Cup', 2023): '2024-01-12',
    ('Asian Cup', 2027): '2027-01-07',  # planned, recheck

    ('Gold Cup', 2000): '2000-02-12',
    ('Gold Cup', 2002): '2002-01-18',
    ('Gold Cup', 2003): '2003-07-12',
    ('Gold Cup', 2005): '2005-07-06',
    ('Gold Cup', 2007): '2007-06-06',
    ('Gold Cup', 2009): '2009-07-03',
    ('Gold Cup', 2011): '2011-06-05',
    ('Gold Cup', 2013): '2013-07-07',
    ('Gold Cup', 2015): '2015-07-07',
    ('Gold Cup', 2017): '2017-07-07',
    ('Gold Cup', 2019): '2019-06-15',
    ('Gold Cup', 2021): '2021-07-10',
    ('Gold Cup', 2023): '2023-06-24',
    ('Gold Cup', 2025): '2025-06-14',

    # 2004: group stage in Adelaide, the final was two-legged (home-away) in October
    ('OFC Nations Cup', 2004): ('2004-05-29', '2004-06-06'),
    ('OFC Nations Cup', 2012): '2012-06-01',
    ('OFC Nations Cup', 2016): '2016-05-28',
    ('OFC Nations Cup', 2024): '2024-06-15',

    # explicit dates: 2025 qualifiers in Doha (Nov 25-26) must not leak into the final stage
    ('FIFA Arab Cup', 2012): '2012-06-22',
    ('FIFA Arab Cup', 2021): '2021-11-30',
    ('FIFA Arab Cup', 2025): '2025-12-01',

    ('UEFA Nations League', 2018): '2019-06-05',
    ('UEFA Nations League', 2020): '2021-10-06',
    ('UEFA Nations League', 2022): ('2023-06-14', '2023-06-18'),
    ('UEFA Nations League', 2024): '2025-06-04',

    ('CONCACAF Nations League', 2019): '2021-06-03',
    ('CONCACAF Nations League', 2022): '2023-06-15',
    ('CONCACAF Nations League', 2023): '2024-03-21',
    ('CONCACAF Nations League', 2024): '2025-03-20',

    ('Finalissima', 2022): '2022-06-01',
    ('CONCACAF Cup', 2015): '2015-10-10',
}

# (tournament, season, home_team, away_team) -> home_indicator
# multi-host tournaments: hosts playing outside their country.
# Keys are in the orientation after _handle_visit_stadium (as in the preprocessed data).
FINAL_STAGE_HOME_INDICATOR_OVERRIDES = {
    # Euro 2020: 11 host countries, hosts played only group matches at home
    ('Euro', 2020, 'Russia', 'Denmark'): -1,  # Copenhagen
    ('Euro', 2020, 'Italy', 'Austria'): 0,  # Wembley
    ('Euro', 2020, 'Netherlands', 'Czech Republic'): 0,  # Budapest
    ('Euro', 2020, 'Wales', 'Denmark'): 0,  # Amsterdam
    ('Euro', 2020, 'Croatia', 'Spain'): 0,  # Copenhagen
    ('Euro', 2020, 'Switzerland', 'Spain'): 0,  # St Petersburg
    ('Euro', 2020, 'Belgium', 'Italy'): 0,  # Munich
    ('Euro', 2020, 'Czech Republic', 'Denmark'): 0,  # Baku
    ('Euro', 2020, 'Ukraine', 'England'): 0,  # Rome
    ('Euro', 2020, 'Italy', 'Spain'): 0,  # Wembley
    ('Euro', 2020, 'Italy', 'England'): -1,  # final, Wembley

    # Asian Cup 2007: the quarter-final in Bangkok
    ('Asian Cup', 2007, 'Iraq', 'Vietnam'): 0,

    # Gold Cup 2015: Canada hosted only the last group matchday (Toronto)
    ('Gold Cup', 2015, 'El Salvador', 'Canada'): 0,  # Carson
    ('Gold Cup', 2015, 'Jamaica', 'Canada'): 0,  # Houston

    # Gold Cup 2019: Costa Rica and Jamaica hosted one group doubleheader each
    ('Gold Cup', 2019, 'Costa Rica', 'Bermuda'): 0,  # Frisco
    ('Gold Cup', 2019, 'Haiti', 'Costa Rica'): 0,
    ('Gold Cup', 2019, 'Mexico', 'Costa Rica'): 0,  # quarter-final, Houston
    ('Gold Cup', 2019, 'El Salvador', 'Jamaica'): 0,  # Houston
    ('Gold Cup', 2019, 'Jamaica', 'Curacao'): 0,
    ('Gold Cup', 2019, 'Jamaica', 'Panama'): 0,  # quarter-final, Philadelphia
    ('Gold Cup', 2019, 'Jamaica', 'USA'): -1,  # semi-final, Nashville

    # Gold Cup 2023: Canada hosted only the first group match (Toronto)
    ('Gold Cup', 2023, 'Guatemala', 'Canada'): 0,  # Houston
    ('Gold Cup', 2023, 'Canada', 'Cuba'): 0,

    # Gold Cup 2025: Canada played in Vancouver only in the group stage
    ('Gold Cup', 2025, 'Canada', 'El Salvador'): 0,
    ('Gold Cup', 2025, 'Canada', 'Guatemala'): 0,  # quarter-final
}

# Final stages without a group stage: every final stage match is a knockout match
KNOCKOUT_ONLY_TOURNAMENTS = {
    'UEFA Nations League',
    'CONCACAF Nations League',
    'Finalissima',
    'CONCACAF Cup',
}

# clubs named as countries: collide with national teams in the union of clubs and national teams
CLUBS_NAMED_AS_COUNTRIES = {
    'Liberia': 'Liberia Costa Rica',
    'Andorra': 'Andorra Spain',
}
