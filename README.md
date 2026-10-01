# SportsRatings

Data supporting the research on league-aware Glicko ratings for soccer clubs and national teams.

The goal of the project is a single probabilistic rating scale that makes teams comparable
across leagues, countries and confederations. Ratings for other sports will be added later.

## Data

`datasets/soccer/matches.feather` — about 1.2 million matches in one table:

| Source | Coverage |
|---|---|
| Clubs | first and second divisions of nearly every country, domestic cups and super cups, continental club competitions, FIFA Club World Cup; seasons from 2000 |
| National teams | World Cup and continental championships with qualifiers, Nations Leagues, friendlies; from 1998 (qualifiers for Euro 2000 and the 2002 World Cup) |

### Loading

```python
import pandas as pd

matches = pd.read_feather('datasets/soccer/matches.feather')
clubs = matches[matches['source'] == 'clubs']
national = matches[matches['source'] == 'national']
```

### Main columns

| Column | Description |
|---|---|
| `match_id` | stable id: hash of source, match day, home and away team |
| `source` | `clubs` or `national` |
| `date` | kick-off date and time |
| `season` | season start year (for national teams: year of the final tournament) |
| `country`, `tournament`, `tournament_name`, `tournament_type` | competition |
| `home_team`, `away_team` | team names, unique across countries and sources |
| `home_score`, `away_score` | score after extra time if played; penalty shootouts are not included in the score |
| `note` | `AET` (decided in extra time), `Pen` (decided on penalties) |
| `leg`, `first_leg_home`, `first_leg_away`, `aggregate_home`, `aggregate_away` | two-legged ties |
| `outcome` | `H` / `D` / `A` by the final score |
| `outcome_regular_time` | `H` / `D` / `A` after 90 minutes |
| `outcome_5` | regular time win/draw, extra time win, draw after extra time (penalties) |
| `is_extra_time`, `is_extra_time_possible` | extra time played / would be played in case of a draw |
| `is_neutral`, `home_indicator` | neutral venue; for national teams `1` / `0` / `-1` = home / neutral / away team at home |
| `is_championship`, `is_playoff`, `is_relegation_group` | league matches, promotion/relegation playoffs, relegation groups |
| `home_team_league`, `away_team_league` | league of each club in the season |

### Preprocessing

- penalty shootout goals removed from the match score;
- teams with identical names in different countries disambiguated (e.g. `River Plate Argentina`, `River Plate Uruguay`);
- forfeits, walkovers, annulled and abandoned matches removed;
- neutral venues, matches played at the visiting team's stadium and tournament hosts annotated;
- promotion/relegation playoffs separated from regular league matches.

### Known issues

Older seasons contain a small number of inconsistencies: approximate match dates
(several matches of one team on the same day), a few two-legged cup ties without a leg
label, and rare duplicates. They affect a negligible share of matches.

## Source and terms of use

Match results were collected from [Flashscore](https://www.flashscore.com), an excellent source
of soccer results worldwide. The data is shared for non-commercial research and reproducibility
purposes only. If you are a rights holder and have concerns, please open an issue and the data
will be adjusted or removed.

## Citation

- Journal version (club ratings methodology): *link will be added*
- arXiv preprint: *link will be added*