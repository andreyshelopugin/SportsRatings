# SportsRatings

Code and data for **"A Modified Glicko Model for Rating Soccer Clubs and National Teams Worldwide"**
(submitted to *SN Computer Science*). An arXiv preprint will be available soon.

The model puts clubs and national teams of all countries on a single probabilistic rating scale:
a Glicko rating system with strength-dependent draws and home advantage, league-aware initialization
(a club enters at the level of its league, a national team at the level of its confederation),
promotion and relegation rules, and goals expectations of a gradient boosting model.

## Ratings

The current ratings are in the root of the repository:

| File | Content |
|---|---|
| `club_ratings.csv` | clubs: rank, rating, league |
| `league_ratings.csv` | leagues: the weighted mean of the top 8 clubs, top 5 / 7 / 10, the ratings of the places 1-10; the links of the league to the other leagues in the last three seasons (matches against foreign clubs, against other leagues, promotions and relegations) and `comparable`: the level of the league is identified by the data |
| `national_ratings.csv` | national teams: rating, rating deviation, confederation, the last match |

## Code

```bash
pip install -r requirements.txt
```

Run everything from the root of the repository.

| File | Purpose |
|---|---|
| `run.py` | the full pipeline: matches, features, goals expectations (LightGBM), ratings. `REFIT = False` uses the fitted parameters in `data/soccer/params/` (1-2 hours), `REFIT = True` refits all parameters (about 30 hours) |
| `paper.py` | all tables of the paper from the fitted parameters (`data/soccer/tables/`) |
| `evaluation_pipeline.py` | the fit on the training seasons (before 2021) and the test predictions |
| `soccer/club_ratings/`, `soccer/national_ratings/` | the rating models, their preprocessing and the fit |
| `soccer/boostings/` | features and the goals model |
| `soccer/matrix_models/` | the baselines: Skellam, Dixon-Coles, Markov |
| `diagnostics/` | ablations, baselines, calibration |

The raw match data (`data/soccer/club_matches.feather`, `national_team_matches.feather`) are not distributed:
without them `run.py` skips the preprocessing of the raw matches and starts from `datasets/soccer/matches.feather` (see below).

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
| `home_score_90`, `away_score_90`, `home_score_et`, `away_score_et` | matches decided after 90 minutes (`AET`, `Pen`): the score after 90 minutes and the goals of extra time, where the match page gives them |
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
- a team is identified by its Flashscore team id: clubs with identical names are told apart (e.g. `River Plate Argentina`, `River Plate Uruguay`), renamed clubs are merged;
- forfeits, walkovers, annulled and abandoned matches removed;
- neutral venues, matches played at the visiting team's stadium and tournament hosts annotated;
- promotion/relegation playoffs separated from regular league matches;
- a penalty shootout directly after 90 minutes (the score after 90 minutes is known, the score of extra time is not) is a draw of the regular time without extra time;
- leagues without any link to the other leagues in any season (no promotion, no cup or international matches) removed.

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