import warnings
from pathlib import Path

from config import config
from full_pipeline import ALL_SEASONS, LEAGUE_FIT_ITERATIONS, SHARED_SHIFTS
from national_pipeline import production
from soccer.boostings.features import Features
from soccer.boostings.outcomes_lgbm import OutcomesLGBM
from soccer.club_ratings.preprocessing import ClubPreprocessing
from soccer.club_ratings.ratings import Ratings
from soccer.club_ratings.train_glicko import TrainGlicko
from soccer.national_ratings.preprocessing import NationalPreprocessing
from soccer.national_ratings.ratings import NationalRatings
from soccer.preprocessing import MatchesPreprocessor

warnings.filterwarnings('ignore')

# False: the data, the lgbm and the ratings with the saved params (~1-2 hours), True: and a refit of all params (~30 hours)
REFIT = False


def main(refit: bool):
    # the raw matches of Flashscore are not public: without them the pipeline starts from datasets/soccer/matches.feather
    if Path(config.soccer['club_matches']).exists():
        MatchesPreprocessor().transform()

    Features().transform()

    OutcomesLGBM().save_model()
    OutcomesLGBM().predict_chunks()

    ClubPreprocessing().transform()
    NationalPreprocessing().transform()

    if refit:
        # all stages (see TrainGlicko.fit), the params are saved after every improvement:
        # the fit can be stopped, then normalize_params() and Ratings().transform() alone
        train = TrainGlicko(train_test_split_season=ALL_SEASONS)
        if Path(config.soccer['params']['league_params']).exists():
            train.fit(SHARED_SHIFTS, LEAGUE_FIT_ITERATIONS)
        else:
            train.fit_from_scratch(SHARED_SHIFTS, league_fit_iterations=LEAGUE_FIT_ITERATIONS)
        TrainGlicko(train_test_split_season=ALL_SEASONS).normalize_params()

    Ratings().transform()

    if refit:
        # global params, country offsets, normalization and national_ratings.csv
        production()
    else:
        NationalRatings().transform()


if __name__ == '__main__':
    main(refit=REFIT)
