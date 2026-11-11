"""Extensions of the national model that did not reduce the test loss (the list of the paper):
    periods: rating periods by the calendar of the national teams (every international window and every match day
             of a final stage) instead of the calendar quarters, the global params refitted on the train matches.
Test loss: matches since 2021-01-01, 90 minutes, as in diagnostics/national_ablation.py. Every variant writes its own national glicko matches and params
(data/soccer/national_extensions/<variant>), the production files are not changed.
Results: data/soccer/tables/national_extensions.csv.
Run from the root of the project."""
import shutil
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from config import config
from diagnostics.national_ablation import GLOBAL_ITERATIONS, TABLES, national_params, test_log_loss, with_increase
from national_pipeline import COUNTRY_PENALTY, DISCOUNT_PER_YEAR, TRAIN_FIT_FOLDER
from soccer.national_ratings.preprocessing import NationalPreprocessing
from soccer.national_ratings.train_glicko import TrainNationalGlicko
from utils.functions import stage

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / 'data/soccer/national_extensions'


@contextmanager
def glicko_matches(path: Path):
    """The national glicko matches from the path instead of the production file."""

    path.parent.mkdir(parents=True, exist_ok=True)

    production = config.soccer['national_glicko_matches']
    config.soccer['national_glicko_matches'] = str(path)
    try:
        yield path
    finally:
        config.soccer['national_glicko_matches'] = production


class NationalPreprocessingWindows(NationalPreprocessing):
    """Rating periods by the calendar of the national teams:
        regular stage: consecutive match dates without a gap > window_gap_days, not longer than max_window_days
                       (international windows; qualifiers of other confederations during a final stage
                       must not chain into one long period)
        final stage: (tournament, season, match day), a team plays at most once a day.
    Periods may overlap in time (a final stage and qualifiers of another confederation), a team plays only in one of them."""

    def __init__(self):
        super().__init__()
        self.window_gap_days = 10
        self.max_window_days = 14

    def _rating_periods(self, matches: pd.DataFrame) -> pd.DataFrame:
        match_day = matches['date'].dt.normalize()

        regular_days = np.sort(match_day[matches['stage'] == 'regular'].unique())

        window_of_day = dict()
        window = 0
        window_start = None
        previous_day = None
        for day in regular_days:
            if previous_day is not None and ((day - previous_day) / np.timedelta64(1, 'D') > self.window_gap_days
                                             or (day - window_start) / np.timedelta64(1, 'D') >= self.max_window_days):
                window += 1
                window_start = None

            if window_start is None:
                window_start = day

            window_of_day[day] = window
            previous_day = day

        matches['period_key'] = np.where(matches['stage'] == 'final',
                                    'final|' + matches['tournament'] + '|' + matches['season'].astype(str)
                                    + '|' + match_day.dt.strftime('%Y-%m-%d'),
                                    'window|' + match_day.map(window_of_day).astype('Int64').astype(str))

        return self._number_periods(matches, match_day)


def periods() -> pd.DataFrame:
    """Quarters (the control: the same refit) vs the windows, the global params refitted from the train fit."""

    reports = []
    for variant, preprocessing in [('full', NationalPreprocessing), ('windows', NationalPreprocessingWindows)]:
        folder = FOLDER / f'periods_{variant}'
        shutil.copytree(TRAIN_FIT_FOLDER, folder / 'params', dirs_exist_ok=True)

        with glicko_matches(folder / 'national_glicko_matches.feather'), national_params(folder / 'params'):
            stage(f'periods: {variant}')
            preprocessing().transform()

            train = TrainNationalGlicko()
            train.discount_per_year = DISCOUNT_PER_YEAR
            train.country_penalty = COUNTRY_PENALTY
            train.fit_global_params(number_iterations=GLOBAL_ITERATIONS)

            reports.append(test_log_loss().assign(variant=variant))

    return with_increase(pd.concat(reports)).assign(experiment='periods')


def main():
    reports = []
    for experiment in [periods]:
        reports.append(experiment())

        # saved after every experiment
        pd.concat(reports).to_csv(TABLES / 'national_extensions.csv', index=False)

    report = pd.concat(reports)
    print(report.pivot_table(index=['experiment', 'variant'], columns='group', values='increase').to_string())


if __name__ == '__main__':
    main()
