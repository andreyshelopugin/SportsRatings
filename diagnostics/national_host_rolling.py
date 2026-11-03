"""Rolling origin for the host advantage of the final stages (28.10): is the host advantage of the old tournaments
(Korea 2002, ...) the same as of the new ones (VAR since the World Cup 2018, Qatar 2022, ...)?

For every split date the global params and the final stage param gamma (28.10: instead of alpha, host, beta) are fitted
from the defaults on the matches before the split only (no params that saw the future), country offsets are not fitted.
Then the test loss after the split: the fitted gamma vs gamma 0, 0.5 and 1.
If the fitted gamma falls with the split date, the host advantage of the final stages changed with time.

The saved national params are restored at the end (also after an error).
Run:
    .venv_core/bin/python diagnostics/national_host_rolling.py
"""
import sys
import warnings
from math import sqrt
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import config
from soccer.national_ratings.train_glicko import TrainNationalGlicko, _backup, _restore
from utils.functions import load_pickle, save_pickle

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)

SPLIT_DATES = ['2006-01-01', '2010-01-01', '2014-01-01', '2018-01-01', '2022-01-01']


def reset_to_defaults(train: TrainNationalGlicko, paths: dict):
    """The defaults of NationalRatings: the fit of a split starts from the same point and does not see the future."""

    save_pickle({**train.init_global_params, 'rd_growth': sqrt(train.rd_growth_squared)}, paths['global'])
    save_pickle(dict(), paths['confederation'])
    save_pickle(dict(train.init_final_stage_params), paths['final_stage'])
    save_pickle(dict(), paths['country'])


def main():
    paths = config.soccer['national_params']
    backups = _backup(paths, 'before_host_rolling')

    fitted = []
    try:
        for split_date in SPLIT_DATES:
            print(f'\n========== split {split_date} ==========', flush=True)

            reset_to_defaults(TrainNationalGlicko(train_test_split_date=split_date), paths)

            train = TrainNationalGlicko(train_test_split_date=split_date)
            train.fit_global_params(number_iterations=100, fit_final_stage=True)

            final_stage_params = load_pickle(paths['final_stage'])
            fitted.append({'split_date': split_date, **final_stage_params})
            print(f'fitted final stage params: {final_stage_params}', flush=True)

            variants = {f'gamma_{gamma}': {'gamma': gamma} for gamma in (0., 0.5, 1.)}
            TrainNationalGlicko(train_test_split_date=split_date).compare_losses(variants)

    finally:
        _restore(backups)
        print('\nsaved national params restored')

    print('\n===== fitted gamma by the split date =====')
    print(pd.DataFrame(fitted).to_string(index=False))


if __name__ == '__main__':
    main()
