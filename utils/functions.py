from datetime import datetime
from pathlib import Path
from pickle import dump, load

import pandas as pd


def save_pickle(variable, path: str):
    with open(path, 'wb') as f:
        dump(variable, f, protocol=5)


def load_pickle(path: str):
    with open(path, 'rb') as f:
        return load(f)


FEATHER_KW = dict(compression="zstd", compression_level=9)


def save_feather(df: pd.DataFrame, path, **kw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_feather(path, **{**FEATHER_KW, **kw})


def stage(name: str):
    """The header of a stage of a pipeline with the time."""
    print(f'\n[{datetime.now():%H:%M:%S}] ===== {name} =====', flush=True)
