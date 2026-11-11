import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import config

# the figure of the paper (non-stationarity, the discount of the loss): home advantage by season,
# domestic league matches of the clubs, neutral venues excluded, 90 minutes outcome

POINTS = {'H': (3, 0), 'D': (1, 1), 'A': (0, 3)}

LEVEL_COLORS = {
    'all': '#2f80ed',
    'first_league': '#27ae60',
    'second_league': '#f2b800',
    # the top 5 first leagues, the leagues with VAR since 2017-2019
    'top_5': '#c0392b',
}

TOP_5_COUNTRIES = ['England', 'Spain', 'Germany', 'Italy', 'France']


def home_advantage_by_season(matches: pd.DataFrame) -> pd.DataFrame:
    """Home points minus away points per match and the home win share minus the away win share, by season."""

    home_points = matches['outcome_regular_time'].map(lambda outcome: POINTS[outcome][0])
    away_points = matches['outcome_regular_time'].map(lambda outcome: POINTS[outcome][1])

    return (matches
            .assign(points_difference=home_points - away_points,
                    win_difference=(matches['outcome_regular_time'] == 'H').astype(float)
                                   - (matches['outcome_regular_time'] == 'A').astype(float))
            .groupby('season')
            .agg(n_matches=('points_difference', 'size'),
                 points_difference=('points_difference', 'mean'),
                 points_difference_std=('points_difference', 'std'),
                 win_difference=('win_difference', 'mean')))


def plot_home_advantage(save_path: str | None = 'Fig_home_advantage.pdf',
                        font_family: str = 'serif',
                        figsize: tuple = (7.2, 3.3),
                        min_season: int = 2001,
                        max_season: int = 2025,
                        dpi: int = 300):
    """Home points minus away points per match by season for all, first and second leagues,
    95% intervals (2 standard errors), the pandemic seasons shaded."""

    plt.rcParams['font.family'] = font_family

    columns = ['season', 'country', 'tournament_type', 'is_neutral', 'outcome_regular_time', 'is_pandemic']
    matches = pd.read_feather(config.soccer['preprocessed_club_matches'], columns=columns)

    matches = matches.loc[matches['tournament_type'].isin(['first_league', 'second_league'])
                          & (matches['is_neutral'] != True)
                          & matches['outcome_regular_time'].isin(list(POINTS))
                          & matches['season'].between(min_season, max_season)]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # # the seasons with matches without spectators (2019/2020 - 2020/2021 in the start year notation)
    # ax.axvspan(2019.5, 2021.5, color='#e0e0e0', zorder=0)
    # ax.text(2020.5, 0.02, 'COVID-19', ha='center', va='bottom', fontsize=9, color='#555555')

    # from the data, the share of the pandemic matches (is_pandemic) of every season sets the darkness of its band:
    # 2019: 13%, 2020: 93%, 2021: 9%
    pandemic_share = matches.groupby('season')['is_pandemic'].mean()
    for season, share in pandemic_share.items():
        if share > 0.01:
            ax.axvspan(season - 0.5, season + 0.5, color='#9e9e9e', alpha=0.6 * share, linewidth=0, zorder=0)
    ax.text(2020, 0.02, 'COVID-19', ha='center', va='bottom', fontsize=9, color='#555555')

    # VAR in the top leagues: Bundesliga and Serie A 2017/2018, La Liga and Ligue 1 2018/2019, Premier League 2019/2020
    ax.axvline(2017, color='#555555', linestyle=':', linewidth=1, zorder=1)
    ax.text(2016.8, 0.02, 'VAR in top leagues', ha='right', va='bottom', fontsize=9, color='#555555')

    for level, color in LEVEL_COLORS.items():
        if level == 'all':
            part = matches
        elif level == 'top_5':
            part = matches.loc[(matches['tournament_type'] == 'first_league') & matches['country'].isin(TOP_5_COUNTRIES)]
        else:
            part = matches.loc[matches['tournament_type'] == level]
        by_season = home_advantage_by_season(part)

        error = 2 * by_season['points_difference_std'] / np.sqrt(by_season['n_matches'])
        label = {'all': 'All leagues', 'first_league': 'First divisions', 'second_league': 'Second divisions',
                 'top_5': 'Top 5 European leagues'}[level]

        ax.plot(by_season.index, by_season['points_difference'], marker='o', markersize=3,
                color=color, label=label, linewidth=1.5 if level == 'all' else 1.)
        # no band for the top 5 (~1,800 matches per season, +-0.12): the band hides the other lines
        if level != 'top_5':
            ax.fill_between(by_season.index, by_season['points_difference'] - error, by_season['points_difference'] + error,
                            color=color, alpha=0.15, linewidth=0)

        print(level)
        print(by_season.round(4).to_string())

    ax.set_xlabel('Season (start year)')
    ax.set_ylabel('Home minus away points per match')
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=dpi)

    return fig


if __name__ == '__main__':
    plot_home_advantage()
