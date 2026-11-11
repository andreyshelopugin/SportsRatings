import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from scipy.stats import poisson

# the colors of the paper: home win, draw, away win
REGION_COLORS = {
    "home": "#1C5985",
    "draw": "#F28C38",
    "away": "#7B3FA0",
}
TEXT_COLOR = "#37474F"


def plot_poisson_draws(
        expected_goals: tuple = (1.1, 1.6),
        max_goals: int = 5,
        font_family: tuple = ("TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"),
        save_path: str | None = "Fig_draws.pdf",
        dpi: int = 300,
        figsize: tuple = (9, 4.6),
        cell_size: float = 0.94,
):
    """Score matrices of two equally strong teams under independent Poisson goals:
    equal teams, but the draw probability falls as the expected goals grow.
    A cell is darker for a more probable score, the draws (the diagonal) are bold."""

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = list(font_family)
    # the fonts are embedded as TrueType, not Type 3
    plt.rcParams["pdf.fonttype"] = 42

    goals = np.arange(max_goals + 1)
    n = len(goals)

    fig, axes = plt.subplots(1, len(expected_goals), figsize=figsize)

    for panel, (ax, mu) in enumerate(zip(axes, expected_goals)):
        matrix = np.outer(poisson.pmf(goals, mu), poisson.pmf(goals, mu))
        # the draw probability over all scores, not only the plotted ones
        draw_probability = (poisson.pmf(np.arange(50), mu) ** 2).sum()

        for i in goals:
            for j in goals:
                region = "draw" if i == j else "home" if i > j else "away"

                # white -> region color, the darker the more likely
                intensity = 0.12 + 0.88 * (matrix[i, j] / matrix.max()) ** 0.7
                color = intensity * np.array(to_rgb(REGION_COLORS[region])) + (1 - intensity) * np.ones(3)

                ax.add_patch(Rectangle((j, i), cell_size, cell_size, color=color, linewidth=0))
                ax.text(
                    j + cell_size / 2,
                    i + cell_size / 2,
                    f"{100 * matrix[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color="white" if intensity > 0.55 else TEXT_COLOR,
                    fontweight="bold" if i == j else "normal",
                )

        ax.set_xlim(-0.06, n)
        ax.set_ylim(n, -0.06)
        ax.set_aspect(1)

        # the away goals on top (as a score table), the panel label below
        ax.set_xticks(goals + cell_size / 2, goals)
        ax.set_yticks(goals + cell_size / 2, goals)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.tick_params(length=0, labelsize=8.5, colors=TEXT_COLOR)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xlabel("Away goals", fontsize=9, color=TEXT_COLOR)
        ax.set_ylabel("Home goals", fontsize=9, color=TEXT_COLOR)
        ax.set_title(
            f"({'abcdefgh'[panel]})  both teams expect {mu} goals,  P(draw) = {100 * draw_probability:.1f}%",
            fontsize=9.5,
            y=-0.1,
        )

        print(f"mu = {mu}: P(draw) = {draw_probability:.4f}")

    handles = [Rectangle((0, 0), 1, 1, color=REGION_COLORS[region]) for region in ("home", "draw", "away")]
    fig.legend(handles, ["home win", "draw", "away win"], loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout(rect=(0, 0.02, 1, 1))

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, facecolor="white", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    plot_poisson_draws()
