import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from scipy.stats import poisson


REGION_COLORS = {
    "draw": "#f2b800",       # draw: yellow
    "home": "#2f80ed",       # home win: blue
    "away": "#27ae60",       # away win: green
}


def plot_poisson_draws(
        expected_goals: tuple = (1.1, 1.6),
        max_goals: int = 5,
        font_family: str = "serif",
        save_path: str | None = "Fig_draws.pdf",
        dpi: int = 300,
        figsize: tuple = (7.2, 3.5),
        font_scale: float = 1.,
        max_probability: float = 0.14,
):
    """Score matrices of two equally strong teams under independent Poisson goals:
    equal teams, but the draw probability falls as the expected goals grow (28.10)."""

    plt.rcParams["font.family"] = font_family
    goals = np.arange(max_goals + 1)

    fig, axes = plt.subplots(1, len(expected_goals), figsize=figsize, constrained_layout=True)

    for panel, (ax, mu) in enumerate(zip(axes, expected_goals)):
        matrix = np.outer(poisson.pmf(goals, mu), poisson.pmf(goals, mu))
        # the draw probability over all scores, not only the plotted ones
        draw_probability = (poisson.pmf(np.arange(50), mu) ** 2).sum()

        # white -> region color, the darker the more likely
        rgb = np.ones((len(goals), len(goals), 3))

        for i in goals:
            for j in goals:
                region = "draw" if i == j else "home" if i > j else "away"
                alpha = min(matrix[i, j] / max_probability, 1)
                rgb[i, j] = 1 - alpha * (1 - np.array(to_rgb(REGION_COLORS[region])))

                ax.text(
                    j,
                    i,
                    f"{100 * matrix[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.5 * font_scale,
                    color="white" if alpha > 0.6 and i != j else "black",
                )

            # Draw cells
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.4))

        ax.imshow(rgb)

        ax.set_xticks(goals)
        ax.set_yticks(goals)
        ax.tick_params(labelsize=9 * font_scale)
        ax.set_xlabel("Away goals", fontsize=9 * font_scale)
        ax.set_ylabel("Home goals", fontsize=9 * font_scale)

        # 31.10: the away goals on top (as a score table), the panel label below
        # ax.set_title(
        #     f"({'abcdefgh'[panel]}) $\\mu_1=\\mu_2={mu}$, P(draw) = {100 * draw_probability:.1f}%",
        #     fontsize=10 * font_scale,
        # )
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_title(
            f"({'abcdefgh'[panel]}) $\\mu_1=\\mu_2={mu}$, P(draw) = {100 * draw_probability:.1f}%",
            fontsize=10 * font_scale,
            y=-0.1,
        )

        print(f"mu = {mu}: P(draw) = {draw_probability:.4f}")

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, facecolor="white")

    plt.show()


plot_poisson_draws()
