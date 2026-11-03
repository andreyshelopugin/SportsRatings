import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle


COLOR_THEMES = {
    "presentation": {
        "draw": "#ffd966",       # draw: yellow
        "home": "#4da3ff",       # home win: blue
        "away": "#65d46e",       # away win: green
        "grid": "#4f4f4f",
        "text": "#111111",
        "dc_edge": "#111111",
        "dc_arrow": "#111111",
    },
    "contrast": {
        "draw": "#ffcc00",
        "home": "#2f80ed",
        "away": "#27ae60",
        "grid": "#333333",
        "text": "#000000",
        "dc_edge": "#000000",
        "dc_arrow": "#000000",
    },
}


def plot_score_probability_matrix(
        n: int = 15,
        theme: str = "presentation",
        font_family: str = "Helvetica Neue",
        save_path: str | None = "score_probability_matrix_dixon_coles_arrows.png",
        dpi: int = 300,
        show_dc_adjustments: bool = True,
        title: str | None = r"Dixon–Coles Adjustment in the Score Matrix",
        figsize: tuple = (9, 9),
        font_scale: float = 1.,
        show_sums: bool = False,
):
    """font_scale multiplies all font sizes, title=None for a figure in a paper (the caption is in the text),
    show_sums adds the sums of the regions (p_H, p_D, p_A) to their labels (28.10)."""

    if theme not in COLOR_THEMES:
        raise ValueError(
            f"Unknown theme '{theme}'. Available themes: {list(COLOR_THEMES.keys())}"
        )

    colors = COLOR_THEMES[theme]
    plt.rcParams["font.family"] = font_family

    matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if i > j:
                matrix[i, j] = 1      # Home win
            elif i < j:
                matrix[i, j] = 2      # Away win
            else:
                matrix[i, j] = 0      # Draw

    cmap = ListedColormap([
        colors["draw"],
        colors["home"],
        colors["away"],
    ])

    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n + 1)
    y = np.arange(n + 1)

    ax.pcolormesh(
        x,
        y,
        matrix,
        cmap=cmap,
        norm=norm,
        edgecolors=colors["grid"],
        linewidth=0.5,
        antialiased=False,
        shading="flat",
    )

    # Put 0-0 in the top-left corner
    ax.invert_yaxis()

    # Strict square matrix
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)

    # Ticks in the center of cells
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)

    ax.set_xticklabels(np.arange(n), fontsize=12 * font_scale, color=colors["text"])
    ax.set_yticklabels(np.arange(n), fontsize=12 * font_scale, color=colors["text"])

    # Put x-axis on top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_xlabel(
        "Away goals (j)",
        fontsize=18 * font_scale,
        labelpad=14,
        color=colors["text"],
        fontweight="medium",
    )

    ax.set_ylabel(
        "Home goals (i)",
        fontsize=18 * font_scale,
        labelpad=14,
        color=colors["text"],
        fontweight="medium",
    )

    if title is not None:
        ax.set_title(
            title,
            fontsize=22 * font_scale,
            fontweight="bold",
            color=colors["text"],
            pad=55,
        )

    ax.tick_params(axis="both", which="major", length=0)

    text_bbox = dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="none",
        alpha=0.88,
    )

    # Region labels
    ax.text(
        n * 0.70,
        n * 0.22,
        "Away win region\n$(i < j)$" + ("\n$p_A = \\sum p(i,j)$" if show_sums else ""),
        ha="center",
        va="center",
        fontsize=14 * font_scale,
        fontweight="semibold",
        color=colors["text"],
        zorder=4,
        bbox=text_bbox,
    )

    ax.text(
        n * 0.30,
        n * 0.78,
        "Home win region\n$(i > j)$" + ("\n$p_H = \\sum p(i,j)$" if show_sums else ""),
        ha="center",
        va="center",
        fontsize=14 * font_scale,
        fontweight="semibold",
        color=colors["text"],
        zorder=4,
        bbox=text_bbox,
    )

    ax.annotate(
        "Draw region\n$(i = j)$" + ("\n$p_D = \\sum p(i,j)$" if show_sums else ""),
        xy=(n * 0.52, n * 0.52),
        xytext=(n * 0.73, n * 0.57),
        fontsize=14 * font_scale,
        fontweight="semibold",
        color=colors["text"],
        ha="center",
        va="center",
        zorder=5,
        bbox=text_bbox,
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.6,
            color=colors["text"],
            shrinkA=0,
            shrinkB=4,
        ),
    )

    if show_dc_adjustments:
        adjusted_cells = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for i, j in adjusted_cells:
            ax.add_patch(
                Rectangle(
                    (j, i),
                    1,
                    1,
                    fill=False,
                    edgecolor=colors["dc_edge"],
                    linewidth=2.4,
                    zorder=7,
                    clip_on=False,
                )
            )

        formula_bbox = dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="none",
            alpha=0.92,
        )

        # Two arrows from top-right, two from bottom-left
        adjustments = [
            {
                "cell": (0, 0),
                "text": r"$(0,0):\ \times(1-\mu_1\mu_2\rho)$",
                "text_pos": (4.2, 0.45),   # top-right
            },
            {
                "cell": (0, 1),
                "text": r"$(0,1):\ \times(1+\mu_1\rho)$",
                "text_pos": (4., 1.55),   # top-right
            },
            {
                "cell": (1, 0),
                "text": r"$(1,0):\ \times(1+\mu_2\rho)$",
                "text_pos": (0.15, 4.25),
            },
            {
                "cell": (1, 1),
                "text": r"$(1,1):\ \times(1-\rho)$",
                "text_pos": (3.0, 3.0),
            },
        ]

        for adj in adjustments:
            i, j = adj["cell"]
            ax.annotate(
                adj["text"],
                xy=(j + 0.5, i + 0.5),
                xytext=adj["text_pos"],
                fontsize=11.5 * font_scale,
                fontweight="semibold",
                color=colors["text"],
                ha="left",
                va="center",
                zorder=8,
                bbox=formula_bbox,
                arrowprops=dict(
                    arrowstyle="->",
                    linewidth=1.4,
                    color=colors["dc_arrow"],
                    shrinkA=4,
                    shrinkB=5,
                    connectionstyle="arc3,rad=0.10",
                ),
            )

    # Outer border
    border = Rectangle(
        (0, 0),
        n,
        n,
        fill=False,
        edgecolor=colors["text"],
        linewidth=1.8,
        zorder=10,
        clip_on=False,
    )
    ax.add_patch(border)

    # Hide default spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.15,
            facecolor="white",
        )

    plt.show()


plot_score_probability_matrix(
    n=15,
    theme="presentation",
    save_path="score_probability_matrix_dixon_coles_arrows.png",
    show_dc_adjustments=True,
)
