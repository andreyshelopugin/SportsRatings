"""The diagrams of the rating pipeline for the appendix (clubs and national teams),
in the style of the old Fig1.jpg: yellow boxes, blue titles, black italic descriptions, red italic outputs."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

YELLOW = '#F7C613'
BLUE = '#1a1aff'
RED = '#d6004c'

BOX_W, BOX_H = 8.2, 1.75
X0 = 0.9


def box(ax, y, title, text, output, height=BOX_H, x=X0, width=BOX_W):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0.02,rounding_size=0.25',
                                facecolor=YELLOW, edgecolor='#222222', linewidth=1.4))
    cx = x + width / 2
    ax.text(cx, y + height - 0.38, title, ha='center', va='center', fontsize=13, color=BLUE)
    ax.text(cx, y + height / 2 - 0.05, text, ha='center', va='center', fontsize=8.5, style='italic')
    ax.text(cx, y + 0.3, output, ha='center', va='center', fontsize=8.5, style='italic', color=RED)


def arrow(ax, y_from, y_to, x=X0 + BOX_W / 2):
    ax.add_patch(FancyArrowPatch((x, y_from), (x, y_to), arrowstyle='-|>', mutation_scale=14, color='#333333', linewidth=1.2))


def container(ax, y, height, title, text, footer):
    ax.add_patch(FancyBboxPatch((0.1, y), 9.8, height, boxstyle='round,pad=0.02,rounding_size=0.6',
                                facecolor='white', edgecolor='#222222', linewidth1=None) if False else
                 FancyBboxPatch((0.6, y), 8.8, height, boxstyle='round,pad=0.02,rounding_size=0.6',
                                facecolor='white', edgecolor='#222222', linewidth=1.4))
    ax.text(5, y + height - 0.4, title, ha='center', va='center', fontsize=13, color=BLUE)
    ax.text(5, y + height - 1.05, text, ha='center', va='center', fontsize=8.5, style='italic')
    ax.text(5, y + 0.4, footer, ha='center', va='center', fontsize=8.5, style='italic', color=RED)


def loop(ax, y_top, y_bottom, x=X0, width=BOX_W):
    """The arrows between the rating update and the outcome probabilities (as in the old figure)."""
    ax.add_patch(FancyArrowPatch((x, y_top), (x, y_bottom), connectionstyle='arc3,rad=0.5',
                                 arrowstyle='-|>', mutation_scale=12, color='#333333', linewidth=1))
    ax.add_patch(FancyArrowPatch((x + width, y_bottom), (x + width, y_top), connectionstyle='arc3,rad=0.5',
                                 arrowstyle='-|>', mutation_scale=12, color='#333333', linewidth=1))


def diagram(blocks_before, glicko, blocks_after, path):
    n = len(blocks_before) + len(blocks_after)
    container_h = 6.6
    total = n * (BOX_H + 0.55) + container_h + 0.6
    fig, ax = plt.subplots(figsize=(7, total * 0.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, total)
    ax.axis('off')

    y = total - BOX_H - 0.1
    for title, text, output in blocks_before:
        box(ax, y, title, text, output)
        arrow(ax, y, y - 0.55)
        y -= BOX_H + 0.55

    top = y + BOX_H
    y_container = top - container_h
    container(ax, y_container, container_h, glicko['title'], glicko['text'], glicko['footer'])
    y_rating = top - 1.55 - BOX_H
    # the inner boxes are narrower, the loops stay inside the container
    box(ax, y_rating, *glicko['rating'], x=1.4, width=7.2)
    y_probs = y_rating - BOX_H - 0.7
    box(ax, y_probs, *glicko['probabilities'], x=1.4, width=7.2)
    loop(ax, y_rating + BOX_H / 2, y_probs + BOX_H / 2, x=1.4, width=7.2)

    y = y_container
    for title, text, output in blocks_after:
        arrow(ax, y, y - 0.55)
        y -= BOX_H + 0.55
        box(ax, y, title, text, output)

    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


CLUBS = dict(
    before=[('Match data', 'results of first and second divisions, domestic cups\nand continental competitions, venue and spectators',
             'outcome after 90 minutes, extra time, venue status'),
            ('Goal model', 'LightGBM on the recent goals of the teams, the venue and the competition',
             'expected goals of both teams'),
            ('Scoring intensity', 'sum of the expected goals of both teams, clipped and centered', r'$g_{ij}$')],
    glicko=dict(title='Parameter estimation',
                text='coordinate descent on the weighted log loss of the matches:\ninternational matches weighted up, older periods discounted',
                footer='global, country and league parameters',
                rating=('Modified Glicko', 'rating updates: league-aware initialization, promotion and relegation,\nnewcomers, growth of the rating deviation per period and per season',
                        r'ratings $\mu_i$, deviations $\phi_i$'),
                probabilities=('Outcome probabilities', 'home advantage (crowd, travel), strength-dependent draws,\nscoring intensity, extra time as a second stage',
                               'win, draw, loss; log loss')),
    after=[('Club ratings', 'ratings of all clubs with the estimated parameters', 'club ratings and deviations'),
           ('League ratings', 'weighted mean rating of the eight strongest clubs of a league', 'league ratings')])

NATIONAL = dict(
    before=[('Match data', 'World Cup and continental final tournaments,\ntheir qualifiers and the Nations Leagues',
             'outcome after 90 minutes, extra time, host, neutral venue'),
            ('Goal model', 'the same LightGBM model, trained on club and national team matches',
             'expected goals of both teams'),
            ('Scoring intensity', 'sum of the expected goals of both teams, clipped and centered', r'$g_{ij}$')],
    glicko=dict(title='Parameter estimation',
                text='coordinate descent on the weighted log loss of the matches:\nfinal tournaments weighted up, older periods discounted',
                footer='global parameters and levels of the confederations',
                rating=('Modified Glicko', 'rating updates: initialization at the level of the confederation,\ngrowth of the rating deviation with the time between rating periods',
                        r'ratings $\mu_i$, deviations $\phi_i$'),
                probabilities=('Outcome probabilities', 'home advantage of the host, neutral venues of final tournaments,\nstrength-dependent draws, extra time as a second stage',
                               'win, draw, loss; log loss')),
    after=[('National team ratings', 'ratings of all national teams with the estimated parameters',
            'ratings and deviations of the national teams')])

if __name__ == '__main__':
    diagram(CLUBS['before'], CLUBS['glicko'], CLUBS['after'], 'Fig_pipeline_clubs.pdf')
    diagram(NATIONAL['before'], NATIONAL['glicko'], NATIONAL['after'], 'Fig_pipeline_national.pdf')
    diagram(CLUBS['before'], CLUBS['glicko'], CLUBS['after'], 'Fig_pipeline_clubs.png')
    diagram(NATIONAL['before'], NATIONAL['glicko'], NATIONAL['after'], 'Fig_pipeline_national.png')
