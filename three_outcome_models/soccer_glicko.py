from dataclasses import dataclass
from math import log, exp, sqrt


@dataclass(slots=True)
class Rating:
    mu: float = 1500.
    rd: float = 250.


class SoccerGlicko(object):
    """Five-outcome version (90 minutes + extra time).
    Based on the "Proposed ICCF Rating Algorithm"
    and "Paired comparison models with strength-dependent ties and order effects" by Mark E. Glickman

    Nested model: 90 minutes is the three-outcome model, extra time is a separate
    three-outcome model with its own params (params_et).
    """

    def __init__(self, mu=1500.):
        self.mu = mu

        self.q = log(10) / 400
        self.q1 = 400 / log(10)  # 173.71

    def _convert_into_glicko2(self, rating: Rating) -> Rating:
        """Step a.
        Convert the ratings and RDs to the standardized scale
        """
        return Rating((rating.mu - self.mu) * self.q, rating.rd * self.q)

    def _convert_from_glicko2(self, rating: Rating) -> Rating:
        """Step g.
        Convert ratings and RD’s back to original (Elo) scale.
        """
        return Rating(rating.mu * self.q1 + self.mu, rating.rd * self.q1)

    @staticmethod
    def probabilities(mu: float, opp_mu: float, home_indicator: int, goals_expectations: float, params: dict) -> tuple[float, float, float]:
        """Step c.
        Define the probability of a win, draw and loss as a function of the standardized ratings.
        x = 1 if player i plays white and −1 if playing black, 0 if neutral in original paper.

        Params related to home team. Params of home and away teams could be different if they are from different leagues

        kappa scales the whole rating scale (extra time): the strength difference in win / loss
        and the strength level in draw. kappa = 1 for 90 minutes.
        With kappa in all three linear predictors the derivatives in _d1_d2 are exactly kappa * d1, kappa ** 2 * d2.
        """

        kappa = params.get('kappa', 1.)

        avg_mu = (mu + opp_mu) / 2

        advantage = home_indicator * (params['crowd']
                                      + params['logistic']
                                      + params['strength'] * avg_mu
                                      + params['scoring'] * goals_expectations)

        win = exp(kappa * mu + advantage)

        draw = exp(params['draw_frequency'] + params['strength_overall'] * kappa * avg_mu + params['scoring_overall'] * goals_expectations)

        loss = exp(kappa * opp_mu - advantage)

        s = win + draw + loss

        return win / s, draw / s, loss / s

    def five_outcome_probabilities(self, rating: Rating, opp_rating: Rating, home_indicator: int, goals_expectations: float,
                                   params: dict, params_et: dict, goals_expectations_et: float) -> tuple[float, float, float, float, float]:
        """Single-leg knockout: extra time iff draw after 90 minutes.
        For two-legged ties replace P(draw in 90) with P(aggregate tie).

        Returns win in 90, loss in 90, win in extra time, loss in extra time, tie in extra time
        """

        mu = self._convert_into_glicko2(rating).mu
        opp_mu = self._convert_into_glicko2(opp_rating).mu

        win, draw, loss = self.probabilities(mu, opp_mu, home_indicator, goals_expectations, params)

        win_et, draw_et, loss_et = self.probabilities(mu, opp_mu, home_indicator, goals_expectations_et, params_et)

        return win, loss, draw * win_et, draw * loss_et, draw * draw_et

    def _d1_d2(self, rating: Rating, opp_rating: Rating, home_indicator: int, goals_expectations: float,
               params: dict, outcome: float) -> tuple[float, float]:
        """To account for the uncertainty in an opponent’s rating in computing probabilities,
        the probabilities in step (c) are replaced with the average of probabilities evaluated at the opponent’s rating of μj − σj and μj + σj

        outcome is 1, 0.5, 0

        The formulas are the derivatives for Glickman's model: rating enters win with coefficient 1 and draw with 1/2
        (strength_overall = 1, strength = 0). kappa multiplies all three linear predictors in probabilities,
        so d1 is scaled by kappa and d2 by kappa ** 2 (chain rule)"""

        kappa = params.get('kappa', 1.)

        mu = rating.mu

        opp_mu = opp_rating.mu
        opp_rd = opp_rating.rd

        # step d
        pr_win_minus, pr_draw_minus, pr_loss_minus = self.probabilities(mu, opp_mu - opp_rd, home_indicator, goals_expectations, params)

        pr_win_plus, pr_draw_plus, pr_loss_plus = self.probabilities(mu, opp_mu + opp_rd, home_indicator, goals_expectations, params)

        # step e
        w1_minus = pr_win_minus + 0.5 * pr_draw_minus
        w1_plus = pr_win_plus + 0.5 * pr_draw_plus

        w2_minus = pr_win_minus + 0.25 * pr_draw_minus
        w2_plus = pr_win_plus + 0.25 * pr_draw_plus

        if outcome == 1:
            p = pr_win_minus + pr_win_plus

            d1 = (pr_win_minus * (1 - w1_minus) + pr_win_plus * (1 - w1_plus)) / p

            d2 = (pr_win_minus * (1 - w2_minus + 2 * w1_minus * (w1_minus - 1))
                  + pr_win_plus * (1 - w2_plus + 2 * w1_plus * (w1_plus - 1))) / p - d1 ** 2

        elif outcome == 0.5:
            p = pr_draw_minus + pr_draw_plus

            d1 = (pr_draw_minus * (0.5 - w1_minus) + pr_draw_plus * (0.5 - w1_plus)) / p

            d2 = (pr_draw_minus * (0.25 - w2_minus + 2 * w1_minus * (w1_minus - 0.5))
                  + pr_draw_plus * (0.25 - w2_plus + 2 * w1_plus * (w1_plus - 0.5))) / p - d1 ** 2

        else:
            p = pr_loss_minus + pr_loss_plus

            d1 = (- pr_loss_minus * w1_minus - pr_loss_plus * w1_plus) / p

            d2 = (pr_loss_minus * (- w2_minus + 2 * w1_minus * w1_minus)
                  + pr_loss_plus * (- w2_plus + 2 * w1_plus * w1_plus)) / p - d1 ** 2

        return kappa * d1, kappa ** 2 * d2

    def update_rating(self, rating: Rating, results: list[tuple[int, float, dict, float, Rating, tuple | None]]) -> Rating:
        """
            Params are related to our team.

            The last element of a result is None if there was no extra time,
            otherwise (goals_expectations_et, params_et, outcome_et).
            Extra time is taken from data (AET / Pen), not from the 90 minutes outcome,
            because in two-legged ties it follows an aggregate tie.
        """

        rating = self._convert_into_glicko2(rating)

        d1_sum = 0
        d2_sum = 0
        for home_indicator, goals_expectations, params, outcome, opp_rating, extra_time in results:
            opp_rating = self._convert_into_glicko2(opp_rating)

            d1, d2 = self._d1_d2(rating, opp_rating, home_indicator, goals_expectations, params, outcome)

            d1_sum += d1
            d2_sum += d2

            # log-likelihoods of 90 minutes and extra time add up
            if extra_time is not None:
                goals_expectations_et, params_et, outcome_et = extra_time

                d1, d2 = self._d1_d2(rating, opp_rating, home_indicator, goals_expectations_et, params_et, outcome_et)

                d1_sum += d1
                d2_sum += d2

        # step f
        new_rd = 1 / sqrt(1 / rating.rd ** 2 - d2_sum)

        new_mu = rating.mu + new_rd ** 2 * d1_sum

        # step g
        rating = self._convert_from_glicko2(Rating(new_mu, new_rd))

        return rating


def call_example():
    """"""

    rating = Rating(1900, 80)

    home_params = {
        'crowd': 0.15,
        'logistic': 0.15,
        'strength': 0.037,
        'scoring': 0,

        'draw_frequency': -0.471,
        'strength_overall': 1.120,
        'scoring_overall': 0,
    }

    # illustrative values
    home_params_et = {
        'kappa': 0.8,
        'crowd': 0.05,
        'logistic': 0.05,
        'strength': 0,
        'scoring': 0,

        'draw_frequency': 0.8,
        'strength_overall': 1.,
        'scoring_overall': 0,
    }

    results = [(1, 2.5, home_params, 1, Rating(1750, 150), None),
               (1, 2.2, home_params, 0.5, Rating(2000, 70), None),
               (1, 3, home_params, 0, Rating(2300, 50), None),
               (-1, 3.8, home_params, 0, Rating(1500, 50), None),
               (-1, 2.4, home_params, 0.5, Rating(1950, 60), (0.9, home_params_et, 1)),
               (1, 2.6, home_params, 0, Rating(1850, 70), (0.8, home_params_et, 0.5)),
               ]

    venues = {1: 'home', -1: 'away', 0: 'neutral'}
    outcomes = {1: 'win', 0.5: 'draw', 0: 'loss'}

    print(f'Initial rating: mu = {rating.mu:.1f}, rd = {rating.rd:.1f}')
    print()
    print('Matches in the rating period:')
    for i, (home_indicator, goals_expectations, params, outcome, opp_rating, extra_time) in enumerate(results, 1):
        line = (f'  {i}. {venues[home_indicator]:<7} vs opponent {opp_rating.mu:.0f} (rd {opp_rating.rd:.0f}), '
                f'expected goals {goals_expectations}, 90 min: {outcomes[outcome]}')
        if extra_time is not None:
            goals_expectations_et, params_et, outcome_et = extra_time
            line += f' -> extra time (expected goals {goals_expectations_et}): {outcomes[outcome_et]}'
        print(line)
    print()

    # from Proposed ICCF Rating Algorithm (beta1 += 1)
    new_rating = SoccerGlicko().update_rating(rating, results)

    print(f'Updated rating: mu = {new_rating.mu:.1f} ({new_rating.mu - rating.mu:+.1f}), '
          f'rd = {new_rating.rd:.1f} ({new_rating.rd - rating.rd:+.1f})')

    print()

    probs = SoccerGlicko().five_outcome_probabilities(Rating(1900, 80), Rating(1850, 70), 1, 2.5,
                                                      home_params, home_params_et, 0.8)
    labels = ['win in 90', 'loss in 90', 'win in extra time', 'loss in extra time', 'tie in extra time']

    print('Single-leg knockout: 1900 at home vs 1850, expected goals 2.5 (extra time 0.8)')
    for label, p in zip(labels, probs):
        print(f'  {label:<20} {p:.3f}')
    print(f'  {"sum":<20} {sum(probs):.3f}')

    draw_90 = probs[2] + probs[3] + probs[4]
    print()
    print(f'P(extra time) = P(draw in 90) = {draw_90:.3f}')
    print(f'Within extra time: win {probs[2] / draw_90:.3f}, loss {probs[3] / draw_90:.3f}, '
          f'tie {probs[4] / draw_90:.3f}')
    print(f'Overall: win {probs[0] + probs[2]:.3f}, loss {probs[1] + probs[3]:.3f}, '
          f'tie after extra time {probs[4]:.3f}')


def check_derivatives():
    """Compare d1, d2 from _d1_d2 with finite differences of log((P(outcome | opp_mu - rd) + P(outcome | opp_mu + rd)) / 2).
    Glickman's formulas are exact when rating enters draw with 1/2: strength_overall = 1, strength = 0."""

    glicko = SoccerGlicko()

    rating = glicko._convert_into_glicko2(Rating(1900, 80))
    opp_rating = glicko._convert_into_glicko2(Rating(1850, 70))

    params = {
        'kappa': 0.75,
        'crowd': 0.05,
        'logistic': 0.05,
        'strength': 0,
        'scoring': 0,

        'draw_frequency': 0.25,
        'strength_overall': 1.,
        'scoring_overall': 0,
    }

    def log_likelihood(mu: float, outcome: float) -> float:
        index = {1: 0, 0.5: 1, 0: 2}[outcome]
        p_minus = glicko.probabilities(mu, opp_rating.mu - opp_rating.rd, 1, 0.8, params)[index]
        p_plus = glicko.probabilities(mu, opp_rating.mu + opp_rating.rd, 1, 0.8, params)[index]
        return log(p_minus + p_plus)

    h = 1e-4
    print(f'{"outcome":<8} {"d1":>10} {"d1 numeric":>12} {"d2":>10} {"d2 numeric":>12}')
    for outcome in [1, 0.5, 0]:
        d1, d2 = glicko._d1_d2(rating, opp_rating, 1, 0.8, params, outcome)

        mu = rating.mu
        d1_numeric = (log_likelihood(mu + h, outcome) - log_likelihood(mu - h, outcome)) / (2 * h)
        d2_numeric = (log_likelihood(mu + h, outcome) - 2 * log_likelihood(mu, outcome) + log_likelihood(mu - h, outcome)) / h ** 2

        print(f'{outcome:<8} {d1:>10.5f} {d1_numeric:>12.5f} {d2:>10.5f} {d2_numeric:>12.5f}')


if __name__ == '__main__':
    call_example()
    print()
    check_derivatives()