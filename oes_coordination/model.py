"""SYNTHETIC toy model: local disturbances versus system-wide (shared) shocks.

Everything here is hand-written and illustrative. There are no learning
agents, no real infrastructure data and no operational claims.

Each round one hidden regime is drawn:

* ``local``  - every agent faces its own unrelated, independent problem;
  the best response is an individual ``local_fix``.
* ``shared`` - one common shock hits the whole system; the best outcome
  needs *every* agent to switch to ``system_contingency`` together.

Payoffs (per agent, explicit, see ``PAYOFFS``):

=========================  ======================  =====================
action of agent i          regime ``local``        regime ``shared``
=========================  ======================  =====================
``local_fix``              3                       1
``system_contingency``     1 if all agents do it,  4 if all agents do it,
                           else -2 (stranded)      else -2 (stranded)
``hold``                   0                       0
=========================  ======================  =====================

Under ``local`` the unique pure equilibrium is all ``local_fix``. Under
``shared`` the ex-post game is a coordination game with two pure equilibria:
all ``system_contingency`` (payoff 4, efficient) and all ``local_fix``
(payoff 1, inefficient but safe).
"""

from fractions import Fraction
from itertools import product

ACTIONS = ("local_fix", "system_contingency", "hold")
REGIMES = ("local", "shared")
CONDITIONS = ("no_info", "private_info", "shared_info")

PAYOFFS = {
    "local": {"local_fix": 3, "hold": 0, "all_system": 1, "stranded": -2},
    "shared": {"local_fix": 1, "hold": 0, "all_system": 4, "stranded": -2},
}

EFFICIENT_ACTION = {"local": "local_fix", "shared": "system_contingency"}


def utilities(regime, profile):
    """Return the tuple of per-agent payoffs for an action profile."""
    if regime not in PAYOFFS:
        raise ValueError(f"unknown regime {regime!r}")
    table = PAYOFFS[regime]
    profile = tuple(profile)
    if not profile:
        raise ValueError("profile must contain at least one action")
    for action in profile:
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")
    all_system = all(a == "system_contingency" for a in profile)
    out = []
    for action in profile:
        if action == "system_contingency":
            out.append(table["all_system"] if all_system else table["stranded"])
        else:
            out.append(table[action])
    return tuple(out)


def efficient_welfare(regime, n_agents):
    """Mean per-agent payoff of the ex-post efficient profile."""
    return utilities(regime, (EFFICIENT_ACTION[regime],) * n_agents)[0]


def flip(regime):
    return "shared" if regime == "local" else "local"


def observations(condition, regime, private_correct, public_correct):
    """Return each agent's observation (``None`` when there is no signal).

    ``private_correct[i]`` / ``public_correct`` say whether the noisy regime
    signal is correct. ``no_info`` agents observe nothing, ``private_info``
    agents each see an independent noisy signal, ``shared_info`` agents all
    see the same (common, public) noisy signal.
    """
    n = len(private_correct)
    if condition == "no_info":
        return (None,) * n
    if condition == "private_info":
        return tuple(regime if ok else flip(regime) for ok in private_correct)
    if condition == "shared_info":
        sig = regime if public_correct else flip(regime)
        return (sig,) * n
    raise ValueError(f"unknown condition {condition!r}")


def strategies(condition):
    """All pure strategies: maps from observation to action, as tuples.

    ``no_info``: ``(action,)``; otherwise ``(action_if_local, action_if_shared)``.
    """
    if condition == "no_info":
        return [(a,) for a in ACTIONS]
    if condition in ("private_info", "shared_info"):
        return list(product(ACTIONS, repeat=2))
    raise ValueError(f"unknown condition {condition!r}")


def act(strategy, observation):
    if observation is None:
        return strategy[0]
    return strategy[0] if observation == "local" else strategy[1]


def _states(condition, n, p_shared, accuracy):
    """Yield exact (Fraction probability, regime, private_correct, public_correct)."""
    p_shared = Fraction(str(p_shared))
    accuracy = Fraction(str(accuracy))
    for regime in REGIMES:
        pr = p_shared if regime == "shared" else 1 - p_shared
        if pr == 0:
            continue
        if condition == "private_info":
            for bits in product((True, False), repeat=n):
                p = pr
                for b in bits:
                    p *= accuracy if b else 1 - accuracy
                if p:
                    yield p, regime, bits, True
        elif condition == "shared_info":
            for ok in (True, False):
                p = pr * (accuracy if ok else 1 - accuracy)
                if p:
                    yield p, regime, (True,) * n, ok
        else:
            yield pr, regime, (True,) * n, True


def expected_utilities(condition, strategy_profile, p_shared, accuracy):
    """Exact expected per-agent payoffs (as ``Fraction``) of a strategy profile."""
    n = len(strategy_profile)
    totals = [Fraction(0)] * n
    for p, regime, priv, pub in _states(condition, n, p_shared, accuracy):
        obs = observations(condition, regime, priv, pub)
        profile = tuple(act(s, o) for s, o in zip(strategy_profile, obs))
        for i, u in enumerate(utilities(regime, profile)):
            totals[i] += p * u
    return tuple(totals)


def exact_metrics(condition, strategy_profile, p_shared, accuracy):
    """Exact expected metrics (same definitions as the simulation)."""
    n = len(strategy_profile)
    m = dict.fromkeys(("coordination_rate", "miscoordination_rate", "welfare",
                       "miscoordination_cost", "regret"), Fraction(0))
    for p, regime, priv, pub in _states(condition, n, p_shared, accuracy):
        obs = observations(condition, regime, priv, pub)
        r = _round_metrics(regime, tuple(act(s, o) for s, o in zip(strategy_profile, obs)))
        for k in m:
            m[k] += p * r[k]
    return {k: float(v) for k, v in m.items()}


def round_metrics(regime, profile):
    """Metrics for one realised round (per-agent averages), as floats."""
    return {k: float(v) for k, v in _round_metrics(regime, profile).items()}


def _round_metrics(regime, profile):
    n = len(profile)
    welfare = Fraction(sum(utilities(regime, profile)), n)
    best = efficient_welfare(regime, n)
    n_system = sum(a == "system_contingency" for a in profile)
    mis = 0 < n_system < n
    regret = best - welfare
    return {
        "coordination_rate": Fraction(int(all(a == EFFICIENT_ACTION[regime] for a in profile))),
        "miscoordination_rate": Fraction(int(mis)),
        "welfare": welfare,
        "miscoordination_cost": regret if mis else Fraction(0),
        "regret": regret,
    }
