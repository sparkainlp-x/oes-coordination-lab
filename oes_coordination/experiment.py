"""SYNTHETIC three-information-condition experiment (toy model, hand-coded policies).

Conditions: (1) ``no_info`` - no regime information; (2) ``private_info`` -
each agent sees its own independent noisy regime signal; (3) ``shared_info`` -
all agents see one common (public) noisy signal of the same accuracy.

All conditions are evaluated on the *same* seeded disturbances and the same
signal-noise draws (common random numbers), so differences come from the
information structure and the policy, not from different luck.
"""

import math
from itertools import product
import random
import statistics

from finite_game_equilibria import analyze_profile, find_pure_equilibria

from .model import (CONDITIONS, act, exact_metrics, expected_utilities,
                    observations, round_metrics, strategies)

METRICS = ("coordination_rate", "miscoordination_rate", "welfare",
           "miscoordination_cost", "regret")

# Hand-coded baseline policies (one strategy used by every agent).
BASELINES = {
    "no_info": {
        "always_system": ("system_contingency",),
        "always_local": ("local_fix",),
    },
    "private_info": {
        "follow_signal": ("local_fix", "system_contingency"),
    },
    "shared_info": {
        "follow_signal": ("local_fix", "system_contingency"),
    },
}


def draw_rounds(seed, rounds, n_agents, p_shared, accuracy):
    """Seeded hidden regimes plus signal-correctness draws, shared by all conditions."""
    rng = random.Random(seed)
    out = []
    for _ in range(rounds):
        regime = "shared" if rng.random() < p_shared else "local"
        private = tuple(rng.random() < accuracy for _ in range(n_agents))
        public = rng.random() < accuracy
        out.append((regime, private, public))
    return out


def simulate(condition, strategy, draws, n_agents):
    sums = dict.fromkeys(METRICS, 0.0)
    for regime, private, public in draws:
        obs = observations(condition, regime, private, public)
        r = round_metrics(regime, tuple(act(strategy, o) for o in obs[:n_agents]))
        for k in METRICS:
            sums[k] += r[k]
    return {k: v / len(draws) for k, v in sums.items()}


def mean_ci(values, z=1.96, digits=6):
    """Mean over seeds with a normal-approximation 95% CI (seed = unit of replication).

    Rounded to ``digits`` so the committed results file is byte-stable across
    Python versions.
    """
    m = statistics.fmean(values)
    if len(values) < 2:
        return {"mean": round(m, digits), "ci95": None}
    half = z * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": round(m, digits), "ci95": [round(m - half, digits), round(m + half, digits)]}


def strategy_game(condition, n_agents, p_shared, accuracy):
    """Ex-ante normal-form game whose 'actions' are observation->action maps."""
    strats = strategies(condition)
    actions = [strats] * n_agents
    exact = {prof: expected_utilities(condition, prof, p_shared, accuracy)
             for prof in product(strats, repeat=n_agents)}
    # finite-game-equilibria takes int/float utilities; scale the exact
    # Fractions by a common denominator so every comparison stays exact.
    scale = 1
    for us in exact.values():
        for u in us:
            scale = math.lcm(scale, u.denominator)
    payoffs = {prof: tuple(int(u * scale) for u in us) for prof, us in exact.items()}
    return actions, payoffs, scale


def deviation_check(condition, strategy, n_agents, p_shared, accuracy):
    """Unilateral-deviation audit of the symmetric baseline profile.

    Uses ``finite_game_equilibria.analyze_profile`` on the exact ex-ante
    strategy game; also counts all pure equilibria of that game.
    """
    actions, payoffs, scale = strategy_game(condition, n_agents, p_shared, accuracy)
    audit = analyze_profile(actions, payoffs, (strategy,) * n_agents)
    eqs = find_pure_equilibria(actions, payoffs)
    best = max(eqs, key=lambda e: sum(payoffs[tuple(tuple(s) for s in e["profile"])]))
    p0 = audit["players"][0]
    gain = audit["max_deviation_gain"]
    return {
        "max_deviation_gain": round(gain / scale, 6),
        "is_equilibrium": gain <= 0,
        "best_unilateral_response": list(p0["best_response_action"]),
        "pure_equilibria_in_strategy_game": len(eqs),
        "best_equilibrium_profile": [list(s) for s in best["profile"]],
        "best_equilibrium_welfare": round(
            sum(payoffs[tuple(tuple(s) for s in best["profile"])]) / n_agents / scale, 6),
    }


def run(seeds=200, rounds=1000, n_agents=3, p_shared=0.5, accuracy=0.8,
        check_deviations=True, first_seed=0):
    if not (isinstance(seeds, int) and seeds >= 1 and isinstance(rounds, int) and rounds >= 1):
        raise ValueError("seeds and rounds must be positive integers")
    if not (isinstance(n_agents, int) and n_agents >= 1):
        raise ValueError("n_agents must be a positive integer")
    for name, v in (("p_shared", p_shared), ("accuracy", accuracy)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
            raise ValueError(f"{name} must be a number in [0, 1]")
    seed_list = list(range(first_seed, first_seed + seeds))
    all_draws = [draw_rounds(s, rounds, n_agents, p_shared, accuracy) for s in seed_list]
    rows = []
    for condition in CONDITIONS:
        for name, strategy in BASELINES[condition].items():
            per_seed = [simulate(condition, strategy, d, n_agents) for d in all_draws]
            row = {
                "condition": condition,
                "policy": name,
                "strategy": list(strategy),
                "simulated": {k: mean_ci([s[k] for s in per_seed]) for k in METRICS},
                "exact": {k: round(v, 6) for k, v in
                          exact_metrics(condition, (strategy,) * n_agents, p_shared, accuracy).items()},
            }
            if check_deviations:
                row["deviation_check"] = deviation_check(condition, strategy, n_agents,
                                                         p_shared, accuracy)
            rows.append(row)
    return {
        "label": "SYNTHETIC toy model; hand-coded policies, not learning agents",
        "parameters": {"seeds": seeds, "first_seed": first_seed, "rounds_per_seed": rounds,
                       "n_agents": n_agents, "p_shared": p_shared,
                       "signal_accuracy": accuracy},
        "results": rows,
    }


def sweep(n_values=(2, 3, 5), accuracies=(0.6, 0.7, 0.8, 0.9, 0.95, 1.0), p_shared=0.5):
    """Exact (no sampling) welfare of each baseline across group size and signal accuracy."""
    out = []
    for n in n_values:
        for q in accuracies:
            for condition in CONDITIONS:
                for name, strategy in BASELINES[condition].items():
                    m = exact_metrics(condition, (strategy,) * n, p_shared, q)
                    out.append({"n_agents": n, "signal_accuracy": q, "condition": condition,
                                "policy": name, **{k: round(v, 6) for k, v in m.items()}})
    return out
