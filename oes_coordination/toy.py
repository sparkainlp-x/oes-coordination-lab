"""Exhaustive best responses in explicitly finite, separate normal-form games."""

from itertools import product
import math


def analyze_game(
    actions: tuple[tuple[str, ...], ...],
    payoffs: dict[tuple[str, ...], tuple[float, ...]],
) -> dict:
    if not actions or any(
        not choices or len(set(choices)) != len(choices)
        or any(not isinstance(choice, str) for choice in choices)
        for choices in actions
    ):
        raise ValueError("each player needs distinct named actions")
    profiles = tuple(product(*actions))
    if set(payoffs) != set(profiles):
        raise ValueError("payoff table must cover exactly the Cartesian product")
    for values in payoffs.values():
        if len(values) != len(actions) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) for value in values
        ):
            raise ValueError("each payoff must contain one finite number per player")
    checks, equilibria = [], []
    for profile in profiles:
        players = []
        for player, choices in enumerate(actions):
            deviations = {}
            for choice in choices:
                changed = list(profile)
                changed[player] = choice
                deviations[choice] = payoffs[tuple(changed)][player]
            best = max(deviations.values())
            players.append({
                "player": player, "utility": payoffs[profile][player],
                "deviation_utilities": deviations,
                "best_responses": [choice for choice, value in deviations.items() if value == best],
                "improvement": best - payoffs[profile][player],
            })
        equilibrium = all(player["improvement"] == 0 for player in players)
        if equilibrium:
            equilibria.append(profile)
        checks.append({"profile": profile, "players": players, "pure_nash": equilibrium})
    return {
        "scope": "Pure Nash results apply only to this fully enumerated finite toy game.",
        "actions": actions,
        "payoff_table": [{"profile": p, "utilities": payoffs[p]} for p in profiles],
        "checks": checks, "pure_equilibria": equilibria,
        "has_pure_equilibrium": bool(equilibria),
    }


def toy_examples() -> dict:
    actions = (("A", "B"), ("A", "B"))
    return {
        "coordination": analyze_game(actions, {
            ("A", "A"): (4, 4), ("A", "B"): (0, 3),
            ("B", "A"): (3, 0), ("B", "B"): (2, 2),
        }),
        "matching_pennies": analyze_game(actions, {
            ("A", "A"): (1, -1), ("A", "B"): (-1, 1),
            ("B", "A"): (-1, 1), ("B", "B"): (1, -1),
        }),
    }
