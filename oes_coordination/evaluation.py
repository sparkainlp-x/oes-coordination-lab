"""Calibration and held-out paired evaluation, not an equilibrium certificate."""

from dataclasses import asdict, dataclass, field
from itertools import chain
import math
import random
import statistics

from .model import AGENTS, SCENARIOS, Costs, Episode, Simulation
from .model import generate_episode, policy_inputs, utilities
from .policies import LOCAL_POLICIES, SHARED_POLICIES, act


@dataclass(frozen=True)
class Experiment:
    simulation: Simulation = field(default_factory=Simulation)
    costs: Costs = field(default_factory=Costs)
    seeds: tuple[int, ...] = (0, 1, 2)
    calibration_episodes: int = 200
    evaluation_episodes: int = 1000

    def __post_init__(self) -> None:
        if not isinstance(self.simulation, Simulation) or not isinstance(self.costs, Costs):
            raise ValueError("simulation and costs must use their configuration types")
        if not isinstance(self.seeds, tuple) or not self.seeds:
            raise ValueError("seeds must be a nonempty tuple")
        if any(type(seed) is not int for seed in self.seeds):
            raise ValueError("seeds must be integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        for count in (self.calibration_episodes, self.evaluation_episodes):
            if type(count) is not int or count < 1:
                raise ValueError("episode counts must be positive integers")


def episodes(config: Experiment, seed: int, phase: str, scenario: str) -> tuple[Episode, ...]:
    if phase not in ("calibration", "evaluation"):
        raise ValueError("unknown phase")
    count = getattr(config, f"{phase}_episodes")
    rng = random.Random(f"oes-v1:{seed}:{phase}:{scenario}")
    return tuple(generate_episode(rng, scenario, config.simulation) for _ in range(count))


def mean_utilities(
    data: tuple[Episode, ...], profile: tuple[str, ...], sharing: bool, costs: Costs
) -> tuple[float, ...]:
    allowed = SHARED_POLICIES if sharing else LOCAL_POLICIES
    if len(profile) != AGENTS or any(policy not in allowed for policy in profile):
        raise ValueError("profile must contain three policies available in this mode")
    if not data:
        raise ValueError("evaluation episodes cannot be empty")
    totals = [[] for _ in range(AGENTS)]
    for episode in data:
        inputs = policy_inputs(episode.observation, sharing)
        actions = tuple(act(policy, visible) for policy, visible in zip(profile, inputs))
        payoff = utilities(episode.truth, actions, sharing, costs)
        for agent in range(AGENTS):
            totals[agent].append(payoff[agent])
    # statistics.mean uses exact ratio summation, avoiding intermediate overflow.
    return tuple(statistics.mean(values) for values in totals)


def restricted_gap(
    data: tuple[Episode, ...], profile: tuple[str, ...], sharing: bool, costs: Costs
) -> dict:
    baseline = mean_utilities(data, profile, sharing, costs)
    policies = SHARED_POLICIES if sharing else LOCAL_POLICIES
    agents = []
    for agent in range(AGENTS):
        tested = {}
        for policy in policies:
            deviation = list(profile)
            deviation[agent] = policy
            tested[policy] = mean_utilities(data, tuple(deviation), sharing, costs)[agent]
        best = max(tested, key=tested.get)
        improvement = max(0.0, tested[best] - baseline[agent])
        agents.append({
            "agent": agent,
            "baseline_utility": baseline[agent],
            "best_tested_deviation": best,
            "best_tested_deviation_utility": tested[best],
            "improvement": improvement,
            "tested_deviation_utilities": tested,
        })
    return {
        "name": "restricted empirical deviation gap",
        "sample_count": len(data),
        "profile": profile,
        "policy_class": policies,
        "agents": agents,
        "maximum_improvement": max(agent["improvement"] for agent in agents),
    }


def select_profile(data: tuple[Episode, ...], sharing: bool, costs: Costs) -> tuple[str, ...]:
    """Per-agent calibration argmax; first declared policy wins exact ties."""
    policies = SHARED_POLICIES if sharing else LOCAL_POLICIES
    scores = {
        policy: mean_utilities(data, (policy,) * AGENTS, sharing, costs)
        for policy in policies
    }
    return tuple(max(policies, key=lambda policy: scores[policy][i]) for i in range(AGENTS))


def standard_error(values: tuple[float, ...]) -> float | None:
    if len(values) < 2:
        return None
    scale = max(abs(value) for value in values)
    if scale == 0:
        return 0.0
    # Normalize first: the standard deviation can overflow even when its SE cannot.
    normalized = tuple(value / scale for value in values)
    return (statistics.stdev(normalized) / math.sqrt(len(values))) * scale


def run_experiment(config: Experiment) -> dict:
    calibration = tuple(chain.from_iterable(
        episodes(config, seed, "calibration", scenario)
        for seed in config.seeds for scenario in SCENARIOS
    ))
    profiles = {
        "no_sharing": select_profile(calibration, False, config.costs),
        "sharing": select_profile(calibration, True, config.costs),
    }
    results = []
    differences = []
    for seed in config.seeds:
        seed_differences = []
        for scenario in SCENARIOS:
            data = episodes(config, seed, "evaluation", scenario)
            modes = {
                mode: restricted_gap(data, profile, mode == "sharing", config.costs)
                for mode, profile in profiles.items()
            }
            paired = [
                modes["sharing"]["agents"][i]["baseline_utility"]
                - modes["no_sharing"]["agents"][i]["baseline_utility"]
                for i in range(AGENTS)
            ]
            seed_differences.append(paired)
            results.append({
                "seed": seed, "scenario": scenario, "modes": modes,
                "paired_utility_difference_sharing_minus_no_sharing": paired,
            })
        differences.append([
            statistics.mean(row[i] for row in seed_differences) for i in range(AGENTS)
        ])
    return {
        "schema_version": 1,
        "interpretation": "Synthetic finite-sample comparisons, not proofs or unrestricted Nash certificates.",
        "configuration": asdict(config),
        "reporting_protocol": "Mandatory truthful simultaneous broadcast of each observed bit; each sender pays communication once per episode.",
        "sample_counts": {
            "calibration_per_seed_per_scenario": config.calibration_episodes,
            "evaluation_per_seed_per_scenario": config.evaluation_episodes,
            "calibration_total": len(calibration),
            "evaluation_total": len(config.seeds) * len(SCENARIOS) * config.evaluation_episodes,
        },
        "selected_profiles": profiles,
        "results": results,
        "paired_summary": {
            "seed_count": len(config.seeds),
            "per_seed_agent_mean_differences": differences,
            "agent_mean_difference": [
                statistics.mean(row[i] for row in differences) for i in range(AGENTS)
            ],
            "agent_standard_error_over_seed_means": [
                standard_error(tuple(row[i] for row in differences)) for i in range(AGENTS)
            ],
            "caveat": "Descriptive seed variability conditional on the pooled calibrated profiles; not a significance test.",
        },
    }
