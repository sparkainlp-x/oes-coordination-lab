"""Hidden simulation state, observable data, and synthetic utility."""

from dataclasses import dataclass
from enum import Enum
import math
import random


AGENTS = 3
SCENARIOS = ("stable", "localized", "shared")


def finite_nonnegative(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")


def probability(value: float, name: str) -> None:
    finite_nonnegative(value, name)
    if value > 1:
        raise ValueError(f"{name} must lie in [0, 1]")


def bits(values: tuple[int, ...], name: str) -> None:
    if not isinstance(values, tuple) or len(values) != AGENTS:
        raise ValueError(f"{name} must be a tuple of three binary integers")
    if any(type(value) is not int or value not in (0, 1) for value in values):
        raise ValueError(f"{name} must be a tuple of three binary integers")


@dataclass(frozen=True)
class Costs:
    missed: float = 5.0
    false_isolation: float = 1.0
    communication: float = 0.1

    def __post_init__(self) -> None:
        for name in ("missed", "false_isolation", "communication"):
            finite_nonnegative(getattr(self, name), name)
        if not math.isfinite(max(self.missed, self.false_isolation) + self.communication):
            raise ValueError("combined payoff magnitude must be finite")


@dataclass(frozen=True)
class Simulation:
    noise: float = 0.1
    stable_prior: float = 0.02
    localized_prior: float = 0.3
    shared_prior: float = 0.3

    def __post_init__(self) -> None:
        for name in ("noise", "stable_prior", "localized_prior", "shared_prior"):
            probability(getattr(self, name), name)


@dataclass(frozen=True)
class GroundTruth:
    disturbances: tuple[int, ...]

    def __post_init__(self) -> None:
        bits(self.disturbances, "disturbances")


@dataclass(frozen=True)
class Observation:
    indicators: tuple[int, ...]

    def __post_init__(self) -> None:
        bits(self.indicators, "indicators")


@dataclass(frozen=True)
class Episode:
    truth: GroundTruth
    observation: Observation


@dataclass(frozen=True)
class PolicyInput:
    local: int
    reports: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if type(self.local) is not int or self.local not in (0, 1):
            raise ValueError("local must be a binary integer")
        if self.reports is not None:
            bits(self.reports, "reports")


class Action(str, Enum):
    CONTINUE = "continue"
    ISOLATE = "isolate"


def generate_episode(
    rng: random.Random, scenario: str, config: Simulation
) -> Episode:
    if scenario not in SCENARIOS:
        raise ValueError("unknown scenario")
    if scenario == "stable":
        hidden = tuple(int(rng.random() < config.stable_prior) for _ in range(AGENTS))
    elif scenario == "localized":
        active = rng.random() < config.localized_prior
        location = rng.randrange(AGENTS)
        hidden = tuple(int(active and i == location) for i in range(AGENTS))
    else:
        active = int(rng.random() < config.shared_prior)
        hidden = (active,) * AGENTS
    observed = tuple(bit ^ int(rng.random() < config.noise) for bit in hidden)
    return Episode(GroundTruth(hidden), Observation(observed))


def report(observation: Observation) -> tuple[int, ...]:
    """Mandatory truthful broadcast: one observed bit per agent, no other data."""
    return observation.indicators


def policy_inputs(
    observation: Observation, sharing: bool
) -> tuple[PolicyInput, ...]:
    if type(sharing) is not bool:
        raise ValueError("sharing must be boolean")
    reports = report(observation) if sharing else None
    return tuple(PolicyInput(local, reports) for local in observation.indicators)


def utilities(
    truth: GroundTruth, actions: tuple[Action, ...], sharing: bool, costs: Costs
) -> tuple[float, ...]:
    if len(actions) != AGENTS or any(not isinstance(a, Action) for a in actions):
        raise ValueError("exactly three Action values required")
    if type(sharing) is not bool:
        raise ValueError("sharing must be boolean")
    return tuple(
        -(costs.missed if disturbed and action == Action.CONTINUE else 0.0)
        - (costs.false_isolation if not disturbed and action == Action.ISOLATE else 0.0)
        - (costs.communication if sharing else 0.0)
        for disturbed, action in zip(truth.disturbances, actions)
    )
