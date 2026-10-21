"""Run from the repository root: python -m oes_coordination --help."""

import argparse
import json

from .evaluation import Experiment, run_experiment
from .model import Costs, Simulation
from .toy import toy_examples


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--calibration-episodes", type=int, default=200)
    parser.add_argument("--evaluation-episodes", type=int, default=1000)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--stable-prior", type=float, default=0.02)
    parser.add_argument("--localized-prior", type=float, default=0.3)
    parser.add_argument("--shared-prior", type=float, default=0.3)
    parser.add_argument("--missed-cost", type=float, default=5.0)
    parser.add_argument("--false-isolation-cost", type=float, default=1.0)
    parser.add_argument("--communication-cost", type=float, default=0.1)
    args = parser.parse_args(argv)
    try:
        config = Experiment(
            Simulation(args.noise, args.stable_prior, args.localized_prior, args.shared_prior),
            Costs(args.missed_cost, args.false_isolation_cost, args.communication_cost),
            tuple(args.seeds), args.calibration_episodes, args.evaluation_episodes,
        )
    except ValueError as exc:
        parser.error(str(exc))
    result = run_experiment(config)
    result["toy_games"] = toy_examples()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
