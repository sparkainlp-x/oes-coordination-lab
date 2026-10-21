"""CLI: run the SYNTHETIC three-information-condition experiment, print JSON."""

import argparse
import json
import sys

from .experiment import run, sweep


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="oes-coordination-lab",
        description="SYNTHETIC toy experiment: no / private / shared regime information.")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=1000)
    p.add_argument("--agents", type=int, default=3)
    p.add_argument("--p-shared", type=float, default=0.5)
    p.add_argument("--accuracy", type=float, default=0.8)
    p.add_argument("--no-deviation-check", action="store_true")
    p.add_argument("--sweep", action="store_true",
                   help="also include the exact group-size x accuracy sweep")
    a = p.parse_args(argv)
    try:
        out = run(seeds=a.seeds, rounds=a.rounds, n_agents=a.agents, p_shared=a.p_shared,
                  accuracy=a.accuracy, check_deviations=not a.no_deviation_check,
                  first_seed=a.first_seed)
        if a.sweep:
            out["exact_sweep"] = sweep(p_shared=a.p_shared)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
