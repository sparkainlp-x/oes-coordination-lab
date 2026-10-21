"""Render the results JSON as a Markdown table (used for results/RESULTS.md)."""

import json
import sys

NAMES = {"no_info": "(1) no regime info", "private_info": "(2) private signals",
         "shared_info": "(3) shared signal"}


def fmt(cell):
    m, ci = cell["mean"], cell["ci95"]
    return f"{m:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else f"{m:.3f}"


def table(data):
    lines = ["| condition | policy | coordination rate | welfare | miscoordination cost "
             "| regret | unilateral-deviation gain | equilibrium? |",
             "|---|---|---|---|---|---|---|---|"]
    for r in data["results"]:
        s, d = r["simulated"], r.get("deviation_check", {})
        lines.append(
            f"| {NAMES[r['condition']]} | `{r['policy']}` | {fmt(s['coordination_rate'])} "
            f"| {fmt(s['welfare'])} | {fmt(s['miscoordination_cost'])} | {fmt(s['regret'])} "
            f"| {d.get('max_deviation_gain', float('nan')):.3f} "
            f"| {'yes' if d.get('is_equilibrium') else 'no'} |")
    return "\n".join(lines)


def main():
    data = json.load(open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin)
    print(table(data))


if __name__ == "__main__":
    main()
