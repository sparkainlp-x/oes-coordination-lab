# oes-coordination-lab

Software DOI for v0.1.0: [10.5281/zenodo.23530289](https://doi.org/10.5281/zenodo.23530289) (all versions: [10.5281/zenodo.23530288](https://doi.org/10.5281/zenodo.23530288)).

**SYNTHETIC toy benchmark.** A small, fully explicit model of local disturbances versus system-wide shocks. It asks one question: *if agents are told more clearly whether a disruption is local or shared, do they coordinate better?* The agents are **hand-coded policies, not learning agents**. The payoffs are invented for illustration. Nothing here describes a real system.

## Question

Each round, every agent faces a disruption. It is either

- **local**: each agent has its own unrelated problem, and the right response is an individual `local_fix`; or
- **shared**: one system-wide shock, and the right response is for *every* agent to switch to `system_contingency` **together**. Switching alone strands the agent.

The experiment compares three information conditions:

1. **no regime information**: agents only know the prior;
2. **private information**: each agent gets its *own independent* noisy signal of the regime;
3. **shared information**: all agents see *one shared* noisy signal, with the same accuracy.

Then it asks whether better observability improves coordination, or whether agents still fail even when the structure is explicit.

## Design

Payoffs per agent (`oes_coordination/model.py`, `PAYOFFS`):

| action of agent *i* | regime `local` | regime `shared` |
|---|---|---|
| `local_fix` | 3 | 1 |
| `system_contingency` | 1 if **all** agents choose it, else −2 (stranded) | 4 if **all** agents choose it, else −2 (stranded) |
| `hold` | 0 | 0 |

- In the `local` round game, all-`local_fix` is the only pure equilibrium. In the `shared` round game there are two: all-`system_contingency` (4, efficient) and all-`local_fix` (1, safe but inefficient). That second one is the coordination trap.
- Defaults: 3 agents, P(shared) = 0.5, signal accuracy q = 0.8 (the signal shows the true regime with probability q and the other regime otherwise).
- **Common random numbers:** every condition is evaluated on the *same* seeded regime draws and the *same* signal-noise draws (`draw_rounds`). The only difference between conditions is who sees which signal.
- **Baseline policies** (one strategy used by every agent):
  - no info: `always_system` and `always_local` (the only sensible fixed actions);
  - private info and shared info: `follow_signal` (local signal → `local_fix`, shared signal → `system_contingency`).
- **Metrics:** each is a per-agent average per round, and the efficient profile is the regime's best common action.
  - *coordination rate*: share of rounds in which everyone chose the efficient action;
  - *welfare*: mean payoff;
  - *miscoordination cost*: welfare lost in rounds where some, but not all, agents chose `system_contingency`;
  - *regret*: efficient welfare minus realised welfare.
- **Uncertainty:** 200 seeds × 1,000 rounds. The table shows the mean over seeds with a normal-approximation 95% CI, treating the seed as the unit of replication. The exact expectations are computed by enumerating every regime and signal state with `fractions.Fraction`, and they are also stored in `results/results.json`. The tests check that each simulated CI agrees with its exact value.
- **Unilateral-deviation check:** for each condition, the policy problem becomes an ex-ante finite game. Its "actions" are the full maps from observation to action: 3 per agent with no info, 9 with a signal. Expected payoffs are exact. Each baseline profile is audited with [`finite-game-equilibria`](https://github.com/sparkainlp-x/finite-game-equilibria)'s `analyze_profile` and `find_pure_equilibria`, pinned to commit `f92deb3`. Utilities are scaled to integers by a common denominator, so every comparison is exact. The check covers pure strategies within this toy game only.

## Results (defaults: 3 agents, P(shared) = 0.5, q = 0.8, 200 seeds × 1,000 rounds)

| condition | policy | coordination rate | welfare | miscoordination cost | regret | unilateral-deviation gain | equilibrium? |
|---|---|---|---|---|---|---|---|
| (1) no regime info | `always_system` | 0.499 [0.496, 0.501] | 2.496 [2.489, 2.502] | 0.000 [0.000, 0.000] | 1.003 [0.998, 1.007] | 0.000 | yes |
| (1) no regime info | `always_local` | 0.501 [0.499, 0.504] | 2.003 [1.998, 2.007] | 0.000 [0.000, 0.000] | 1.496 [1.489, 1.502] | 0.000 | yes |
| (2) private signals | `follow_signal` | 0.512 [0.510, 0.514] | 1.845 [1.837, 1.853] | 1.633 [1.624, 1.642] | 1.654 [1.645, 1.663] | 0.152 | no |
| (3) shared signal | `follow_signal` | 0.799 [0.797, 0.801] | 2.997 [2.992, 3.001] | 0.000 [0.000, 0.000] | 0.502 [0.497, 0.506] | 0.000 | yes |

Exact expectations: welfare 2.5 / 2.0 / 1.848 / 3.0, and coordination rate 0.5 / 0.5 / 0.512 / 0.8, in table order. The efficient ceiling (perfect information) is 3.5.

**Takeaway (honest, toy-model only):** information helped coordination only when it was *shared*. A shared signal raised welfare from 2.50 to 3.00. Private signals of the same accuracy made things *worse* than having no information at all: welfare fell to 1.85, because agents acting on their own signals split their choices in 48% of rounds. Following those private signals is not even an equilibrium: an agent gains 0.152 by ignoring its signal and always playing `local_fix`. The only pure equilibrium of that strategy game is the safe all-`local_fix` trap, at welfare 2.0.

Exact sweep (`exact_sweep` in `results/results.json`, P(shared) = 0.5). Welfare of `follow_signal`:

| agents | q | private signals | shared signal | best no-info baseline |
|---|---|---|---|---|
| 2 | 0.8 | 2.280 | 3.000 | 2.5 |
| 3 | 0.8 | 1.848 | 3.000 | 2.5 |
| 5 | 0.8 | 1.284 | 3.000 | 2.5 |
| 3 | 0.6 | 0.844 | 2.500 | 2.5 |
| 3 | 0.95 | 3.022 | 3.375 | 2.5 |
| 3 | 1.0 | 3.500 | 3.500 | 2.5 |

Private information gets worse as the group grows, because all *n* independent signals must be right at once. It beats the no-information baseline only when signals are very accurate. A shared signal never does worse than the best no-information baseline in this sweep. With perfect signals, private and shared information are identical.

## Limitations

- Toy model with one hand-chosen payoff table and hand-coded policies. There is no learning, no communication protocol, no reporting incentives and no real data. Different payoffs can change the ranking.
- Signals are symmetric and binary. Private noise is independent across agents, and the shared signal is a single draw seen by every agent.
- The deviation check covers pure observation-to-action strategies in this finite toy game. It says nothing about mixed strategies, learning dynamics, or any real multi-agent or AI system.
- The CIs use a normal approximation across seeds. The exact values are the reference.

## Reproduce

```sh
python -m pip install .            # installs the pinned finite-game-equilibria dependency
python -m pip install -r requirements-dev.txt
python -W error -m pytest -q
python -m oes_coordination --sweep > results.json   # byte-identical to results/results.json
python -m oes_coordination.report results.json      # the table above
```

CI (Python 3.10–3.13) runs the tests with warnings treated as errors. It also regenerates `results/results.json` and `results/table.md` and diffs them against the committed files. Options: `--seeds`, `--first-seed`, `--rounds`, `--agents`, `--p-shared`, `--accuracy`, `--no-deviation-check`.

## Relation to other work

- [finite-game-equilibria](https://github.com/sparkainlp-x/finite-game-equilibria) is the general, standalone checker for pure-strategy (epsilon-)equilibria of explicit finite games. This lab builds its own strategy games and uses that package for every deviation audit.
- In the OES repositories (for example [oes-resilience](https://github.com/sparkainlp-x/oes-resilience) and [oes32-residual](https://github.com/sparkainlp-x/oes32-residual)), detection is the question: is a disturbance there? This lab asks the next question in the simplest form: once a disturbance is known to be local or shared, can several agents respond together? It does not use OES-32 residuals or telemetry.

## Acknowledgement

The three-condition comparison (no regime information, local/private information, shared information) was suggested in public discussion by Alexandra Bernadotte. Thank you. This acknowledgement does not imply any endorsement of this repository or its results.

## Citation and licence

See `CITATION.cff`. Author: Jean-François Brisson (ORCID [0009-0000-9778-5374](https://orcid.org/0009-0000-9778-5374)). Licensed under AGPL-3.0-only (`LICENSE`).
