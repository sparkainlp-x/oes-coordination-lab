import contextlib
import io
import json
import random
import unittest

from oes_coordination.__main__ import main
from oes_coordination.evaluation import (
    Experiment, episodes, mean_utilities, restricted_gap, run_experiment, select_profile,
    standard_error,
)
from oes_coordination.model import (
    Costs, Episode, GroundTruth, Observation, Simulation, policy_inputs, utilities,
)
from oes_coordination.policies import act


DATA = (
    Episode(GroundTruth((1, 0, 1)), Observation((1, 0, 1))),
    Episode(GroundTruth((0, 1, 0)), Observation((0, 1, 0))),
)


class EvaluationTests(unittest.TestCase):
    def test_exact_restricted_gap(self):
        result = restricted_gap(DATA, ("continue",) * 3, False, Costs(4, 2, 0.5))
        self.assertEqual(result["name"], "restricted empirical deviation gap")
        self.assertEqual(result["maximum_improvement"], 2)
        for agent in result["agents"]:
            self.assertEqual(agent["baseline_utility"], -2)
            self.assertEqual(agent["best_tested_deviation"], "local")
            self.assertEqual(agent["best_tested_deviation_utility"], 0)
            self.assertEqual(agent["improvement"], 2)
            self.assertEqual(agent["tested_deviation_utilities"],
                             {"continue": -2, "isolate": -1, "local": 0})

    def test_identity_deviation_zero_gain(self):
        for sharing in (False, True):
            profile = ("local", "continue", "isolate")
            result = restricted_gap(DATA, profile, sharing, Costs())
            for i, agent in enumerate(result["agents"]):
                self.assertEqual(agent["tested_deviation_utilities"][profile[i]]
                                 - agent["baseline_utility"], 0)
                self.assertGreaterEqual(agent["improvement"], 0)
        self.assertEqual(restricted_gap(DATA, ("local",) * 3, False, Costs())
                         ["maximum_improvement"], 0)

    def test_deviations_hold_opponents_and_episodes_fixed(self):
        costs = Costs(4, 2, 0.5)
        profile = ("continue", "majority_report", "isolate")
        result = restricted_gap(DATA, profile, True, costs)
        for i, agent in enumerate(result["agents"]):
            for candidate, score in agent["tested_deviation_utilities"].items():
                changed = list(profile)
                changed[i] = candidate
                expected = []
                for episode in DATA:
                    actions = tuple(act(p, x) for p, x in
                                    zip(changed, policy_inputs(episode.observation, True)))
                    expected.append(utilities(episode.truth, actions, True, costs)[i])
                self.assertEqual(score, sum(expected) / len(expected))

    def test_calibration_selects_without_evaluation_data(self):
        self.assertEqual(select_profile(DATA, False, Costs()), ("local",) * 3)
        config = Experiment(seeds=(4, 9), calibration_episodes=3, evaluation_episodes=4)
        result = run_experiment(config)
        changed = run_experiment(Experiment(seeds=(4, 9), calibration_episodes=3,
                                            evaluation_episodes=9))
        self.assertEqual(result["selected_profiles"], changed["selected_profiles"])
        self.assertEqual(result["sample_counts"]["calibration_total"], 18)
        self.assertEqual(result["sample_counts"]["evaluation_total"], 24)
        for row in result["results"]:
            for agent in range(3):
                shared = row["modes"]["sharing"]["agents"][agent]["baseline_utility"]
                local = row["modes"]["no_sharing"]["agents"][agent]["baseline_utility"]
                self.assertEqual(
                    row["paired_utility_difference_sharing_minus_no_sharing"][agent],
                    shared - local,
                )

    def test_determinism_and_rng_isolation(self):
        config = Experiment(seeds=(12, 13), calibration_episodes=4, evaluation_episodes=5)
        state = random.getstate()
        first = run_experiment(config)
        self.assertEqual(first, run_experiment(config))
        self.assertEqual(state, random.getstate())
        self.assertEqual(episodes(config, 12, "evaluation", "shared"),
                         episodes(config, 12, "evaluation", "shared"))
        # Different named streams are reproducible and do not share RNG state.
        larger = Experiment(seeds=(12, 13), calibration_episodes=20, evaluation_episodes=5)
        self.assertEqual(episodes(config, 12, "evaluation", "shared"),
                         episodes(larger, 12, "evaluation", "shared"))

    def test_paired_communication_difference_when_actions_identical(self):
        config = Experiment(Simulation(0, 0, 0, 0), Costs(4, 2, 0.25),
                            (1, 2), 2, 3)
        result = run_experiment(config)
        self.assertEqual(result["selected_profiles"]["sharing"], ("continue",) * 3)
        self.assertEqual(result["paired_summary"]["agent_mean_difference"], [-0.25] * 3)
        self.assertEqual(result["paired_summary"]["agent_standard_error_over_seed_means"],
                         [0.0] * 3)

    def test_single_seed_uncertainty_is_null(self):
        result = run_experiment(Experiment(seeds=(0,), calibration_episodes=1,
                                           evaluation_episodes=1))
        self.assertEqual(result["paired_summary"]["agent_standard_error_over_seed_means"],
                         [None] * 3)

    def test_invalid_configuration(self):
        for kwargs in (
            {"seeds": ()}, {"seeds": (0, 0)}, {"seeds": (True,)}, {"seeds": [0]},
            {"calibration_episodes": 0}, {"evaluation_episodes": -1},
            {"evaluation_episodes": 1.5}, {"evaluation_episodes": True},
            {"simulation": None}, {"costs": None},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Experiment(**kwargs)
        with self.assertRaises(ValueError):
            episodes(Experiment(), 0, "unknown", "shared")
        for data, profile in (((), ("local",) * 3), (DATA, ("local",)),
                              (DATA, ("any_report",) * 3)):
            with self.assertRaises(ValueError):
                mean_utilities(data, profile, False, Costs())

    def test_large_finite_costs_produce_finite_json(self):
        self.assertEqual(mean_utilities(DATA * 3, ("continue",) * 3,
                                        False, Costs(1e308, 1e308, 0)), (-5e307,) * 3)
        result = run_experiment(Experiment(Simulation(1, 1, 1, 1), Costs(1e308, 1e308, 0),
                                           seeds=(1, 2), calibration_episodes=2,
                                           evaluation_episodes=3))
        json.dumps(result, allow_nan=False)

    def test_standard_error_avoids_intermediate_overflow(self):
        self.assertEqual(standard_error((-1.7e308, 1.7e308)), 1.7e308)
        self.assertEqual(standard_error((0.0, 0.0)), 0)
        self.assertEqual(standard_error((1.0, 3.0)), 1)
        self.assertIsNone(standard_error((1.0,)))
        result = run_experiment(Experiment(
            Simulation(0.4, 0.5, 0.5, 0.5), Costs(1.7e308, 1.7e308, 0),
            (43, 44), 1, 1,
        ))
        json.dumps(result, allow_nan=False)

    def test_cli_json_and_validation(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["--seeds", "3", "7", "--calibration-episodes", "2",
                  "--evaluation-episodes", "3"])
        result = json.loads(output.getvalue())
        self.assertEqual(result["configuration"]["seeds"], [3, 7])
        self.assertEqual(result["sample_counts"]["evaluation_total"], 18)
        self.assertFalse(result["toy_games"]["matching_pennies"]["has_pure_equilibrium"])
        for args in (["--noise", "nan"], ["--missed-cost", "-1"],
                     ["--evaluation-episodes", "0"], ["--seeds", "1", "1"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                main(args)
            self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
