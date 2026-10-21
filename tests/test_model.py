from dataclasses import fields
import itertools
import random
import unittest

from oes_coordination.model import (
    Action, Costs, Episode, GroundTruth, Observation, PolicyInput, Simulation,
    generate_episode, policy_inputs, report, utilities,
)
from oes_coordination.policies import LOCAL_POLICIES, SHARED_POLICIES, act


class ModelTests(unittest.TestCase):
    def test_payoff_table_exhaustively(self):
        costs = Costs(7, 3, 0.5)
        for hidden, actions, sharing in itertools.product(
            itertools.product((0, 1), repeat=3),
            itertools.product(tuple(Action), repeat=3),
            (False, True),
        ):
            expected = tuple(
                {(0, Action.CONTINUE): 0, (0, Action.ISOLATE): -3,
                 (1, Action.CONTINUE): -7, (1, Action.ISOLATE): 0}[bit, action]
                - (0.5 if sharing else 0)
                for bit, action in zip(hidden, actions)
            )
            self.assertEqual(utilities(GroundTruth(hidden), actions, sharing, costs), expected)

    def test_communication_is_paid_even_when_reports_are_ignored(self):
        hidden = GroundTruth((0, 1, 0))
        actions = (Action.CONTINUE,) * 3
        off = utilities(hidden, actions, False, Costs(4, 2, 0.25))
        on = utilities(hidden, actions, True, Costs(4, 2, 0.25))
        self.assertEqual(tuple(a - b for a, b in zip(on, off)), (-0.25,) * 3)

    def test_information_boundary(self):
        observation = Observation((1, 0, 1))
        a = Episode(GroundTruth((0, 0, 0)), observation)
        b = Episode(GroundTruth((1, 1, 1)), observation)
        for sharing in (False, True):
            self.assertEqual(policy_inputs(a.observation, sharing),
                             policy_inputs(b.observation, sharing))
        self.assertEqual(report(observation), (1, 0, 1))
        self.assertEqual([f.name for f in fields(PolicyInput)], ["local", "reports"])
        self.assertEqual(policy_inputs(observation, False),
                         (PolicyInput(1), PolicyInput(0), PolicyInput(1)))
        self.assertTrue(all(value.reports == (1, 0, 1)
                            for value in policy_inputs(observation, True)))
        # Other agents' observations cannot change a no-sharing action.
        for policy in LOCAL_POLICIES:
            self.assertEqual(act(policy, policy_inputs(observation, False)[0]),
                             act(policy, policy_inputs(Observation((1, 1, 0)), False)[0]))

    def test_truth_is_not_an_observation(self):
        config = Simulation(noise=1, stable_prior=0)
        episode = generate_episode(random.Random(0), "stable", config)
        self.assertEqual(episode.truth.disturbances, (0, 0, 0))
        self.assertEqual(episode.observation.indicators, (1, 1, 1))
        self.assertEqual(report(episode.observation), (1, 1, 1))

    def test_scenario_extremes_and_noise(self):
        for scenario in ("stable", "localized", "shared"):
            for prior in (0, 1):
                config = Simulation(0, prior, prior, prior)
                rng = random.Random(20)
                for _ in range(30):
                    episode = generate_episode(rng, scenario, config)
                    hidden = episode.truth.disturbances
                    self.assertEqual(hidden, episode.observation.indicators)
                    expected_count = prior if scenario == "localized" else 3 * prior
                    self.assertEqual(sum(hidden), expected_count)

    def test_all_policy_truth_tables(self):
        for reports in itertools.product((0, 1), repeat=3):
            for local in (0, 1):
                inputs = PolicyInput(local, reports)
                expected = (False, True, bool(local), any(reports), sum(reports) >= 2)
                for policy, isolate in zip(SHARED_POLICIES, expected):
                    self.assertEqual(act(policy, inputs),
                                     Action.ISOLATE if isolate else Action.CONTINUE)
        for policy in ("any_report", "majority_report", "unknown"):
            with self.assertRaises(ValueError):
                act(policy, PolicyInput(1))

    def test_invalid_probabilities_and_costs(self):
        with self.assertRaises(ValueError):
            Costs(1e308, 0, 1e308)
        for value in (-1, float("nan"), float("inf"), -float("inf"), True, "1"):
            for name in ("missed", "false_isolation", "communication"):
                with self.subTest(value=value, name=name), self.assertRaises(ValueError):
                    Costs(**{name: value})
        for value in (-0.1, 1.1, float("nan"), float("inf"), True):
            for name in ("noise", "stable_prior", "localized_prior", "shared_prior"):
                with self.subTest(value=value, name=name), self.assertRaises(ValueError):
                    Simulation(**{name: value})

    def test_invalid_data_and_actions(self):
        for value in ((0, 1), (0, 1, 2), (False, 1, 0), [0, 0, 0]):
            for constructor in (GroundTruth, Observation):
                with self.assertRaises(ValueError):
                    constructor(value)
            with self.assertRaises(ValueError):
                PolicyInput(0, value)
        with self.assertRaises(ValueError):
            PolicyInput(True)
        with self.assertRaises(ValueError):
            generate_episode(random.Random(0), "unknown", Simulation())
        with self.assertRaises(ValueError):
            utilities(GroundTruth((0, 0, 0)), ("continue",) * 3, False, Costs())
        with self.assertRaises(ValueError):
            utilities(GroundTruth((0, 0, 0)), (Action.CONTINUE,), False, Costs())
        with self.assertRaises(ValueError):
            policy_inputs(Observation((0, 0, 0)), 1)


if __name__ == "__main__":
    unittest.main()
