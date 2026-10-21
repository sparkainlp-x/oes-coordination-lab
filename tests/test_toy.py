import itertools
import unittest

from oes_coordination.toy import analyze_game, toy_examples


class ToyTests(unittest.TestCase):
    def test_exact_coordination_equilibria_and_checks(self):
        game = toy_examples()["coordination"]
        self.assertEqual(game["pure_equilibria"], [("A", "A"), ("B", "B")])
        self.assertTrue(game["has_pure_equilibrium"])
        self.assertEqual(len(game["checks"]), 4)
        table = {row["profile"]: row["utilities"] for row in game["payoff_table"]}
        for check in game["checks"]:
            for i, player in enumerate(check["players"]):
                expected = {}
                for choice in ("A", "B"):
                    changed = list(check["profile"])
                    changed[i] = choice
                    expected[choice] = table[tuple(changed)][i]
                self.assertEqual(player["deviation_utilities"], expected)
                self.assertEqual(player["improvement"],
                                 max(expected.values()) - player["utility"])

    def test_matching_pennies_has_no_pure_equilibrium(self):
        game = toy_examples()["matching_pennies"]
        self.assertEqual(game["pure_equilibria"], [])
        self.assertFalse(game["has_pure_equilibrium"])
        self.assertEqual(len(game["checks"]), 4)
        self.assertTrue(all(any(player["improvement"] == 2 for player in row["players"])
                            for row in game["checks"]))

    def test_ties_and_three_player_cartesian_product(self):
        actions = (("a", "b"), ("x",), ("i", "j"))
        game = analyze_game(actions, {p: (0, 0, 0) for p in itertools.product(*actions)})
        self.assertEqual(len(game["pure_equilibria"]), 4)
        self.assertEqual(game["checks"][0]["players"][0]["best_responses"], ["a", "b"])

    def test_invalid_game(self):
        for actions, table in (
            ((), {}), (((),), {}), ((("a", "a"),), {}),
            ((("a",),), {}), ((("a",),), {("a",): (0, 1)}),
            ((("a",),), {("a",): (float("nan"),)}),
            ((("a",),), {("a",): (float("inf"),)}),
            ((("a",),), {("a",): (True,)}),
            ((("a",),), {("a",): (0,), ("b",): (0,)}),
        ):
            with self.subTest(actions=actions, table=table), self.assertRaises(ValueError):
                analyze_game(actions, table)


if __name__ == "__main__":
    unittest.main()
