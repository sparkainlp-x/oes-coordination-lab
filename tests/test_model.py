from fractions import Fraction

import pytest

from oes_coordination.model import (ACTIONS, CONDITIONS, act, efficient_welfare, exact_metrics,
                                    expected_utilities, observations, round_metrics, strategies,
                                    utilities)

S, L, H = "system_contingency", "local_fix", "hold"


def test_payoff_table_local():
    assert utilities("local", (L, L, L)) == (3, 3, 3)
    assert utilities("local", (S, S, S)) == (1, 1, 1)
    assert utilities("local", (S, L, H)) == (-2, 3, 0)


def test_payoff_table_shared():
    assert utilities("shared", (S, S, S)) == (4, 4, 4)
    assert utilities("shared", (S, S, L)) == (-2, -2, 1)
    assert utilities("shared", (H, L)) == (0, 1)


@pytest.mark.parametrize("bad", [("local", ()), ("other", (L,)), ("local", ("jump",))])
def test_invalid_inputs(bad):
    with pytest.raises(ValueError):
        utilities(*bad)


def test_efficient_welfare():
    assert efficient_welfare("local", 3) == 3
    assert efficient_welfare("shared", 3) == 4


def test_observations():
    assert observations("no_info", "shared", (True, False), True) == (None, None)
    assert observations("private_info", "shared", (True, False), True) == ("shared", "local")
    assert observations("shared_info", "local", (True, True), False) == ("shared", "shared")
    with pytest.raises(ValueError):
        observations("telepathy", "local", (True,), True)


def test_strategies_and_act():
    assert len(strategies("no_info")) == len(ACTIONS)
    assert len(strategies("private_info")) == 9
    assert act((S,), None) == S
    assert act((L, S), "shared") == S and act((L, S), "local") == L
    with pytest.raises(ValueError):
        strategies("x")


def test_expected_utilities_are_exact_fractions():
    u = expected_utilities("private_info", ((L, S),) * 3, 0.5, 0.8)
    assert all(isinstance(x, Fraction) for x in u)
    # local: 0.8*3 + 0.2*(-2 + 0.2**2 * 3) ; shared: 0.512*4 + 0.288*(-2) + 0.2*1
    local = Fraction(4, 5) * 3 + Fraction(1, 5) * (Fraction(24, 25) * -2 + Fraction(1, 25) * 1)
    shared = Fraction(512, 1000) * 4 + Fraction(288, 1000) * -2 + Fraction(1, 5)
    assert u[0] == (local + shared) / 2


def test_hand_computed_metrics():
    assert exact_metrics("no_info", ((S,),) * 3, 0.5, 0.8)["welfare"] == 2.5
    assert exact_metrics("shared_info", ((L, S),) * 3, 0.5, 0.8)["welfare"] == 3.0
    m = exact_metrics("private_info", ((L, S),) * 3, 0.5, 0.8)
    assert m["welfare"] == pytest.approx(1.848)
    assert m["coordination_rate"] == pytest.approx(0.512)
    assert m["miscoordination_rate"] == pytest.approx(0.48)


def test_perfect_signals_reach_ceiling_in_both_signal_conditions():
    for c in ("private_info", "shared_info"):
        m = exact_metrics(c, ((L, S),) * 4, 0.5, 1.0)
        assert m["welfare"] == 3.5 and m["regret"] == 0 and m["coordination_rate"] == 1


def test_round_metrics_definitions():
    r = round_metrics("shared", (S, S, L))
    assert r["miscoordination_rate"] == 1 and r["coordination_rate"] == 0
    assert r["welfare"] == pytest.approx(-1.0) and r["regret"] == pytest.approx(5.0)
    assert r["miscoordination_cost"] == r["regret"]
    r = round_metrics("shared", (L, L, L))
    assert r["miscoordination_rate"] == 0 and r["miscoordination_cost"] == 0 and r["regret"] == 3


def test_regret_identity():
    for c in CONDITIONS:
        for s in strategies(c):
            m = exact_metrics(c, (s,) * 2, 0.3, 0.7)
            # efficient welfare = 0.7*3 + 0.3*4 = 3.3
            assert m["welfare"] + m["regret"] == pytest.approx(3.3)
