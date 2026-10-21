import json

import pytest

from oes_coordination.__main__ import main
from oes_coordination.experiment import (BASELINES, deviation_check, draw_rounds, mean_ci, run,
                                         strategy_game, sweep)
from oes_coordination.report import table

S, L = "system_contingency", "local_fix"


@pytest.fixture(scope="module")
def small():
    return run(seeds=30, rounds=300)


def by(res, cond):
    return next(r for r in res["results"] if r["condition"] == cond)


def test_common_random_numbers_and_determinism():
    assert draw_rounds(7, 50, 3, 0.5, 0.8) == draw_rounds(7, 50, 3, 0.5, 0.8)
    assert draw_rounds(7, 50, 3, 0.5, 0.8) != draw_rounds(8, 50, 3, 0.5, 0.8)
    assert run(seeds=3, rounds=20) == run(seeds=3, rounds=20)


def test_simulation_matches_exact_within_ci(small):
    for r in small["results"]:
        for k, cell in r["simulated"].items():
            lo, hi = cell["ci95"]
            assert lo - 0.02 <= r["exact"][k] <= hi + 0.02, (r["policy"], k)


def test_three_condition_ordering(small):
    w = {r["condition"] + "/" + r["policy"]: r["exact"]["welfare"] for r in small["results"]}
    assert w["shared_info/follow_signal"] > w["no_info/always_system"] > w["private_info/follow_signal"]


def test_deviation_checks(small):
    assert by(small, "no_info")["deviation_check"]["is_equilibrium"]
    assert by(small, "shared_info")["deviation_check"]["is_equilibrium"]
    d = by(small, "private_info")["deviation_check"]
    assert not d["is_equilibrium"]
    assert d["max_deviation_gain"] == pytest.approx(0.152)
    assert d["best_unilateral_response"] == [L, L]
    assert d["pure_equilibria_in_strategy_game"] == 1


def test_strategy_game_scaling_is_exact():
    actions, payoffs, scale = strategy_game("private_info", 2, 0.5, 0.8)
    assert all(isinstance(u, int) for us in payoffs.values() for u in us)
    assert scale > 0 and len(payoffs) == 81 and len(actions) == 2


def test_two_agent_private_follow_signal_is_equilibrium():
    assert deviation_check("private_info", (L, S), 2, 0.5, 0.8)["is_equilibrium"]


def test_mean_ci():
    assert mean_ci([1.0])["ci95"] is None
    c = mean_ci([1.0, 3.0])
    assert c["mean"] == 2 and c["ci95"][0] < 2 < c["ci95"][1]


@pytest.mark.parametrize("kw", [{"seeds": 0}, {"rounds": -1}, {"n_agents": 0},
                                {"p_shared": 1.5}, {"accuracy": True}, {"accuracy": "0.8"}])
def test_invalid_run_arguments(kw):
    with pytest.raises(ValueError):
        run(**kw)


def test_sweep_has_all_cells():
    rows = sweep(n_values=(2,), accuracies=(0.8,))
    assert len(rows) == sum(len(v) for v in BASELINES.values())


def test_output_is_json_safe_and_labelled(small):
    text = json.dumps(small, allow_nan=False)
    assert "SYNTHETIC" in text
    assert "| (3) shared signal |" in table(small)


def test_cli(capsys):
    assert main(["--seeds", "2", "--rounds", "10", "--sweep"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["results"]) == 4 and out["exact_sweep"]
    assert main(["--accuracy", "2"]) == 2
