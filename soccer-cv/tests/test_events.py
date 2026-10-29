"""Event detector vs. synthetic ground truth. Run: python -m pytest -q  (from the poc/ folder)"""
import statistics

import pytest

from pipeline.events import detect
from pipeline.synthetic import make_game

OUTCOMES = [(0, "goal"), (1, "lost"), (0, "wide"), (1, "goal"), (0, "goal"), (0, "lost"), (1, "wide"), (1, "goal")]


def _count(events, etype, team):
    return sum(1 for e in events if e["type"] == etype and e["team"] == team)


@pytest.mark.parametrize("chaos", [0, 8.0])   # textbook formation and "all over the place"
def test_matches_ground_truth(chaos):
    for seed in range(40):
        tracks, truth = make_game(seed, OUTCOMES, chaos=chaos)
        ev = detect(tracks)

        goals = [e for e in ev if e["type"] == "goal"]
        assert len(goals) == len(truth["goals"]), f"seed {seed}: goals"
        for got, want in zip(goals, truth["goals"]):
            assert got["team"] == want["team"]
            assert got["player"] == want["scorer"]
            assert got["data"]["buildup_passes"] == want["buildup_passes"], f"seed {seed}: buildup"
            assert got["data"]["scorer_touches"] == want["scorer_touches"], f"seed {seed}: scorer touches"
            assert len(got["data"]["attackers"]) == 11 and len(got["data"]["defenders"]) == 11

        for team in (0, 1):
            assert _count(ev, "shot", team) == truth["shots"][team], f"seed {seed}: shots team {team}"
            assert _count(ev, "pass", team) == truth["passes"][team], f"seed {seed}: passes team {team}"
            assert _count(ev, "turnover", team) == truth["turnovers"][team], f"seed {seed}: turnovers team {team}"


def test_first_time_passes_are_exercised_and_counted():
    quick = sum(make_game(s, OUTCOMES)[1]["one_touch"] for s in range(10))
    assert quick > 10, "generator should produce plenty of first-time passes"
    # (the exact pass counts in test_matches_ground_truth include them, so they are detected as touches)


def test_goal_positions_are_normalised_to_attack_right():
    tracks, _ = make_game(3, [(0, "goal"), (1, "goal")])
    for g in (e for e in detect(tracks) if e["type"] == "goal"):
        xs = [p[0] for p in g["data"]["attackers"]]
        assert sum(xs) / len(xs) > 40, "scoring team should be mirrored to attack towards x=105"


def test_goal_positions_vary_when_chaos_is_on():
    def spread(chaos):
        xs, ys = [], []
        for seed in range(6):
            tracks, _ = make_game(seed, [(0, "goal")] * 3, chaos=chaos)
            for g in (e for e in detect(tracks) if e["type"] == "goal"):
                xs += [p[0] for p in g["data"]["attackers"]]
                ys += [p[1] for p in g["data"]["attackers"]]
        return statistics.pstdev(xs), statistics.pstdev(ys)

    (nx, ny), (cx, cy) = spread(0), spread(8.0)
    assert cx > nx and cy > ny, "chaos should spread the positions out compared with the neat formation"
    assert cx > 18 and cy > 14
