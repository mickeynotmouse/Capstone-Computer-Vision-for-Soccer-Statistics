"""Scoring pipeline events against hand-logged ones (scripts/score_events.py, scripts/log_events.py)."""
from scripts import log_events, score_events as se


def test_match_pairs_same_type_closest_first_and_uses_each_once():
    truth = [{"t": 1.0, "type": "pass"}, {"t": 1.6, "type": "pass"}, {"t": 5.0, "type": "turnover"}]
    found = [{"t": 1.5, "type": "pass"}, {"t": 5.2, "type": "pass"}, {"t": 9.0, "type": "pass"}]
    pairs, miss, extra = se.match(truth, found, tol=1.0)
    assert pairs == [(1, 0)]            # 1.6 is closer to 1.5 than 1.0 is; 5.2 is a pass, not a turnover
    assert miss == [0, 2] and extra == [1, 2]


def test_score_counts_only_the_watched_part(monkeypatch):
    found = [{"t": 2.0, "type": "pass", "team": 0}, {"t": 4.0, "type": "turnover", "team": 1},
             {"t": 30.0, "type": "pass", "team": 0}]           # after reviewed_until: not scored
    monkeypatch.setattr(se.ev, "detect", lambda tracks: [dict(e, frame=0, player=1, data={}) for e in found])
    truth = {"reviewed_until": 10.0, "events": [{"t": 2.3, "type": "pass", "team": 1},
                                                 {"t": 7.0, "type": "pass", "team": 0}]}
    frames = [{"t": i / 10, "ball": {"x": 0, "y": 0, "filled": i % 2 == 0}} for i in range(100)]
    s = se.score(truth, {"frames": frames})
    assert s["rows"]["pass"] == {"yours": 2, "pipeline": 1, "matched": 1, "team_ok": 0}
    assert [e["t"] for e in s["misses"]] == [7.0] and [e["t"] for e in s["false_alarms"]] == [4.0]
    assert abs(s["misses"][0]["ball_seen"] - 0.5) < 0.1 and len(s["wrong_team"]) == 1


def test_logger_keeps_only_well_formed_events():
    body = {"reviewed_until": 12.345, "events": [{"t": 3.21, "type": "pass", "team": 1},
                                                 {"t": 1.0, "type": "turnover", "team": 0},
                                                 {"t": 2.0, "type": "dribble", "team": 0},
                                                 {"t": 2.0, "type": "pass", "team": 5}]}
    out = log_events.clean(body, "clips/test.mp4")
    assert out["video"] == "test.mp4" and out["reviewed_until"] == 12.35
    assert out["events"] == [{"t": 1.0, "type": "turnover", "team": 0}, {"t": 3.21, "type": "pass", "team": 1}]
