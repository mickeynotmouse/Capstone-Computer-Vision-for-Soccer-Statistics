"""Score the pipeline's events against the ones you logged by hand (scripts/log_events.py).

    python -m scripts.score_events truth/test_events.json data/test_tracks.json

A pipeline event is FOUND when it has the same type as one of yours and is within --tol seconds of it (each event
is used once, closest pairs first). Only the part of the clip you watched is scored. For each type:
    found      share of YOUR events the pipeline also has (recall)
    correct    share of the PIPELINE's events that are real (precision)
    team ok    of the found ones, how many have the right team
Every miss and false alarm is listed with its time, and with how much of the ball was really SEEN (not guessed
between sightings) in the second around it: a low number there means the ball detector is the one to blame.
"""
import argparse
import json

from pipeline import events as ev

TYPES = ("pass", "turnover", "shot", "goal")
TOL_S = 1.0


def match(truth, found, tol=TOL_S):
    """-> (pairs [(i_truth, i_found)], unmatched truth indices, unmatched found indices). Same type only, closest in
    time first, each event used at most once."""
    cands = sorted((abs(t["t"] - f["t"]), i, j) for i, t in enumerate(truth) for j, f in enumerate(found)
                   if t["type"] == f["type"] and abs(t["t"] - f["t"]) <= tol)
    used_t, used_f, pairs = set(), set(), []
    for _, i, j in cands:
        if i not in used_t and j not in used_f:
            used_t.add(i), used_f.add(j), pairs.append((i, j))
    return (sorted(pairs), [i for i in range(len(truth)) if i not in used_t],
            [j for j in range(len(found)) if j not in used_f])


def ball_seen(frames, t, half=0.5):
    """Share of frames within `half` s of t where the ball was detected, not filled in by the tracker."""
    near = [f for f in frames if abs(f["t"] - t) <= half]
    if not near:
        return None
    return sum(1 for f in near if f.get("ball") and not f["ball"].get("filled")) / len(near)


def score(truth_doc, tracks, tol=TOL_S):
    upto = truth_doc.get("reviewed_until") or float("inf")
    truth = [e for e in truth_doc["events"] if e["t"] <= upto]
    found = [{"t": e["t"], "type": e["type"], "team": e["team"]} for e in ev.detect(tracks)
             if e["type"] in TYPES and e["t"] <= upto]
    pairs, miss, extra = match(truth, found, tol)
    rows = {}
    for k in TYPES:
        nt = sum(1 for e in truth if e["type"] == k)
        nf = sum(1 for e in found if e["type"] == k)
        m = [(i, j) for i, j in pairs if truth[i]["type"] == k]
        rows[k] = {"yours": nt, "pipeline": nf, "matched": len(m),
                   "team_ok": sum(1 for i, j in m if truth[i]["team"] == found[j]["team"])}
    frames = tracks["frames"]
    return {"upto": upto, "rows": rows,
            "misses": [{**truth[i], "ball_seen": ball_seen(frames, truth[i]["t"])} for i in miss],
            "false_alarms": [{**found[j], "ball_seen": ball_seen(frames, found[j]["t"])} for j in extra],
            "wrong_team": [{**truth[i], "pipeline_t": found[j]["t"]} for i, j in pairs
                           if truth[i]["team"] != found[j]["team"]]}


def pct(a, b):
    return f"{100 * a / b:3.0f} %" if b else "   -"


def report(s):
    team = {0: "light", 1: "dark"}
    print(f"scored the first {s['upto']:.1f} s\n")
    print(f"{'':10}{'yours':>7}{'pipeline':>10}{'found':>8}{'correct':>9}{'team ok':>9}")
    for k, r in s["rows"].items():
        if r["yours"] or r["pipeline"]:
            print(f"{k:10}{r['yours']:7}{r['pipeline']:10}{pct(r['matched'], r['yours']):>8}"
                  f"{pct(r['matched'], r['pipeline']):>9}{pct(r['team_ok'], r['matched']):>9}")
    seen = lambda e: "   ?" if e["ball_seen"] is None else f"{100 * e['ball_seen']:3.0f} %"   # noqa: E731
    for title, key in (("MISSED (yours, not the pipeline's)", "misses"),
                       ("FALSE ALARMS (the pipeline's, not yours)", "false_alarms")):
        if s[key]:
            print(f"\n{title}:   time   type       team   ball seen in the second around it")
            for e in s[key]:
                print(f"  {e['t']:7.2f} s  {e['type']:10} {team[e['team']]:6} {seen(e)}")
    if s["wrong_team"]:
        print("\nFOUND BUT WRONG TEAM:")
        for e in s["wrong_team"]:
            print(f"  {e['t']:7.2f} s  {e['type']:10} yours: {team[e['team']]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="score pipeline events against hand-logged ones")
    ap.add_argument("truth", help="truth/<clip>_events.json from scripts.log_events")
    ap.add_argument("tracks", help="data/<clip>_tracks.json from pipeline.track")
    ap.add_argument("--tol", type=float, default=TOL_S, help="seconds apart that still count as the same event")
    a = ap.parse_args(argv)
    report(score(json.load(open(a.truth)), json.load(open(a.tracks)), a.tol))


if __name__ == "__main__":
    main()
