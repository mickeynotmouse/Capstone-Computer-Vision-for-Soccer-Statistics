"""Sanity-check helper: compare what the demo generator INJECTED with what the detector FOUND, and write CSVs
you can open in Excel to check by hand.

    python -m scripts.inspect_demo        # all demo games
    python -m scripts.inspect_demo 2      # just game 2 (numbering matches the dashboard after `seed_demo --reset`)

Where the data comes from: pipeline/synthetic.py invents it (random but seeded, so it's identical every run) using the
hard-coded GAMES list in scripts/seed_demo.py. Nothing is downloaded and nothing comes from real matches.
Output goes to data/inspect/.
"""
import csv
import sys
from pathlib import Path

from pipeline.events import detect
from pipeline.synthetic import make_game
from scripts.seed_demo import GAMES

OUT = Path(__file__).resolve().parent.parent / "data" / "inspect"
TEAM = {0: "us", 1: "opp"}


def _mmss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def inspect(n, name, seed, outcomes):
    tracks, truth = make_game(seed, outcomes)
    events = detect(tracks)
    fr = tracks["frames"]
    print(f"\n=== Game {n}: {name}  (generator seed {seed}) ===")
    print(f"{'seq':>3} {'who':>4} {'injected':>8} {'passes':>6} {'1-touch':>7} | {'found':>8} {'passes':>6} | ok?")
    all_ok = True
    rows = []
    for s in truth["sequences"]:
        inwin = [e for e in events if s["start_frame"] <= e["frame"] <= s["end_frame"] + 1]
        passes = sum(1 for e in inwin if e["type"] == "pass" and e["team"] == s["team"])
        found = ("goal" if any(e["type"] == "goal" for e in inwin) else
                 "wide" if any(e["type"] == "shot" for e in inwin) else
                 "lost" if any(e["type"] == "turnover" for e in inwin) else "?")
        ok = found == s["outcome"] and passes == s["passes"]
        all_ok &= ok
        print(f"{s['seq']:>3} {TEAM[s['team']]:>4} {s['outcome']:>8} {s['passes']:>6} {s['one_touch_passes']:>7} | "
              f"{found:>8} {passes:>6} | {'OK' if ok else 'MISMATCH'}")
        rows.append([s["seq"], TEAM[s["team"]], s["outcome"], s["passes"], s["one_touch_passes"], s["player"] or "",
                     _mmss(fr[s["start_frame"]]["t"]), _mmss(fr[s["end_frame"]]["t"]), found, passes])
    goals = [e for e in events if e["type"] == "goal"]
    for g, want in zip(goals, truth["goals"]):
        gok = (g["data"]["buildup_passes"], g["data"]["scorer_touches"]) == (want["buildup_passes"], want["scorer_touches"])
        all_ok &= gok
        print(f"  goal at {_mmss(g['t'])} by {TEAM[g['team']]} player {g['player']}: "
              f"build-up passes {g['data']['buildup_passes']} (injected {want['buildup_passes']}), "
              f"scorer touches {g['data']['scorer_touches']} (injected {want['scorer_touches']})  {'OK' if gok else 'MISMATCH'}")
    print("  ->", "everything matches" if all_ok else "MISMATCHES FOUND")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"game{n}_injected_vs_found.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence", "team", "injected_outcome", "injected_completed_passes", "injected_first_time_passes",
                    "shooter_id", "starts_at", "ends_at", "detected_outcome", "detected_passes"])
        w.writerows(rows)
    with open(OUT / f"game{n}_detected_events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "seconds", "frame", "type", "team", "player", "detail"])
        for e in events:
            d = e["data"]
            detail = {"pass": f"to player {d.get('to')}", "turnover": f"ball won by player {d.get('to')}",
                      "shot": "scored" if d.get("goal") else "no goal",
                      "goal": f"scorer touches {d.get('scorer_touches')}, build-up passes {d.get('buildup_passes')}"}[e["type"]]
            w.writerow([_mmss(e["t"]), e["t"], e["frame"], e["type"], TEAM[e["team"]], e["player"], detail])
    with open(OUT / f"game{n}_ball_track.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "seconds", "ball_x_m", "ball_y_m"])
        w.writerows([[x["f"], round(x["t"], 2), x["ball"]["x"], x["ball"]["y"]] for x in fr if x["ball"]])
    with open(OUT / f"game{n}_goal_positions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["goal_time", "scoring_team", "group", "x_m", "y_m"])
        for g in goals:
            for grp, pts in (("scoring team", g["data"]["attackers"]), ("defending team", g["data"]["defenders"])):
                w.writerows([[_mmss(g["t"]), TEAM[g["team"]], grp, x, y] for x, y in pts])
    return all_ok


if __name__ == "__main__":
    only = int(sys.argv[1]) if len(sys.argv) > 1 else None
    ok = all([inspect(n, *g[:1], *g[2:]) for n, g in enumerate(GAMES, 1) if only in (None, n)])
    print(f"\nCSV files written to {OUT}")
    sys.exit(0 if ok else 1)
