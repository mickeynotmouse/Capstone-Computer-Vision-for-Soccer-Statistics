"""Show, for each demo match: the scripted input, the generator's ground truth, what the detector found,
and the totals the dashboard will display -- so you can check the numbers by hand.

Run from poc/:   python -m scripts.show_truth
Also writes data/events_game<N>.csv (open in Excel/Numbers) listing every detected event with its timestamp.
"""
import csv
import json
import pathlib

from backend import stats
from pipeline.events import detect
from pipeline.labels import game_label
from pipeline.synthetic import make_game
from scripts.seed_demo import GAMES

OUT = pathlib.Path("data")
OUT.mkdir(exist_ok=True)
WHO = {0: "us", 1: "opp"}

for n, (opp, venue, date, seed, outcomes) in enumerate(GAMES, 1):
    label = game_label(opp, venue)
    tracks, truth = make_game(seed, outcomes, one_touch=0.3)   # same call as seed_demo -> identical match
    ev = detect(tracks)

    print(f"\n=== GAME {n}: {label}  ({date}, random seed {seed}) ===")
    print("Scripted possessions (hardcoded in scripts/seed_demo.py):")
    for i, (team, outcome) in enumerate(outcomes, 1):
        print(f"  {i}. {WHO[team]:<3} -> {outcome}")

    goals = [e for e in ev if e["type"] == "goal"]
    print("Goals: ground truth (what the generator did)  vs  detected (what the algorithm found from the tracks)")
    for i, (w, g) in enumerate(zip(truth["goals"], goals), 1):
        ok = (w["team"], w["scorer"], w["buildup_passes"], w["scorer_touches"]) == (
            g["team"], g["player"], g["data"]["buildup_passes"], g["data"]["scorer_touches"])
        print(f"  goal {i}: {WHO[w['team']]:<3} scorer #{w['scorer']:<3} build-up passes {w['buildup_passes']}, scorer touches {w['scorer_touches']}"
              f"   |   detected at {g['t']:.1f}s: {WHO[g['team']]:<3} #{g['player']:<3} {g['data']['buildup_passes']}, {g['data']['scorer_touches']}"
              f"   {'OK' if ok else 'MISMATCH'}")
    print(f"First-time (one-touch) passes the generator played: {truth['one_touch']}")

    row = stats.summarize([{"id": n, "name": label, "date": date, "our_team": 0}], {n: ev})["games"][0]
    print("Dashboard totals for this game:", {k: v for k, v in row.items() if k not in ("id", "name", "date")})

    path = OUT / f"events_game{n}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "type", "team", "player", "details"])
        for e in ev:
            slim = {k: v for k, v in e["data"].items() if k not in ("attackers", "defenders")}
            w.writerow([e["t"], e["type"], WHO[e["team"]], e["player"], json.dumps(slim)])
    print("  full event list ->", path)
