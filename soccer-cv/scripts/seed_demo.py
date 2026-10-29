"""Fill the DB with SYNTHETIC matches so the dashboard has something to show.
None of this is real: team names, dates and results are invented; the data comes from pipeline/synthetic.py.

Run from poc/:
    python -m scripts.seed_demo --reset     # wipes the database first (demo only!), then loads the 4 games
"""
import sys

from backend import db
from pipeline.events import detect
from pipeline.synthetic import make_game

# Label format: home games "vs. Team (H)", away games "at Team (A)".
# Each outcome list = one possession sequence: (team that has the ball, how it ends). Team 0 is "us".
GAMES = [
    ("vs. Linfield (H)", "2026-09-05", 1, [(0, "goal"), (1, "lost"), (0, "wide"), (0, "goal"), (1, "goal"), (0, "lost")]),
    ("at Pacific (A)", "2026-09-12", 2, [(1, "goal"), (0, "goal"), (1, "wide"), (0, "lost"), (0, "goal"), (1, "lost")]),
    ("vs. Whitworth (H)", "2026-09-19", 3, [(0, "goal"), (0, "wide"), (0, "goal"), (1, "lost"), (0, "goal"), (0, "goal")]),
    ("at Puget Sound (A)", "2026-09-26", 4, [(1, "goal"), (1, "goal"), (0, "lost"), (0, "goal"), (1, "wide"), (0, "wide")]),
]

if __name__ == "__main__":
    db.init()
    if "--reset" in sys.argv:
        db.reset()
    elif db.list_games():
        sys.exit("The database already has games. Re-run with --reset to wipe it and load the demo data.")
    for name, date, seed, outcomes in GAMES:
        tracks, truth = make_game(seed, outcomes)
        events = detect(tracks)
        gid = db.add_game(name, date, our_team=0, events=events)
        print(f"game {gid}: {name:20s} goals detected {sum(e['type'] == 'goal' for e in events)} "
              f"(injected {len(truth['goals'])}), events stored {len(events)}")
