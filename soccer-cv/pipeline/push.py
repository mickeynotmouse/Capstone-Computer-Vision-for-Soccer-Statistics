"""Step 3: tracks.json -> events -> POST to the backend (only structured stats leave your machine).

    python -m pipeline.push data/g1_tracks.json --opponent Linfield --venue H --date 2026-09-05 --our-team 0
    python -m pipeline.push data/g2_tracks.json --opponent Pacific  --venue A --date 2026-09-12
    python -m pipeline.push tracks.json --opponent X --venue H --date 2026-09-05 --dry-run   # just writes events json

Game names are built for you so they're always consistent: "vs. Linfield (H)" at home, "at Pacific (A)" away.
"""
import argparse
import json
import urllib.request

from .events import detect


def game_name(opponent, venue):
    return f"vs. {opponent} (H)" if venue == "H" else f"at {opponent} (A)"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks")
    ap.add_argument("--opponent", required=True, help='e.g. "Linfield"')
    ap.add_argument("--venue", required=True, choices=["H", "A"], help="H = home, A = away")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--our-team", type=int, default=0, choices=[0, 1], help="which jersey cluster is Willamette")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    events = detect(json.load(open(a.tracks)))
    name = game_name(a.opponent, a.venue)
    print(f"{name}: {len(events)} events, {sum(e['type'] == 'goal' for e in events)} goals")
    payload = {"name": name, "date": a.date, "our_team": a.our_team, "events": events}
    if a.dry_run:
        out = a.tracks.replace(".json", "_events.json")
        json.dump(payload, open(out, "w"))
        print("wrote", out)
    else:
        req = urllib.request.Request(f"{a.api}/api/games", json.dumps(payload).encode(), {"Content-Type": "application/json"})
        print(urllib.request.urlopen(req).read().decode())
