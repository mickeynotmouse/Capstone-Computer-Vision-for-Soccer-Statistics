"""Step 3: tracks.json -> events -> the database (only structured stats leave your machine, never video).

Online (Supabase: what the website's Results page shows). Needs soccer-cv/.env, see README "Online database":
    python -m pipeline.push data/g1_tracks.json --opponent Linfield --venue H --date 2026-09-05 --our-team 1 --to supabase
    python -m pipeline.push --list --to supabase              # what's online, with ids
    python -m pipeline.push --delete 3 --to supabase          # remove game 3 (and its events)

Local (backend/ on your own machine, http://localhost:8000):
    python -m pipeline.push data/g1_tracks.json --opponent Linfield --venue H --date 2026-09-05 --our-team 1
    python -m pipeline.push tracks.json --opponent X --venue H --date 2026-09-05 --dry-run   # just writes events json

Game names are built for you so they're always consistent: "vs. Linfield (H)" at home, "at Pacific (A)" away.
--our-team is which jersey cluster is Willamette: 0 = the lighter shirts, 1 = the darker ones.
"""
import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from .events import detect

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
CHUNK = 500   # events per request


def game_name(opponent, venue):
    return f"vs. {opponent} (H)" if venue == "H" else f"at {opponent} (A)"


def settings(path=ENV_FILE):
    """SUPABASE_URL and SUPABASE_SECRET_KEY, from the environment or soccer-cv/.env (KEY=value lines)."""
    vals = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip().strip('"').strip("'")
    vals.update({k: os.environ[k] for k in ("SUPABASE_URL", "SUPABASE_SECRET_KEY") if k in os.environ})
    missing = [k for k in ("SUPABASE_URL", "SUPABASE_SECRET_KEY") if not vals.get(k)]
    if missing:
        raise SystemExit(f"missing {', '.join(missing)}: put them in {path} (README: 'Online database')")
    url = vals["SUPABASE_URL"].rstrip("/").removesuffix("/rest/v1")   # the API address is fine too
    return url, vals["SUPABASE_SECRET_KEY"]


class Supabase:
    """The few calls we need from Supabase's REST API (no extra Python packages)."""

    def __init__(self, url, key):
        self.base, self.headers = f"{url}/rest/v1", {"apikey": key, "Content-Type": "application/json"}
        if key.startswith("eyJ"):            # an older-style (JWT) key also goes in Authorization
            self.headers["Authorization"] = f"Bearer {key}"

    def call(self, method, path, body=None, prefer=None):
        headers = dict(self.headers, **({"Prefer": prefer} if prefer else {}))
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(f"{self.base}/{path}", data, headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                text = r.read().decode()
        except urllib.error.HTTPError as e:
            raise SystemExit(f"Supabase said {e.code} to {method} {path}: {e.read().decode()[:300]}")
        except urllib.error.URLError as e:
            raise SystemExit(f"can't reach Supabase ({e.reason}): check SUPABASE_URL and your internet")
        return json.loads(text) if text else None

    def add_game(self, payload):
        """-> new game id. If the events can't be stored, the game is removed again, so nothing half-stored stays."""
        game = {k: payload[k] for k in ("name", "date", "our_team")}
        gid = self.call("POST", "games", [game], prefer="return=representation")[0]["id"]
        rows = [{"game_id": gid, "type": e["type"], "frame": e["frame"], "t": e["t"], "team": e["team"],
                 "player": e.get("player"), "data": e.get("data", {})} for e in payload["events"]]
        try:
            for i in range(0, len(rows), CHUNK):
                self.call("POST", "events", rows[i:i + CHUNK], prefer="return=minimal")
        except SystemExit:
            self.delete_game(gid)
            raise
        return gid

    def games(self):
        return self.call("GET", "games?select=id,name,date,our_team&order=date,id")

    def delete_game(self, gid):
        return self.call("DELETE", f"games?id=eq.{int(gid)}", prefer="return=representation")


def main(argv=None):
    ap = argparse.ArgumentParser(description="send a game's events to the database")
    ap.add_argument("tracks", nargs="?", help="tracks.json from pipeline.track")
    ap.add_argument("--opponent", help='e.g. "Linfield"')
    ap.add_argument("--venue", choices=["H", "A"], help="H = home, A = away")
    ap.add_argument("--date", help="YYYY-MM-DD")
    ap.add_argument("--our-team", type=int, default=0, choices=[0, 1],
                    help="which jersey cluster is Willamette: 0 = lighter shirts, 1 = darker")
    ap.add_argument("--to", choices=["local", "supabase"], default="local", help="where to send it")
    ap.add_argument("--api", default="http://localhost:8000", help="the local backend (with --to local)")
    ap.add_argument("--dry-run", action="store_true", help="only write the events to a json file")
    ap.add_argument("--list", action="store_true", help="list the games in the online database")
    ap.add_argument("--delete", type=int, metavar="ID", help="delete a game from the online database")
    a = ap.parse_args(argv)

    if a.list or a.delete is not None:
        if a.to != "supabase":
            raise SystemExit("--list and --delete are for the online database: add --to supabase")
        sb = Supabase(*settings())
        if a.delete is not None:
            gone = sb.delete_game(a.delete)
            print(f"deleted game {a.delete}" if gone else f"no game with id {a.delete}")
        for g in sb.games():
            print(f"  {g['id']:4}  {g['date']}  {g['name']}  (Willamette = team {g['our_team']})")
        return

    if not (a.tracks and a.opponent and a.venue and a.date):
        ap.error("give the tracks file, --opponent, --venue and --date")
    events = detect(json.load(open(a.tracks)))
    name = game_name(a.opponent, a.venue)
    print(f"{name}: {len(events)} events, {sum(e['type'] == 'goal' for e in events)} goals")
    payload = {"name": name, "date": a.date, "our_team": a.our_team, "events": events}
    if a.dry_run:
        out = a.tracks.replace(".json", "_events.json")
        json.dump(payload, open(out, "w"))
        print("wrote", out)
    elif a.to == "supabase":
        gid = Supabase(*settings()).add_game(payload)
        print(f"stored online as game {gid}: it shows on the website's Results page now")
    else:
        req = urllib.request.Request(f"{a.api}/api/games", json.dumps(payload).encode(), {"Content-Type": "application/json"})
        print(urllib.request.urlopen(req).read().decode())


if __name__ == "__main__":
    main()
