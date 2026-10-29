"""SQLite for the PoC (zero setup). The schema is plain SQL, so moving to Supabase/Postgres later is a driver swap."""
import json
import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.environ.get("POC_DB", Path(__file__).resolve().parent.parent / "data" / "poc.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    date TEXT NOT NULL,
    our_team INTEGER NOT NULL,          -- which track team label (0/1) is Willamette
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    frame INTEGER NOT NULL,
    t REAL NOT NULL,
    team INTEGER NOT NULL,
    player INTEGER,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_game_type ON events(game_id, type);
"""


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def init():
    with _conn() as c:
        c.executescript(SCHEMA)


def reset():
    """Delete the whole database file. Demo/dev only."""
    DB_PATH.unlink(missing_ok=True)
    init()


def add_game(name, date, our_team, events):
    with _conn() as c:
        gid = c.execute("INSERT INTO games(name, date, our_team) VALUES (?,?,?)", (name, date, our_team)).lastrowid
        c.executemany(
            "INSERT INTO events(game_id, type, frame, t, team, player, data) VALUES (?,?,?,?,?,?,?)",
            [(gid, e["type"], e["frame"], e["t"], e["team"], e.get("player"), json.dumps(e["data"])) for e in events])
    return gid


def list_games(ids=None):
    with _conn() as c:
        rows = c.execute("SELECT id, name, date, our_team FROM games ORDER BY date, id").fetchall()
    rows = [dict(r) for r in rows]
    return [r for r in rows if ids is None or r["id"] in ids]


def events_for(game_ids):
    out = {gid: [] for gid in game_ids}
    if not game_ids:
        return out
    q = ",".join("?" * len(game_ids))
    with _conn() as c:
        for r in c.execute(f"SELECT * FROM events WHERE game_id IN ({q}) ORDER BY frame", list(game_ids)):
            out[r["game_id"]].append({"type": r["type"], "frame": r["frame"], "t": r["t"], "team": r["team"],
                                      "player": r["player"], "data": json.loads(r["data"])})
    return out


def delete_game(gid):
    with _conn() as c:
        return c.execute("DELETE FROM games WHERE id = ?", (gid,)).rowcount > 0
