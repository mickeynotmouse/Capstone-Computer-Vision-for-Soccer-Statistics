"""FastAPI backend.  Run from poc/:  uvicorn backend.main:app --reload"""
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, stats

app = FastAPI(title="Soccer CV PoC")
db.init()


class GameIn(BaseModel):
    name: str
    date: str            # YYYY-MM-DD
    our_team: int = 0    # 0 or 1: which jersey cluster is us
    events: list[dict]   # output of pipeline.events.detect (structured stats only -- heavy video work stays offline)


def _selected(games: Optional[str]):
    """?games=1,3 -> those games; omitted -> every game (season-wide aggregate)."""
    ids = None
    if games:
        try:
            ids = {int(x) for x in games.split(",") if x.strip()}
        except ValueError:
            raise HTTPException(400, "games must be comma-separated integers")
    sel = db.list_games(ids)
    return sel, db.events_for([g["id"] for g in sel])


@app.get("/api/games")
def games():
    return db.list_games()


@app.post("/api/games", status_code=201)
def add_game(g: GameIn):
    if g.our_team not in (0, 1):
        raise HTTPException(400, "our_team must be 0 or 1")
    return {"id": db.add_game(g.name, g.date, g.our_team, g.events)}


@app.delete("/api/games/{gid}")
def delete_game(gid: int):
    if not db.delete_game(gid):
        raise HTTPException(404, "no such game")
    return {"deleted": gid}


@app.get("/api/stats/summary")
def summary(games: Optional[str] = None):
    sel, ev = _selected(games)
    return stats.summarize(sel, ev)


@app.get("/api/stats/goals")
def goals(games: Optional[str] = None, kind: str = Query("for", pattern="^(for|against)$")):
    """'Where were we standing when a goal was scored?' -- positions, build-up, scorer touches, heatmap."""
    sel, ev = _selected(games)
    gl = stats.goal_list(sel, ev, kind)
    return {"kind": kind, "count": len(gl), "goals": gl, "heatmap": stats.heatmap(gl)}


# static dashboard last, so it doesn't shadow /api
app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent.parent / "frontend", html=True), name="frontend")
