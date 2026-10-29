"""Aggregation over any set of games -- the thing HUDL can't do."""

GRID_COLS, GRID_ROWS = 21, 14   # 5 m x ~4.9 m cells over the 105 x 68 pitch
L, W = 105.0, 68.0


def _blank():
    return {"goals_for": 0, "goals_against": 0, "shots_for": 0, "shots_against": 0,
            "passes_for": 0, "passes_against": 0, "turnovers": 0}


def summarize(games, events_by_game):
    rows, totals = [], _blank()
    for g in games:
        ours, s = g["our_team"], _blank()
        for e in events_by_game.get(g["id"], []):
            mine = e["team"] == ours
            if e["type"] == "goal":
                s["goals_for" if mine else "goals_against"] += 1
            elif e["type"] == "shot":
                s["shots_for" if mine else "shots_against"] += 1
            elif e["type"] == "pass":
                s["passes_for" if mine else "passes_against"] += 1
            elif e["type"] == "turnover" and mine:
                s["turnovers"] += 1  # possessions we lost
        rows.append({"id": g["id"], "name": g["name"], "date": g["date"], **s})
        for k, v in s.items():
            totals[k] += v

    goals = goal_list(games, events_by_game, "for")
    n = len(goals)
    totals["avg_buildup_passes"] = round(sum(x["buildup_passes"] for x in goals) / n, 2) if n else None
    totals["avg_scorer_touches"] = round(sum(x["scorer_touches"] for x in goals) / n, 2) if n else None
    totals["games"] = len(games)
    return {"games": rows, "totals": totals}


def goal_list(games, events_by_game, kind="for"):
    """kind='for': our goals (positions = our attackers). kind='against': goals conceded (positions = our defenders).
    Coordinates are mirrored so the scoring team always attacks towards x=105."""
    out = []
    for g in games:
        for e in events_by_game.get(g["id"], []):
            if e["type"] != "goal" or (e["team"] == g["our_team"]) != (kind == "for"):
                continue
            d = e["data"]
            ours, theirs = (d["attackers"], d["defenders"]) if kind == "for" else (d["defenders"], d["attackers"])
            out.append({"id": f"{g['id']}-{e['frame']}",   # stable id the dashboard uses to select individual goals
                        "game_id": g["id"], "game": g["name"], "t": e["t"], "minute": int(e["t"] // 60) + 1,
                        "scorer_touches": d["scorer_touches"], "buildup_passes": d["buildup_passes"],
                        "ball": d["ball"], "ours": ours, "theirs": theirs})
    return out


def heatmap(goals):
    cells = [[0] * GRID_COLS for _ in range(GRID_ROWS)]
    for g in goals:
        for x, y in g["ours"]:
            cx = min(GRID_COLS - 1, max(0, int(x / L * GRID_COLS)))
            cy = min(GRID_ROWS - 1, max(0, int(y / W * GRID_ROWS)))
            cells[cy][cx] += 1
    return {"cols": GRID_COLS, "rows": GRID_ROWS, "cells": cells}
