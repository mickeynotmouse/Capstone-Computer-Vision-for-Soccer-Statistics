"""Step 2: turn per-frame pitch-coordinate tracks into soccer events.

Input  : the JSON written by pipeline.track  {"fps": .., "frames": [{"t", "players": [{id,x,y,team}], "ball": {x,y}|None}]}
Output : a flat list of events (touch-based passes, turnovers, shots, goals) ready to store in the DB.

Pitch coordinates are metres, 105 x 68, origin top-left. Everything here is deliberately simple,
rule-based and tunable via the constants below -- it is the part of the PoC to iterate on with real footage.

Known PoC simplifications (see README):
  * a "touch" = an unbroken run of frames where one player is the closest to the ball (so a dribble is 1 touch)
  * a goal needs the ball to be seen beyond the goal line; own goals are credited to the last toucher's team
  * all time thresholds are in seconds, so they behave the same at 10, 25 or 30 fps
"""
import math

L, W = 105.0, 68.0
GOAL_HALF = 7.32 / 2

POSSESS_R = 1.5        # m: ball closer than this to a player = "on the ball"
MIN_RUN_S = 0.3        # s on the ball for a *controlled* touch; shorter contacts (first-time passes) need a kick, see _is_contact
WINDOW_S = 0.3         # s either side of a touch used to measure how the ball's velocity changed
MERGE_GAP_S = 1.0      # same player re-touching within this gap is merged into one touch
DEAD_BALL_S = 5.0      # gap between touches above which play is considered stopped
CONTROL_SPEED = 5.0    # m/s: a ball that stays with a player (net speed over the run) is under control
CHANGE_V = 5.0         # m/s: a velocity change this large across the run means the player actually played the ball
SHOT_SPEED = 16.0      # m/s ball speed right after the touch
SHOT_MAX_DIST = 40.0   # m from the goal line
SHOT_Y_TOL = 4.0       # m beyond the posts that still counts as "on its way to goal"
GOAL_COOLDOWN_S = 10.0


def _velocity_change(frames, i0, i1, k):
    """|v_out - v_in| (m/s) of the ball across the run i0..i1; None if the ball isn't tracked on both sides of it."""
    if i0 - k < 0 or i1 + k >= len(frames):
        return None
    p0, p1, q0, q1 = frames[i0 - k]["ball"], frames[i0]["ball"], frames[i1]["ball"], frames[i1 + k]["ball"]
    if not (p0 and q1):
        return None
    t_in = frames[i0]["t"] - frames[i0 - k]["t"]
    t_out = frames[i1 + k]["t"] - frames[i1]["t"]
    vin = ((p1["x"] - p0["x"]) / t_in, (p1["y"] - p0["y"]) / t_in)
    vout = ((q1["x"] - q0["x"]) / t_out, (q1["y"] - q0["y"]) / t_out)
    return math.hypot(vout[0] - vin[0], vout[1] - vin[1])


def _in_play(b):
    return b is not None and 0 <= b["x"] <= L and 0 <= b["y"] <= W


def _free_flight(frames, holders, firm, i0, i1, k):
    """True if, in the k frames before and after the run, the ball was in play and no OTHER player was clearly
    playing it (`firm` marks frames inside a long, confirmed touch). Otherwise a velocity change can't be pinned
    on this player (it may be the previous kick, or the net/boards). A bystander the ball merely flew past
    doesn't count."""
    me = holders[i0]["id"]
    for f in list(range(i0 - k, i0)) + list(range(i1 + 1, i1 + k + 1)):
        if not 0 <= f < len(frames):
            return False
        if not _in_play(frames[f]["ball"]):
            return False
        if firm[f] and holders[f]["id"] != me:
            return False
    return True


def _is_contact(frames, holders, firm, i0, i1, k):
    """Did the player at the ball really play it, or did the ball just fly through his radius?

    * Long run (>= MIN_RUN_S): he controlled it (ball moves slowly with him) OR the ball's velocity changed sharply.
    * Quick run (< MIN_RUN_S), e.g. a first-time pass: there is no time to see control, so the ONLY evidence is a
      sharp change in ball velocity, measured over a window of free flight either side. A fly-by (constant
      velocity) is rejected; a deflection or flick that changes direction/speed is accepted.
    Limits: a first-time pass that sends the ball on in exactly the same direction at the same speed is
    kinematically identical to a fly-by and can't be detected this way; and a first-time pass played within
    ~WINDOW_S of the previous player's touch is skipped because the windows overlap.
    """
    span = frames[i1]["t"] - frames[i0]["t"]
    dv = _velocity_change(frames, i0, i1, k)
    if span < MIN_RUN_S - 1e-6:
        return dv is not None and dv >= CHANGE_V and _free_flight(frames, holders, firm, i0, i1, k)
    # controlled = the ball slowed down and stayed with him: slow over the whole run, or at least over its second half
    # (a fast ball that he then traps looks fast on average but slow at the end)
    mid = (i0 + i1) // 2
    for s, e in ((i0, i1), (mid, i1)):
        a, b, dt = frames[s]["ball"], frames[e]["ball"], frames[e]["t"] - frames[s]["t"]
        if dt > 0 and math.hypot(b["x"] - a["x"], b["y"] - a["y"]) / dt < CONTROL_SPEED:
            return True
    return dv is None or dv >= CHANGE_V  # ball lost on one side of the run (e.g. out of view): keep it


def find_touches(frames, k=3):
    holders = []
    for fr in frames:
        b, best = fr["ball"], None
        if b:
            for p in fr["players"]:
                d = math.hypot(p["x"] - b["x"], p["y"] - b["y"])
                if d <= POSSESS_R and (best is None or d < best[0]):
                    best = (d, p)
        holders.append(best[1] if best else None)

    n = len(frames)
    runs, i = [], 0                      # unbroken runs of one player being closest to the ball
    while i < n:
        h = holders[i]
        if h is None:
            i += 1
            continue
        j = i
        while j + 1 < n and holders[j + 1] is not None and holders[j + 1]["id"] == h["id"]:
            j += 1
        runs.append((i, j, h))
        i = j + 1
    firm = [False] * n                   # frames inside a long run that is clearly a real touch
    for i, j, _ in runs:
        if frames[j]["t"] - frames[i]["t"] >= MIN_RUN_S - 1e-6 and _is_contact(frames, holders, firm, i, j, k):
            for f in range(i, j + 1):
                firm[f] = True

    touches = []
    for i, j, h in runs:
        if _is_contact(frames, holders, firm, i, j, k):
            for f in range(i, j + 1):   # an accepted quick touch also explains velocity changes right next to it
                firm[f] = True
            if (touches and touches[-1]["player"] == h["id"]
                    and frames[i]["t"] - frames[touches[-1]["i1"]]["t"] <= MERGE_GAP_S):
                touches[-1]["i1"] = j
            else:
                touches.append({"i0": i, "i1": j, "player": h["id"], "team": h["team"]})
    return touches


def _kick(frames, touch, k):
    """If the ball leaves this touch fast and towards a goal, return shot info, else None."""
    i1 = touch["i1"]
    if i1 + 1 + k >= len(frames):
        return None
    # measure the flight just AFTER the ball has left the player: the ball's position while it is still at his
    # feet is a poor starting point for estimating direction (a few cm of error becomes metres at the goal line)
    touch_ball, a, c = frames[i1]["ball"], frames[i1 + 1]["ball"], frames[i1 + 1 + k]["ball"]
    if not touch_ball or not a or not c:
        return None
    dt = frames[i1 + 1 + k]["t"] - frames[i1 + 1]["t"]
    vx, vy = (c["x"] - a["x"]) / dt, (c["y"] - a["y"]) / dt
    if math.hypot(vx, vy) < SHOT_SPEED or abs(vx) < 1e-6:
        return None
    goal_x = L if vx > 0 else 0.0
    t_line = (goal_x - a["x"]) / vx
    if t_line <= 0 or t_line > 3 or abs(goal_x - a["x"]) > SHOT_MAX_DIST:
        return None
    y_at = a["y"] + vy * t_line
    if abs(y_at - W / 2) > GOAL_HALF + SHOT_Y_TOL:
        return None
    return {"side": 1 if vx > 0 else -1, "target_y": round(y_at, 2),
            "speed": round(math.hypot(vx, vy), 1), "x": touch_ball["x"], "y": touch_ball["y"]}


def _chain_start(frames, touches, idx):
    """Index of the first touch of the possession chain that ends at touches[idx]."""
    j = idx
    while (j > 0 and touches[j - 1]["team"] == touches[idx]["team"]
           and frames[touches[j]["i0"]]["t"] - frames[touches[j - 1]["i1"]]["t"] <= DEAD_BALL_S):
        j -= 1
    return j


def _ev(frames, i, etype, team, player, **data):
    return {"type": etype, "frame": frames[i]["f"], "t": round(frames[i]["t"], 2),
            "team": team, "player": player, "data": data}


def detect(tracks):
    frames = tracks["frames"]
    dt = (frames[1]["t"] - frames[0]["t"]) if len(frames) > 1 else 0.1
    k = max(1, round(WINDOW_S / dt))   # frames in the velocity window
    touches = find_touches(frames, k)
    events = []

    # passes and turnovers: consecutive touches with the ball in play between them
    for a, b in zip(touches, touches[1:]):
        if frames[b["i0"]]["t"] - frames[a["i1"]]["t"] > DEAD_BALL_S:
            continue
        ba, bb = frames[a["i1"]]["ball"], frames[b["i0"]]["ball"]
        if a["team"] == b["team"] and a["player"] != b["player"]:
            events.append(_ev(frames, a["i1"], "pass", a["team"], a["player"], to=b["player"],
                              frm_xy=[ba["x"], ba["y"]], to_xy=[bb["x"], bb["y"]]))
        elif a["team"] != b["team"]:
            events.append(_ev(frames, a["i1"], "turnover", a["team"], a["player"], to=b["player"]))

    # shots
    shot_by_touch = {}
    for ti, tch in enumerate(touches):
        kick = _kick(frames, tch, k)
        if kick:
            e = _ev(frames, tch["i1"], "shot", tch["team"], tch["player"], goal=False, **kick)
            events.append(e)
            shot_by_touch[ti] = e

    # goals: ball seen beyond the goal line inside the posts
    last_goal_t = -1e9
    for i, fr in enumerate(frames):
        b = fr["ball"]
        if not b or not (b["x"] > L or b["x"] < 0):
            continue
        if fr["t"] - last_goal_t < GOAL_COOLDOWN_S:
            continue
        side = 1 if b["x"] > L else -1
        # where did the ball cross the line? interpolate from the last in-play frame (falls back to this frame's y)
        goal_x, y_cross, a = (L if side > 0 else 0.0), b["y"], frames[i - 1]["ball"] if i > 0 else None
        if a and not (0 <= a["x"] <= L):
            continue  # already past the line a frame ago: the crossing was judged then (a miss that skims the line stays a miss)
        if a and abs(b["x"] - a["x"]) > 1e-6:
            y_cross = a["y"] + (b["y"] - a["y"]) * (goal_x - a["x"]) / (b["x"] - a["x"])
        if abs(y_cross - W / 2) > GOAL_HALF:
            continue
        prev = [ti for ti, t in enumerate(touches) if t["i1"] < i and fr["t"] - frames[t["i1"]]["t"] <= DEAD_BALL_S]
        if not prev:
            continue
        ti = prev[-1]
        last_goal_t = fr["t"]
        tch = touches[ti]
        start = _chain_start(frames, touches, ti)
        chain = touches[start:ti + 1]
        team = tch["team"]
        if ti in shot_by_touch:
            shot_by_touch[ti]["data"]["goal"] = True
        else:  # e.g. a deflection or header the speed rule missed -> still a shot on target
            events.append(_ev(frames, tch["i1"], "shot", team, tch["player"], goal=True, side=side))

        # player positions at the goal frame, mirrored so the scoring team always attacks towards x = L
        flip = (lambda p: [round(L - p["x"], 1), round(W - p["y"], 1)]) if side < 0 else (lambda p: [round(p["x"], 1), round(p["y"], 1)])
        bx, by = (L - b["x"], W - b["y"]) if side < 0 else (b["x"], b["y"])
        events.append(_ev(frames, i, "goal", team, tch["player"], side=side,
                          scorer_touches=sum(1 for c in chain if c["player"] == tch["player"]),
                          buildup_passes=len(chain) - 1,
                          ball=[round(min(bx, L), 1), round(by, 1)],
                          attackers=[flip(p) for p in fr["players"] if p["team"] == team],
                          defenders=[flip(p) for p in fr["players"] if p["team"] != team and p["team"] >= 0]))

    return sorted(events, key=lambda e: e["frame"])
