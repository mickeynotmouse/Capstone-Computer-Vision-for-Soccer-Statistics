"""Synthetic match generator: lets the whole pipeline (events -> DB -> API -> dashboard) be developed and tested
without any video. It emits tracks in the same format as pipeline.track and returns the ground truth
(everything it injected), so the event detector can be tested against known answers.

NOTHING HERE IS REAL MATCH DATA. Scores, team names and dates are invented; positions are random-ish.
"""
import math
import random

L, W = 105.0, 68.0
# 4-4-2 for team 0 (attacks towards +x); team 1 is the mirror image and attacks towards -x
FORMATION = [(5, 34), (25, 10), (25, 26), (25, 42), (25, 58), (50, 10), (50, 26), (50, 42), (50, 58), (75, 24), (75, 44)]
SHOT_V, PASS_V = 24.0, 10.0  # m/s
MIN_SEP = 8.0                # m: players are nudged apart so nobody stands on top of someone else
                             # (if two players are 3 m apart, "who touched it" is ambiguous even for a human)


def _home(team, k):
    x, y = FORMATION[k]
    return (x, y) if team == 0 else (L - x, W - y)


def pid(team, k):
    return k + 1 + 100 * team


def _turn_deg(u, v):
    nu, nv = math.hypot(*u), math.hypot(*v)
    if nu < 1e-6 or nv < 1e-6:
        return 0.0
    c = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / (nu * nv)))
    return math.degrees(math.acos(c))


def make_game(seed, outcomes, fps=10, chaos=8.0, one_touch_prob=0.35):
    """outcomes: list of (team, 'goal' | 'wide' | 'lost') -- one possession sequence each.

    chaos          : std-dev (m) of how far players drift from the neat 4-4-2 for each sequence (0 = textbook shape)
    one_touch_prob : chance that an intermediate receiver plays the ball first-time instead of controlling it
    """
    rng = random.Random(seed)
    home = {(t, k): _home(t, k) for t in (0, 1) for k in range(11)}
    base = {key: list(h) for key, h in home.items()}   # where each player "wants" to be right now
    pos = {key: list(h) for key, h in home.items()}
    frames = []
    truth = {"goals": [], "shots": {0: 0, 1: 0}, "passes": {0: 0, 1: 0}, "turnovers": {0: 0, 1: 0},
             "one_touch": 0, "sequences": []}

    def step(ball):
        bx = ball[0] if ball else L / 2
        dx = max(-15, min(15, (bx - L / 2) * 0.3))
        for key, p in pos.items():
            tx = min(max(base[key][0] + dx * (0.2 if key[1] == 0 else 1.0), 2), L - 2)
            ty = min(max(base[key][1], 2), W - 2)
            p[0] += (tx - p[0]) * 0.15 + rng.gauss(0, 0.12)
            p[1] += (ty - p[1]) * 0.15 + rng.gauss(0, 0.12)
        n = len(frames)
        frames.append({
            "f": n, "t": n / fps,
            "players": [{"id": pid(t, k), "x": round(p[0], 2), "y": round(p[1], 2), "team": t} for (t, k), p in pos.items()],
            "ball": {"x": round(ball[0], 2), "y": round(ball[1], 2), "conf": 0.9} if ball else None,
        })

    def reshuffle(team, shooter_k):
        """New 'situation': everyone drifts away from the formation; the shooter ends up somewhere he can shoot from."""
        if chaos <= 0:
            return
        for key, (hx, hy) in home.items():
            base[key] = [hx + rng.gauss(0, chaos), hy + rng.gauss(0, chaos)]
        if shooter_k is not None:
            want_x = rng.uniform(72, 86)   # 19-33 m from goal (the team shifts up the pitch with the ball, hence not closer)
            base[(team, shooter_k)] = [want_x if team == 0 else L - want_x, rng.uniform(14, 54)]
        keys = list(base)
        for _ in range(40):  # push apart anyone closer than MIN_SEP
            moved = False
            for i, a in enumerate(keys):
                for b in keys[i + 1:]:
                    ddx, ddy = base[a][0] - base[b][0], base[a][1] - base[b][1]
                    dist = math.hypot(ddx, ddy)
                    if dist < MIN_SEP:
                        if dist < 1e-6:
                            ddx, ddy, dist = 1.0, 0.0, 1.0
                        push = (MIN_SEP - dist) / 2
                        ux, uy = ddx / dist, ddy / dist
                        base[a][0] += ux * push
                        base[a][1] += uy * push
                        base[b][0] -= ux * push
                        base[b][1] -= uy * push
                        moved = True
            if not moved:
                break

    def hold(key, d):
        for _ in range(rng.randint(6, 8)):
            p = pos[key]
            step((p[0] + 0.4 * d, p[1]))

    def fly(src, dst_key, speed=None):
        dist = math.hypot(pos[dst_key][0] - src[0], pos[dst_key][1] - src[1])
        if speed is None:  # long balls travel faster so they never exceed the 5 s dead-ball gap
            speed = min(15.5, max(PASS_V, dist / 4.0))
        n = max(4, int(dist / (speed / fps)))
        for i in range(1, n + 1):
            dst, f = pos[dst_key], i / n
            step((src[0] + (dst[0] - src[0]) * f, src[1] + (dst[1] - src[1]) * f))

    for seq_no, (team, outcome) in enumerate(outcomes, 1):
        d = 1 if team == 0 else -1
        goal_x = L if d == 1 else 0.0
        n_passes = rng.randint(2, 7)
        attackers = list(range(5, 11))
        holders = [rng.choice(attackers)]
        for _ in range(n_passes if outcome != "lost" else n_passes - 1):
            holders.append(rng.choice([k for k in attackers if k != holders[-1]]))
        if outcome != "lost":  # the chain always ends with a forward who shoots
            while holders[-1] not in (9, 10) or holders[-1] == holders[-2]:
                holders[-1] = rng.choice([9, 10])

        reshuffle(team, holders[-1] if outcome != "lost" else None)
        for _ in range(40):          # let everyone walk to their new spots before the ball is in play
            step(None)

        seq_start = len(frames)
        hold((team, holders[0]), d)
        completed, quick_count, prev_quick = 0, 0, False
        for idx in range(len(holders) - 1):
            a, b = holders[idx], holders[idx + 1]
            p = pos[(team, a)]
            src = (p[0], p[1]) if prev_quick else (p[0] + 0.4 * d, p[1])
            fly(src, (team, b))
            completed += 1
            truth["passes"][team] += 1
            # first-time pass: the receiver redirects the ball without controlling it (needs a real change of direction)
            prev_quick = False
            if idx + 1 < len(holders) - 1 and rng.random() < one_touch_prob:
                bp, nxt = pos[(team, b)], pos[(team, holders[idx + 2])]
                prev_quick = _turn_deg((bp[0] - src[0], bp[1] - src[1]), (nxt[0] - bp[0], nxt[1] - bp[1])) >= 40
            if prev_quick:
                quick_count += 1
            else:
                hold((team, b), d)
        truth["one_touch"] += quick_count

        player = None
        if outcome == "lost":
            p = pos[(team, holders[-1])]
            dists = {k: math.hypot(pos[(1 - team, k)][0] - p[0], pos[(1 - team, k)][1] - p[1]) for k in range(1, 11)}
            near = [k for k, v in dists.items() if 8 < v < 30] or [min(dists, key=dists.get)]
            opp = (1 - team, rng.choice(near))
            fly((p[0] + 0.4 * d, p[1]), opp, PASS_V)
            hold(opp, -d)
            truth["turnovers"][team] += 1
        else:
            shooter = (team, holders[-1])
            player = pid(*shooter)
            p = pos[shooter]
            src = (p[0] + 0.4 * d, p[1])
            # y where the ball crosses the goal line: inside the posts for a goal, 2-3 m outside them for a miss
            y_cross = 34 + rng.uniform(-3, 3) if outcome == "goal" else 34 + rng.choice([-1, 1]) * rng.uniform(5.8, 6.8)
            dxc = abs(goal_x - src[0])
            tgt = (goal_x + 2 * d, src[1] + (y_cross - src[1]) * (dxc + 2) / dxc)  # same straight line, 2 m past the line
            n = math.ceil(math.hypot(tgt[0] - src[0], tgt[1] - src[1]) / (SHOT_V / fps))
            for i in range(1, n + 1):
                f = i / n
                step((src[0] + (tgt[0] - src[0]) * f, src[1] + (tgt[1] - src[1]) * f))
            for _ in range(4):
                step(tgt)
            truth["shots"][team] += 1
            if outcome == "goal":
                truth["goals"].append({"team": team, "scorer": player, "buildup_passes": len(holders) - 1,
                                       "scorer_touches": holders.count(holders[-1])})
        truth["sequences"].append({"seq": seq_no, "team": team, "outcome": outcome, "passes": completed,
                                   "one_touch_passes": quick_count, "player": player,
                                   "start_frame": seq_start, "end_frame": len(frames) - 1})
        for _ in range(80):  # dead ball: ball not visible for 8 s
            step(None)

    return {"fps": fps, "frames": frames}, truth
