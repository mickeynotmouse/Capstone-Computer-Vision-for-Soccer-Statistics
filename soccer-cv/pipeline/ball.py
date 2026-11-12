"""Pick the one real ball out of noisy per-frame detections, using the whole clip.

The detector sees the ball in many frames, and also sees "balls" that aren't: a white boot, a head, a marker disc, and the
spare balls that sit beside the pitch all game. Taken a frame at a time those can't be told apart. Over time they can:
  * anything that stays put for seconds is set aside (on real footage these were spare balls by the corner flag and the
    stands, and a disc on the pitch). The cost: a ball resting before a free kick is only picked up once it's kicked.
  * the real ball forms a physically possible path. This finds the path through the remaining candidates with the highest
    total score (dynamic programming over a graph of candidates), then fills in short gaps along a straight line.
Longer gaps stay empty: no ball is better than an invented one.

Everything here is in pitch metres and seconds, with candidates as (frame, x, y, conf), so it needs no video to test.
"""
import numpy as np

MAX_BALL_SPEED = 40.0  # m/s: about 90 mph, faster than any shot at this level (a ball in the air appears to move faster)
SLACK_M = 2.0          # m: position noise, more at the far side
LINK_GAP_S = 0.8       # s: furthest two detections can be apart and still count as the same flight
JUMP_PENALTY = 1.5     # score given up to join two detections with no possible flight between them (worth ~5 clear sightings)
MIN_JUMP_S = 0.3       # ... and only after the ball has been out of sight this long: never a frame-to-frame teleport
GAP_PENALTY = 0.02     # score given up for each frame without a detection
MOVE_PENALTY = 0.15    # score given up for flying at top speed: of two possible paths, prefer the less frantic one
BASE_REWARD = 0.05     # subtracted from conf, so a very weak lone detection isn't worth a detour
STATIC_M = 1.0         # m: candidates this close together ...
STATIC_S = 2.0         # ... over at least this long ...
STATIC_SHARE = 0.25    # ... seen at least this share of that time, are something lying still, and are set aside
FILL_GAP_S = 0.5       # s: gaps up to this are filled in along a straight line


def static_mask(cands, fps):
    """True for each candidate that is part of something lying still (see STATIC_*). Candidates are bucketed into a grid
    of STATIC_M cells so only neighbours are compared: linear in the number of candidates, so a 45-minute half is fine."""
    cands = np.asarray(cands, dtype=np.float64).reshape(-1, 4)
    out = np.zeros(len(cands), bool)
    if not len(cands):
        return out
    xy = cands[:, 1:3]
    cell = np.floor(xy / STATIC_M).astype(np.int64)
    buckets = {}
    for i, (cx, cy) in enumerate(cell):
        buckets.setdefault((cx, cy), []).append(i)
    rate = _detection_rate(cands[:, 0], fps)
    for i in range(len(cands)):
        if out[i]:
            continue
        cx, cy = cell[i]
        idx = np.array([j for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j in buckets.get((cx + dx, cy + dy), ())])
        near = idx[np.linalg.norm(xy[idx] - xy[i], axis=1) < STATIC_M]
        fr = cands[near, 0]
        span = (fr.max() - fr.min()) / fps
        if span >= STATIC_S and len(np.unique(fr)) >= STATIC_SHARE * span * rate:
            out[near] = True
    return out


def _detection_rate(frames, fps):
    """Frames per second that were actually looked at (fps, or less when only every other frame is detected)."""
    u = np.unique(frames)
    if len(u) < 2:
        return fps
    step = np.min(np.diff(u))
    return fps / max(step, 1)


def best_path(cands, fps):
    """cands (k,4) [frame, x, y, conf] -> indices of the candidates on the best path, in frame order (at most 1 per frame)."""
    cands = np.asarray(cands, dtype=np.float64).reshape(-1, 4)
    moving = np.flatnonzero(~static_mask(cands, fps))
    if not len(moving):
        return []
    order = moving[np.argsort(cands[moving, 0], kind="stable")]
    c = cands[order]
    reward = c[:, 3] - BASE_REWARD
    n = len(c)
    best, prev = reward.copy(), np.full(n, -1)
    frames = c[:, 0]
    # the best score of any path ending at or before each candidate, for long jumps
    top_score, top_idx = -np.inf, -1
    j_done = 0
    for i in range(n):
        # strictly earlier frames only, so a path can never loop back on itself, whatever MIN_JUMP_S is set to
        while j_done < n and frames[j_done] < frames[i] and frames[j_done] <= frames[i] - MIN_JUMP_S * fps:
            if best[j_done] > top_score:
                top_score, top_idx = best[j_done], j_done
            j_done += 1
        if top_idx >= 0 and top_score - JUMP_PENALTY + reward[i] > best[i]:
            best[i], prev[i] = top_score - JUMP_PENALTY + reward[i], top_idx
        # flights: earlier candidates within LINK_GAP_S
        lo = np.searchsorted(frames, frames[i] - LINK_GAP_S * fps, side="left")
        hi = np.searchsorted(frames, frames[i], side="left")
        if hi > lo:
            dt = (frames[i] - frames[lo:hi]) / fps
            dist = np.linalg.norm(c[lo:hi, 1:3] - c[i, 1:3], axis=1)
            limit = MAX_BALL_SPEED * dt + SLACK_M
            ok = dist <= limit
            if ok.any():
                link = best[lo:hi] - GAP_PENALTY * (frames[i] - frames[lo:hi] - 1) - MOVE_PENALTY * dist / limit + reward[i]
                link[~ok] = -np.inf
                k = int(np.argmax(link))
                if link[k] > best[i]:
                    best[i], prev[i] = link[k], lo + k
    path, i = [], int(np.argmax(best))
    while i >= 0:
        path.append(int(order[i]))
        i = int(prev[i])
    return path[::-1]


def ball_track(cands, total_frames, fps, fill_gap_s=FILL_GAP_S):
    """-> list of total_frames entries, each None or {"x", "y", "conf"}; "filled": True marks an estimated position."""
    cands = np.asarray(cands, dtype=np.float64).reshape(-1, 4)
    chosen = cands[best_path(cands, fps)] if len(cands) else np.zeros((0, 4))
    out = [None] * total_frames
    for f, x, y, conf in chosen:
        out[int(f)] = {"x": float(x), "y": float(y), "conf": float(conf)}
    limit = fill_gap_s * fps
    for (f0, x0, y0, _), (f1, x1, y1, _) in zip(chosen[:-1], chosen[1:]):
        gap = f1 - f0
        if 1 < gap <= limit and np.hypot(x1 - x0, y1 - y0) <= MAX_BALL_SPEED * gap / fps + SLACK_M:
            for k in range(1, int(gap)):
                w = k / gap
                out[int(f0) + k] = {"x": float(x0 + w * (x1 - x0)), "y": float(y0 + w * (y1 - y0)), "conf": 0.0, "filled": True}
    return out
