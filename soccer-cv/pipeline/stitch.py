"""Clean up tracker output, working on whole tracks rather than one frame at a time.

Every step uses pitch metres and seconds, and only needs numpy, so it's all testable without any video.
A Tracklet is one tracker id: `frames` (sorted ints), `xy` ((n,2) metres). Steps, in the order track.py runs them:

  remove_duplicates   two boxes on one person give two ids that move as one: drop the shorter
  stitch              the tracker lost someone and gave them a new id: join the two when only one explanation fits
  smooth_and_fill     steady the feet (the box bottom jitters), and carry the id across short gaps
"""
import numpy as np

MAX_SPEED = 9.0        # m/s a player could cover while out of sight (a sprint is about 9-10)
REACH_SLACK = 2.0      # m on top of that: feet positions are noisy, more so at the far side
MAX_GAP_S = 6.0        # s: longer than this and too much could have happened
AMBIGUITY = 1.6        # the best candidate must be this much better than the next, or it's a coin toss: don't join
DUPLICATE_M = 1.0      # m: two tracks this close ...
DUPLICATE_SHARE = 0.8  # ... for this share of the shorter one's life are one person
SMOOTH_S = 0.2         # s: width of the median filter on feet positions
FILL_S = 1.0           # s: gaps up to this are filled in by straight line, keeping the id in every frame


class Tracklet:
    def __init__(self, tid, frames, xy, samples=(), sample_frames=()):
        order = np.argsort(frames, kind="stable")
        self.id = tid
        self.frames = np.asarray(frames, dtype=np.int64)[order]
        self.xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)[order]
        self.samples = list(samples)                 # shirt samples (pipeline.teams.shirt_feature) ...
        self.sample_frames = list(sample_frames)     # ... and the frame each was taken in
        self.team = -1
        self.role = -1
        self.merged_from = [tid]

    def split(self, at_frames, new_ids):
        """Cut into pieces at the given frames -> list of Tracklets; the first keeps this id, later ones take new_ids."""
        bounds = [self.start] + sorted(at_frames) + [self.end + 1]
        out = []
        for k, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
            sel = (self.frames >= lo) & (self.frames < hi)
            if not sel.any():
                continue
            ss = [i for i, f in enumerate(self.sample_frames) if lo <= f < hi]
            out.append(Tracklet(self.id if not out else next(new_ids), self.frames[sel], self.xy[sel],
                                [self.samples[i] for i in ss], [self.sample_frames[i] for i in ss]))
        return out

    def __len__(self):
        return len(self.frames)

    @property
    def start(self):
        return int(self.frames[0])

    @property
    def end(self):
        return int(self.frames[-1])

    def velocity_at_end(self, fps, tail_s=0.5, end=True):
        """Metres per second over the last (or, with end=False, first) half second, for projecting across a gap."""
        k = min(len(self), max(2, int(tail_s * fps)))
        if k < 2:
            return np.zeros(2)
        a, b = (self.frames[-k], self.frames[-1]) if end else (self.frames[0], self.frames[k - 1])
        pa, pb = (self.xy[-k], self.xy[-1]) if end else (self.xy[0], self.xy[k - 1])
        return (pb - pa) / max((b - a) / fps, 1e-6)


def remove_duplicates(tracks, near_m=DUPLICATE_M, share=DUPLICATE_SHARE):
    """Drop any track that spends most of its life on top of a longer one."""
    ordered = sorted(tracks, key=len, reverse=True)
    kept = []
    for t in ordered:
        dup = False
        for k in kept:
            if k.end < t.start or k.start > t.end:      # never on screen together: cheap to rule out
                continue
            common, ia, ib = np.intersect1d(t.frames, k.frames, return_indices=True)
            if not len(common) or len(common) < share * len(t):
                continue
            close = np.linalg.norm(t.xy[ia] - k.xy[ib], axis=1) < near_m
            if close.sum() >= share * len(t):
                dup = True
                break
        if not dup:
            kept.append(t)
    return kept


def _cost(a, b, fps, max_speed, slack, max_gap_s):
    """How awkward "b is a, re-found" is: 0 is perfect, 1 is at the limit of possible; None if impossible."""
    if b.start <= a.end or (a.team != b.team and a.team != -1 and b.team != -1):
        return None
    gap = (b.start - a.end) / fps
    if gap > max_gap_s:
        return None
    reach = max_speed * gap + slack
    dist = float(np.linalg.norm(b.xy[0] - a.xy[-1]))
    if dist > reach:
        return None
    # where a was heading, kept up for no more than one second: nobody runs in a straight line for long
    predicted = a.xy[-1] + a.velocity_at_end(fps) * min(gap, 1.0)
    miss = float(np.linalg.norm(b.xy[0] - predicted))
    return min(dist, miss) / reach + 0.02 * gap


def stitch(tracks, fps, max_speed=MAX_SPEED, slack=REACH_SLACK, max_gap_s=MAX_GAP_S, ambiguity=AMBIGUITY):
    """Join tracks that are one person -> new list. Greedy, cheapest join first. A join needs to be clear-cut both ways:
    no other track a might have become, and no other track b might have been, may come close to it in cost. A missed
    join only costs an extra id, whereas a wrong join silently hands one player's stats to another.
    Only pairs that could follow each other within max_gap_s are ever compared, and after each join only the new track's
    pairs are worked out again, so this stays quick on a 45-minute half with thousands of tracks."""
    alive = dict(enumerate(tracks))
    nxt = len(tracks)
    out_of, into = {}, {}          # i -> {j: cost} (i could become j), j -> {i: cost}

    def add_pairs(i):
        a = alive[i]
        for j, b in alive.items():
            if j == i:
                continue
            for p, q in ((i, j), (j, i)):
                c = _cost(alive[p], alive[q], fps, max_speed, slack, max_gap_s)
                if c is not None:
                    out_of.setdefault(p, {})[q] = c
                    into.setdefault(q, {})[p] = c

    # first pass: compare only tracks whose end and start lie within max_gap_s, using start-sorted order
    order = sorted(alive, key=lambda k: alive[k].start)
    starts = np.array([alive[k].start for k in order])
    for i, a in alive.items():
        lo = np.searchsorted(starts, a.end, side="right")
        hi = np.searchsorted(starts, a.end + max_gap_s * fps, side="right")
        for j in order[lo:hi]:
            c = _cost(a, alive[j], fps, max_speed, slack, max_gap_s)
            if c is not None:
                out_of.setdefault(i, {})[j] = c
                into.setdefault(j, {})[i] = c

    def drop(k):
        for q in out_of.pop(k, {}):
            into.get(q, {}).pop(k, None)
        for p in into.pop(k, {}):
            out_of.get(p, {}).pop(k, None)

    while True:
        chosen = None
        pairs = sorted(((c, i, j) for i, d in out_of.items() for j, c in d.items()))
        for c, i, j in pairs:
            rivals = [v for q, v in out_of.get(i, {}).items() if q != j] + [v for p, v in into.get(j, {}).items() if p != i]
            if all(v >= ambiguity * max(c, 0.05) for v in rivals):
                chosen = (i, j)
                break
        if chosen is None:
            return list(alive.values())
        i, j = chosen
        a, b = alive[i], alive[j]
        merged = Tracklet(a.id, np.r_[a.frames, b.frames], np.r_[a.xy, b.xy], a.samples + b.samples,
                          a.sample_frames + b.sample_frames)
        merged.team = a.team if a.team != -1 else b.team
        merged.role = a.role if a.role != -1 else b.role
        merged.merged_from = a.merged_from + b.merged_from
        drop(i)
        drop(j)
        del alive[i], alive[j]
        alive[nxt] = merged
        add_pairs(nxt)
        nxt += 1


def smooth_and_fill(t, fps, smooth_s=SMOOTH_S, fill_s=FILL_S):
    """-> (frames, xy, was_seen): median-filtered positions, with gaps up to fill_s filled in along a straight line."""
    k = max(1, int(smooth_s * fps)) | 1
    if len(t) >= k and k > 1:
        pad = k // 2
        idx = np.clip(np.arange(-pad, len(t) + pad), 0, len(t) - 1)
        win = np.lib.stride_tricks.sliding_window_view(t.xy[idx], k, axis=0)
        xy = np.median(win, axis=2)
    else:
        xy = t.xy
    limit = fill_s * fps
    frames, out, seen = [t.frames[:1]], [xy[:1]], [np.ones(1, bool)]
    for i in range(1, len(t)):
        gap = t.frames[i] - t.frames[i - 1]
        if 1 < gap <= limit:
            fill = np.arange(t.frames[i - 1] + 1, t.frames[i])
            w = ((fill - t.frames[i - 1]) / gap)[:, None]
            frames.append(fill)
            out.append(xy[i - 1] + w * (xy[i] - xy[i - 1]))
            seen.append(np.zeros(len(fill), bool))
        frames.append(t.frames[i:i + 1])
        out.append(xy[i:i + 1])
        seen.append(np.ones(1, bool))
    return np.concatenate(frames), np.concatenate(out), np.concatenate(seen)


def is_sideline_bystander(t, width, fps, near_m=2.5, still_m=1.5, min_s=2.0):
    """A coach, substitute or ball-boy: lives right on a touchline and hardly moves. Players on the line keep moving."""
    if len(t) < min_s * fps:
        return False
    y = float(np.median(t.xy[:, 1]))
    on_line = min(abs(y - (34 + width / 2)), abs(y - (34 - width / 2))) < near_m
    spread = np.percentile(np.linalg.norm(t.xy - np.median(t.xy, axis=0), axis=1), 90)
    return on_line and spread < still_m


def on_pitch_share(t, width, length, margin):
    """Share of a track's frames with the feet on the pitch, or within `margin` metres of it."""
    near, far, goal = 34 + width / 2, 34 - width / 2, 105 - length
    x, y = t.xy[:, 0], t.xy[:, 1]
    return float(np.mean((x >= goal - margin) & (x <= 105 + margin) & (y >= far - margin) & (y <= near + margin)))
