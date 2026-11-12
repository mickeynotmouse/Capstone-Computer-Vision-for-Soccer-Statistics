"""Snap a calibration onto the WHITE soccer lines.

This turf also carries red, orange and yellow lines for other sports, some of them running right beside the white
touchline. Clicks bunched at one goal leave the far end of the fit free to drift onto one of those. Here, each soccer line
is projected through the current fit, white paint is looked for either side of it, and the fit is redone with what's found.

Whiteness is the MINIMUM of the three colour channels, which is high only when all three are: white paint scores high,
whereas red, orange, yellow and green all have almost no blue. A coloured line therefore can't be picked up at all.
"""
import cv2
import numpy as np

from . import homography as hg

TOPHAT_PX = 41         # wider than the widest painted line in the frame (about 20 px in the foreground of 2880 px video)
MIN_WHITE = 28         # how much brighter than its surroundings (0-255) paint must be
MAX_LINE_PX = 34       # a bright run wider than this is not paint (a shirt, a bench)
SAMPLES = 60           # places to look along each line
END_GAP_M = 1.0        # keep away from where lines meet
RANSAC_PX = 3.0        # paint must sit this close to one straight image line to count
MIN_HITS = 6           # fewer found spots than this: the line is not used
MIN_HIT_RATIO = 0.25   # ... or fewer than this share of the visible samples
PASSES_M = (4.0, 2.0, 1.0, 0.5)   # search distance either side of the line, in pitch metres, narrowing each pass
ALIGN_PX = 4.0         # a stretch of line counts as "on the paint" if white is this close


def soccer_lines(width=68.0, length=105.0, both_halves=True):
    """(name, [a, b, c], p0, p1): the straight white lines, in pitch metres. The half with the goal at x = 105 comes first;
    with both_halves, the other half's lines follow (named "... (other end)"), for cameras that pan the whole field."""
    near, far, half = 34 + width / 2, 34 - width / 2, 105 - length / 2
    goal = 105 - length
    out = [
        ("near touchline", hg.line("y", near), (goal, near), (105, near)),
        ("far touchline", hg.line("y", far), (goal, far), (105, far)),
        ("goal line", hg.line("x", 105), (105, far), (105, near)),
        ("halfway line", hg.line("x", half), (half, far), (half, near)),
        ("18-yd front", hg.line("x", 88.5), (88.5, 13.84), (88.5, 54.16)),
        ("18-yd near side", hg.line("y", 54.16), (88.5, 54.16), (105, 54.16)),
        ("18-yd far side", hg.line("y", 13.84), (88.5, 13.84), (105, 13.84)),
        ("6-yd front", hg.line("x", 99.5), (99.5, 24.84), (99.5, 43.16)),
    ]
    if both_halves:
        mirror = lambda x: 2 * half - x     # noqa: E731   reflects x across the halfway line
        for name, (a, b, c), p0, p1 in out[2:]:
            if name == "halfway line":
                continue
            q0, q1 = (mirror(p0[0]), p0[1]), (mirror(p1[0]), p1[1])
            abc = hg.line("x", q0[0]) if a else hg.line("y", q0[1])
            out.append((name + " (other end)", abc, q0, q1))
    return out


def whiteness(frame):
    """Per-pixel 'white paint' score: min(B, G, R), minus the local background so only thin bright things remain."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_PX, TOPHAT_PX))
    return cv2.morphologyEx(frame.min(axis=2), cv2.MORPH_TOPHAT, k)


def _peaks(profile):
    """Centre of each bright run along a whiteness profile, as (index, peak height), brightest first."""
    runs, start = [], None
    on = profile >= MIN_WHITE
    for i, b in enumerate(np.r_[on, False]):
        if b and start is None:
            start = i
        elif not b and start is not None:
            if i - start <= MAX_LINE_PX:
                runs.append(((start + i - 1) / 2, float(profile[start:i].max())))
            start = None
    return sorted(runs, key=lambda r: -r[1])


def _visible_samples(H, p0, p1, box, n=SAMPLES):
    """Points along the pitch segment p0-p1 that land in the picture -> (image points, pitch points, image normal)."""
    p0, p1 = np.float64(p0), np.float64(p1)
    length = np.linalg.norm(p1 - p0)
    t = np.linspace(END_GAP_M / length, 1 - END_GAP_M / length, n)[:, None]
    pit = p0 + t * (p1 - p0)
    Hinv = np.linalg.inv(H)
    q = np.c_[pit, np.ones(n)] @ Hinv.T
    # the side of the horizon holding the picture: points past it would project backwards
    w_ref = (np.array([(box[0] + box[2]) / 2, box[3], 1.0]) @ H.T)[2]
    front = np.sign(q[:, 2]) == np.sign(1 / w_ref) if w_ref else np.ones(n, bool)
    img = q[:, :2] / q[:, 2:]
    ok = front & (img[:, 0] >= box[0] + 8) & (img[:, 0] < box[2] - 8) & (img[:, 1] >= box[1] + 8) & (img[:, 1] < box[3] - 8)
    return img[ok], pit[ok]


def _search_radius_px(H, pit, a_b, metres):
    """How many image pixels `metres` across the line comes to, at each sample."""
    Hinv = np.linalg.inv(H)
    n = np.float64(a_b)[:2]
    d = hg.to_pitch(Hinv, pit + metres * n) - hg.to_pitch(Hinv, pit)
    return np.linalg.norm(d, axis=1)


def _find_paint(white, H, seg, box, metres, rng):
    """White spots along one projected line that agree on a single straight image line -> (k,2) pixels, or None."""
    _, abc, p0, p1 = seg
    img, pit = _visible_samples(H, p0, p1, box)
    if len(img) < MIN_HITS:
        return None
    u = img[-1] - img[0]
    if np.linalg.norm(u) < 40:
        return None
    u /= np.linalg.norm(u)
    normal = np.array([-u[1], u[0]])
    radii = np.clip(_search_radius_px(H, pit, abc, metres), 3, 400)

    cands = []        # (x, y, sample index)
    for k, (c, r) in enumerate(zip(img, radii)):
        s = np.arange(-r, r + 1)
        xs, ys = c[0] + s * normal[0], c[1] + s * normal[1]
        prof = cv2.remap(white, np.float32(xs).reshape(-1, 1), np.float32(ys).reshape(-1, 1), cv2.INTER_LINEAR).ravel()
        for idx, _ in _peaks(prof)[:3]:
            cands.append((c[0] + (idx - r) * normal[0], c[1] + (idx - r) * normal[1], k))
    if len(cands) < MIN_HITS:
        return None
    pts = np.float64(cands)

    best = None
    for _ in range(300):        # the straight line most found spots agree on: legs and shirts won't line up
        a, b = pts[rng.choice(len(pts), 2, replace=False), :2]
        d = b - a
        if np.linalg.norm(d) < 60:
            continue
        nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        inl = np.abs((pts[:, :2] - a) @ nrm) < RANSAC_PX
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None:
        return None
    # at most one spot per sample, so a double-painted patch can't outvote the rest of the line
    keep = {}
    for x, y, k in pts[best]:
        keep.setdefault(int(k), (x, y))
    if len(keep) < max(MIN_HITS, MIN_HIT_RATIO * len(img)):
        return None
    return np.float64(list(keep.values()))


def refine(frame, obs, H, width, length, box=None, seed=0, keep_hand_lines=True):
    """Re-fit H using white paint along every soccer line it can find -> (H, Obs including the new line spots, report).
    Clicked points stay in as they pin down position along the lines; report = {line name: spots used}, and is empty if
    nothing beat the fit it started with (which is then returned unchanged)."""
    box = box or (0, 0, frame.shape[1], frame.shape[0])
    white = whiteness(frame)
    rng = np.random.default_rng(seed)
    base = obs if keep_hand_lines else hg.Obs(obs.pts, obs.pit)
    best = (_score(alignment(frame, H, width, length, box, white=white)), H, obs, {})
    for metres in PASSES_M:
        lpx, lco, used = [], [], {}
        for seg in soccer_lines(width, length):
            spots = _find_paint(white, best[1], seg, box, metres, rng)
            if spots is not None:
                lpx += spots.tolist()
                lco += [seg[1]] * len(spots)
                used[seg[0]] = len(spots)
        if not lpx:
            continue
        full = hg.Obs(base.pts, base.pit, np.r_[base.lpx, np.float64(lpx)], np.r_[base.lco, np.float64(lco)])
        H2 = hg.fit(full, init=best[1])
        if H2 is None:
            continue
        s = _score(alignment(frame, H2, width, length, box, white=white))
        if s > best[0]:      # a pass that latched onto the wrong paint scores lower, and is thrown away
            best = (s, H2, full, used)
    return best[1], best[2], best[3]


def _score(a):
    """One number for a fit: the average share of each visible line that sits on white paint, within ALIGN_PX."""
    return float(np.mean([share for share, _ in a.values()])) if a else 0.0


def alignment(frame, H, width, length, box=None, tol_px=ALIGN_PX, white=None):
    """How well the fit sits on white paint: for each line, the share of its visible length with white within tol_px of
    the projected line, and the median offset to it. {line name: (share, median px)}. Lines out of shot are left out."""
    box = box or (0, 0, frame.shape[1], frame.shape[0])
    white = whiteness(frame) if white is None else white
    out = {}
    for name, abc, p0, p1 in soccer_lines(width, length):
        img, pit = _visible_samples(H, p0, p1, box)
        if len(img) < MIN_HITS or np.linalg.norm(img[-1] - img[0]) < 40:
            continue
        u = (img[-1] - img[0]) / np.linalg.norm(img[-1] - img[0])
        normal = np.array([-u[1], u[0]])
        R = 60
        s = np.arange(-R, R + 1)
        offs = []
        for c in img:
            prof = cv2.remap(white, np.float32(c[0] + s * normal[0]).reshape(-1, 1),
                             np.float32(c[1] + s * normal[1]).reshape(-1, 1), cv2.INTER_LINEAR).ravel()
            pk = _peaks(prof)
            offs.append(min((abs(i - R) for i, _ in pk), default=np.inf))
        offs = np.array(offs)
        found = offs[np.isfinite(offs)]
        out[name] = (float(np.mean(offs <= tol_px)), float(np.median(found)) if len(found) else float("nan"))
    return out


def snap(white, x, y, radius=14.0):
    """The white-paint ridge nearest a click, or None if there's no paint within `radius` pixels."""
    x0, y0 = int(max(x - radius, 0)), int(max(y - radius, 0))
    win = white[y0:int(y + radius) + 1, x0:int(x + radius) + 1]
    if win.size == 0:
        return None
    blur = cv2.GaussianBlur(win, (0, 0), 2.0)
    if blur.max() < MIN_WHITE * 0.7:
        return None
    gy, gx = np.mgrid[0:win.shape[0], 0:win.shape[1]]
    score = blur.astype(np.float64) - 0.25 * np.hypot(gx + x0 - x, gy + y0 - y)   # the ridge, with a slight pull to the click
    j, i = np.unravel_index(np.argmax(score), score.shape)
    return float(i + x0), float(j + y0)
