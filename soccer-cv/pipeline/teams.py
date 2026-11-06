"""Who is who, from shirts and position: team 0, team 1, goalkeeper, or official.

Works per TRACK, over many frames: one crop of a far-away player is a dozen blurry pixels, but the median over a track is
reliable. Measured on real footage, three things matter:
  * striped shirts average to mid-grey from a distance, as does a white shirt in shade. So the feature includes how DARK
    the darkest part of the shirt is (20th percentile of lightness): stripes have dark in them, plain white doesn't.
  * grass shows through the crop. Pixels near this clip's grass colour are removed first; even so, a blurred striped shirt
    keeps a green TINT (a and b move together, the way grass does), which must not be mistaken for a coloured shirt.
  * officials wear neon. The share of shirt pixels that are strongly yellow-green is 30-75 % for them and under 25 % for
    everyone else on the pitch, so that alone picks them out.

Anyone who is neither team's colour is a goalkeeper if they live in a penalty area, otherwise an official. A goalkeeper is
given to the team defending that goal (see `keeper_team`): it's position, not shirt colour, that says whose keeper he is.
"""
import cv2
import numpy as np

# torso, as a share of the person's box
TORSO_Y = (0.18, 0.50)
TORSO_X = (0.25, 0.75)
GRASS_DIST = 16.0        # Lab distance from the clip's grass colour within which a pixel counts as grass
MIN_SHIRT_PIXELS = 12
MIN_SAMPLES = 5          # a track with fewer shirt samples than this isn't given a role from them
NEON_B_SHARE = 0.7       # a pixel is "neon" if its yellowness b is at least this share of the grass's, and it is not red
NEON_MAX_A = 5.0         # ... (a above this): red and orange are yellowish too, but neon is yellow-GREEN
NEON_MEDIAN = 0.30       # ... and a track is an official/keeper candidate if this share of its shirt is neon (median).
                         #     Measured on two games: officials 0.33-1.0, players at most 0.24. (Looking at a track's
                         #     clearest frames instead wrongly picked out red players, whose shirts have yellowish pixels.)
NEON_SAMPLE = 0.40       # one shirt sample this neon counts as "other" when looking for a tracker swap
CHROMA_LIMIT = 9.0       # how far from its team's shirt colour (a, b in Lab) a shirt must be to be a third colour
SPREAD_LIMIT = 2.5       # ... or that many of the team's own spreads, whichever is larger
CHROMATIC = 12.0         # a team whose shirt colour (a, b) is at least this strong is a coloured kit (red, blue ...): ...
HUE_LIMIT_DEG = 35.0     # ... and a third colour is then a different HUE, not a stronger or weaker shade of the same one
                         #     (measured: red players vary up to 25 deg in hue with light; a pale-green keeper is 40-55 deg
                         #     from his light-blue team). White, black and striped kits use the plain distance (CHROMA_LIMIT).
TINT_SHARE = 0.6         # colour lying along the grass direction, at least this strongly, is grass tint, not shirt
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16   # penalty area, metres


def grass_colour(frame, box=None):
    """This clip's grass colour in Lab (OpenCV's 0-255 scale): the median green in the lower half of the picture."""
    x0, y0, x1, y1 = box or (0, 0, frame.shape[1], frame.shape[0])
    crop = frame[y0 + (y1 - y0) // 2:y1, x0:x1]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    green = lab[(lab[:, 1] < 118) & (lab[:, 2] > 135)]     # a < 0 and b > 0 in signed Lab
    return np.median(green if len(green) > 100 else lab, axis=0)


def shirt_feature(frame, box, grass):
    """[L median, L 20th percentile, a, b, neon share] of the shirt (a, b signed), or None if too little shirt is visible."""
    x1, y1, x2, y2 = (float(v) for v in box)
    h, w = y2 - y1, x2 - x1
    ya, yb = max(int(y1 + TORSO_Y[0] * h), 0), int(y1 + TORSO_Y[1] * h)
    xa, xb = max(int(x1 + TORSO_X[0] * w), 0), int(x1 + TORSO_X[1] * w)
    crop = frame[ya:yb, xa:xb]
    if crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    shirt = lab[np.linalg.norm(lab - grass, axis=1) > GRASS_DIST]
    if len(shirt) < MIN_SHIRT_PIXELS:
        return None
    L, a, b = shirt[:, 0], shirt[:, 1] - 128, shirt[:, 2] - 128
    neon = float(np.mean((b > NEON_B_SHARE * (grass[2] - 128)) & (a < NEON_MAX_A)))
    return [float(np.median(L)), float(np.percentile(L, 20)), float(np.median(a)), float(np.median(b)), neon]


def _chroma(ab, grass):
    """Colour of a shirt (a, b), with any grass tint taken out.
    Grass blended into a grey shirt adds yellowness b AND greenness a in the grass's own proportion. A yellow shirt has
    the yellowness without the greenness. So a shirt counts as tinted grey only if it is at least TINT_SHARE as green as
    grass would make it at that yellowness (measured: blurred stripes a=-7 at b=12; yellow keeper a=+1 at b=12)."""
    ga, gb = float(grass[1]) - 128, float(grass[2]) - 128
    ab = np.atleast_2d(np.float64(ab))
    out = ab.copy()
    if gb > 1 and ga < 0:
        tinted = (ab[:, 1] > 0) & (ab[:, 0] <= TINT_SHARE * ab[:, 1] * ga / gb)
        out[tinted] = 0.0      # a green-tinted grey shirt counts as grey
    return out


def _off_team(chroma, centre, limit):
    """How far outside its team's colour each shirt is, in units where above 1 means "a third colour".
    chroma: (k, 2) shirt (a, b); centre: (k, 2) the team colour each is compared with; limit: (k,) plain-distance limits."""
    chroma, centre = np.atleast_2d(np.float64(chroma)), np.atleast_2d(np.float64(centre))
    plain = np.linalg.norm(chroma - centre, axis=1) / limit
    strength = np.linalg.norm(centre, axis=1)
    ang = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(chroma[:, 1], chroma[:, 0])
                                                  - np.arctan2(centre[:, 1], centre[:, 0]))))))
    weak = np.linalg.norm(chroma, axis=1) < 0.4 * strength           # a grey shirt beside a coloured team
    hue = np.where(weak, 2.0, ang / HUE_LIMIT_DEG)
    return np.where(strength >= CHROMATIC, hue, plain)


def _two_means(data):
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    _, labels, centres = cv2.kmeans(np.float32(data), 2, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    return labels.ravel(), centres


def in_box(x, y, length=105.0):
    """True if a pitch point is inside either penalty area."""
    return abs(y - 34) < BOX_HALF_WIDTH and (abs(105 - x) < BOX_DEPTH or abs(x - (105 - length)) < BOX_DEPTH)


def _not_a_team(f):
    """Neon test on a track's samples (rows = samples)."""
    return np.median(np.float32(f)[:, 4]) >= NEON_MEDIAN


def assign(features, positions=None, grass=(0, 128, 128), length=105.0):
    """{track id: [shirt features]} -> ({track id: role}, colours); role is 0, 1, "gk", "official" or -1 (unknown).

    positions: {track id: (median x, median y)} in pitch metres, to tell goalkeepers from officials.
    colours: {"centres": {0: .., 1: ..}, "limit": ..}, for `sample_label`.
    """
    positions = positions or {}
    feats = {i: np.float32(f) for i, f in features.items() if len(f) >= MIN_SAMPLES}
    roles = {i: -1 for i in features}
    if len(feats) < 2:
        return roles, {}
    ids = list(feats)
    med = np.array([np.median(feats[i], axis=0) for i in ids])
    neon = np.array([_not_a_team(feats[i]) for i in ids])
    chroma = _chroma(med[:, 2:4], grass)
    data = np.c_[med[:, :2], chroma]

    keep, centres = ~neon, None
    for _ in range(4):
        if keep.sum() < 2:
            break
        _, centres = _two_means(data[keep])
        nearer = np.linalg.norm(data[:, None, :] - centres[None], axis=2).argmin(1)
        dist = np.linalg.norm(data[:, 2:] - centres[nearer][:, 2:], axis=1)
        limit = np.zeros(2)
        for t in (0, 1):
            own = dist[keep & (nearer == t)]
            mad = 1.4826 * np.median(np.abs(own - np.median(own))) if len(own) else 0.0
            limit[t] = max(CHROMA_LIMIT, (np.median(own) + SPREAD_LIMIT * mad) if len(own) else 0.0)
        off = _off_team(data[:, 2:], centres[nearer][:, 2:], limit[nearer])
        new_keep = ~neon & (off <= 1.0)
        if (new_keep == keep).all():
            break
        keep = new_keep
    if centres is None:
        return roles, {}

    nearer = np.linalg.norm(data[:, None, :] - centres[None], axis=2).argmin(1)
    if centres[0][0] < centres[1][0]:            # team 0 is always the lighter shirt, so re-runs number the same way
        nearer, centres = 1 - nearer, centres[::-1]
    for i, k, t in zip(ids, keep, nearer):
        if k:
            roles[i] = int(t)
        else:
            x, y = positions.get(i, (None, None))
            roles[i] = "gk" if x is not None and in_box(x, y, length) else "official"
    return roles, {"centres": {0: centres[0], 1: centres[1]}, "limits": {0: float(limit[0]), 1: float(limit[1])},
                   "limit": float(limit.max()), "grass": np.float32(grass)}


def sample_label(feature, colours):
    """One shirt sample -> 0, 1 or "other" (neon, or a third colour). Noisy on its own: used in runs of samples."""
    if not colours:
        return -1
    f = np.float32(feature)
    if f[4] >= NEON_SAMPLE:
        return "other"
    v = np.r_[f[:2], _chroma(f[2:4], colours["grass"])[0]]
    c = colours["centres"]
    d = [np.linalg.norm(v - c[0]), np.linalg.norm(v - c[1])]
    t = int(d[1] < d[0])
    limit = colours.get("limits", {}).get(t, colours["limit"])
    return "other" if _off_team(v[2:], c[t][2:], [limit])[0] > 1.0 else t


def kit_changes(sample_frames, labels, fps, min_run_s=1.0, purity=0.8):
    """Frames at which a track's shirt changes for good: the tracker jumped from one person to another.
    labels: one per sample. A change only counts if at least min_run_s of mostly-one-label samples sit either side,
    so a single blurry sample, or a moment hidden behind someone, never splits a track."""
    if len(labels) < 4:
        return []
    fr = np.asarray(sample_frames)
    lab = list(labels)
    win = max(1, int(min_run_s * fps))
    # majority label over the next/previous min_run_s
    def majority(lo, hi):
        sel = [lab[k] for k in range(len(lab)) if lo <= fr[k] < hi]
        if not sel:
            return None
        best = max(set(sel), key=sel.count)
        return best if sel.count(best) >= purity * len(sel) else None

    cuts, last_cut = [], fr[0]
    for k in range(1, len(lab)):
        if lab[k] == lab[k - 1]:
            continue
        before = majority(max(last_cut, fr[k] - win), fr[k])
        after = majority(fr[k], fr[k] + win)
        if before is not None and after is not None and before != after \
                and fr[k] - last_cut >= win and fr[-1] - fr[k] >= win * 0.8:
            cuts.append(int(fr[k]))
            last_cut = fr[k]
    return cuts


KEEPER_MARGIN_M = 3.0   # one team's players must sit this much nearer a goal, on average, to call it the defending team


def keeper_team(goal_x, outfield_x_by_team, margin=KEEPER_MARGIN_M):
    """Team a keeper plays for, from position: the team whose outfield players are, over the clip, nearer his goal.
    Attackers always leave some players back, so the defending side sits deeper even when a corner fills the box.
    outfield_x_by_team: {0: [x, ...], 1: [x, ...]}. -> 0, 1, or -1 when it's too close to call (use --keepers then)."""
    d = {t: float(np.median(np.abs(np.float64(xs) - goal_x))) for t, xs in outfield_x_by_team.items() if len(xs) >= 10}
    if len(d) < 2 or abs(d[0] - d[1]) < margin:
        return -1
    return min(d, key=d.get)


def goal_of(x, length=105.0):
    """x of the goal line nearer to pitch x."""
    return 105.0 if abs(105.0 - x) <= abs(x - (105.0 - length)) else 105.0 - length
