"""Look at what the pipeline did on a real clip, so you can judge it with your own eyes.

    python -m pipeline.review clip.mp4 --calib calib.json --tracks data/goal_tracks.json

Writes into review/ :
  calib_check.png   LEFT: your frame with the real pitch lines projected on it (they should sit on the painted lines).
                    RIGHT: the frame flattened to a top-down pitch (should look like a flat, correctly proportioned pitch).
  review.mp4        LEFT: video with a dot at every tracked player's feet (colour = the team's measured shirt colour, number = track id; yellow ring = ball,
                    orange ring = ball position estimated between detections; "GK" = goalkeeper; hollow R = official, not tracked as a player).
                    RIGHT: the same positions on a top-down pitch, with detected events (PASS / SHOT / GOAL) flashed on top.
  stills/           PNG snapshots of that every few seconds and at every shot/goal (easiest way to inspect).
Also prints a summary of how healthy the tracking is.
"""
import argparse
import json
import os
from collections import Counter

import cv2
import numpy as np

from . import homography
from .events import detect

L, W, PX, MARGIN = 105.0, 68.0, 8, 32          # top-down panel: 8 px per metre, 4 m of margin around the pitch
TEAM_BGR = {0: (60, 60, 230), 1: (230, 120, 40), -1: (150, 150, 150)}   # team 0 red, team 1 blue, unknown grey
FONT = cv2.FONT_HERSHEY_SIMPLEX


def pitch_lines(width=W, length=L):
    """The pitch's lines in metres. A field smaller than 105 x 68 stays centred on y = 34 with its goal line at x = 105."""
    def arc(cx, cy, r, a0, a1):
        a = np.linspace(a0, a1, 60)
        return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)
    x0, x1, y0, y1, mid = L - length, L, 34 - width / 2, 34 + width / 2, L - length / 2
    lines = [np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]), np.array([[mid, y0], [mid, y1]]), arc(mid, 34, 9.15, 0, 2 * np.pi)]
    for x0, s in ((x0, 1), (x1, -1)):
        lines.append(np.array([[x0, 13.84], [x0 + s * 16.5, 13.84], [x0 + s * 16.5, 54.16], [x0, 54.16]]))   # penalty area
        lines.append(np.array([[x0, 24.84], [x0 + s * 5.5, 24.84], [x0 + s * 5.5, 43.16], [x0, 43.16]]))     # six-yard box
    return [l.astype(np.float32) for l in lines]


def poly(pts):
    return np.round(np.clip(pts, -30000, 30000)).astype(np.int32)


def td(x, y):   # pitch metres -> pixel in the top-down panel
    return int(np.clip(x * PX + MARGIN, -30000, 30000)), int(np.clip(y * PX + MARGIN, -30000, 30000))


def project_line(line, Hinv):
    """A pitch polyline in image pixels, split where it passes behind the camera (which would draw it wrapping across the sky)."""
    dense = np.vstack([np.linspace(a, b, 24) for a, b in zip(line[:-1], line[1:])]).astype(np.float64)
    q = np.c_[dense, np.ones(len(dense))] @ Hinv.T
    front = np.sign(q[:, 2]) == np.sign(Hinv[2, 2])
    px = q[:, :2] / q[:, 2:]
    parts, cur = [], []
    for p, ok in zip(px, front):
        if ok:
            cur.append(p)
        elif cur:
            parts.append(np.array(cur))
            cur = []
    if cur:
        parts.append(np.array(cur))
    return [p for p in parts if len(p) > 1]


def topdown_base(width=W, length=L):
    img = np.full((int(W * PX) + 2 * MARGIN, int(L * PX) + 2 * MARGIN, 3), (60, 125, 50), np.uint8)
    for line in pitch_lines(width, length):
        cv2.polylines(img, [poly(line * PX + MARGIN)], False, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def fit(img, h):
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))


def calib_check(frame, H, Hinv, width=W, length=L):
    left = frame.copy()
    for line in pitch_lines(width, length):
        for p in project_line(line, Hinv):
            cv2.polylines(left, [poly(p)], False, (0, 0, 255), 2, cv2.LINE_AA)
    S = np.float64([[PX, 0, MARGIN], [0, PX, MARGIN], [0, 0, 1]])
    right = cv2.warpPerspective(frame, S @ H, (int(L * PX) + 2 * MARGIN, int(W * PX) + 2 * MARGIN))
    for line in pitch_lines(width, length):
        cv2.polylines(right, [poly(line * PX + MARGIN)], False, (0, 0, 255), 1, cv2.LINE_AA)
    h = 600
    return np.hstack([fit(left, h), np.full((h, 8, 3), 255, np.uint8), fit(right, h)])


def shirt_colours(tracks):
    """Each team's dot in its own shirt colour (from the Lab colours track.py measured), else red / blue."""
    lab = tracks.get("meta", {}).get("team_colours_lab")
    if not lab:
        return dict(TEAM_BGR)
    out = dict(TEAM_BGR)
    for t, (L, _, a, b, *_) in lab.items():
        px = np.uint8([[[np.clip(L, 0, 255), np.clip(a + 128, 0, 255), np.clip(b + 128, 0, 255)]]])
        out[int(t)] = tuple(int(v) for v in cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0])
    return out


def render(frame, fr, Hinv, base, banner, team_bgr=TEAM_BGR):
    top = base.copy()
    h = top.shape[0]
    s = h / frame.shape[0]
    left = cv2.resize(frame, (int(frame.shape[1] * s), h))

    def on_video(x, y):
        ix, iy = cv2.perspectiveTransform(np.float32([[[x, y]]]), Hinv)[0][0]
        return int(np.clip(ix * s, -30000, 30000)), int(np.clip(iy * s, -30000, 30000))

    for p in fr["players"]:
        col, label = team_bgr.get(p["team"], TEAM_BGR[-1]), ("GK " if p.get("role") == "gk" else "") + str(p["id"])
        for panel, c, size in ((left, on_video(p["x"], p["y"]), 0.5), (top, td(p["x"], p["y"]), 0.45)):
            cv2.circle(panel, c, 6, col, -1)
            cv2.circle(panel, c, 6, (255, 255, 255), 1)
            cv2.putText(panel, label, (c[0] + 7, c[1] - 7), FONT, size, (255, 255, 255), 1, cv2.LINE_AA)
    for p in fr.get("others", ()):
        label = "R"
        for panel, c in ((left, on_video(p["x"], p["y"])), (top, td(p["x"], p["y"]))):
            cv2.circle(panel, c, 6, (230, 230, 230), 2)
            cv2.putText(panel, label, (c[0] + 7, c[1] - 7), FONT, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    b = fr["ball"]
    if b:
        col = (0, 140, 255) if b.get("filled") else (0, 255, 255)
        cv2.circle(left, on_video(b["x"], b["y"]), 10, col, 2)
        cv2.circle(top, td(b["x"], b["y"]), 8, col, 2)
    out = np.hstack([left, top])
    cv2.putText(out, f"t = {fr['t']:.1f}s", (10, 24), FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    if banner:
        cv2.putText(out, banner, (10, 56), FONT, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def summary(tracks, events):
    fr = tracks["frames"]
    dt = fr[1]["t"] - fr[0]["t"] if len(fr) > 1 else 0.1
    seen, teams = Counter(), {}
    inside = total = 0
    meta = tracks.get("meta", {})
    width, length = meta.get("pitch_width", W), meta.get("pitch_length", L)
    for f in fr:
        for p in f["players"]:
            seen[p["id"]] += 1
            teams.setdefault(p["id"], p["team"])
            total += 1
            inside += L - length - 2 <= p["x"] <= L + 2 and 34 - width / 2 - 2 <= p["y"] <= 34 + width / 2 + 2
    per = sorted(len(f["players"]) for f in fr)
    print("\n=== TRACKING SUMMARY ===")
    print(f"frames processed        : {len(fr)}  ({fr[-1]['t'] - fr[0]['t']:.1f} s)")
    seen_ball = sum(1 for f in fr if f["ball"] and not f["ball"].get("filled"))
    filled_ball = sum(1 for f in fr if f["ball"] and f["ball"].get("filled"))
    print(f"ball                    : detected in {100 * seen_ball / len(fr):.0f}% of frames, estimated in a further {100 * filled_ball / len(fr):.0f}%")
    print(f"players per frame       : median {per[len(per) // 2]}   (outfield players only: 20 at most)")
    other_ids = {p["id"] for f in fr for p in f.get("others", ())}
    keepers = {p["id"]: p["team"] for f in fr for p in f["players"] if p.get("role") == "gk"}
    print(f"goalkeepers             : {keepers or 'none found'}   (id: team; -1 = couldn't tell whose keeper)")
    print(f"officials left out      : {len(other_ids)} ids   (shown hollow R in the video)")
    print(f"distinct track ids      : {len(seen)}   (ideal ~ the number of people; far more = tracker lost people and restarted)")
    print(f"ids tracked for >= 3 s  : {sum(1 for v in seen.values() if v * dt >= 3)}")
    print(f"ids per team cluster    : {dict(Counter(teams.values()))}   (-1 = no colour found)")
    print(f"detections inside pitch : {100 * inside / max(total, 1):.0f}%   (low = calibration is off, or lots of people off the pitch)")
    print(f"events detected         : {dict(Counter(e['type'] for e in events))}")
    for e in events:
        if e["type"] in ("shot", "goal"):
            print(f"   {e['type']:5s} at {e['t']:.1f}s  team {e['team']}  player id {e['player']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--calib-frame", type=int, default=0, help="frame number you calibrated on")
    ap.add_argument("--still-every", type=float, default=3.0, help="seconds between snapshot PNGs")
    ap.add_argument("--out", default="review")
    a = ap.parse_args()

    calib, tracks = json.load(open(a.calib)), json.load(open(a.tracks))
    H = homography.load(a.calib)
    Hinv = np.linalg.inv(H)
    width, length = calib.get("pitch_width", W), calib.get("pitch_length", L)
    moves = tracks.get("meta", {}).get("camera_to_ref")      # older tracks files have no camera path: treat as locked off
    events = detect(tracks)
    fr = tracks["frames"]
    os.makedirs(os.path.join(a.out, "stills"), exist_ok=True)
    summary(tracks, events)

    cap = cv2.VideoCapture(a.video)
    for _ in range(calib.get("frame", a.calib_frame)):      # step through: seeking is unreliable on screen recordings
        cap.grab()
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("could not read the calibration frame")
    cv2.imwrite(os.path.join(a.out, "calib_check.png"), calib_check(frame, H, Hinv, width, length))
    cap.release()
    cap = cv2.VideoCapture(a.video)

    by_f = {f["f"]: f for f in fr}
    stride = (fr[1]["f"] - fr[0]["f"]) if len(fr) > 1 else 1
    base = topdown_base(width, length)
    colours = shirt_colours(tracks)
    writer, next_still = None, 0.0
    key_times = [e["t"] for e in events if e["type"] in ("shot", "goal")]
    for n in range(fr[-1]["f"] + 1):
        ok, frame = cap.read()
        if not ok:
            break
        f = by_f.get(n)
        if f is None:
            continue
        banner = " | ".join(f"{e['type'].upper()} team {e['team']}" for e in events if 0 <= f["t"] - e["t"] <= 1.5 and e["type"] != "turnover")
        Hinv_n = np.linalg.inv(H @ np.array(moves[n]).reshape(3, 3)) if moves else Hinv
        img = render(frame, f, Hinv_n, base, banner, colours)
        if writer is None:
            writer = cv2.VideoWriter(os.path.join(a.out, "review.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), tracks["fps"] / stride, (img.shape[1], img.shape[0]))
        writer.write(img)
        if f["t"] >= next_still or any(0 <= f["t"] - t < 1.0 / tracks["fps"] * stride for t in key_times):
            cv2.imwrite(os.path.join(a.out, "stills", f"t{f['t']:05.1f}s.png"), img)
            next_still = max(next_still, f["t"]) + a.still_every
    if writer:
        writer.release()
    print(f"\nwrote {a.out}/calib_check.png, {a.out}/review.mp4 and snapshots in {a.out}/stills/")


if __name__ == "__main__":
    main()