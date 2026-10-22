"""Look at what the pipeline did on a real clip, so you can judge it with your own eyes.

    python -m pipeline.review clip.mp4 --calib calib.json --tracks data/goal_tracks.json

Writes into review/ :
  calib_check.png   LEFT: your frame with the real pitch lines projected on it (they should sit on the painted lines).
                    RIGHT: the frame flattened to a top-down pitch (should look like a flat, correctly proportioned pitch).
  review.mp4        LEFT: video with a dot at every tracked player's feet (colour = team, number = track id; yellow ring = ball).
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

from .events import detect

L, W, PX, MARGIN = 105.0, 68.0, 8, 32          # top-down panel: 8 px per metre, 4 m of margin around the pitch
TEAM_BGR = {0: (60, 60, 230), 1: (230, 120, 40), -1: (150, 150, 150)}   # team 0 red, team 1 blue, unknown grey
FONT = cv2.FONT_HERSHEY_SIMPLEX


def pitch_lines():
    def arc(cx, cy, r, a0, a1):
        a = np.linspace(a0, a1, 60)
        return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)
    lines = [np.array([[0, 0], [L, 0], [L, W], [0, W], [0, 0]]), np.array([[L / 2, 0], [L / 2, W]]), arc(L / 2, W / 2, 9.15, 0, 2 * np.pi)]
    for x0, s in ((0, 1), (L, -1)):
        lines.append(np.array([[x0, 13.84], [x0 + s * 16.5, 13.84], [x0 + s * 16.5, 54.16], [x0, 54.16]]))   # penalty area
        lines.append(np.array([[x0, 24.84], [x0 + s * 5.5, 24.84], [x0 + s * 5.5, 43.16], [x0, 43.16]]))     # six-yard box
    return [l.astype(np.float32) for l in lines]


def poly(pts):
    return np.round(np.clip(pts, -30000, 30000)).astype(np.int32)


def td(x, y):   # pitch metres -> pixel in the top-down panel
    return int(np.clip(x * PX + MARGIN, -30000, 30000)), int(np.clip(y * PX + MARGIN, -30000, 30000))


def topdown_base():
    img = np.full((int(W * PX) + 2 * MARGIN, int(L * PX) + 2 * MARGIN, 3), (60, 125, 50), np.uint8)
    for line in pitch_lines():
        cv2.polylines(img, [poly(line * PX + MARGIN)], False, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def fit(img, h):
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))


def calib_check(frame, H, Hinv):
    left = frame.copy()
    for line in pitch_lines():
        p = cv2.perspectiveTransform(line.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
        cv2.polylines(left, [poly(p)], False, (0, 0, 255), 2, cv2.LINE_AA)
    S = np.float64([[PX, 0, MARGIN], [0, PX, MARGIN], [0, 0, 1]])
    right = cv2.warpPerspective(frame, S @ H, (int(L * PX) + 2 * MARGIN, int(W * PX) + 2 * MARGIN))
    for line in pitch_lines():
        cv2.polylines(right, [poly(line * PX + MARGIN)], False, (0, 0, 255), 1, cv2.LINE_AA)
    h = 600
    return np.hstack([fit(left, h), np.full((h, 8, 3), 255, np.uint8), fit(right, h)])


def render(frame, fr, Hinv, base, banner):
    top = base.copy()
    h = top.shape[0]
    s = h / frame.shape[0]
    left = cv2.resize(frame, (int(frame.shape[1] * s), h))
    marks = [(p["x"], p["y"], TEAM_BGR.get(p["team"], TEAM_BGR[-1]), str(p["id"])) for p in fr["players"]]
    for x, y, col, label in marks:
        ix, iy = cv2.perspectiveTransform(np.float32([[[x, y]]]), Hinv)[0][0]
        ix, iy = int(np.clip(ix * s, -30000, 30000)), int(np.clip(iy * s, -30000, 30000))
        cv2.circle(left, (ix, iy), 6, col, -1)
        cv2.circle(left, (ix, iy), 6, (255, 255, 255), 1)
        cv2.putText(left, label, (ix + 7, iy - 7), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tx, ty = td(x, y)
        cv2.circle(top, (tx, ty), 6, col, -1)
        cv2.circle(top, (tx, ty), 6, (255, 255, 255), 1)
        cv2.putText(top, label, (tx + 7, ty - 7), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    b = fr["ball"]
    if b:
        ix, iy = cv2.perspectiveTransform(np.float32([[[b["x"], b["y"]]]]), Hinv)[0][0]
        cv2.circle(left, (int(np.clip(ix * s, -30000, 30000)), int(np.clip(iy * s, -30000, 30000))), 10, (0, 255, 255), 2)
        cv2.circle(top, td(b["x"], b["y"]), 8, (0, 255, 255), 2)
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
    for f in fr:
        for p in f["players"]:
            seen[p["id"]] += 1
            teams.setdefault(p["id"], p["team"])
            total += 1
            inside += -2 <= p["x"] <= L + 2 and -2 <= p["y"] <= W + 2
    per = sorted(len(f["players"]) for f in fr)
    print("\n=== TRACKING SUMMARY ===")
    print(f"frames processed        : {len(fr)}  ({fr[-1]['t'] - fr[0]['t']:.1f} s)")
    print(f"ball detected           : {100 * sum(1 for f in fr if f['ball']) / len(fr):.0f}% of frames   (small ball in a wide shot -> expect this to be low)")
    print(f"players per frame       : median {per[len(per) // 2]}   (22 players + referees would be ~24 if all are seen)")
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
    H, _ = cv2.findHomography(np.float32(calib["image"]), np.float32(calib["pitch"]))
    Hinv = np.linalg.inv(H)
    events = detect(tracks)
    fr = tracks["frames"]
    os.makedirs(os.path.join(a.out, "stills"), exist_ok=True)
    summary(tracks, events)

    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, calib.get("frame", a.calib_frame))
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("could not read the calibration frame")
    cv2.imwrite(os.path.join(a.out, "calib_check.png"), calib_check(frame, H, Hinv))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    by_f = {f["f"]: f for f in fr}
    stride = (fr[1]["f"] - fr[0]["f"]) if len(fr) > 1 else 1
    base = topdown_base()
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
        img = render(frame, f, Hinv, base, banner)
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