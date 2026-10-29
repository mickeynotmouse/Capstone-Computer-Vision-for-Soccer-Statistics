"""Step 1: run YOLO + ByteTrack on a clip and write per-frame tracks in PITCH coordinates.

Usage:
    python -m pipeline.track clip.mp4 --calib calib.json --out data/game1_tracks.json

calib.json: written by pipeline.calibrate. Image points (pixels) with their pitch points (metres), plus optional
clicks along pitch lines, e.g.
    {"image": [[x,y], ...], "pitch": [[105,68], ...], "line_image": [[x,y], ...], "line_pitch": [[0,1,-68], ...]}
The homography is fitted from all of them by pipeline.homography.
"""
import argparse
import json

import cv2
import numpy as np

from . import homography

PERSON, BALL = 0, 32  # COCO class ids used by the pretrained YOLO weights

L, W = 105.0, 68.0  # pitch dimensions in metres


def _in_pitch(x, y, margin=3.0):
    """True if a pitch-coordinate point is on the pitch or within `margin` metres of it."""
    return -margin <= x <= L + margin and -margin <= y <= W + margin


def load_homography(path):
    return homography.load(path)


def to_pitch(H, x, y):
    p = cv2.perspectiveTransform(np.float32([[[x, y]]]), H)[0][0]
    return float(p[0]), float(p[1])


def jersey_color(frame, box):
    """Mean hue/sat/val of the torso (upper-middle of the box), robust to grass via median."""
    x1, y1, x2, y2 = map(int, box)
    h, w = y2 - y1, x2 - x1
    crop = frame[y1 + int(h * 0.2): y1 + int(h * 0.5), x1 + int(w * 0.2): x2 - int(w * 0.2)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return np.median(hsv, axis=0).tolist()


def assign_teams(colors_by_id):
    """2-means on jersey colour per track id -> {id: 0|1}. (Referees/GK will be noise in the PoC.)"""
    ids = [i for i, c in colors_by_id.items() if c]
    if len(ids) < 2:
        return {i: 0 for i in ids}
    data = np.float32([colors_by_id[i] for i in ids])
    _, labels, _ = cv2.kmeans(data, 2, None,
                              (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1),
                              5, cv2.KMEANS_PP_CENTERS)
    return {i: int(l) for i, l in zip(ids, labels.flatten())}


def run(video, calib, out, model_name="yolov8m.pt", stride=1):
    from ultralytics import YOLO  # imported lazily so the rest of the package works without it

    H = load_homography(calib)
    model = YOLO(model_name)
    fps = cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS) or 30
    frames, colors = [], {}

    # stream=True keeps memory flat on long matches; tracker="bytetrack.yaml" is bundled with ultralytics
    for n, r in enumerate(model.track(video, stream=True, tracker="bytetrack.yaml",
                                      classes=[PERSON, BALL], persist=True, verbose=False)):
        if n % stride:
            continue
        players, ball = [], None
        if r.boxes is not None:
            for b in r.boxes:
                cls = int(b.cls)
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                if cls == BALL:
                    px, py = to_pitch(H, (x1 + x2) / 2, (y1 + y2) / 2)
                    if _in_pitch(px, py, margin=8.0) and (ball is None or float(b.conf) > ball["conf"]):
                        ball = {"x": px, "y": py, "conf": float(b.conf)}
                elif b.id is not None:
                    tid = int(b.id)
                    px, py = to_pitch(H, (x1 + x2) / 2, y2)  # feet = bottom-centre of box
                    if _in_pitch(px, py):
                        players.append({"id": tid, "x": px, "y": py})
                    if tid not in colors or n % 30 == 0:
                        colors[tid] = jersey_color(r.orig_img, (x1, y1, x2, y2)) or colors.get(tid)
        frames.append({"f": n, "t": n / fps, "players": players, "ball": ball})

    teams = assign_teams(colors)
    for fr in frames:
        for p in fr["players"]:
            p["team"] = teams.get(p["id"], -1)
    json.dump({"fps": fps, "frames": frames}, open(out, "w"))
    print(f"wrote {len(frames)} frames -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    run(a.video, a.calib, a.out, a.model, a.stride)
