"""Step 1: run YOLO + ByteTrack on a clip and write per-frame tracks in PITCH coordinates.

Usage:
    python -m pipeline.track clip.mp4 --calib calib.json --out data/game1_tracks.json

calib.json: written by pipeline.calibrate or pipeline.refine. Image points (pixels) with their pitch points (metres), plus
optional clicks along pitch lines, e.g.
    {"image": [[x,y], ...], "pitch": [[105,68], ...], "line_image": [[x,y], ...], "line_pitch": [[0,1,-68], ...]}
The homography is fitted from all of them by pipeline.homography.

What happens, in order:
  camera    follow the camera's pan and zoom, from the background (pipeline.camera)
  detect    YOLO at high resolution (the ball is invisible at the default 640 px), only on the rows that can hold the
            pitch, and only on frames that differ from the one before (screen recordings repeat about 1 frame in 4)
  track     ByteTrack on boxes moved into the calibration frame's view, so to the tracker the camera never moves
  people    keep the people on the pitch; sort shirts into team 0 / team 1 / goalkeeper / official (pipeline.teams);
            cut a track where its shirt changes for good (the tracker jumped from one person to another)
  stitch    remove duplicates, re-join people the tracker lost (pipeline.stitch)
  ball      find the one physically possible ball path among the detections (pipeline.ball)

In the output, "players" holds both teams' players INCLUDING goalkeepers, who carry "role": "gk" and the team of the goal
they defend. events.py counts their touches (saves, goal kicks) but leaves them out of the positions it records at goals,
which the dashboard averages. Officials go in "others", where the review video shows them and events.py doesn't look.
"""
import argparse
import json
from types import SimpleNamespace

import cv2
import numpy as np

from . import ball as ball_mod
from . import camera, homography, stitch, teams

PERSON, BALL = 0, 32  # COCO class ids used by the pretrained YOLO weights

L, W = 105.0, 68.0    # default pitch dimensions in metres, if the calibration doesn't say

IMGSZ = 1920          # detector input width: at 640 px the ball is too small to see in a wide shot
DET_CONF = 0.08       # keep weak detections: ByteTrack's second stage and the ball path can both make use of them
DET_IOU = 0.5         # suppress overlapping boxes harder than YOLO's default, since a person often gets two
PRE_MARGIN = 6.0      # m beyond the lines: people further out never even reach the tracker (the crowd)
ON_PITCH_MARGIN = 1.0 # m: a track must spend most of its life within this of the pitch ...
ON_PITCH_SHARE = 0.6  # ... this share of it. Fans stay off; a player may step out to take a throw
BALL_MARGIN = 5.0     # m beyond the lines where a ball can still be seen
MIN_TRACK_S = 0.5     # shorter tracks are noise
SHIRT_EVERY = 3       # sample shirts every this many frames
REPEAT_DIFF = 0.6     # mean grey-level change (0-255) below which a frame is a repeat of the one before
DETECT_EVERY = 2      # look at every other frame: positions in between come from the tracker; about half the time
CHUNK_FRAMES = 1800   # detections are saved this many frames at a time (1 minute at 30 fps), so a stopped run resumes
HEADROOM = 0.10       # share of the picture height kept above the far touchline, for the heads of far players
TRACKER = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=0.35, track_low_thresh=0.1, new_track_thresh=0.45,
                          match_thresh=0.85, fuse_score=True, track_buffer=0)   # track_buffer is set from the fps


def _dims(calib_path):
    c = json.load(open(calib_path))
    return c.get("pitch_width", W), c.get("pitch_length", L), c.get("frame", 0)


def load_homography(path):
    return homography.load(path)


def to_pitch(H, x, y):
    p = cv2.perspectiveTransform(np.float32([[[x, y]]]), H)[0][0]
    return float(p[0]), float(p[1])


def inside(xy, width, length, margin):
    """Boolean per point: on the pitch, or within `margin` metres of it."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return ((xy[:, 0] >= 105 - length - margin) & (xy[:, 0] <= 105 + margin)
            & (xy[:, 1] >= 34 - width / 2 - margin) & (xy[:, 1] <= 34 + width / 2 + margin))


def in_front(H, px):
    """Boolean per pixel: on the picture's side of the horizon. Pixels past it land on a mirrored, meaningless pitch spot."""
    w = (np.c_[px, np.ones(len(px))] @ H.T)[:, 2]
    return np.sign(w) == np.sign(np.median(w)) if len(w) else np.zeros(0, bool)


def pitch_rows(Ht, width, length, box, margin=PRE_MARGIN):
    """(top, bottom) picture rows that can hold someone standing on the pitch, given frame->pitch homography Ht."""
    x0, x1 = 105 - length - margin, 105 + margin
    y0, y1 = 34 - width / 2 - margin, 34 + width / 2 + margin
    edge = np.r_[np.c_[np.linspace(x0, x1, 40), np.full(40, y0)], np.c_[np.full(20, x0), np.linspace(y0, y1, 20)],
                 np.c_[np.full(20, x1), np.linspace(y0, y1, 20)]]
    Hinv = np.linalg.inv(Ht)
    q = np.c_[edge, np.ones(len(edge))] @ Hinv.T
    px = q[:, :2] / q[:, 2:]
    ok = (np.sign(q[:, 2]) == np.sign(Hinv[2, 2])) & (px[:, 0] > box[0]) & (px[:, 0] < box[2])
    if not ok.any():
        return box[1], box[3]
    top = int(px[ok, 1].min() - HEADROOM * (box[3] - box[1]))
    return max(box[1], min(top, box[3] - 64)), box[3]


def best_device():
    """The fastest place to run YOLO here. Left to itself, ultralytics uses the CPU on a Mac: about 6x slower on an M1."""
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detect(video, path=None, H=None, width=W, length=L, model_name="yolov8m.pt", imgsz=IMGSZ, device=None,
           every=DETECT_EVERY, work=None):
    """-> (per-frame detections, fps, repeats). Each entry is an array [x1, y1, x2, y2, conf, cls] in video pixels, or None
    for a frame that wasn't looked at (only every `every`-th frame is). Black (buffering) frames get an empty array.
    With the camera path and calibration, only the rows that can hold the pitch are looked at. A frame that repeats the
    previous one (screen recordings) reuses its detections; `repeats` counts them.
    work: a folder to save detections in, CHUNK_FRAMES at a time. Chunks already there are loaded rather than redone,
    so a run that stops part-way (a crash, a closed laptop, a Colab disconnect) carries on where it was."""
    import os

    model, cap = None, cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out, box, prev, repeats, chunk = [], None, None, 0, []
    n = 0
    while True:
        k = n // CHUNK_FRAMES
        saved = os.path.join(work, f"dets_{k:05d}.npz") if work else None
        if saved and n % CHUNK_FRAMES == 0 and os.path.exists(saved):
            c = np.load(saved, allow_pickle=False)
            got = [c[str(i)] if c[f"seen{i}"] else None for i in range(int(c["n"]))]
            ok = True
            for _ in got:                        # step past the frames this chunk covers
                ok = cap.grab() and ok
            out += got
            n += len(got)
            prev = None
            if len(got) < CHUNK_FRAMES:
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        if box is None:
            box = camera.content_box(frame)
        x0, y0, x1, y1 = box
        if n % every:
            d = None
        else:
            small = cv2.cvtColor(frame[y0:y1:8, x0:x1:8], cv2.COLOR_BGR2GRAY).astype(np.int16)
            if prev is not None and np.abs(small - prev[1]).mean() < REPEAT_DIFF:
                d = prev[0].copy()
                repeats += 1
            elif small.mean() < 25:          # a black buffering screen
                d = np.zeros((0, 6))
            else:
                if model is None:
                    from ultralytics import YOLO  # imported lazily so the rest of the package works without it
                    model, device = YOLO(model_name), device or best_device()
                    print(f"  running the detector on {device}", flush=True)
                if path is not None and H is not None and n < len(path):
                    y0, y1 = pitch_rows(H @ path[n], width, length, box)
                r = model.predict(frame[y0:y1, x0:x1], classes=[PERSON, BALL], conf=DET_CONF, iou=DET_IOU, imgsz=imgsz,
                                  verbose=False, device=device)[0]
                d = r.boxes.data.cpu().numpy()[:, :6].astype(np.float64)
                d[:, [0, 2]] += x0
                d[:, [1, 3]] += y0
            prev = (d, small)
        out.append(d)
        chunk.append(d)
        n += 1
        if len(chunk) == CHUNK_FRAMES:
            _save_chunk(saved, chunk)
            chunk = []
            print(f"  detected {n} frames ({n / fps / 60:.1f} min of video)", flush=True)
    if chunk:
        _save_chunk(os.path.join(work, f"dets_{(n - 1) // CHUNK_FRAMES:05d}.npz") if work else None, chunk)
    cap.release()
    return out, fps, repeats


def _save_chunk(path, chunk):
    if path:
        np.savez_compressed(path, n=len(chunk), **{str(i): (d if d is not None else np.zeros((0, 6))) for i, d in enumerate(chunk)},
                            **{f"seen{i}": d is not None for i, d in enumerate(chunk)})


def _every(dets):
    """How often frames were looked at (1 = every frame)."""
    seen = [n for n, d in enumerate(dets[:200]) if d is not None]
    return max(1, int(np.min(np.diff(seen)))) if len(seen) > 1 else 1


def track_people(dets, path, H, fps, width, length):
    """ByteTrack in the calibration frame's view. -> {track id: {"frames", "xy", "det"}}, det = index into dets[frame]."""
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker

    args = SimpleNamespace(**vars(TRACKER))
    every = _every(dets)
    args.track_buffer = int(2 * fps / every)    # 2 s: longer gaps are left to pipeline.stitch, which can see both ends
    tracker = BYTETracker(args=args)
    tracks = {}
    for n, d in enumerate(dets):
        if d is None:                            # a frame that wasn't looked at: the tracker just sees the next one
            continue
        people = np.flatnonzero(d[:, 5] == PERSON) if len(d) else np.zeros(0, int)
        T = path[n]
        Ht = H @ T
        rows, xy_by_row = [], []
        if len(people):
            b = d[people, :4]
            feet = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]]
            xy = homography.to_pitch(Ht, feet)
            ok = in_front(Ht, feet) & inside(xy, width, length, PRE_MARGIN)
            people, xy = people[ok], xy[ok]
            stable = camera.warp_boxes(T, d[people, :4])
            rows = np.c_[stable, d[people, 4], np.zeros(len(people))]
            xy_by_row = xy
        if len(rows):
            res = tracker.update(Boxes(rows, (4000, 6000)))
        else:
            res = tracker.update(Boxes(np.zeros((0, 6)), (4000, 6000)))
        for row in np.atleast_2d(res) if len(res) else ():
            tid, idx = int(row[4]), int(row[7])
            t = tracks.setdefault(tid, {"frames": [], "xy": [], "det": []})
            t["frames"].append(n)
            t["xy"].append(xy_by_row[idx])
            t["det"].append(int(people[idx]))
    return tracks


def shirts(video, dets, raw_tracks, box=None):
    """Shirt samples for every track -> ({track id: [(frame, feature)]}, grass colour)."""
    wanted = {}
    for tid, t in raw_tracks.items():
        for n, di in zip(t["frames"], t["det"]):
            if n % SHIRT_EVERY == 0:      # tracks only hold frames that were looked at
                wanted.setdefault(n, []).append((tid, di))
    out = {tid: [] for tid in raw_tracks}
    cap = cv2.VideoCapture(video)
    grass, n = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if grass is None:
            grass = teams.grass_colour(frame, box or camera.content_box(frame))
        for tid, di in wanted.get(n, ()):
            f = teams.shirt_feature(frame, dets[n][di, :4], grass)
            if f is not None:
                out[tid].append((n, f))
        n += 1
    cap.release()
    return out, grass


def ball_candidates(dets, path, H, width, length):
    """Every ball detection on or near the pitch, as (k,4) [frame, x, y, conf] in pitch metres."""
    rows = []
    for n, d in enumerate(dets):
        if d is None:
            continue
        balls = d[d[:, 5] == BALL] if len(d) else d
        if not len(balls):
            continue
        Ht = H @ path[n]
        c = np.c_[(balls[:, 0] + balls[:, 2]) / 2, (balls[:, 1] + balls[:, 3]) / 2]
        xy = homography.to_pitch(Ht, c)
        ok = in_front(Ht, c) & inside(xy, width, length, BALL_MARGIN)
        rows += [[n, x, y, conf] for (x, y), conf in zip(xy[ok], balls[ok, 4])]
    return np.array(rows, dtype=np.float64).reshape(-1, 4)


def _classify(tracks, grass, length):
    roles, colours = teams.assign({t.id: t.samples for t in tracks},
                                  {t.id: tuple(np.median(t.xy, axis=0)) for t in tracks}, grass, length)
    for t in tracks:
        t.role = roles.get(t.id, -1)
        t.team = t.role if t.role in (0, 1) else -1
    return colours


def build_people(raw, samples, fps, width, length, grass=(0, 128, 128), keeper_teams=None):
    """Tracker output + shirt samples -> (player Tracklets incl. keepers, official Tracklets, shirt colours).
    keeper_teams: optional {goal line x: team}, overriding the automatic call of whose keeper is whose.
    Needs no video, so the rules can be tested on made-up tracks."""
    keeper_teams = keeper_teams or {}
    tracks = []
    for tid, t in raw.items():
        s = samples.get(tid, ())
        tk = stitch.Tracklet(tid, t["frames"], t["xy"], [f for _, f in s], [n for n, _ in s])
        if (len(tk) / fps >= MIN_TRACK_S and stitch.on_pitch_share(tk, width, length, ON_PITCH_MARGIN) >= ON_PITCH_SHARE
                and not stitch.is_sideline_bystander(tk, width, fps)):
            tracks.append(tk)
    tracks = stitch.remove_duplicates(tracks)

    # a first pass learns the shirt colours; then any track whose shirt changes for good is cut there (the tracker
    # swapped people, e.g. a keeper diving across a defender), and everything is classified again
    colours = _classify(tracks, grass, length)
    new_ids = iter(range(max(raw, default=0) + 1, 10 ** 9))
    pieces = []
    for t in tracks:
        cuts = teams.kit_changes(t.sample_frames, [teams.sample_label(f, colours) for f in t.samples], fps) if colours else []
        pieces += t.split(cuts, new_ids) if cuts else [t]
    tracks = [t for t in pieces if len(t) / fps >= MIN_TRACK_S]
    colours = _classify(tracks, grass, length)

    # join within each kind separately: a referee can't become a player, nor one team the other
    players = []
    for kind in (0, 1, -1):
        players += stitch.stitch([t for t in tracks if t.role == kind], fps)
    players = [t for t in players if t.team in (0, 1)]   # unknown shirts are kept only if joined to a known one
    keepers = stitch.stitch([t for t in tracks if t.role == "gk"], fps)
    officials = stitch.stitch([t for t in tracks if t.role == "official"], fps)
    for t in keepers:
        t.role = "gk"
    for t in officials:
        t.role = "official"

    # one keeper per goal: pieces at the same goal that never overlap are one person
    by_goal = {}
    for k in sorted(keepers, key=lambda t: t.start):
        by_goal.setdefault(teams.goal_of(float(np.median(k.xy[:, 0])), length), []).append(k)
    keepers = []
    for gx, ks in by_goal.items():
        merged = [ks[0]]
        for k in ks[1:]:
            m = merged[-1]
            if k.start > m.end:
                joined = stitch.Tracklet(m.id, np.r_[m.frames, k.frames], np.r_[m.xy, k.xy], m.samples + k.samples,
                                         m.sample_frames + k.sample_frames)
                joined.role, joined.merged_from = "gk", m.merged_from + k.merged_from
                merged[-1] = joined
            else:
                merged.append(k)
        # whose keeper: given on the command line, else the team that sits nearer this goal over the clip
        outfield = {t: np.concatenate([p.xy[:, 0] for p in players if p.team == t] or [np.zeros(0)]) for t in (0, 1)}
        team = keeper_teams.get(gx, teams.keeper_team(gx, outfield))
        for k in merged:
            k.team = team
        keepers += merged
    return players + keepers, officials, colours


def to_frames(players, others, ball, total, fps):
    """Tracklets + ball path -> the frame list that events.py and review.py read."""
    frames = [{"f": n, "t": n / fps, "players": [], "others": [], "ball": ball[n]} for n in range(total)]
    for group, key in ((players, "players"), (others, "others")):
        for t in group:
            fr, xy, seen = stitch.smooth_and_fill(t, fps)
            for n, (x, y), s in zip(fr, xy, seen):
                if n >= total:
                    continue
                p = {"id": int(t.id), "x": float(x), "y": float(y), "team": int(t.team) if key == "players" else -1}
                if t.role in ("gk", "official"):
                    p["role"] = t.role
                if not s:
                    p["filled"] = True
                frames[int(n)][key].append(p)
    return frames


def run(video, calib, out, model_name="yolov8m.pt", stride=1, imgsz=IMGSZ, work=None, device=None, keepers=None,
        every=DETECT_EVERY):
    import os
    import time

    H = load_homography(calib)
    width, length, ref_frame = _dims(calib)
    t0 = time.time()
    if work:
        os.makedirs(work, exist_ok=True)
        stamp = {"video": os.path.abspath(video), "size": os.path.getsize(video), "calib_frame": ref_frame,
                 "model": model_name, "imgsz": imgsz, "every": every}
        old = os.path.join(work, "run.json")
        if os.path.exists(old) and json.load(open(old)) != stamp:
            raise SystemExit(f"{work} holds work from a different video or settings: use another --work folder")
        json.dump(stamp, open(old, "w"))

    path_file = os.path.join(work, "camera_path.npz") if work else None
    if path_file and os.path.exists(path_file):
        path = camera.CameraPath.load(path_file)
        print("camera path: loaded from the work folder")
    else:
        print("following the camera ...", flush=True)
        path = camera.CameraPath.from_video(video, ref_frame=ref_frame,
                                            progress=lambda n: print(f"  {n} frames", flush=True) if n % 9000 == 0 else None)
        if path_file:
            path.save(path_file)
    t1 = time.time()

    print("detecting ...", flush=True)
    dets, fps, repeats = detect(video, path, H, width, length, model_name, imgsz, device, every, work)
    if repeats:
        print(f"  {repeats} frames repeated the one before and were not detected again")
    t2 = time.time()

    print("tracking ...", flush=True)
    raw = track_people(dets, path, H, fps, width, length)
    samples, grass = shirts(video, dets, raw)
    players, officials, colours = build_people(raw, samples, fps, width, length, grass, keepers)
    ball = ball_mod.ball_track(ball_candidates(dets, path, H, width, length), len(dets), fps)
    t3 = time.time()

    frames = to_frames(players, officials, ball, len(dets), fps)
    centres = colours.get("centres", {}) if colours else {}
    meta = {"pitch_width": width, "pitch_length": length,
            "team_colours_lab": {k: np.round(v, 1).tolist() for k, v in centres.items()},
            "keepers": {int(t.id): int(t.team) for t in players if t.role == "gk"},
            "raw_track_ids": len(raw), "detect_every": every,
            "seconds": {"camera": round(t1 - t0, 1), "detect": round(t2 - t1, 1), "track": round(t3 - t2, 1)},
            "camera_to_ref": np.round(path.to_ref.reshape(len(path), 9), 7).tolist()}
    json.dump({"fps": fps, "meta": meta, "frames": frames[::stride]}, open(out, "w"))
    print(f"wrote {len(frames[::stride])} frames -> {out}   (camera {t1 - t0:.0f} s, detection {t2 - t1:.0f} s, "
          f"tracking {t3 - t2:.0f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--work", help="folder for the camera path and detections, saved as they're made: re-running "
                                   "resumes a stopped run, and re-tuning tracking takes seconds")
    ap.add_argument("--detect-every", type=int, default=DETECT_EVERY,
                    help="look at every n-th frame (default 2; 1 = every frame, slower, a little more of the ball)")
    ap.add_argument("--device", help="mps, cpu, cuda:0 ...; default: the fastest one available")
    ap.add_argument("--keepers", help='whose keeper is whose, if the automatic call is wrong: "105:1" = the keeper of the '
                                      'goal at x=105 plays for team 1; for both goals "105:1,0:0"')
    a = ap.parse_args()
    kt = {float(g): int(t) for g, t in (p.split(":") for p in a.keepers.split(","))} if a.keepers else None
    run(a.video, a.calib, a.out, a.model, a.stride, a.imgsz, a.work, a.device, kt, a.detect_every)
