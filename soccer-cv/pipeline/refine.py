"""Make a calibration for a clip, with no clicking, from one good calibration on the same camera mount; then snap to white.

    python -m pipeline.refine goal.mp4 --from-calib calib_wide_v2.json --from-video wide.mp4 --out calib_goal_v2.json
    python -m pipeline.refine wide.mp4 --from-calib calib_wide_v2.json --out calib_wide_v3.json      # just snap to white

The clicks are moved onto this clip's frame through a background match (pipeline.camera.register), so the field size and
clicks carry over. A new mount needs a fresh pipeline.calibrate. Writes the same calib.json format everything else reads.
"""
import argparse
import json

import cv2
import numpy as np

from . import camera, lines
from . import homography as hg
from .calibrate import read_frame

MISCLICK_PX = 10.0     # a clicked point further than this from where the paint-fitted pitch puts it is dropped


def quality(obs, H, size, width, length):
    """The honest numbers from pipeline.calibrate, for a calib.json."""
    grid = hg.pitch_grid(H, size, (105 - length, 105, 34 - width / 2, 34 + width / 2))
    err = hg.error_map(obs, H, grid, trials=15)
    return {"fit_rms_px": round(float(np.sqrt(np.mean(hg.residuals_px(H, obs) ** 2))), 2),
            "expected_error_m": {"median": round(float(np.nanmedian(err)), 3), "p90": round(float(np.nanpercentile(err, 90)), 3)}}


def move_clicks(c, to_new, box):
    """Clicks of calib c, moved through homography to_new; those landing outside the picture are dropped."""
    def inside(p):
        return box[0] <= p[0] < box[2] and box[1] <= p[1] < box[3]

    pts = [(q.tolist(), pit) for q, pit in zip(camera.apply(to_new, c["image"]), c["pitch"]) if inside(q)]
    li = [(q.tolist(), co) for q, co in zip(camera.apply(to_new, c.get("line_image", [])), c.get("line_pitch", [])) if inside(q)]
    return hg.Obs([p for p, _ in pts], [x for _, x in pts], [p for p, _ in li], [x for _, x in li])


def print_alignment(label, a):
    print(f"  {label}")
    for name, (share, off) in a.items():
        print(f"    {name:18s} {100 * share:4.0f}% on white paint, typically {off:.1f} px away")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--from-calib", required=True, help="calib.json to start from")
    ap.add_argument("--from-video", help="the clip that calib.json was made on, if it isn't this one")
    ap.add_argument("--frame", type=int, default=0, help="frame of this clip to calibrate on")
    ap.add_argument("--width", type=float, help="field width in metres, if the source file is wrong")
    ap.add_argument("--length", type=float, help="field length in metres, if the source file is wrong")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    c = json.load(open(a.from_calib))
    width, length = a.width or c.get("pitch_width", 68.0), a.length or c.get("pitch_length", 105.0)
    frame = read_frame(a.video, a.frame)
    box = camera.content_box(frame)
    if a.from_video:
        T, inliers = camera.register(read_frame(a.from_video, c.get("frame", 0)), frame)
        if T is None:
            raise SystemExit("couldn't match the two clips' backgrounds: are they really from the same camera mount?")
        print(f"matched the clips' backgrounds on {inliers} features")
        obs = move_clicks(c, T, box)
    else:
        obs = hg.Obs.from_calib(c)
    H0 = hg.fit(obs)
    if H0 is None:
        raise SystemExit("too few clicks are visible in this frame to start from")

    print_alignment("before", lines.alignment(frame, H0, width, length, box))
    H, full, used = lines.refine(frame, obs, H0, width, length, box)
    print_alignment("after", lines.alignment(frame, H, width, length, box))
    print("  white paint found on:", ", ".join(f"{k} ({v})" for k, v in used.items()) or "nothing")

    # a clicked point the white paint disagrees with is a misclick (or a misjudged landmark): drop it and refit
    res = hg.residuals_px(H, full)[:len(full.pts)]
    bad = res > MISCLICK_PX
    if bad.any() and (~bad).sum() >= 2:
        for (x, y), (X, Y), e in zip(full.pts[bad], full.pit[bad], res[bad]):
            print(f"  dropped the point clicked at ({x:.0f}, {y:.0f}) for pitch ({X:g}, {Y:g}): the paint puts it {e:.0f} px away")
        full = hg.Obs(full.pts[~bad], full.pit[~bad], full.lpx, full.lco)
        H = hg.fit(full, init=H)
    # likewise spots along a line that the rest of the paint says belong to some other line
    off = hg.residuals_px(H, full)[len(full.pts):] > MISCLICK_PX
    if off.any():
        print(f"  dropped {int(off.sum())} line spots that sit on other paint")
        full = hg.Obs(full.pts, full.pit, full.lpx[~off], full.lco[~off])
        H = hg.fit(full, init=H)
    if H is None:
        raise SystemExit("the clicks and the paint don't agree on one pitch: re-run pipeline.calibrate on this frame")

    r = lambda p: [round(float(p[0]), 1), round(float(p[1]), 1)]   # noqa: E731
    size = frame.shape[1::-1]
    out = {"image": [r(p) for p in full.pts], "pitch": full.pit.tolist(),
           "line_image": [r(p) for p in full.lpx], "line_pitch": full.lco.tolist(),
           "frame": a.frame, "frame_size": list(size), "pitch_width": width, "pitch_length": length,
           "H": (H / np.linalg.norm(H)).tolist(), **quality(full, H, size, width, length)}
    json.dump(out, open(a.out, "w"))
    print(f"wrote {a.out}  ({len(full.pts)} points + {len(full.lpx)} line spots)  fit {out['fit_rms_px']} px  "
          f"expected error median {out['expected_error_m']['median']} m, 90% {out['expected_error_m']['p90']} m")


if __name__ == "__main__":
    main()
