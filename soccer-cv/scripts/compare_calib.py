"""Old fit vs new fit on REAL clicks: run it on any calib.json to see how far to trust it.

    python -m scripts.compare_calib calib.json calib_wide.json

For each file and each method (old = cv2.findHomography in pitch metres, points only; new = pipeline.homography):
  fit px          how far the clicks sit from the fitted pitch, in pixels (in-sample, so flattering)
  held-out m      refit without each click and measure how far off it lands: median and worst
  expected m      position error across the visible pitch from 2 px of click noise: median and 90th percentile
If the file has midfield clicks (halfway line, centre circle, ...), it also fits on the goal-end points ONLY (what the
old calibrator offered) and measures how far off that is at the midfield clicks: the error the old workflow hid.
"""
import argparse
import json

import cv2
import numpy as np

from pipeline import homography as hg


def old_method(obs):
    if len(obs.pts) < 4:
        return None
    H, _ = cv2.findHomography(np.float32(obs.pts), np.float32(obs.pit), 0)
    return H


def miss_m(H, obs):
    """Metres between where H puts each click and where it truly is (point) or the line it is on."""
    p = hg.to_pitch(H, obs.pts)
    q = hg.to_pitch(H, obs.lpx) if len(obs.lpx) else np.zeros((0, 2))
    return np.r_[np.linalg.norm(p - obs.pit, axis=1),
                 np.abs((np.c_[q, np.ones(len(q))] * obs.lco).sum(1)) / np.hypot(obs.lco[:, 0], obs.lco[:, 1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calib", nargs="+")
    ap.add_argument("--noise", type=float, default=hg.CLICK_NOISE_PX)
    a = ap.parse_args()
    for path in a.calib:
        c = json.load(open(path))
        obs = hg.Obs.from_calib(c)
        width, length = c.get("pitch_width", 68.0), c.get("pitch_length", 105.0)
        H_new = hg.fit(obs)
        grid = hg.pitch_grid(H_new, tuple(c.get("frame_size", (2880, 1800))), (105 - length, 105, 34 - width / 2, 34 + width / 2))
        print(f"\n=== {path}: {len(obs.pts)} points + {len(obs.lpx)} line clicks")
        print(f"{'method':8}{'fit px':>10}{'held-out m (median / worst)':>32}{'expected m (median / 90%)':>30}")
        for name, fitter in (("old", old_method), ("new", lambda o: hg.fit(o))):
            H = fitter(obs)
            loo = hg.leave_one_out(obs, H, fitter)
            err = hg.error_map(obs, H, grid, a.noise, fitter=fitter)
            rms = np.sqrt(np.mean(hg.residuals_px(H, obs) ** 2))
            print(f"{name:8}{rms:10.1f}{np.nanmedian(loo):16.2f} / {np.nanmax(loo):5.2f}"
                  f"{np.nanmedian(err):19.2f} / {np.percentile(err, 90):5.2f}")

        mid = obs.pit[:, 0] < 85
        mid_lines = hg.to_pitch(H_new, obs.lpx)[:, 0] < 85 if len(obs.lpx) else np.zeros(0, bool)
        if mid.any() or mid_lines.any():
            goal_only = obs.subset(~mid, np.zeros(len(obs.lco), bool))
            held = obs.subset(mid, mid_lines)
            H_goal = old_method(goal_only)
            print(f"goal-end points only (old workflow), checked against {len(held)} midfield clicks it never saw: "
                  f"median {np.median(miss_m(H_goal, held)):.2f} m, worst {np.max(miss_m(H_goal, held)):.2f} m")


if __name__ == "__main__":
    main()
