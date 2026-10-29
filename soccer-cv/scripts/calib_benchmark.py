"""How accurate is each calibration method, measured against a KNOWN answer?

    python -m scripts.calib_benchmark [--calib calib_wide.json] [--trials 300] [--noise 2]

On real footage nobody knows where each pixel truly is on the pitch, so this builds a virtual camera that sees the
pitch the way the clip in --calib does (same frame size, same view), then simulates calibrating it many times with
realistic click error. Because the virtual camera is exact, the true pitch position of every pixel is known, and each
method's error can be measured everywhere on the visible pitch, not only at the clicked landmarks.

Methods compared:
  old            goal-end points only, cv2.findHomography fitted in pitch metres (the code before this change)
  fix 3          same goal-end points, robust fit in image pixels (pipeline.homography)
  fix 2          goal-end + midfield points + line clicks, old fitting method
  fix 2 + 3      goal-end + midfield points + line clicks, robust pixel fit  (what pipeline.calibrate now does)
                 (also shown without the line clicks, to see what clicking along lines adds)
Each is also run with one misclick (one goal-end point off by 15 px) to show what the robust loss buys.
"""
import argparse
import json

import cv2
import numpy as np

from pipeline import homography as hg
from pipeline.calibrate import landmarks

CONTENT = (20, 110, 2860, 1630)   # usable video area (x0, y0, x1, y1): the Hudl player bars cover the rest


def virtual_camera(calib, size):
    """Pinhole camera (square pixels, centred principal point) that best reproduces the clicked view."""
    obs = hg.Obs.from_calib(calib)
    H = hg.fit(obs)
    pit = np.array([[x, y] for x in np.linspace(80, 105, 6) for y in np.linspace(5, 63, 6)])
    img = hg.to_image(H, pit)
    obj = np.c_[pit, np.zeros(len(pit))]
    best = None
    for f in np.geomspace(800, 20000, 120):
        K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]])
        ok, rv, tv = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        err = np.sqrt(np.mean(np.sum((cv2.projectPoints(obj, rv, tv, K, None)[0].reshape(-1, 2) - img) ** 2, 1)))
        if best is None or err < best[0]:
            best = (err, K, rv, tv)
    _, K, rv, tv = best
    R = cv2.Rodrigues(rv)[0]
    G = K @ np.c_[R[:, 0], R[:, 1], tv.ravel()]   # pitch -> image for the ground plane
    centre = -R.T @ tv.ravel()
    return np.linalg.inv(G), K[0, 0], centre, best[0]


def visible(H_true, pit):
    """Which pitch points land inside the usable frame, in front of the camera."""
    G = np.linalg.inv(H_true)
    q = np.c_[pit, np.ones(len(pit))] @ G.T
    px = q[:, :2] / q[:, 2:]
    x0, y0, x1, y1 = CONTENT
    return (q[:, 2] * np.sign(np.linalg.det(G)) > 0) & (px[:, 0] > x0) & (px[:, 0] < x1) & (px[:, 1] > y0) & (px[:, 1] < y1), px


def line_clicks(H_true, items, per_line=3):
    """Simulated clicks along every painted line that is in view: per_line spots spread over its visible part."""
    extent = {"NEAR touchline": ("y", 0, 105), "FAR touchline": ("y", 0, 105), "HALFWAY": ("x", 0, 68),
              "GOAL": ("x", 0, 68), "18-yd box FRONT": ("x", 13.84, 54.16), "NEAR 18-yd box SIDE": ("y", 88.5, 105),
              "FAR 18-yd box SIDE": ("y", 88.5, 105), "6-yd box FRONT": ("x", 24.84, 43.16)}
    px, co = [], []
    for name, kind, value in items:
        if kind != "line":
            continue
        a, b, c = value
        axis, lo, hi = next(v for k, v in extent.items() if name.startswith(k))
        t = np.linspace(lo, hi, 400)
        pit = np.c_[np.full_like(t, -c), t] if axis == "x" else np.c_[t, np.full_like(t, -c)]
        ok, img = visible(H_true, pit)
        if ok.sum() < 20:
            continue
        img = img[ok]
        for i in np.linspace(0.1, 0.9, per_line) * (len(img) - 1):
            px.append(img[int(i)])
            co.append([a, b, c])
    return np.array(px), np.array(co)


def old_method(obs):
    if len(obs.pts) < 4:
        return None
    H, _ = cv2.findHomography(np.float32(np.r_[obs.pts]), np.float32(obs.pit), 0)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="calib_wide.json")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--noise", type=float, default=2.0, help="click error, pixels (standard deviation per axis)")
    ap.add_argument("--size", default="2880x1800")
    a = ap.parse_args()
    size = tuple(int(v) for v in a.size.split("x"))
    calib = json.load(open(a.calib))

    H_true, f, centre, pnp_err = virtual_camera(calib, size)
    print(f"virtual camera for {a.calib}: focal length {f:.0f} px, standing at pitch ({centre[0]:.1f}, {centre[1]:.1f}) m, "
          f"{abs(centre[2]):.1f} m up (camera fit {pnp_err:.2f} px)")

    items = landmarks()
    goal_end = np.array(calib["pitch"], float)                       # the landmarks actually clicked on this clip
    all_pts = np.array([v for _, kind, v in items if kind == "point"], float)
    ok, _ = visible(H_true, all_pts)
    mid_pts = np.array([p for p in all_pts[ok] if not any(np.allclose(p, g) for g in goal_end)])
    lpx, lco = line_clicks(H_true, items)
    print(f"in view: {len(goal_end)} goal-end points (as clicked), {len(mid_pts)} extra points "
          f"{[(float(x), float(y)) for x, y in np.round(mid_pts, 1)]}, {len(lpx)} line clicks")

    exact = lambda pit: hg.to_image(H_true, pit)   # noqa: E731
    obs_goal = hg.Obs(exact(goal_end), goal_end)
    both = np.r_[goal_end, mid_pts] if len(mid_pts) else goal_end
    obs_all = hg.Obs(exact(both), both, lpx, lco)
    obs_all_old = hg.Obs(exact(both), both)                         # the old method cannot use line clicks

    # evaluation grid: every pixel of the usable frame that truly lands on the pitch
    x0, y0, x1, y1 = CONTENT
    xs, ys = np.meshgrid(np.arange(x0, x1, 40.0), np.arange(y0, y1, 40.0))
    grid = np.c_[xs.ravel(), ys.ravel()]
    truth = hg.to_pitch(H_true, grid)
    on = (truth[:, 0] > 0) & (truth[:, 0] < 105) & (truth[:, 1] > 0) & (truth[:, 1] < 68)
    grid, truth = grid[on], truth[on]
    fg = grid[:, 1] > (y0 + y1) / 2                                 # foreground = bottom half of the picture

    methods = [("old", obs_goal, old_method), ("fix 3", obs_goal, hg.fit),
               ("fix 2", obs_all_old, old_method), ("fix 2 + 3, no line clicks", obs_all_old, hg.fit),
               ("fix 2 + 3", obs_all, hg.fit)]
    print(f"\ntrue position error over the visible pitch, {a.trials} simulated calibrations, click error {a.noise} px:")
    print(f"{'':26}{'':>12}{'whole pitch':>24}{'foreground (bottom half)':>30}")
    print(f"{'method':26}{'misclick?':>12}{'median':>12}{'90%':>12}{'median':>15}{'90%':>15}")
    rng = np.random.default_rng(1)
    for name, obs, fitter in methods:
        for misclick in (False, True):
            errs, fails = [], 0
            for _ in range(a.trials):
                o = obs.jitter(rng, a.noise)
                if misclick:
                    i = rng.integers(len(goal_end))
                    ang = rng.uniform(0, 2 * np.pi)
                    o.pts[i] += 15 * np.array([np.cos(ang), np.sin(ang)])
                H = fitter(o)
                if H is None:
                    fails += 1
                    continue
                errs.append(np.linalg.norm(hg.to_pitch(H, grid) - truth, axis=1))
            e = np.array(errs)
            print(f"{name:26}{'1 x 15 px' if misclick else 'no':>12}{np.median(e):11.2f}m{np.percentile(e, 90):11.2f}m"
                  f"{np.median(e[:, fg]):14.2f}m{np.percentile(e[:, fg], 90):14.2f}m" + (f"   ({fails} failed)" if fails else ""))


if __name__ == "__main__":
    main()
