"""Homography fit and calibrator logic on synthetic clicks with a known answer. Run: python -m pytest -q"""
import json

import cv2

import numpy as np

from pipeline import calibrate, homography as hg

# a camera beside the near touchline looking at the right-hand goal (pitch -> image), roughly like the test clips
G_TRUE = np.array([[-12.0, 30.0, 1800.0], [1.5, 6.0, 300.0], [-0.002, 0.012, 1.0]])
H_TRUE = np.linalg.inv(G_TRUE)
PTS = np.array([[105, 68], [105, 37.66], [94, 34], [88.5, 54.16], [99.5, 24.84], [88.5, 13.84], [52.5, 68], [52.5, 34]])


def _close(H, at=((60, 60), (80, 40), (100, 20)), tol=1e-6):
    at = np.float64(at)
    return np.abs(hg.to_pitch(H, hg.to_image(H_TRUE, at)) - at).max() < tol


def test_exact_recovery_points_and_lines():
    lines = np.array([[70, 68], [60, 68], [52.5, 50]])
    obs = hg.Obs(hg.to_image(H_TRUE, PTS[:4]), PTS[:4], hg.to_image(H_TRUE, lines),
                 [hg.line("y", 68), hg.line("y", 68), hg.line("x", 52.5)])
    assert _close(hg.fit(obs))
    assert hg.residuals_px(hg.fit(obs), obs).max() < 1e-6


def test_line_clicks_stand_in_for_hidden_corners():
    """3 points + line clicks (not enough points alone) still pin the pitch down."""
    lines = np.array([[70, 68], [60, 68], [52.5, 50], [52.5, 20], [100, 13.84]])
    obs = hg.Obs(hg.to_image(H_TRUE, PTS[:3]), PTS[:3], hg.to_image(H_TRUE, lines),
                 [hg.line("y", 68)] * 2 + [hg.line("x", 52.5)] * 2 + [hg.line("y", 13.84)])
    assert _close(hg.fit(obs))


def test_goal_line_only_is_rejected():
    on_line = np.array([[105, 68], [105, 37.66], [105, 30.34], [105, 0]])
    obs = hg.Obs(hg.to_image(H_TRUE, on_line), on_line)
    assert obs.off_goal_line() == 0
    assert hg.fit(obs) is None


def test_one_misclick_barely_moves_the_robust_fit():
    rng = np.random.default_rng(0)
    obs = hg.Obs(hg.to_image(H_TRUE, PTS) + rng.normal(0, 1, (len(PTS), 2)), PTS)
    obs.pts[2] += [25, -10]
    lsq = hg.Obs(obs.pts, obs.pit)
    H_old, _ = cv2.findHomography(np.float32(lsq.pts), np.float32(lsq.pit), 0)
    at = np.float64([[60, 60], [80, 40]])
    err = lambda H: np.abs(hg.to_pitch(H, hg.to_image(H_TRUE, at)) - at).max()   # noqa: E731
    assert err(hg.fit(obs)) < 0.5 < err(H_old)
    assert np.argmax(hg.residuals_px(hg.fit(obs), obs)) == 2   # the misclick stands out instead of being spread around


def test_leave_one_out_flags_the_misclick():
    obs = hg.Obs(hg.to_image(H_TRUE, PTS), PTS)
    obs.pts[3] += [20, 0]
    loo = hg.leave_one_out(obs, hg.fit(obs))
    assert np.nanargmax(loo) == 3


def test_error_map_shrinks_with_midfield_clicks():
    goal = hg.Obs(hg.to_image(H_TRUE, PTS[:6]), PTS[:6])
    both = hg.Obs(hg.to_image(H_TRUE, PTS), PTS)
    px = hg.to_image(H_TRUE, np.float64([[60, 60], [70, 50], [55, 40]]))
    assert np.median(hg.error_map(both, hg.fit(both), px)) < np.median(hg.error_map(goal, hg.fit(goal), px))


def test_calibrator_flow_and_saved_file_round_trip(tmp_path):
    calibrate.LANDMARKS[:] = calibrate.landmarks()
    cal = calibrate.Calibrator(np.zeros((1800, 2880, 3), np.uint8))
    names = [name for name, _, _ in calibrate.LANDMARKS]
    clicks = {names.index("near corner flag (where the pole meets the ground)"): (105, 68),
              names.index("penalty spot"): (94, 34),
              names.index("near 18-yd box: FRONT corner (out on the pitch)"): (88.5, 54.16),
              names.index("far 18-yd box: FRONT corner (out on the pitch)"): (88.5, 13.84),
              names.index("centre spot"): (52.5, 34)}
    near_tl = next(i for i, (_, kind, v) in enumerate(calibrate.LANDMARKS) if kind == "line" and v == hg.line("y", 68))
    while cal.current() is not None:
        if cal.k in clicks:
            cal.click(*hg.to_image(H_TRUE, np.float64([clicks[cal.k]]))[0] * cal.s)
        elif cal.k == near_tl:
            for x in (60, 75, 90):
                cal.click(*hg.to_image(H_TRUE, np.float64([[x, 68]]))[0] * cal.s)
            assert cal.k == near_tl                     # line clicks don't advance
            cal.skip()
        else:
            cal.not_visible()
    assert len(cal.pts) == 5 and len(cal.line_pts) == 3
    cal.undo()                                          # undo the last "not visible", then step forward again
    cal.not_visible()
    path = tmp_path / "calib.json"
    json.dump(cal.to_json(0, 68.0, 105.0), open(path, "w"))
    saved = json.load(open(path))
    assert saved["line_pitch"] == [hg.line("y", 68)] * 3
    assert saved["fit_rms_px"] < 0.2                    # only rounding of the saved pixels
    assert _close(hg.load(path), tol=0.05)


def test_old_calibration_files_still_load():
    assert hg.load("calib.json").shape == (3, 3)
    assert hg.load("calib_wide.json").shape == (3, 3)
