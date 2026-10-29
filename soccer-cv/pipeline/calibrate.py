"""Pick known pitch landmarks and lines on ONE frame -> calib.json (the pixel <-> pitch mapping used by track.py).

    python -m pipeline.calibrate goal.mp4 --out calib.json            # add --frame N to use a different frame

The window names one item at a time (top of the window). Two kinds:
  POINTS  click the landmark itself (a corner, a spot). Clicking moves on to the next item.
  LINES   click as many spots as you like ALONG that painted line (anywhere: no need to find a corner), then press n.
          Lines are the easiest thing to click accurately and are rarely hidden by players: use them.

A magnifier in the top corner shows the full-resolution picture under the mouse; the centre of its red crosshair is
exactly where the click lands. If you can't SEE a corner (hidden by a player, too far away), press v rather than
guessing: a guessed corner hurts more than a missing one, and clicks along the lines will make up for it.

Spread your clicks over the WHOLE frame: the goal end AND midfield (halfway line, centre circle, near touchline).
Clicks bunched at the far goal give a fit that looks fine there but is metres off in the foreground.

The status bar shows three honest numbers (see pipeline/homography.py):
  fit             how far, in pixels, the clicks sit from the fitted pitch (big = a misclick, or wrong field size)
  held-out worst  refit without each click and see how far off it lands, in metres
  expected error  how far a player's position could be off, across the visible pitch (median / 90th percentile).
                  Press e to see it as coloured dots: green < 0.5 m, yellow < 1.5 m, red beyond. Add clicks where it's red.

Keys:  n = skip / next line   v = not visible in frame   u = undo   l = red pitch lines   e = error map
       s = save and quit   esc = cancel

Pitch frame: the goal you are calibrating is at x = 105, the halfway line at x = 105 - length/2, and "near" means
the touchline the CAMERA stands beside (y = 34 + width/2). The turf may carry other sports' lines: only use the soccer lines.
"""
import argparse
import json

import cv2
import numpy as np

from . import homography as hg
from .review import pitch_lines, poly

ARC_Y = float(np.sqrt(9.15 ** 2 - 5.5 ** 2))   # the penalty arc meets the 18-yd box line this far either side of centre


def landmarks(width=68.0, length=105.0):
    """(name, kind, value): kind "point" -> value = (x, y) metres;  kind "line" -> value = [a, b, c].
    Only the touchlines/flags depend on the field's width and the halfway line on its length; everything else is
    fixed by the laws of the game. Order matters: the points that pin down DEPTH come right after the flags and posts."""
    near, far, half = 34 + width / 2, 34 - width / 2, 105 - length / 2
    pt = lambda name, x, y: (name, "point", (x, y))        # noqa: E731
    ln = lambda name, axis, v: (name, "line", hg.line(axis, v))   # noqa: E731
    return [
        pt("near corner flag (where the pole meets the ground)", 105, near),
        pt("near goalpost (where it meets the ground)", 105, 37.66),
        pt("far goalpost (where it meets the ground)", 105, 30.34),
        pt("far corner flag (where the pole meets the ground)", 105, far),
        pt("penalty spot", 94, 34),
        pt("far 18-yd box: FRONT corner (out on the pitch)", 88.5, 13.84),
        pt("near 18-yd box: FRONT corner (out on the pitch)", 88.5, 54.16),
        pt("far 6-yd box: FRONT corner", 99.5, 24.84),
        pt("near 6-yd box: FRONT corner", 99.5, 43.16),
        pt("near 18-yd box: corner on the GOAL LINE", 105, 54.16),
        pt("far 18-yd box: corner on the GOAL LINE", 105, 13.84),
        pt("near 6-yd box: corner on the GOAL LINE", 105, 43.16),
        pt("far 6-yd box: corner on the GOAL LINE", 105, 24.84),
        pt("penalty ARC meets the 18-yd box line, NEAR side", 88.5, 34 + ARC_Y),
        pt("penalty ARC meets the 18-yd box line, FAR side", 88.5, 34 - ARC_Y),
        pt("HALFWAY line meets the NEAR touchline", half, near),
        pt("HALFWAY line meets the FAR touchline", half, far),
        pt("centre spot", half, 34),
        pt("centre CIRCLE meets the halfway line, NEAR side", half, 34 + 9.15),
        pt("centre CIRCLE meets the halfway line, FAR side", half, 34 - 9.15),
        ln("NEAR touchline (the long line on the camera's side)", "y", near),
        ln("FAR touchline", "y", far),
        ln("HALFWAY line", "x", half),
        ln("GOAL line", "x", 105),
        ln("18-yd box FRONT line (parallel to the goal line)", "x", 88.5),
        ln("NEAR 18-yd box SIDE line", "y", 54.16),
        ln("FAR 18-yd box SIDE line", "y", 13.84),
        ln("6-yd box FRONT line", "x", 99.5),
    ]


LANDMARKS = landmarks()   # replaced in main() when --width / --length are given


def read_frame(path, index=0):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {index} from {path}")
    return img


class Calibrator:
    """All the logic, separate from the window so it can be tested. Clicks arrive in DISPLAY pixels and are converted
    back to real video pixels, so it is correct on any screen (including Retina) and any video size (including 4K)."""

    def __init__(self, frame, max_w=1400, max_h=800):
        h, w = frame.shape[:2]
        self.size = (w, h)
        self.s = min(1.0, max_w / w, max_h / h)
        self.base = cv2.resize(frame, (int(w * self.s), int(h * self.s))) if self.s < 1 else frame.copy()
        self.k = 0
        self.hist = []          # ("click", k, (px, py)), ("skip", k, None) or ("not_visible", k, None)
        self.show_lines = False # red check-lines are hidden while clicking; press l to preview them
        self.show_errors = False
        self._cache = {}
        self.mouse = None       # display pixel under the mouse, for the magnifier
        self.padded = cv2.copyMakeBorder(frame, 36, 36, 36, 36, cv2.BORDER_CONSTANT)   # = the magnifier's `half`

    def _clicks(self, kind):
        return [(k, p) for kind_, k, p in self.hist if kind_ == "click" and LANDMARKS[k][1] == kind]

    @property
    def pts(self):
        return self._clicks("point")

    @property
    def line_pts(self):
        return self._clicks("line")

    def current(self):
        return LANDMARKS[self.k] if self.k < len(LANDMARKS) else None

    def click(self, x, y):
        if self.current() is not None:
            self.hist.append(("click", self.k, (x / self.s, y / self.s)))
            if self.current()[1] == "point":   # a line takes as many clicks as you like; n moves on
                self.k += 1

    def skip(self):
        if self.current() is not None:
            self.hist.append(("skip", self.k, None))
            self.k += 1

    def not_visible(self):
        """Mark the current item as 'not visible in frame' (off-camera / panned out)."""
        if self.current() is not None:
            self.hist.append(("not_visible", self.k, None))
            self.k += 1

    def undo(self):
        if self.hist:
            self.k = self.hist.pop()[1]

    def obs(self):
        return hg.Obs([p for _, p in self.pts], [LANDMARKS[k][2] for k, _ in self.pts],
                      [p for _, p in self.line_pts], [LANDMARKS[k][2] for k, _ in self.line_pts])

    def bounds(self):
        """Whole pitch (x0, x1, y0, y1) in metres: halfway is the lowest landmark x, the far goal line twice as far."""
        xs = [v[0] for _, kind, v in LANDMARKS if kind == "point"]
        ys = [v[1] for _, kind, v in LANDMARKS if kind == "point"]
        return 2 * min(xs) - 105, max(xs), min(ys), max(ys)

    def solve(self):
        """-> {"H", "rms_px", "loo_m", "grid", "err_m"} once the clicks pin the pitch down, else None. Cached per click."""
        key = tuple((k, p) for kind, k, p in self.hist if kind == "click")
        if key not in self._cache:
            self._cache[key] = self._solve()
        return self._cache[key]

    def _solve(self):
        obs = self.obs()
        if obs.off_goal_line() < 4:
            return None   # flags + posts all sit on the goal line: need 2 points (or line clicks) OFF it for depth
        H = hg.fit(obs)
        if H is None:
            return None
        grid = hg.pitch_grid(H, self.size, self.bounds())
        return {"H": H, "rms_px": float(np.sqrt(np.mean(hg.residuals_px(H, obs) ** 2))),
                "loo_m": hg.leave_one_out(obs, H), "grid": grid, "err_m": hg.error_map(obs, H, grid)}

    def render(self):
        view = self.base.copy()
        sol = self.solve()
        if sol and self.show_errors:
            for (gx, gy), e in zip(sol["grid"], sol["err_m"]):
                col = (0, 200, 0) if e < 0.5 else (0, 220, 255) if e < 1.5 else (0, 0, 255)
                cv2.circle(view, (int(gx * self.s), int(gy * self.s)), 4, col, -1, cv2.LINE_AA)
        if sol and self.show_lines:
            try:
                Hinv = np.linalg.inv(sol["H"])
                for line in pitch_lines():
                    p = cv2.perspectiveTransform(line.reshape(-1, 1, 2), Hinv).reshape(-1, 2) * self.s
                    cv2.polylines(view, [poly(p)], False, (0, 0, 255), 2, cv2.LINE_AA)
            except np.linalg.LinAlgError:
                pass
        for _, (px, py) in self.pts:     # small dots so earlier clicks don't hide the lines still to be clicked
            cv2.circle(view, (int(px * self.s), int(py * self.s)), 2, (0, 255, 255), -1, cv2.LINE_AA)
        for _, (px, py) in self.line_pts:
            cv2.circle(view, (int(px * self.s), int(py * self.s)), 2, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.rectangle(view, (0, 0), (view.shape[1], 88), (0, 0, 0), -1)
        cur = self.current()
        if cur is None:
            l1 = "no more items - press s to save"
        elif cur[1] == "point":
            l1 = f"{self.k + 1}/{len(LANDMARKS)} POINT: {cur[0]}"
        else:
            n = sum(1 for k, _ in self.line_pts if k == self.k)
            l1 = f"{self.k + 1}/{len(LANDMARKS)} LINE: click spots along the {cur[0]}  ({n} so far, n = next line)"
        if sol:
            loo = sol["loo_m"][np.isfinite(sol["loo_m"])]
            l2 = (f"fit {sol['rms_px']:.1f} px over {len(self.pts)} points + {len(self.line_pts)} line clicks   "
                  + (f"held-out worst {loo.max():.2f} m" if len(loo) else "held-out n/a (need more clicks)"))
            l3 = (f"expected error on the pitch: median {np.nanmedian(sol['err_m']):.2f} m, "
                  f"90% {np.nanpercentile(sol['err_m'], 90):.2f} m   (e = map)" if len(sol["grid"]) else "")
        else:
            l2, l3 = f"{len(self.pts)} points, {len(self.line_pts)} line clicks - need 4 points, 2 of them off the goal line (or line clicks)", ""
        cv2.putText(view, l1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(view, l2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(view, l3 + "   [n skip/next | v not visible | u undo | l lines | e errors | s save | esc]",
                    (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        self._draw_loupe(view)
        return view

    def _draw_loupe(self, view, half=36, zoom=4):
        """Close-up of the FULL-resolution frame around the mouse, in a top corner away from it. The crosshair centre
        is exactly where a click lands, so corners can be placed precisely instead of guessed."""
        if self.mouse is None:
            return
        mx = min(max(self.mouse[0] / self.s, 0), self.size[0] - 1)
        my = min(max(self.mouse[1] / self.s, 0), self.size[1] - 1)
        patch = self.padded[int(my):int(my) + 2 * half, int(mx):int(mx) + 2 * half]   # padded by `half`: no edge cases
        big = cv2.resize(patch, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        for (px, py), col in [(p, (0, 255, 255)) for _, p in self.pts] + [(p, (255, 0, 255)) for _, p in self.line_pts]:
            dx, dy = (px - mx + half) * zoom, (py - my + half) * zoom
            if 0 <= dx < big.shape[1] and 0 <= dy < big.shape[0]:
                cv2.circle(big, (int(dx), int(dy)), 3, col, -1, cv2.LINE_AA)
        c, gap = half * zoom, 8
        for a, b in (((0, c), (c - gap, c)), ((c + gap, c), (2 * c, c)), ((c, 0), (c, c - gap)), ((c, c + gap), (c, 2 * c))):
            cv2.line(big, a, b, (0, 0, 255), 1)
        n = big.shape[0]
        x0 = 10 if self.mouse[0] > view.shape[1] - n - 40 else view.shape[1] - n - 10
        y0 = 96
        if y0 + n <= view.shape[0] and x0 + n <= view.shape[1]:
            view[y0:y0 + n, x0:x0 + n] = big
            cv2.rectangle(view, (x0, y0), (x0 + n, y0 + n), (255, 255, 255), 1)

    def to_json(self, frame_index, width, length):
        sol = self.solve()
        loo = sol["loo_m"][np.isfinite(sol["loo_m"])]
        r = lambda p: [round(p[0], 1), round(p[1], 1)]   # noqa: E731
        return {"image": [r(p) for _, p in self.pts],
                "pitch": [list(LANDMARKS[k][2]) for k, _ in self.pts],
                "line_image": [r(p) for _, p in self.line_pts],
                "line_pitch": [list(LANDMARKS[k][2]) for k, _ in self.line_pts],
                "not_visible": [k for kind, k, _ in self.hist if kind == "not_visible"],
                "frame": frame_index, "frame_size": list(self.size), "pitch_width": width, "pitch_length": length,
                "fit_rms_px": round(sol["rms_px"], 2),
                "held_out_worst_m": round(float(loo.max()), 3) if len(loo) else None,
                "expected_error_m": {"median": round(float(np.nanmedian(sol["err_m"])), 3),
                                     "p90": round(float(np.nanpercentile(sol["err_m"], 90)), 3)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="calib.json")
    ap.add_argument("--frame", type=int, default=0, help="which frame to calibrate on")
    ap.add_argument("--width", type=float, default=68.0, help="real field width in metres (touchline to touchline)")
    ap.add_argument("--length", type=float, default=105.0, help="real field length in metres (goal line to goal line)")
    a = ap.parse_args()
    LANDMARKS[:] = landmarks(a.width, a.length)

    cal = Calibrator(read_frame(a.video, a.frame))

    def on_click(ev, x, y, *_):
        cal.mouse = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN and cal.current():
            name = cal.current()[0]
            cal.click(x, y)
            print(f"{len(cal.hist):2d}. {name}: pixel ({x / cal.s:.0f}, {y / cal.s:.0f})")

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_click)
    while True:
        cv2.imshow("calibrate", cal.render())
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            raise SystemExit("cancelled")
        if key == ord("n"):
            cal.skip()
        elif key == ord("v"):
            cal.not_visible()
        elif key == ord("u"):
            cal.undo()
        elif key == ord("l"):
            cal.show_lines = not cal.show_lines
        elif key == ord("e"):
            cal.show_errors = not cal.show_errors
        elif key == ord("s"):
            if cal.solve() is None:
                print("need at least 4 points (2 off the goal line), or line clicks, first")
                continue
            break
    cv2.destroyAllWindows()

    out = cal.to_json(a.frame, a.width, a.length)
    json.dump(out, open(a.out, "w"))
    print(f"wrote {a.out}  ({len(out['image'])} points + {len(out['line_image'])} line clicks)\n"
          f"  fit {out['fit_rms_px']} px | held-out worst {out['held_out_worst_m']} m | "
          f"expected error median {out['expected_error_m']['median']} m, 90% {out['expected_error_m']['p90']} m")


if __name__ == "__main__":
    main()
