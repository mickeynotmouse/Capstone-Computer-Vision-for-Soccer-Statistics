"""Pick known pitch landmarks on ONE frame -> calib.json (the pixel <-> pitch mapping used by track.py).

    python -m pipeline.calibrate goal.mp4 --out calib.json            # add --frame N to use a different frame

The window names one landmark at a time (top of the window). After 4 clicks the real pitch lines are drawn on the frame
in red and redraw after every click, so you can SEE whether the fit is right. Click as many landmarks as you can see,
spread across the frame (4 points bunched in one corner give a shaky fit).

Keys:  n = skip this landmark   v = not visible in frame   u = undo   s = save and quit   esc = cancel

Pitch frame: 105 x 68 m. The goal you are calibrating is at x = 105, and "near" means the touchline the CAMERA stands
beside (y = 68). The turf may carry other sports' lines (orange/yellow): only use the soccer lines.
"""
import argparse
import json

import cv2
import numpy as np

from .review import pitch_lines, poly

LANDMARKS = [
    ("near corner flag (where the pole meets the ground)", (105, 68)),
    ("near 18-yd box: FRONT corner (out on the pitch)", (88.5, 54.16)),
    ("near 18-yd box: corner on the GOAL LINE", (105, 54.16)),
    ("near 6-yd box: FRONT corner", (99.5, 43.16)),
    ("near 6-yd box: corner on the GOAL LINE", (105, 43.16)),
    ("near goalpost (where it meets the ground)", (105, 37.66)),
    ("penalty spot", (94, 34)),
    ("far goalpost (where it meets the ground)", (105, 30.34)),
    ("far 6-yd box: corner on the GOAL LINE", (105, 24.84)),
    ("far 6-yd box: FRONT corner", (99.5, 24.84)),
    ("far 18-yd box: corner on the GOAL LINE", (105, 13.84)),
    ("far 18-yd box: FRONT corner", (88.5, 13.84)),
    ("far corner flag", (105, 0)),
]


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
        self.s = min(1.0, max_w / w, max_h / h)
        self.base = cv2.resize(frame, (int(w * self.s), int(h * self.s))) if self.s < 1 else frame.copy()
        self.k = 0
        self.hist = []          # ("click", k, (px, py)) or ("skip", k, None)

    @property
    def pts(self):
        return [(k, p) for kind, k, p in self.hist if kind == "click"]

    def current(self):
        return LANDMARKS[self.k] if self.k < len(LANDMARKS) else None

    def click(self, x, y):
        if self.current() is not None:
            self.hist.append(("click", self.k, (x / self.s, y / self.s)))
            self.k += 1

    def skip(self):
        if self.current() is not None:
            self.hist.append(("skip", self.k, None))
            self.k += 1

    def not_visible(self):
        """Mark the current landmark as 'not visible in frame' (off-camera / panned out)."""
        if self.current() is not None:
            self.hist.append(("not_visible", self.k, None))
            self.k += 1

    def undo(self):
        if self.hist:
            self.k = self.hist.pop()[1]

    def solve(self):
        """-> (H, rms_error_m) once there are >= 4 points, else None."""
        if len(self.pts) < 4:
            return None
        img = np.float32([p for _, p in self.pts])
        pit = np.float32([LANDMARKS[k][1] for k, _ in self.pts])
        H, _ = cv2.findHomography(img, pit, 0)
        if H is None:
            return None
        got = cv2.perspectiveTransform(img.reshape(-1, 1, 2), H).reshape(-1, 2)
        return H, float(np.sqrt(np.mean(np.sum((got - pit) ** 2, axis=1))))

    def render(self):
        view = self.base.copy()
        sol = self.solve()
        if sol:
            try:
                Hinv = np.linalg.inv(sol[0])
                for line in pitch_lines():
                    p = cv2.perspectiveTransform(line.reshape(-1, 1, 2), Hinv).reshape(-1, 2) * self.s
                    cv2.polylines(view, [poly(p)], False, (0, 0, 255), 2, cv2.LINE_AA)
            except np.linalg.LinAlgError:
                pass
        for i, (k, (px, py)) in enumerate(self.pts):
            c = (int(px * self.s), int(py * self.s))
            cv2.circle(view, c, 6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(view, str(i + 1), (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(view, (0, 0), (view.shape[1], 62), (0, 0, 0), -1)
        cur = self.current()
        l1 = f"click {self.k + 1}/{len(LANDMARKS)}: {cur[0]}" if cur else "no more landmarks - press s to save"
        l2 = (f"fit error {sol[1]:.2f} m over {len(self.pts)} points - red lines should sit on the painted lines"
              if sol else f"{len(self.pts)}/4 points before the fit appears")
        cv2.putText(view, l1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(view, l2 + "   [n skip | v not visible | u undo | s save | esc cancel]", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="calib.json")
    ap.add_argument("--frame", type=int, default=0, help="which frame to calibrate on")
    a = ap.parse_args()

    cal = Calibrator(read_frame(a.video, a.frame))

    def on_click(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and cal.current():
            name = cal.current()[0]
            cal.click(x, y)
            print(f"{len(cal.pts):2d}. {name}: pixel ({x / cal.s:.0f}, {y / cal.s:.0f})")

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
        elif key == ord("s"):
            if cal.solve() is None:
                print("need at least 4 points first")
                continue
            break
    cv2.destroyAllWindows()

    pts = cal.pts
    not_visible = [k for kind, k, _ in cal.hist if kind == "not_visible"]
    json.dump({"image": [[round(p[0], 1), round(p[1], 1)] for _, p in pts],
               "pitch": [list(LANDMARKS[k][1]) for k, _ in pts],
               "not_visible": not_visible,
               "frame": a.frame, "fit_error_m": round(cal.solve()[1], 3)}, open(a.out, "w"))
    print(f"wrote {a.out}  ({len(pts)} points, fit error {cal.solve()[1]:.2f} m)")


if __name__ == "__main__":
    main()