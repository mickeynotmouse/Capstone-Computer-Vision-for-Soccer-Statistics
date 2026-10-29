"""Fit the pixel <-> pitch homography from clicked landmarks and lines, and measure how far to trust it.

Two kinds of click:
  points  a pixel whose pitch position is known (corner flag, penalty spot, centre spot, ...)
  lines   a pixel known only to lie somewhere ON a pitch line (any spot along the near touchline means y = 68).
          Lines are long and rarely hidden by players, so they are easier to click accurately than corners.

The fit minimises error in IMAGE PIXELS, where click noise is the same everywhere. (Fitting in pitch metres, as
cv2.findHomography(image, pitch) does, lets a 2 px slip on a far landmark count as metres and dominate the fit.)
A Huber loss stops a single misclick from dragging the whole fit.

The quality checks are honest ones, not the in-sample fit residual:
  leave_one_out  refit without each click and measure how far off that click lands, in metres
  error_map      re-fit many times with simulated click noise and measure how much each pixel's pitch position
                 moves: shows where on the pitch the calibration can be trusted
"""
import json

import numpy as np

HUBER_PX = 3.0         # a click further off than this counts linearly, so one misclick can't dominate
CLICK_NOISE_PX = 2.0   # assumed click accuracy (about 1 screen pixel in the calibrator window)
GOAL_LINE_X = 105.0

# fixed scaling of pitch metres to numbers of order 1, so the solve is well conditioned
_TP = np.array([[1 / 30, 0, -80 / 30], [0, 1 / 30, -34 / 30], [0, 0, 1]])


def line(axis, value):
    """Pitch line x = value or y = value as [a, b, c] with a*X + b*Y + c = 0."""
    return [1.0, 0.0, -value] if axis == "x" else [0.0, 1.0, -value]


class Obs:
    """Clicked observations.  pts (n,2) pixels at known pitch points pit (n,2) metres;
    lpx (m,2) pixels lying on the pitch lines lco (m,3) = [a, b, c]."""

    def __init__(self, pts=(), pit=(), lpx=(), lco=()):
        self.pts, self.pit = np.float64(pts).reshape(-1, 2), np.float64(pit).reshape(-1, 2)
        self.lpx, self.lco = np.float64(lpx).reshape(-1, 2), np.float64(lco).reshape(-1, 3)

    @classmethod
    def from_calib(cls, c):
        return cls(c["image"], c["pitch"], c.get("line_image", ()), c.get("line_pitch", ()))

    def __len__(self):
        return len(self.pts) + len(self.lpx)

    def without(self, i):
        """Copy with observation i removed (points first, then line clicks)."""
        n = len(self.pts)
        if i < n:
            return Obs(np.delete(self.pts, i, 0), np.delete(self.pit, i, 0), self.lpx, self.lco)
        return Obs(self.pts, self.pit, np.delete(self.lpx, i - n, 0), np.delete(self.lco, i - n, 0))

    def subset(self, keep_pts, keep_lines):
        return Obs(self.pts[keep_pts], self.pit[keep_pts], self.lpx[keep_lines], self.lco[keep_lines])

    def jitter(self, rng, sigma):
        return Obs(self.pts + rng.normal(0, sigma, self.pts.shape), self.pit,
                   self.lpx + rng.normal(0, sigma, self.lpx.shape), self.lco)

    def off_goal_line(self):
        """Equations that pin down depth: points off the goal line count 2, clicks on other lines count 1."""
        return (2 * int(np.sum(np.abs(self.pit[:, 0] - GOAL_LINE_X) > 2.0))
                + int(np.sum(~((np.abs(self.lco[:, 0]) > 0.5) & (np.abs(self.lco[:, 2] + GOAL_LINE_X) < 2.0)))))


def to_pitch(H, px):
    """(k,2) pixels -> (k,2) pitch metres."""
    q = np.c_[px, np.ones(len(px))] @ H.T
    return q[:, :2] / q[:, 2:]


def to_image(H, pit):
    """(k,2) pitch metres -> (k,2) pixels."""
    return to_pitch(np.linalg.inv(H), pit)


def _img_norm(px):
    c = px.mean(0)
    s = np.sqrt(2) / max(np.mean(np.linalg.norm(px - c, axis=1)), 1e-9)
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])


def dlt(obs):
    """Linear least-squares solve (no iteration) -> image->pitch H, or None if the clicks don't pin it down."""
    if len(obs.pts) + len(obs.lpx) == 0 or 2 * len(obs.pts) + len(obs.lpx) < 8:
        return None
    Ti = _img_norm(np.r_[obs.pts, obs.lpx])
    u = np.c_[np.r_[obs.pts, obs.lpx], np.ones(len(obs))] @ Ti.T
    P = np.c_[obs.pit, np.ones(len(obs.pit))] @ _TP.T
    L = obs.lco @ np.linalg.inv(_TP)          # lines transform with the inverse scaling
    rows, z = [], np.zeros(3)
    for (X, Y, _), ui in zip(P, u[:len(obs.pts)]):
        rows += [np.r_[ui, z, -X * ui], np.r_[z, ui, -Y * ui]]
    for (a, b, c), ui in zip(L, u[len(obs.pts):]):
        rows.append(np.r_[a * ui, b * ui, c * ui])
    A = np.array(rows)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    _, s, Vt = np.linalg.svd(A)
    if s[7] / s[0] < 1e-6:
        return None                            # degenerate, e.g. every click on the goal line
    H = np.linalg.inv(_TP) @ Vt[-1].reshape(3, 3) @ Ti
    return H / np.linalg.norm(H)


def _residuals(H, obs):
    """Pixel residuals: 2 per point (reprojection), 1 per line click (distance to the projected line)."""
    r_pts = to_image(H, obs.pit) - obs.pts
    l_img = obs.lco @ H                        # each pitch line, as a line in the image
    r_lin = (np.c_[obs.lpx, np.ones(len(obs.lpx))] * l_img).sum(1) / np.linalg.norm(l_img[:, :2], axis=1)
    return r_pts, r_lin


def residuals_px(H, obs):
    """Per-observation error in pixels (points first, then line clicks)."""
    r_pts, r_lin = _residuals(H, obs)
    return np.r_[np.linalg.norm(r_pts, axis=1), np.abs(r_lin)]


def fit(obs, init=None, iters=50):
    """Robust pixel-space fit -> image->pitch H (3x3), or None if the clicks don't pin it down."""
    H0 = dlt(obs) if init is None else init
    if H0 is None:
        return None
    Ti = _img_norm(np.r_[obs.pts, obs.lpx])
    Tpi, Tii = np.linalg.inv(_TP), np.linalg.inv(Ti)
    Hn = _TP @ H0 @ Tii
    p = (Hn / Hn[2, 2]).ravel()[:8]
    unpack = lambda p: Tpi @ np.r_[p, 1.0].reshape(3, 3) @ Ti   # noqa: E731

    def weighted(p):
        r_pts, r_lin = _residuals(unpack(p), obs)
        mag = np.r_[np.linalg.norm(r_pts, axis=1), np.abs(r_lin)]
        w = np.sqrt(np.minimum(1.0, HUBER_PX / np.maximum(mag, 1e-12)))   # IRLS weights for the Huber loss
        cost = np.sum(np.where(mag <= HUBER_PX, mag ** 2 / 2, HUBER_PX * (mag - HUBER_PX / 2)))
        return np.r_[(r_pts * w[:len(r_pts), None]).ravel(), r_lin * w[len(r_pts):]], cost

    r, cost, lam = *weighted(p), 1e-3
    for _ in range(iters):
        J = np.empty((len(r), 8))
        for j in range(8):
            d = np.zeros(8)
            d[j] = 1e-7 * max(1.0, abs(p[j]))
            J[:, j] = (weighted(p + d)[0] - r) / d[j]
        A, g = J.T @ J, J.T @ r
        while True:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            r2, cost2 = weighted(p + step)
            if np.isfinite(cost2) and cost2 <= cost:
                p, r, lam = p + step, r2, max(lam / 10, 1e-9)
                break
            lam *= 10
            if lam > 1e8:
                break
        if lam > 1e8 or abs(cost - cost2) < 1e-10 * max(cost, 1e-12):
            cost = min(cost, cost2)
            break
        cost = cost2
    H = unpack(p)
    # every click must be on the camera's side of the horizon, or the fit has flipped
    w = (np.c_[np.r_[obs.pts, obs.lpx], np.ones(len(obs))] @ H.T)[:, 2]
    return H / np.linalg.norm(H) if (np.all(w > 0) or np.all(w < 0)) else None


def leave_one_out(obs, H, fitter=None):
    """Refit without each click in turn; how far (metres) does that click land from where it should?
    Points: distance to their known spot.  Line clicks: distance to their line.  nan = can't refit without it."""
    fitter = fitter or (lambda o: fit(o, init=H))
    out = []
    for i in range(len(obs)):
        Hi = fitter(obs.without(i)) if obs.without(i).off_goal_line() >= 4 else None
        if Hi is None:
            out.append(np.nan)
        elif i < len(obs.pts):
            out.append(float(np.linalg.norm(to_pitch(Hi, obs.pts[i:i + 1])[0] - obs.pit[i])))
        else:
            (X, Y), (a, b, c) = to_pitch(Hi, obs.lpx[i - len(obs.pts):][:1])[0], obs.lco[i - len(obs.pts)]
            out.append(float(abs(a * X + b * Y + c) / np.hypot(a, b)))
    return np.array(out)


def pitch_grid(H, size, bounds, step=48):
    """Pixels on a grid over the frame that land on the pitch (bounds = x0, x1, y0, y1 in metres, 2 m margin)."""
    w, h = size
    xs, ys = np.meshgrid(np.arange(step / 2, w, step), np.arange(step / 2, h, step))
    px = np.c_[xs.ravel(), ys.ravel()]
    q = np.c_[px, np.ones(len(px))] @ H.T
    pit = q[:, :2] / q[:, 2:]
    x0, x1, y0, y1 = bounds
    ok = (pit[:, 0] > x0 - 2) & (pit[:, 0] < x1 + 2) & (pit[:, 1] > y0 - 2) & (pit[:, 1] < y1 + 2)
    return px[ok]


def error_map(obs, H, px, sigma=CLICK_NOISE_PX, trials=40, fitter=None, seed=0):
    """Expected position error (metres, 1 standard deviation) at each pixel in px, from re-fitting with simulated
    click noise of `sigma` pixels. Depends on WHERE the clicks are, not on how well they were clicked."""
    fitter = fitter or (lambda o: fit(o, init=H))
    rng = np.random.default_rng(seed)
    runs = [to_pitch(Hk, px) for Hk in (fitter(obs.jitter(rng, sigma)) for _ in range(trials)) if Hk is not None]
    if len(runs) < 2:
        return np.full(len(px), np.nan)
    runs = np.array(runs)
    return np.sqrt(runs.var(0).sum(1))


def load(path):
    """calib.json -> image->pitch H, using every point and line click in the file."""
    obs = Obs.from_calib(json.load(open(path)))
    H = fit(obs)
    if H is None:
        raise SystemExit(f"{path}: these clicks don't pin down the pitch; re-run pipeline.calibrate")
    return H
