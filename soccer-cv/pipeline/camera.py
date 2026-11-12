"""Follow a moving camera: per-frame pixel <-> reference-frame homographies, from the background alone.

The HUDL camera pans and zooms (up to 22 % zoom and 700 px of shift inside a 12 s clip), so one calibration is only true
for the frame it was clicked on. A camera that turns on a fixed mount maps any view onto any other by a homography, so:

    frame t --(CameraPath)--> calibration frame --(calib.json fit)--> pitch metres

The same trick moves a calibration onto another clip from the same mount (see `register`).

Clips are screen recordings of the HUDL player, so `content_box` also finds the area holding the video, leaving out the
player's controls and browser bars.
"""
import cv2
import numpy as np

WORK_WIDTH = 1440       # features are found at this width: plenty for a homography, and far quicker than 2880
RATIO = 0.75            # Lowe's ratio test
RANSAC_PX = 2.0         # at WORK_WIDTH
MIN_INLIERS = 40        # fewer than this is not trusted
REANCHOR_INLIERS = 150  # below this the current anchor is losing overlap with the view, so hand over to a newer frame
BANNER_PX = 120         # the top of the content can hold the browser's "exit full screen" banner: never use it
LIBRARY_TRIES = 4       # stored views tried when the current one stops fitting
MAX_LIBRARY = 150       # stored views kept (about 1.5 MB each)
JITTER_PX = 3.0         # smoothing the camera path may move the picture at most this much


def content_box(frame, dark=40):
    """(x0, y0, x1, y1) of the video inside a screen recording: the longest run of rows that aren't dark UI bars.
    Uses each row's MEDIAN brightness, so icons and text on a bar don't make it look like picture."""
    bright = np.median(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, ::4], axis=1) > dark
    best, start = (0, 0), None
    for y, b in enumerate(np.r_[bright, False]):
        if b and start is None:
            start = y
        elif not b and start is not None:
            best, start = max(best, (start, y), key=lambda r: r[1] - r[0]), None
    return 0, best[0], frame.shape[1], best[1]


class _Features:
    """SIFT on the picture only, ignoring the UI, the banner and (when given) boxes around moving people."""

    def __init__(self, frame_size):
        self.scale = min(1.0, WORK_WIDTH / frame_size[0])
        self.sift = cv2.SIFT_create(3000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def __call__(self, frame, box, people=()):
        s = self.scale
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame
        mask = np.zeros(small.shape[:2], np.uint8)
        x0, y0, x1, y1 = box
        banner = BANNER_PX if y0 > 20 else 0     # only a screen recording (UI bar above the video) has the banner
        mask[int((y0 + banner) * s):int(y1 * s), int(x0 * s):int(x1 * s)] = 255
        for b in people:
            cv2.rectangle(mask, (int(b[0] * s) - 4, int(b[1] * s) - 4), (int(b[2] * s) + 4, int(b[3] * s) + 4), 0, -1)
        kp, des = self.sift.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), mask)
        return np.float32([k.pt for k in kp]) / s, des

    def match(self, a, b):
        """Homography taking features a's pixels to b's, and how many matches agree: (H or None, inliers)."""
        (pa, da), (pb, db) = a, b
        if da is None or db is None or len(da) < 8 or len(db) < 8:
            return None, 0
        good = [m[0] for m in self.matcher.knnMatch(da, db, k=2) if len(m) == 2 and m[0].distance < RATIO * m[1].distance]
        if len(good) < MIN_INLIERS:
            return None, 0
        src, dst = pa[[m.queryIdx for m in good]], pb[[m.trainIdx for m in good]]
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX / self.scale, maxIters=4000, confidence=0.999)
        n = int(inl.sum()) if inl is not None else 0
        return (H / H[2, 2], n) if H is not None and n >= MIN_INLIERS else (None, n)


def register(frame_from, frame_to):
    """Homography taking pixels of frame_from to pixels of frame_to (two views from the same camera mount), or None."""
    f = _Features(frame_from.shape[1::-1])
    H, n = f.match(f(frame_from, content_box(frame_from)), f(frame_to, content_box(frame_to)))
    return H, n


class CameraPath:
    """to_ref[n] takes pixels of frame n to pixels of the reference frame. Frames between samples are interpolated."""

    def __init__(self, to_ref, inliers):
        self.to_ref, self.inliers = to_ref, inliers

    def __len__(self):
        return len(self.to_ref)

    def __getitem__(self, n):
        return self.to_ref[n]

    @classmethod
    def from_video(cls, video, ref_frame=0, every=3, people_by_frame=None, smooth=2, progress=None):
        """Follow the camera through the whole clip in ONE pass, holding only a small library of views in memory, so a
        45-minute half costs no more memory than a 10-second clip.

        Each sampled frame (every `every`-th) is matched to the current anchor view. When the camera has moved so far
        that the overlap gets thin, it tries the stored views nearest to where the camera now points (so returning to a
        part of the pitch seen before reconnects to the same view, which stops errors building up), and only if none
        fits does the latest frame become a new stored view. Positions are found relative to frame 0, then expressed
        relative to `ref_frame` (the calibration frame), which can be anywhere in the clip.
        people_by_frame: optional {frame: boxes} so players aren't used as background."""
        # always read straight through: on screen recordings (variable frame rate) seeking can land on the wrong frame
        cap = cv2.VideoCapture(video)
        people_by_frame = people_by_frame or {}
        feats, box, n = None, None, 0
        known = {}                 # frame -> (pixels of that frame -> pixels of frame 0, inliers)
        library = []               # stored views: (features, to_world, centre of the view in frame-0 pixels)
        anchor = last = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if feats is None:
                feats, box = _Features(frame.shape[1::-1]), content_box(frame)
                centre = np.float64([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]])
            sample = n % every == 0 or n == ref_frame
            if sample:
                cur = feats(frame, box, people_by_frame.get(n, ()))
                if anchor is None:
                    T, inl = np.eye(3), 9999
                    anchor = (cur, T, apply(T, centre)[0])
                    library.append(anchor)
                else:
                    H, inl = feats.match(cur, anchor[0])
                    T = anchor[1] @ H if H is not None else None
                    if inl < REANCHOR_INLIERS:
                        # where is the camera pointing now? try the stored views nearest to that
                        guess = apply(last[1], centre)[0]
                        near = sorted(library, key=lambda v: np.linalg.norm(v[2] - guess))[:LIBRARY_TRIES]
                        for view in near:
                            if view is anchor:
                                continue
                            H2, inl2 = feats.match(cur, view[0])
                            if H2 is not None and inl2 > inl:
                                T, inl, anchor = view[1] @ H2, inl2, view
                        if inl < REANCHOR_INLIERS and last[0] is not anchor[0]:
                            anchor = (last[0], last[1], apply(last[1], centre)[0])
                            library.append(anchor)
                            del library[:-MAX_LIBRARY]
                            H, inl = feats.match(cur, anchor[0])
                            T = anchor[1] @ H if H is not None else None
                    if T is None:
                        n += 1
                        continue
                    T = T / T[2, 2]
                known[n] = (T, inl)
                last = (cur, T)
            n += 1
            if progress and n % 1000 == 0:
                progress(n)
        cap.release()
        if ref_frame not in known:
            raise SystemExit(f"could not follow the camera to frame {ref_frame} of {video}")
        to_ref_world = np.linalg.inv(known[ref_frame][0])
        rel = {f: (to_ref_world @ T / (to_ref_world @ T)[2, 2], inl) for f, (T, inl) in known.items()}
        return cls.from_samples(rel, n, smooth)

    @classmethod
    def from_samples(cls, known, total, smooth=2):
        """{frame: (H, inliers)} -> a path for all frames: smoothed over +-`smooth` samples, then linearly interpolated.
        Comparing entries is fair because every H is scaled so H[2,2] = 1 and the motion is small between samples."""
        ns = np.array(sorted(known))
        Hs = np.array([known[n][0].ravel() for n in ns])
        if smooth and len(ns) > 2 * smooth + 1:
            # a local quadratic fit (Savitzky-Golay), not a plain average: it takes out jitter but follows a steady pan
            # and a change of direction exactly, where an average cuts the corner (40 px off on a test pan). The
            # samples nearest the ends, which have no neighbours on one side, are left as measured.
            # Smoothing may only take out jitter: where it would move the picture by more than JITTER_PX (a sudden
            # change of direction), the measured value is kept.
            k = _savgol(smooth)
            inner = np.stack([np.convolve(Hs[:, j], k[::-1], mode="valid") for j in range(9)], 1)
            raw = Hs[smooth:len(ns) - smooth]
            probe = np.float64([[0, 0], [1000, 0], [0, 600], [1000, 600]])
            moved = np.array([np.abs(apply(a.reshape(3, 3), probe) - apply(b.reshape(3, 3), probe)).max()
                              for a, b in zip(inner, raw)])
            Hs = Hs.copy()
            Hs[smooth:len(ns) - smooth] = np.where((moved <= JITTER_PX)[:, None], inner, raw)
        all_n = np.arange(total)
        out = np.stack([np.interp(all_n, ns, Hs[:, j]) for j in range(9)], 1).reshape(-1, 3, 3)
        inl = np.interp(all_n, ns, [known[n][1] for n in ns])
        return cls(out, inl)

    def save(self, path):
        np.savez_compressed(path, to_ref=self.to_ref, inliers=self.inliers)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["to_ref"], d["inliers"])


def _savgol(half):
    """Weights of a quadratic least-squares fit over 2*half+1 evenly spaced samples, evaluated at the middle one."""
    x = np.arange(-half, half + 1, dtype=np.float64)
    A = np.c_[np.ones_like(x), x, x ** 2]
    return np.linalg.pinv(A)[0]


def apply(H, pts):
    """(k,2) points through a 3x3 homography."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = np.c_[pts, np.ones(len(pts))] @ H.T
    return q[:, :2] / q[:, 2:]


def warp_boxes(H, xyxy):
    """(k,4) boxes through a homography, as axis-aligned boxes around the moved corners."""
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    if not len(xyxy):
        return xyxy
    c = xyxy[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 2)
    p = apply(H, c).reshape(-1, 4, 2)
    return np.c_[p.min(1), p.max(1)]
