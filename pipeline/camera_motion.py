"""Does the camera move during the clip?  (One calibration is only valid if it doesn't.)

    python -m pipeline.camera_motion goal.mp4 [--every 1.0]

Compares the background of each sampled frame (trees, buildings, fences, goal frame: anything that doesn't move) with
frame 0 and prints how far the image has shifted and zoomed. Players are ignored (RANSAC keeps only the consistent background).

Reading it:  shift under ~3 px and zoom within ~0.5%  -> locked-off camera, one calibration works for the whole clip.
             anything bigger -> the operator is panning/zooming; positions in metres will drift unless the calibration follows the camera.
"""
import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between samples")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, ref = cap.read()
    if not ok:
        raise SystemExit("could not read the video")
    h, w = ref.shape[:2]
    s = min(1.0, 1280 / w)
    prep = lambda im: cv2.cvtColor(cv2.resize(im, None, fx=s, fy=s) if s < 1 else im, cv2.COLOR_BGR2GRAY)  # noqa: E731

    orb = cv2.ORB_create(4000)
    kp0, d0 = orb.detectAndCompute(prep(ref), None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    centre = np.float32([[[w * s / 2, h * s / 2]]])
    print(f"video {w}x{h}, {fps:.1f} fps.  Shift = where the middle of each frame sits relative to frame 0 (video pixels).")
    print(f"{'time':>6} {'shift x':>8} {'shift y':>8} {'zoom':>8} {'inliers':>8}")
    worst_shift = worst_zoom = 0.0
    for i in range(max(1, int(round(fps * a.every))), total, max(1, int(round(fps * a.every)))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            break
        kp, d = orb.detectAndCompute(prep(fr), None)
        matches = sorted(bf.match(d, d0), key=lambda m: m.distance)[:800] if d is not None and d0 is not None else []
        H = None
        if len(matches) >= 12:
            src = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst = np.float32([kp0[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None:
            print(f"{i / fps:5.1f}s   could not match this frame to frame 0")
            continue
        shift = (cv2.perspectiveTransform(centre, H)[0][0] - centre[0][0]) / s
        zoom = 1.0 / float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))   # >1 = zoomed in compared with frame 0
        worst_shift = max(worst_shift, float(np.hypot(*shift)))
        worst_zoom = max(worst_zoom, abs(zoom - 1))
        print(f"{i / fps:5.1f}s {shift[0]:8.1f} {shift[1]:8.1f} {zoom:8.3f} {int(inl.sum()):8d}")
    print(f"\nlargest shift {worst_shift:.1f} px, largest zoom change {100 * worst_zoom:.1f}%  ->  " +
          ("camera looks locked off: one calibration is fine." if worst_shift < 3 and worst_zoom < 0.005 else
           "CAMERA MOVES: a single calibration will give drifting positions."))


if __name__ == "__main__":
    main()