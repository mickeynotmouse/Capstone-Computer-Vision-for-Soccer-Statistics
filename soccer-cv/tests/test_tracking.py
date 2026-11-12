"""The video-side rules, on made-up data with a known answer: no footage or model weights needed."""
import cv2
import numpy as np

from pipeline import ball, camera, lines, stitch, teams
from pipeline import homography as hg

FPS = 30.0


# ---------- calibration: snap to WHITE paint, never to the coloured lines beside it ----------

def _synthetic_pitch(H_true, width=65.0, length=104.0, size=(1600, 1000)):
    """A grass-green picture with the soccer lines painted white, and an ORANGE line 2.5 m outside the near touchline."""
    img = np.full((size[1], size[0], 3), (40, 120, 50), np.uint8)
    Hinv = np.linalg.inv(H_true)

    def paint(p0, p1, colour, px=5):
        pts = np.linspace(p0, p1, 80)
        q = np.c_[pts, np.ones(len(pts))] @ Hinv.T
        cv2.polylines(img, [np.int32(q[:, :2] / q[:, 2:])], False, colour, px, cv2.LINE_AA)

    for _, _, p0, p1 in lines.soccer_lines(width, length):
        paint(p0, p1, (255, 255, 255))
    near = 34 + width / 2
    paint((1, near + 2.5), (105, near + 2.5), (0, 140, 255), 7)     # orange (BGR)
    return img


def _camera_H():
    """A plausible image->pitch homography: camera beside the near touchline, looking at the x = 105 goal."""
    pit = np.float64([[60, 66.5], [105, 66.5], [105, 1.5], [60, 1.5]])
    img = np.float64([[150, 950], [1500, 700], [1150, 300], [350, 330]])
    H, _ = cv2.findHomography(img, pit)
    return H / np.linalg.norm(H)


def test_refine_snaps_to_white_touchline_not_orange_one():
    H_true = _camera_H()
    frame = _synthetic_pitch(H_true)
    # a fit that, like the old calib.json, has the near touchline lying on the orange line
    shift = np.array([[1, 0, 0], [0, 1, -2.5], [0, 0, 1]], float)
    H_bad = shift @ H_true
    pts = hg.to_image(H_true, np.float64([[105, 37.66], [105, 30.34], [94, 34], [88.5, 13.84], [99.5, 24.84]]))
    obs = hg.Obs(pts, [[105, 37.66], [105, 30.34], [94, 34], [88.5, 13.84], [99.5, 24.84]])
    before = lines.alignment(frame, H_bad, 65, 104)["near touchline"][0]
    H, _, used = lines.refine(frame, obs, H_bad, 65, 104)
    after = lines.alignment(frame, H, 65, 104)["near touchline"][0]
    assert before < 0.2 and after > 0.8, (before, after)
    # and the near touchline really is back where it belongs, not merely on some paint
    err = np.abs(hg.to_pitch(H, hg.to_image(H_true, [[80, 66.5]]))[0] - [80, 66.5]).max()
    assert err < 0.4


def test_refine_never_makes_a_good_fit_worse():
    H_true = _camera_H()
    frame = _synthetic_pitch(H_true)
    obs = hg.Obs(hg.to_image(H_true, np.float64([[105, 37.66], [105, 30.34], [94, 34], [88.5, 13.84]])),
                 [[105, 37.66], [105, 30.34], [94, 34], [88.5, 13.84]])
    H, _, _ = lines.refine(frame, obs, H_true, 65, 104)
    assert lines._score(lines.alignment(frame, H, 65, 104)) >= lines._score(lines.alignment(frame, H_true, 65, 104)) - 1e-9


def test_whiteness_ignores_coloured_paint():
    img = np.full((60, 60, 3), (40, 120, 50), np.uint8)
    img[:, 10:14] = (255, 255, 255)        # white
    img[:, 40:44] = (0, 140, 255)          # orange
    w = lines.whiteness(img)
    assert w[30, 12] > 100 and w[30, 42] < 20


def test_calibrator_line_clicks_snap_to_white_and_flag_coloured_paint():
    from pipeline import calibrate
    H = _camera_H()
    calibrate.LANDMARKS[:] = calibrate.landmarks(65, 104)
    cal = calibrate.Calibrator(_synthetic_pitch(H))
    cal.k = next(i for i, (n, _, _) in enumerate(calibrate.LANDMARKS) if n.startswith("NEAR touchline"))
    white, orange = hg.to_image(H, [[80, 66.5], [80, 69.0]])
    assert cal.click(*((white + [0, 4]) * cal.s)) == "snapped"
    assert np.linalg.norm(np.float64(cal.hist[-1][2]) - white) < 1.5
    assert cal.click(*(orange * cal.s)) == "no paint"
    calibrate.LANDMARKS[:] = calibrate.landmarks()


# ---------- camera ----------

def test_content_box_finds_the_video_inside_a_screen_recording():
    frame = np.zeros((200, 300, 3), np.uint8)
    frame[20:170] = (40, 120, 50)
    frame[180:190, 10:200] = 255           # a progress bar with text on the dark control strip
    assert camera.content_box(frame) == (0, 20, 300, 170)


def test_camera_path_interpolates_between_samples():
    shift = lambda dx: np.array([[1, 0, dx], [0, 1, 0], [0, 0, 1]], float)   # noqa: E731
    path = camera.CameraPath.from_samples({0: (shift(0), 500), 10: (shift(10), 500)}, 11, smooth=0)
    assert np.allclose(path[5], shift(5))


# ---------- people: roles, teams, keepers ----------

GRASS = np.float32([130, 113, 159])       # this footage's grass in OpenCV Lab: a = -15, b = +31


def _samples(L, L20, a, b, neon, n=20, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return [[L + rng.normal(0, noise), L20 + rng.normal(0, noise), a + rng.normal(0, 0.5), b + rng.normal(0, 0.5), neon]
            for _ in range(n)]


def _match_squad(seed=0):
    """Feature medians measured on the real clips: white team, striped team, a neon referee, a yellow keeper, and a striped
    player so blurred his shirt looks green."""
    f = {}
    for i in range(8):
        f[i] = _samples(115, 100, -1, 0, 0.0, seed=seed + i)            # white
        f[10 + i] = _samples(80, 58, 2, 2, 0.0, seed=seed + 20 + i)     # stripes
    f[30] = _samples(100, 70, -15, 26, 0.55, seed=seed + 40)            # neon referee
    f[31] = _samples(100, 95, 1, 12, 0.0, seed=seed + 41)               # yellow keeper
    f[32] = _samples(88, 65, -7, 12, 0.05, seed=seed + 42)              # blurred, grass-tinted stripes
    pos = {i: (60.0, 30.0) for i in f}
    pos[31] = (101.0, 36.0)                                             # in the penalty area
    pos[30] = (80.0, 40.0)
    return f, pos


def test_roles_from_shirts_and_position():
    f, pos = _match_squad()
    roles, colours = teams.assign(f, pos, GRASS)
    assert {roles[i] for i in range(8)} == {0} and {roles[10 + i] for i in range(8)} == {1}   # 0 = lighter shirt
    assert roles[30] == "official"
    assert roles[31] == "gk"
    assert roles[32] == 1          # green tint from the grass doesn't make a striped player a referee


def test_a_neon_shirt_in_the_box_is_still_a_keeper_and_out_of_it_an_official():
    f, pos = _match_squad()
    pos[30] = (102.0, 34.0)
    assert teams.assign(f, pos, GRASS)[0][30] == "gk"


def test_tint_rule_keeps_yellow_but_removes_grass_tint():
    out = teams._chroma([[1, 12], [-7, 12], [-2, 14]], GRASS)
    assert np.allclose(out[1], 0) and out[0][1] == 12 and out[2][1] == 14


def test_kit_change_cuts_a_swapped_track_once():
    frames = list(range(0, 300, 3))
    labels = [0] * 50 + [1] * 50
    assert teams.kit_changes(frames, labels, FPS) == [150]


def test_kit_change_ignores_flicker():
    rng = np.random.default_rng(1)
    frames = list(range(0, 300, 3))
    labels = [int(x) for x in rng.random(100) < 0.15]       # mostly 0, with noisy single samples of 1
    assert teams.kit_changes(frames, labels, FPS) == []


def test_keeper_belongs_to_the_team_sitting_deeper():
    defenders = list(np.random.default_rng(0).uniform(85, 104, 200))   # crowded around their own goal at x = 105
    attackers = list(np.random.default_rng(1).uniform(60, 104, 200))   # some in the box, some left back
    assert teams.keeper_team(105.0, {0: attackers, 1: defenders}) == 1
    assert teams.keeper_team(105.0, {0: defenders, 1: defenders}) == -1     # too close to call: leave it to --keepers


# ---------- stitching ----------

def _run(tid, f0, f1, start, vel):
    fr = np.arange(f0, f1)
    return stitch.Tracklet(tid, fr, np.float64(start) + np.outer((fr - f0) / FPS, vel))


def test_stitch_joins_a_player_the_tracker_lost():
    a = _run(1, 0, 60, (50, 30), (3, 0))                 # ends near (56, 30)
    b = _run(2, 90, 150, (59, 30), (3, 0))               # picks up a second later, 3 m further on
    out = stitch.stitch([a, b], FPS)
    assert len(out) == 1 and sorted(out[0].merged_from) == [1, 2]


def test_stitch_refuses_a_coin_toss():
    a = _run(1, 0, 60, (50, 30), (0, 0))
    b = _run(2, 90, 150, (51, 31), (0, 0))               # two people re-appear, equally close to where a was lost
    c = _run(3, 90, 150, (51, 29), (0, 0))
    assert len(stitch.stitch([a, b, c], FPS)) == 3


def test_stitch_never_joins_different_teams():
    a, b = _run(1, 0, 60, (50, 30), (3, 0)), _run(2, 70, 150, (56.5, 30), (3, 0))
    a.team, b.team = 0, 1
    assert len(stitch.stitch([a, b], FPS)) == 2


def test_duplicate_boxes_on_one_person_become_one_track():
    a = _run(1, 0, 100, (50, 30), (2, 0))
    b = _run(2, 10, 60, (50.3, 30.2), (2, 0))
    b.frames = b.frames
    b.xy = np.float64((50.3, 30.2)) + np.outer((b.frames - 0) / FPS, (2, 0))
    assert [t.id for t in stitch.remove_duplicates([a, b])] == [1]


def test_sideline_bystander_is_spotted_but_a_winger_is_not():
    coach = _run(1, 0, 150, (70, 66.0), (0, 0))
    winger = _run(2, 0, 150, (60, 65.5), (4, 0))
    assert stitch.is_sideline_bystander(coach, 65, FPS) and not stitch.is_sideline_bystander(winger, 65, FPS)


# ---------- ball ----------

def _flight(f0, f1, start, vel, conf=0.4):
    return [[f, start[0] + vel[0] * (f - f0) / FPS, start[1] + vel[1] * (f - f0) / FPS, conf] for f in range(f0, f1)]


def test_ball_ignores_spare_balls_lying_still():
    real = _flight(0, 90, (70, 30), (8, 0), conf=0.3)
    spare = [[f, 100.0, 68.0, 0.6] for f in range(0, 90)]           # a spare ball by the corner flag, seen better
    path = np.float64(real + spare)[ball.best_path(real + spare, FPS)]
    assert len(path) and np.all(path[:, 2] < 40)


def test_ball_never_teleports_between_two_candidates():
    real = _flight(0, 60, (70, 30), (10, 0), conf=0.35)
    fake = [[f, 90.0 + 0.3 * np.sin(f), 50.0, 0.95] for f in range(0, 60, 4)]   # a confident, flickering false "ball"
    c = np.float64(real + fake)
    path = c[ball.best_path(c, FPS)]
    steps = np.hypot(*np.diff(path[:, 1:3], axis=0).T) / (np.diff(path[:, 0]) / FPS)
    assert steps.max() < 60


def test_ball_track_fills_short_gaps_only():
    c = _flight(0, 10, (50, 30), (10, 0)) + _flight(20, 30, (50 + 10 * 20 / FPS, 30), (10, 0)) \
        + _flight(100, 110, (60, 30), (10, 0))
    out = ball.ball_track(c, 120, FPS)
    assert out[15] is not None and out[15].get("filled")          # 0.3 s gap: filled
    assert out[60] is None                                        # 2.3 s gap: left empty


# ---------- long clips ----------

def _panning_video(path, xs, size=(480, 270)):
    """A video of a camera panning across a fixed textured scene: frame n shows the scene from x = xs[n]."""
    rng = np.random.default_rng(0)
    scene = cv2.GaussianBlur(rng.integers(0, 255, (size[1], 2400, 3), dtype=np.uint8), (0, 0), 2)
    for _ in range(400):                       # blobs and edges give the feature finder something to hold on to
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.circle(scene, (int(rng.integers(0, 2400)), int(rng.integers(0, size[1]))), int(rng.integers(3, 15)), c, -1)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    for x in xs:
        w.write(np.ascontiguousarray(scene[:, int(x):int(x) + size[0]]))
    w.release()


def test_camera_follows_a_long_pan_and_back_with_reference_mid_clip(tmp_path):
    # out across the scene and back again, slowing down to turn the way a camera operator does (an instant reversal at
    # full speed would put frames between two measured samples a few pixels off, for a frame or two)
    xs = 750 * (1 - np.cos(np.linspace(0, 2 * np.pi, 240)))
    video = tmp_path / "pan.mp4"
    _panning_video(video, xs)
    ref = 60
    path = camera.CameraPath.from_video(str(video), ref_frame=ref, every=3)
    assert len(path) == len(xs)
    for n in (0, 30, 119, 180, 239):
        moved = camera.apply(path[n], [[100, 100]])[0]                   # pixel of frame n -> pixel of frame `ref`
        # within 1 % of how far the camera panned between the two frames (the point may lie far outside the reference
        # frame's picture, where small errors in the estimated perspective grow), and never worse than 4 px
        tol = max(4.0, 0.01 * abs(xs[n] - xs[ref]))
        assert abs(moved[0] - (100 + xs[n] - xs[ref])) < tol and abs(moved[1] - 100) < tol, (n, moved)


def test_static_check_on_a_grid_matches_comparing_everything():
    rng = np.random.default_rng(3)
    spare = [[f, 100.0 + rng.normal(0, 0.1), 66.0, 0.5] for f in range(0, 300, 2)]
    moving = _flight(0, 300, (40, 30), (6, 1))
    noise = [[f, *rng.uniform([0, 0], [105, 68]), 0.2] for f in range(0, 300, 3)]
    c = np.float64(spare + moving + noise)
    brute = np.zeros(len(c), bool)
    for i in range(len(c)):
        near = np.linalg.norm(c[:, 1:3] - c[i, 1:3], axis=1) < ball.STATIC_M
        fr = c[near, 0]
        span = (fr.max() - fr.min()) / FPS
        if span >= ball.STATIC_S and len(np.unique(fr)) >= ball.STATIC_SHARE * span * FPS:
            brute[near] = True
    assert (ball.static_mask(c, FPS) == brute).all() and brute[:len(spare)].all()


def test_stitching_thousands_of_tracks_is_quick():
    import time
    rng = np.random.default_rng(0)
    tracks = []
    for k in range(3000):                      # 45 minutes of short, scattered tracks
        f0 = int(rng.integers(0, 81000))
        tracks.append(_run(k, f0, f0 + 40, rng.uniform([0, 0], [105, 68]), rng.normal(0, 2, 2)))
    t = time.time()
    stitch.stitch(tracks, FPS)
    stitch.remove_duplicates(tracks)
    assert time.time() - t < 20
