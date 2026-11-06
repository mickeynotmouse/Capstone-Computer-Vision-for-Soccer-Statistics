# Soccer CV – Proof of Concept

Footage → detection/tracking → pitch coordinates → events → database → season-wide dashboard.

Run every command below from inside this `soccer-cv/` folder. Test videos (`*.mp4`) and YOLO weights (`*.pt`) are
not in the repo (too big for GitHub): keep them in this folder on your own machine; `.gitignore` stops them being committed.

```
pipeline/track.py      detection + tracking on a clip, teams/keepers/officials, ball path, pitch metres   -> tracks.json
pipeline/calibrate.py  click pitch landmarks + spots along lines once per camera mount (snaps to WHITE)    -> calib.json
pipeline/refine.py     no clicking: move a calibration onto another clip from the same mount, snap to white -> calib.json
pipeline/homography.py robust pixel-space fit of calib.json, held-out check and expected-error map
pipeline/camera.py     follows the camera's pan/zoom from the background; finds the video inside a screen recording
pipeline/lines.py      finds WHITE paint (never the red/orange/yellow lines of other sports) and snaps the fit to it
pipeline/teams.py      shirts -> team 0 / team 1 / goalkeeper / official; spots a track that jumped between people
pipeline/stitch.py     removes duplicate tracks, re-joins people the tracker lost (only when unambiguous)
pipeline/ball.py       picks the one physically possible ball path; ignores balls lying still (spares, discs)
pipeline/review.py     overlay video + stills + tracking summary, to check any of the above by eye
pipeline/events.py     rule-based touches / passes / turnovers / shots / goals from tracks.json
pipeline/push.py       tracks.json -> events -> POST to the backend (only structured stats are uploaded)
pipeline/synthetic.py  FAKE matches with known ground truth (develop & test with no video)
backend/               FastAPI + SQLite (swap for Supabase/Postgres later) + aggregation
frontend/index.html    dashboard: tick any games, get aggregated stats + "where were we standing when a goal was scored"
scripts/               seed_demo.py (load fake games), inspect_demo.py (check injected vs. detected, writes CSVs)
tests/                 event detector vs. synthetic ground truth; tracking rules on made-up data (no video needed)
CLAUDE.md              project context for Claude Code (see below)
```

## Run the demo (no video needed)

```bash
python3 -m venv .venv && source .venv/bin/activate        # once; re-run the `source` line in each new terminal
python -m pip install fastapi uvicorn httpx pytest numpy
python -m pytest -q                       # detector vs. ground truth on 40 synthetic matches x 2 layouts
python -m scripts.seed_demo --reset       # wipes the DB (demo only) and loads 4 fake games
uvicorn backend.main:app --reload         # open http://localhost:8000
```

## Where the demo numbers come from (nothing is real)

- Team names, dates and results are **invented** (the `GAMES` list in `scripts/seed_demo.py`).
- `pipeline/synthetic.py` generates the tracks: a seeded random simulation, so it is identical on every run. It also
  returns the **ground truth** of what it injected (every possession sequence, passes, first-time passes, shooter, outcome).
- The detector (`pipeline/events.py`) then runs on those tracks exactly as it would on real ones. The test suite and
  `scripts/inspect_demo.py` compare what was injected with what was detected.
- Nothing is downloaded from any website or API.

Check it by hand:

```bash
python -m scripts.inspect_demo 2          # game 2: injected vs. found, plus CSVs in data/inspect/
```

CSV files per game: `*_injected_vs_found.csv`, `*_detected_events.csv` (with timestamps), `*_ball_track.csv`,
`*_goal_positions.csv`.

## Run on real footage (your machine)

```bash
python -m pip install opencv-python ultralytics
python -m pipeline.calibrate clip.mp4 --out calib.json --width 68 --length 105   # use the REAL field size
python -m scripts.compare_calib calib.json                    # old vs new fit: held-out and expected error
python -m pipeline.track clip.mp4 --calib calib.json --out data/g1_tracks.json
python -m pipeline.push data/g1_tracks.json --opponent Linfield --venue H --date 2026-09-05 --our-team 0
```

Game names are built for you: home → `vs. Linfield (H)`, away → `at Pacific (A)`.
`--our-team` is which jersey cluster (0/1) is Willamette – check a few track ids against the video the first time.

## What is verified vs. not

| Part | Status |
|---|---|
| Event detection logic, aggregation, API | Tested on synthetic matches with known answers |
| Dashboard | JavaScript syntax-checked; not yet looked at in a browser by me |
| Homography maths, jersey-colour team split | Sanity-checked on toy inputs |
| `track.py` with YOLO/ByteTrack on real HUDL video | **Written, not yet run** – needs your footage + `ultralytics` |
| Detection accuracy on real footage | **Unknown** – synthetic data is clean; real ball tracking is the hard part |

## How a touch / first-time pass is detected

A *touch* is a run of frames in which one player is the closest to the ball (within 1.5 m).
- **Controlled touch** (≥ 0.3 s on the ball): accepted if the ball slows down and stays with him, or its velocity changes sharply.
- **Quick touch** (< 0.3 s, e.g. a first-time pass): there is no time to see control, so the only evidence is a sharp change in the
  ball's velocity, measured over a clean window of free flight either side. A ball that just flies past a defender at constant
  velocity is rejected; a flick/redirect is accepted.
- All thresholds are in seconds (not frames), so they behave the same at 10, 25 or 30 fps. They are constants at the top of `events.py`.

Passes are consecutive touches by different players of the same team; a turnover is a consecutive touch by the other team.

## Real footage: calibrate, track, review

```bash
# once per camera mount: click landmarks + lines (line clicks snap to white paint; saving snaps the whole fit to it)
python -m pipeline.calibrate wide.mp4 --width 65 --length 104 --out calib_wide.json
# another clip from the SAME mount: no clicking, the calibration is carried over and re-snapped to the white lines
python -m pipeline.refine goal.mp4 --from-calib calib_wide_v2.json --from-video wide.mp4 --out calib_goal_v2.json
# tracking: a 78 s 1080p clip takes about 7 minutes on an M1 (8 GB). --work saves the camera path and detections
# as they're made: if a run stops, the same command carries on where it was; re-tuning tracking afterwards takes seconds
python -m pipeline.track goal.mp4 --calib calib_goal_v2.json --work work/goal --out data/goal_tracks.json
python -m pipeline.review goal.mp4 --calib calib_goal_v2.json --tracks data/goal_tracks.json
```

The camera pans and zooms, so `track.py` follows it frame by frame from the background (trees, buildings, fences);
the calibration only has to be right on its own frame, which can be anywhere in the clip. It reads the video in one pass
and keeps only a small library of views, so a 45-minute half needs no more memory than a short clip.

Speed: only every other frame goes through the detector (`--detect-every 2`, the default); the tracker fills in the rest.
Measured on a 78 s clip with the same detections: about half the detection time, for ~15 % more ids (60 -> 69) and a
little less of the ball (77 % -> 70 % of frames). Use `--detect-every 1` when that matters more than time. `refine.py` prints how much of each line sits on white paint,
before and after: use it to check any calibration.

Who is who in `tracks.json`:
- `players` holds both teams, **including goalkeepers** (`"role": "gk"`). A keeper's team is the side defending that goal,
  judged from where each team's players stand over the clip. If that's wrong, override it: `--keepers "105:1"` (the keeper
  of the goal at x = 105 plays for team 1). Keepers' touches count in events.py; they are left out of the positions
  recorded at goals, which the dashboard averages.
- `others` holds officials (neon shirts), shown as a hollow R in the review video. Fans and people standing still on the
  touchline (coaches, substitutes) are dropped.
- Team 0 is always the lighter shirt.

## Known simplifications (the to-do list for real footage)

- A first-time pass that sends the ball on in the *same direction at the same speed* looks identical to a fly-by and is missed;
  a first-time pass within ~0.3 s of the previous touch is also skipped (the measurement windows overlap).
- The ball is seen in about 25-30 % of frames on the test clips (plus short gaps filled in). A ball lying still is ignored
  (on the test clips those were spare balls and a disc), so a dead ball is only picked up once it's kicked. A ball in
  the air is placed where it would be on the ground, so lofted balls look faster and further than they are.
- A "touch" is an unbroken run, so a dribble counts as one touch.
- Goals are judged by where the ball crosses the goal line; own goals are credited to the last toucher's team.
- Shirt rules are tuned on one match (white v maroon stripes, neon officials, yellow keeper) and need checking on others.
- When two players cross, the tracker can swap them. A swap between TEAMS is caught (the shirt changes) and the track is
  cut; a swap between two players of the same team is not.
- The calibration and line snapping only know the half with the goal at x = 105; a camera that pans to the other half
  needs the other half's lines added.
- Detections, tracking and stats only exist for the part of the pitch in view, so averages favour where the camera points.
- Dashboard is plain HTML/JS for speed; port to React + TypeScript per the proposal once the data model settles.

## Working on this with Claude Code in VS Code

`CLAUDE.md` at the top of this folder is read by Claude Code at the start of every session, so a fresh session already knows what
the project is, how it is laid out, and the conventions. Open the `poc` folder in VS Code, start Claude Code, and ask for changes.

## Suggested next steps (maps to the proposal timeline)

1. Export one short clip from HUDL, calibrate, run `track.py`, and eyeball the tracks overlaid on the video.
2. Hand-label the 2–3 validation clips and compute ball-position error + event precision/recall against `events.py`.
3. Tune thresholds / swap in a ball-specific detector if recall is poor.
4. Move the DB to Supabase and the dashboard to React.
