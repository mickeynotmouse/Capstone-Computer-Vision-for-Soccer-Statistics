# Soccer CV – Proof of Concept

Footage → detection/tracking → pitch coordinates → events → database → season-wide dashboard.

Run every command below from inside this `soccer-cv/` folder. Test videos (`*.mp4`) and YOLO weights (`*.pt`) are
not in the repo (too big for GitHub): keep them in this folder on your own machine; `.gitignore` stops them being committed.

```
pipeline/track.py      YOLO + ByteTrack on a clip, jersey-colour team split, homography to pitch metres  -> tracks.json
pipeline/calibrate.py  click pitch landmarks + spots along lines once per camera setup                     -> calib.json
pipeline/homography.py robust pixel-space fit of calib.json, held-out check and expected-error map
pipeline/events.py     rule-based touches / passes / turnovers / shots / goals from tracks.json
pipeline/push.py       tracks.json -> events -> POST to the backend (only structured stats are uploaded)
pipeline/synthetic.py  FAKE matches with known ground truth (develop & test with no video)
backend/               FastAPI + SQLite (swap for Supabase/Postgres later) + aggregation
frontend/index.html    dashboard: tick any games, get aggregated stats + "where were we standing when a goal was scored"
scripts/               seed_demo.py (load fake games), inspect_demo.py (check injected vs. detected, writes CSVs)
tests/                 event detector vs. synthetic ground truth
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

## Known simplifications (the to-do list for real footage)

- A first-time pass that sends the ball on in the *same direction at the same speed* looks identical to a fly-by and is missed;
  a first-time pass within ~0.3 s of the previous touch is also skipped (the measurement windows overlap).
- Ball detection from a wide angle will drop frames; expect to add ball interpolation/smoothing and re-tune the thresholds.
- A "touch" is an unbroken run, so a dribble counts as one touch.
- Goals are judged by where the ball crosses the goal line; own goals are credited to the last toucher's team.
- Referees and goalkeepers can pollute the 2-cluster jersey split (add a third cluster / goalkeeper rule).
- Tracker ID switches will split one player into two ids.
- A fixed camera is assumed (one homography per game); a panning camera needs per-frame calibration.
- Dashboard is plain HTML/JS for speed; port to React + TypeScript per the proposal once the data model settles.

## Working on this with Claude Code in VS Code

`CLAUDE.md` at the top of this folder is read by Claude Code at the start of every session, so a fresh session already knows what
the project is, how it is laid out, and the conventions. Open the `poc` folder in VS Code, start Claude Code, and ask for changes.

## Suggested next steps (maps to the proposal timeline)

1. Export one short clip from HUDL, calibrate, run `track.py`, and eyeball the tracks overlaid on the video.
2. Hand-label the 2–3 validation clips and compute ball-position error + event precision/recall against `events.py`.
3. Tune thresholds / swap in a ball-specific detector if recall is poor.
4. Move the DB to Supabase and the dashboard to React.
