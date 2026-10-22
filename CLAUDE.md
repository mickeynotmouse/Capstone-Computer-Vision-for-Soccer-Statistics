# Project context

Senior capstone (Willamette University CS, student: Mikkel Jensen, advisor Fred Agbo): an application that uses computer vision on
recorded Willamette Men's Soccer matches (downloaded from HUDL) to detect on-field events and turn them into visualizations and
statistics for coaches and players. The key differentiator vs. HUDL: **statistics aggregate over any selection of games / the whole
season**, and the main question to answer is *"Which positions did we have on the field when a goal was scored?"* (plus passes
leading to a goal, touches by the scorer before the goal). Manual labour should be minimised. Full proposal: Proposal.docx
(in the Claude project "Capstone"). Proposal timeline: working prototype by Dec 2026, demo by spring break, handoff May 2027.

In scope: backend importing footage-derived data + DB of on-field events; dashboard with visualizations; automatic collection incl.
team separation by jersey colour. Out of scope: per-player stats, highlight reels, live tracking, opponent scouting.

## This folder: proof of concept

Pipeline: `pipeline/track.py` (YOLO + ByteTrack + homography + jersey-colour teams, offline) -> tracks.json ->
`pipeline/events.py` (rule-based touches/passes/turnovers/shots/goals) -> `pipeline/push.py` -> FastAPI (`backend/`, SQLite for now,
Supabase/Postgres planned) -> `frontend/index.html` (plain HTML/JS for now; React + TypeScript planned).
`pipeline/synthetic.py` generates FAKE matches with ground truth so everything can be developed and tested without video.
Heavy video processing stays offline; only structured events are uploaded (free-tier hosting limits).

## Conventions

- Pitch coordinates are metres, 105 x 68, origin top-left. Track team labels are 0/1; `games.our_team` says which is Willamette.
- Goal events store player positions mirrored so the scoring team always attacks towards x = 105.
- Time thresholds in `events.py` are in seconds, not frames.
- Game labels: home `vs. Team (H)`, away `at Team (A)` (`pipeline/push.py` builds them from --opponent and --venue).
- Dashboard: us = red circles, opponents = blue triangles (optional via "show opponents"); no ball graphic.
- Demo numbers are synthetic. Never present them as real results.

## Commands (from this folder, inside the venv: `source .venv/bin/activate`)

- Tests: `python -m pytest -q`
- Demo data: `python -m scripts.seed_demo --reset` (wipes the DB), then `uvicorn backend.main:app --reload` -> http://localhost:8000
- Check injected vs. detected: `python -m scripts.inspect_demo [game_number]` (writes CSVs to data/inspect/)

## Status / known gaps

- `track.py` has not been run on real footage yet; detection accuracy on real video is unknown.
- Validation plan from the proposal: hand-label 2-3 short clips, compare with the tool, report ball-position error and event precision/recall.
- See README.md "Known simplifications" for the list of detector limitations.

## Working style

The student is new to Python tooling (macOS, uses a venv; explain terminal commands). Prefer small, tested changes: run
`python -m pytest -q` after touching `pipeline/`, and keep the tests in sync with behaviour changes.
