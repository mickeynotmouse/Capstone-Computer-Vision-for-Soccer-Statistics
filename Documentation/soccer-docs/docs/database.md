---
sidebar_position: 5
title: Database
description: Where the pipeline's statistics are stored, how they get there, how the website reads them, and who is allowed to do what.
---

# Database

The database sits between the backend and the frontend. It bridges together the heavy pipeline and the frontned that users will see. The pipeline runs locally and only the detection results are written to the database (shots, passes, turnovers, goals etc.). The frontend then reads from the database in order to display the results. This way the frontend remains light, fast and simply and never touches the backend. This should increase user experience. 

## How the pieces connect

```text
 Laptop (offline)                          Online                              Website (Vercel)
 ┌───────────────────────┐   writes   ┌──────────────────────┐   reads   ┌──────────────────────┐
 │ pipeline.track        │  ───────▶  │  Supabase (Postgres) │  ◀──────  │  Results page        │
 │   video → tracks      │   secret   │  tables:             │  public   │  /results            │
 │ pipeline.push         │    key     │    games             │   key     │  totals, per-game    │
 │   tracks → events     │            │    events            │           │  table, goal map     │
 └───────────────────────┘            └──────────────────────┘           └──────────────────────┘
```

1. `pipeline.track` turns a match video into player and ball tracks (`tracks.json`) (locally).
2. `pipeline.push` turns the tracks into events and sends them to the database, using a **secret key** that only exists locally.
3. The website's **Results** page reads the games and events straight from the database in the user's browser, using a **public key** that can only read. It then adds up the statistics for whichever games the user chooses.


There's no other server sepearate from this, the website reads straight from the database. Supabase provides a web API for the tables automatically, so the website talks to the database directly. This keeps hosting free. 

## Why Supabase

- **It's plain Postgres.** The tables are ordinary SQL, easy to manage.
- **It comes with a web API and access rules.** Reading from a website needs no backend code, and the rules make sure the public key can only read.
- **The free plan is enough.** A full season is a few thousand events per game at most, far below the free plan's 500 MB.

## Tables

There are two tables. Every event belongs to one game, which is what lets the website add up any combination of games.

**`games`**: one row per match.

| Column | Meaning |
| --- | --- |
| `id` | Number given by the database |
| `name` | For example `vs. Linfield (H)` (home) or `at Pacific (A)` (away), built the same way every time |
| `date` | Match date |
| `our_team` | Which of the two shirt colours is Willamette: `0` = the lighter shirts, `1` = the darker ones. The pipeline only knows "light team" and "dark team", so this is how the website knows whose stats are whose |
| `created_at` | When the game was added |

**`events`**: one row per thing that happened.

| Column | Meaning |
| --- | --- |
| `game_id` | The game it belongs to. Deleting a game deletes its events too |
| `type` | `pass`, `turnover`, `shot` or `goal` |
| `t`, `frame` | When it happened, in seconds and as a video frame number |
| `team` | Which team did it (`0` light, `1` dark). For a turnover we note the team that **lost** the ball |
| `player` | The tracker's id for the player. No names are stored |
| `data` | Extra detail. For a goal: where every player and the ball were, which is what the goal map on the Results page draws |


## Who can do what

| Who | Key | Can |
| --- | --- | --- |
| Anyone visiting the website | Public (publishable) key, written into the website's code | **Read** games and events |
| `pipeline.push` (locally) | Secret key, in `soccer-cv/.env` | Read, **add and delete** |

The database's access rules allow the public key nothing but reading. The secret key has edit access, so it is kept in `soccer-cv/.env`, which git ignores, and never goes into the website.

:::note
Once a game is in the database, its statistics are visible to anyone who opens the website. No video and no player names are stored, only team-level numbers and positions.
:::

## Adding and removing games

From `soccer-cv/`, after tracking a match:

```bash
# add a game (--our-team: 0/1 = light/dark shirts)
python -m pipeline.push data/g1_tracks.json --opponent Linfield --venue H --date 2026-09-05 --our-team 1 --to supabase

# see what's in the database
python -m pipeline.push --list --to supabase

# remove a game that went in wrong (its events go with it), then add it again
python -m pipeline.push --delete 3 --to supabase
```

A game is added in one go: if sending its events fails part-way, the game is removed again, so the website never shows a half-stored match.
   ```
