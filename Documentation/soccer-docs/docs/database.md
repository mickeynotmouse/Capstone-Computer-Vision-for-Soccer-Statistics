---
sidebar_position: 5
title: Database
description: Where the pipeline's statistics are stored, how they get there, how the website reads them, and who is allowed to do what.
---

# Database

The database sits at the centre of the project. The video pipeline on a laptop writes into it, and the website reads from it. Only small, structured statistics are stored there (events such as passes, shots and goals, with times and pitch positions). Video never leaves the laptop.

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

1. `pipeline.track` turns a match video into player and ball tracks (`tracks.json`), on the laptop.
2. `pipeline.push` turns the tracks into events and sends them to the database, using a **secret key** that only exists on the laptop.
3. The website's **Results** page fetches the games and events straight from the database in the visitor's browser, using a **public key** that can only read. It then adds up the statistics for whichever games the visitor ticks.

There is no separate server between the database and the website. Supabase provides a web API for the tables automatically, so the website talks to the database directly. This keeps hosting free: the website is a static site on Vercel, and the database is on Supabase's free plan.

## Why Supabase

- **It's plain Postgres.** The tables are ordinary SQL, so the data isn't locked into one company. The same schema already runs locally in SQLite (see [Local version](#local-version)).
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
| `team` | Which team did it (`0` light, `1` dark). For a turnover: the team that **lost** the ball |
| `player` | The tracker's id for the player. No names are stored |
| `data` | Extra detail. For a goal: where every player and the ball were, which is what the goal map on the Results page draws |

The exact definition is in `soccer-cv/backend/supabase_schema.sql`.

## Who can do what

| Who | Key | Can |
| --- | --- | --- |
| Anyone visiting the website | Public (publishable) key, written into the website's code | **Read** games and events |
| `pipeline.push` on the laptop | Secret key, in `soccer-cv/.env` | Read, **add and delete** |

The public key is meant to be public: the database's access rules allow it nothing but reading. The secret key can change everything, so it is kept in `soccer-cv/.env`, which git ignores, and never goes into the website or the repository.

:::note
Once a game is in the database, its statistics are visible to anyone who opens the website. No video and no player names are stored, only team-level numbers and positions.
:::

## Adding and removing games

From `soccer-cv/`, after tracking a match:

```bash
# add a game (--our-team: 0 if Willamette wore the lighter shirts, 1 if the darker ones)
python -m pipeline.push data/g1_tracks.json --opponent Linfield --venue H --date 2026-09-05 --our-team 1 --to supabase

# see what's in the database
python -m pipeline.push --list --to supabase

# remove a game that went in wrong (its events go with it), then add it again
python -m pipeline.push --delete 3 --to supabase
```

A game is added in one go: if sending its events fails part-way, the game is removed again, so the website never shows a half-stored match.

## Setting it up from scratch

Needed only once, or for a new copy of the project.

1. Create a free project at [supabase.com](https://supabase.com) (region: West US).
2. In **SQL Editor**, paste and run the whole of `soccer-cv/backend/supabase_schema.sql`. It creates the tables and the access rules, and can safely be run again.
3. Under **Project Settings → API Keys**, copy the **publishable** key and the project URL into `customFields` in `Documentation/soccer-docs/docusaurus.config.ts`.
4. Copy the project URL and the **secret** key into `soccer-cv/.env`:

   ```text
   SUPABASE_URL=https://<project>.supabase.co
   SUPABASE_SECRET_KEY=sb_secret_...
   ```

## Local version

For development without internet, the same two tables also exist in a local SQLite file (`soccer-cv/data/poc.db`), served by a small FastAPI backend with its own dashboard at `http://localhost:8000`. `pipeline.push` without `--to supabase` sends games there instead. It is only for testing on one laptop. What the website shows always comes from Supabase.
