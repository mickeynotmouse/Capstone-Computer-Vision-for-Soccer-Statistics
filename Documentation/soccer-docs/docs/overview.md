---
sidebar_position: 1
title: Project Overview
description: What the Soccer Statistics project is, why it exists, and where it is headed.
---

# Project Overview

**Using Computer Vision to Automate Soccer Statistics** is a senior capstone project for the Department of Computer Science at Willamette University. It is a web platform that watches match footage, detects on-field events automatically, and turns them into season-long statistics and visualizations for the Willamette Men's Soccer coaching staff and players.

| | |
| --- | --- |
| **Student** | Mikkel Jensen |
| **Faculty advisor** | Fred Agbo |
| **Course** | CS480W |


## The Problem

All Willamette Men's Soccer matches are recorded, and coaches and players review the footage to learn, coach, and reflect. That review is manual. Reaching a conclusion about what went well and what did not can take hours of dissecting film, and the effort is not always worth the reward.

The coaching staff currently uses HUDL, which generates match reports after a game is imported. Those reports are essentially a count of events (shots, saves, corner kicks) and have two major weaknesses:

- **They cannot be aggregated.** Each report covers one match. There is no way to look at all home games, all away games, or the whole season together.
- **They are too shallow.** They do not show where players were when a goal was scored, how many touches a striker took before scoring, or which areas of the pitch are most dangerous.

According to the coaching staff, the reports are rarely used, partly because HUDL takes a long time to process footage and generate them, and partly because the reports themselves do not offer valuable information.

## The Idea

Use computer vision to collect on-field events automatically, store them as structured data, and aggregate them across games to reveal trends over a whole season.

Instead of someone watching hours of film to answer a question, the platform answers it directly. Examples of questions it is designed to answer without anyone having to ask first:

- Which positions did we have on the field when a goal was scored?
- How many passes led to a goal?
- How many touches did the goalscorer take before the goal?
- Which areas of the pitch produce the most dangerous chances, or the most fouls?

The main benefit is the **longitudinal view**: a holistic picture of the team's tendencies across the season, not only a snapshot of one match.

## Goals

- Users can include or exclude any number of games for any statistic.
- The application answers the central question: "Which positions on the field did we have when a goal was scored?"
- On-field events such as goals, saves, and shots are collected automatically as data points.
- Visualizations are generated from the collected data.
- Visualizations update automatically when new game footage is added.
- Statistics can be aggregated over any period of time.

## Scope

**In scope**

- A simple backend that imports video files and handles data collection, including a database of on-field events
- A user-friendly front end with a dashboard of visualizations and key game statistics
- Automatic data collection, including team separation by jersey color
- Individual player tracking statistics *(stretch goal)*

**Out of scope**

- Automatic highlight reels
- Live, real-time tracking during a match
- Opponent scouting

## How It Works (briefly described)

1. Match footage is downloaded from HUDL (semi-automatic procedure, because HUDL's terms of service prohibit automated scraping).
2. An offline computer vision pipeline detects the players and the ball, tracks them across frames, maps them onto the pitch, separates the teams by jersey color, and extracts events.
3. The structured events are stored in a database.
4. A web dashboard lets coaches and players filter games and view aggregated statistics and visualizations.

Heavy computation runs offline, and only the resulting statistics are uploaded. This keeps hosting free without compromising functionality. The full breakdown, including the data flow diagram and wireframes, is on the [Architecture](./architecture.md) page.

## Technology (Planned)

| Area | Tool |
| --- | --- |
| Detection | Ultralytics YOLO |
| Tracking | ByteTrack |
| Front end | React and TypeScript, with SVG visualizations |
| Back end | FastAPI |
| Database | Supabase (Postgres) |
| Hosting | Free tiers such as Vercel or Fly.io |
| Project management | Scrum-style sprints, tracked on a Monday.com kanban board |

:::note
These are the current plans. They may change as the project progresses, and changes will be recorded in the Dev Log.
:::

## Validation

- **Computer vision accuracy:** results from 2-3 short clips are compared with manual tracking, which also gives an error percentage for the average ball position.
- **Statistics and visualizations:** open datasets such as StatsBomb are used to check that the dashboard's numbers match the underlying data.

## Timeline

| Target date | Milestone | Status |
| --- | --- | --- |
| October 18 | Technical environment complete and set up | Completed |
| November 1 | Paper prototype and data workflow (flowchart) | Completed |
| December 2026 | Working prototype with limited abilities | In progress |
| Spring Break | Working demo, testing still needed | Not started |
| May 2027 | Project handoff | Not started |

## Key Risks

- **Event detection** is the hardest part. Getting YOLO and ByteTrack to reliably identify goals, shots, and passes is a significant challenge.
- **Pitch mapping and team assignment** require homography estimation and jersey-color tracking.
- **Training data** is limited. Open datasets like StatsBomb provide event data but not video, so models need to be checked against manually labeled Willamette footage (time consuming).
- **Getting footage out of HUDL** is slow and cannot be fully automated.
- **Free hosting** has size limits, which is why processing is done offline.

## Deliverables

- A working platform that takes in match footage, analyzes and tags events, and generates visualizations in the front end
- All source code in a GitHub repository
- This documentation site, updated throughout the project

## Where to Go Next

- [Architecture](./architecture.md): components, data flow, and wireframes
