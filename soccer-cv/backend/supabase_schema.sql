-- The online database (Supabase / Postgres).
-- Who can do what:
--   anyone (the website, with the public "publishable" key)  READ games and events
--   pipeline.push on your Mac (with the SECRET key)          WRITE: the secret key bypasses these rules

create table if not exists games (
    id          bigint generated always as identity primary key,
    name        text not null,                   -- "vs. Linfield (H)"
    date        date not null,
    our_team    smallint not null check (our_team in (0, 1)),   -- which jersey cluster (0 = lighter) is Willamette
    created_at  timestamptz not null default now()
);

create table if not exists events (
    id       bigint generated always as identity primary key,
    game_id  bigint not null references games(id) on delete cascade,
    type     text not null,                      -- pass, turnover, shot, goal
    frame    integer not null,
    t        real not null,                      -- seconds into the video
    team     smallint not null,
    player   integer,                            -- track id (no names are stored)
    data     jsonb not null default '{}'::jsonb  -- extra detail, e.g. player positions at a goal
);
create index if not exists events_game_type on events (game_id, type);

-- Newer Supabase projects give no key access to new tables until it's granted here:
-- the website's key may only READ, pipeline.push's secret key may read and write.
grant select on games, events to anon, authenticated;
grant select, insert, update, delete on games, events to service_role;

alter table games  enable row level security;
alter table events enable row level security;

drop policy if exists "public read" on games;
create policy "public read" on games  for select to anon, authenticated using (true);
drop policy if exists "public read" on events;
create policy "public read" on events for select to anon, authenticated using (true);
