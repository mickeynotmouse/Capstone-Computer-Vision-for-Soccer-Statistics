/**
 * Results: the pipeline's stats, read straight from the online database (Supabase).
 *
 * pipeline.push (on the Mac, with the SECRET key) writes games + events; this page reads them with the public
 * "publishable" key, which can only read (see soccer-cv/backend/supabase_schema.sql). The totals are worked out the
 * same way as soccer-cv/backend/stats.py.
 */
import React, {useEffect, useMemo, useState} from 'react';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './results.module.css';

type Game = {id: number; name: string; date: string; our_team: number};
type GoalData = {attackers?: number[][]; defenders?: number[][]; ball?: number[];
                 buildup_passes?: number; scorer_touches?: number};
type Event = {game_id: number; type: string; t: number; team: number; data: GoalData};
type Totals = {goalsFor: number; goalsAgainst: number; shotsFor: number; shotsAgainst: number;
               passesFor: number; passesAgainst: number; turnovers: number};

const L = 105, W = 68;
const PAGE = 1000;   // Supabase returns at most 1000 rows per request

function blank(): Totals {
  return {goalsFor: 0, goalsAgainst: 0, shotsFor: 0, shotsAgainst: 0, passesFor: 0, passesAgainst: 0, turnovers: 0};
}

function tally(game: Game, events: Event[]): Totals {
  const s = blank();
  for (const e of events) {
    const ours = e.team === game.our_team;
    if (e.type === 'goal') ours ? s.goalsFor++ : s.goalsAgainst++;
    else if (e.type === 'shot') ours ? s.shotsFor++ : s.shotsAgainst++;
    else if (e.type === 'pass') ours ? s.passesFor++ : s.passesAgainst++;
    else if (e.type === 'turnover' && ours) s.turnovers++;   // possessions we lost
  }
  return s;
}

async function rest<T>(url: string, key: string, path: string): Promise<T[]> {
  const out: T[] = [];
  for (let offset = 0; ; offset += PAGE) {
    const sep = path.includes('?') ? '&' : '?';
    const r = await fetch(`${url}/rest/v1/${path}${sep}limit=${PAGE}&offset=${offset}`, {headers: {apikey: key}});
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    const rows: T[] = await r.json();
    out.push(...rows);
    if (rows.length < PAGE) return out;
  }
}

function Pitch({goals, kind, showOpp}: {goals: {game: Game; e: Event}[]; kind: 'for' | 'against'; showOpp: boolean}) {
  // goal events store positions with the scoring team attacking towards x = 105 (pipeline/events.py)
  const ours: number[][] = [], theirs: number[][] = [], balls: number[][] = [];
  for (const {e} of goals) {
    const att = e.data.attackers ?? [], def = e.data.defenders ?? [];
    ours.push(...(kind === 'for' ? att : def));
    theirs.push(...(kind === 'for' ? def : att));
    if (e.data.ball) balls.push(e.data.ball);
  }
  const line = {fill: 'none', stroke: 'rgba(255,255,255,.75)', strokeWidth: 0.4};
  return (
    <svg className={styles.pitch} viewBox={`-3 -3 ${L + 6} ${W + 6}`} role="img"
         aria-label={`player positions at ${goals.length} goals`}>
      <rect x={-3} y={-3} width={L + 6} height={W + 6} fill="#2f7d3a" rx={2} />
      <rect x={0} y={0} width={L} height={W} {...line} />
      <line x1={L / 2} y1={0} x2={L / 2} y2={W} {...line} />
      <circle cx={L / 2} cy={W / 2} r={9.15} {...line} />
      <rect x={0} y={W / 2 - 20.16} width={16.5} height={40.32} {...line} />
      <rect x={L - 16.5} y={W / 2 - 20.16} width={16.5} height={40.32} {...line} />
      <rect x={0} y={W / 2 - 9.16} width={5.5} height={18.32} {...line} />
      <rect x={L - 5.5} y={W / 2 - 9.16} width={5.5} height={18.32} {...line} />
      <rect x={L} y={W / 2 - 3.66} width={1.5} height={7.32} fill="rgba(255,255,255,.75)" />
      {showOpp && theirs.map(([x, y], i) => <circle key={`o${i}`} cx={x} cy={y} r={0.9} fill="#f87171" opacity={0.75} />)}
      {ours.map(([x, y], i) => <circle key={`u${i}`} cx={x} cy={y} r={0.9} fill="#93c5fd" stroke="#1e3a8a" strokeWidth={0.2} />)}
      {balls.map(([x, y], i) => <circle key={`b${i}`} cx={x} cy={y} r={0.7} fill="#fde047" stroke="#000" strokeWidth={0.15} />)}
    </svg>
  );
}

export default function Results(): React.ReactNode {
  const {siteConfig} = useDocusaurusContext();
  const url = String(siteConfig.customFields?.supabaseUrl ?? '').replace(/\/+$/, '').replace(/\/rest\/v1$/, '');
  const key = String(siteConfig.customFields?.supabasePublishableKey ?? '');
  const [games, setGames] = useState<Game[]>([]);
  const [events, setEvents] = useState<Event[]>([]);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [status, setStatus] = useState<'loading' | 'ready' | 'error' | 'unset'>(url && key ? 'loading' : 'unset');
  const [error, setError] = useState('');
  const [kind, setKind] = useState<'for' | 'against'>('for');
  const [showOpp, setShowOpp] = useState(false);

  useEffect(() => {
    if (!url || !key) return;
    (async () => {
      try {
        const g = await rest<Game>(url, key, 'games?select=id,name,date,our_team&order=date.desc,id.desc');
        const ev = await rest<Event>(url, key, 'events?select=game_id,type,t,team,data&order=id');
        setGames(g); setEvents(ev); setSelected(new Set(g.map(x => x.id))); setStatus('ready');
      } catch (err) {
        setError(String(err)); setStatus('error');
      }
    })();
  }, [url, key]);

  const byGame = useMemo(() => {
    const m = new Map<number, Event[]>();
    for (const e of events) (m.get(e.game_id) ?? m.set(e.game_id, []).get(e.game_id)!).push(e);
    return m;
  }, [events]);

  const chosen = games.filter(g => selected.has(g.id));
  const rows = chosen.map(g => ({g, s: tally(g, byGame.get(g.id) ?? [])}));
  const total = rows.reduce((acc, {s}) => {
    (Object.keys(acc) as (keyof Totals)[]).forEach(k => { acc[k] += s[k]; });
    return acc;
  }, blank());
  const goals = chosen.flatMap(g => (byGame.get(g.id) ?? [])
    .filter(e => e.type === 'goal' && (e.team === g.our_team) === (kind === 'for'))
    .map(e => ({game: g, e})));

  const toggle = (id: number) => setSelected(prev => {
    const next = new Set(prev);
    next.has(id) ? next.delete(id) : next.add(id);
    return next;
  });

  return (
    <Layout title="Results" description="Willamette soccer stats from the computer-vision pipeline">
      <main className={`container ${styles.page}`}>
        <h1>Results</h1>
        <p className={styles.lead}>
          Stats the computer-vision pipeline measured, from the online database. Pick any games: every number
          below adds up exactly those.
        </p>

        {status === 'unset' && (
          <div className="alert alert--info">Not connected to the database yet: <code>supabaseUrl</code> and{' '}
            <code>supabasePublishableKey</code> in <code>docusaurus.config.ts</code> are empty.</div>
        )}
        {status === 'loading' && <p>Loading games…</p>}
        {status === 'error' && <div className="alert alert--danger">Couldn't read the database: {error}</div>}
        {status === 'ready' && games.length === 0 && (
          <div className="alert alert--secondary">The database is connected but has no games yet. Add one with{' '}
            <code>python -m pipeline.push … --to supabase</code>.</div>
        )}

        {status === 'ready' && games.length > 0 && (
          <div className={styles.grid}>
            <aside className={styles.card}>
              <h2 className={styles.h2}>Games</h2>
              {games.map(g => (
                <label key={g.id} className={styles.game}>
                  <input type="checkbox" checked={selected.has(g.id)} onChange={() => toggle(g.id)} />
                  <span>{g.name}<small>{g.date}</small></span>
                </label>
              ))}
              <div className={styles.links}>
                <button onClick={() => setSelected(new Set(games.map(g => g.id)))}>All</button>
                <button onClick={() => setSelected(new Set())}>None</button>
              </div>
            </aside>

            <section>
              <div className={styles.stats}>
                {[['Games', chosen.length], ['Goals', `${total.goalsFor} – ${total.goalsAgainst}`],
                  ['Shots', `${total.shotsFor} – ${total.shotsAgainst}`],
                  ['Passes', `${total.passesFor} – ${total.passesAgainst}`],
                  ['Turnovers (ours)', total.turnovers]].map(([label, v]) => (
                  <div key={String(label)} className={styles.stat}><b>{v}</b><span>{label}</span></div>
                ))}
              </div>
              <p className={styles.note}>Goals, shots and passes are Willamette – opponent.</p>

              <div className={styles.card}>
                <h2 className={styles.h2}>Where were we standing when a goal was scored?</h2>
                <div className={styles.controls}>
                  <div className="button-group">
                    <button className={`button button--sm ${kind === 'for' ? 'button--primary' : 'button--secondary'}`}
                            onClick={() => setKind('for')}>Goals scored</button>
                    <button className={`button button--sm ${kind === 'against' ? 'button--primary' : 'button--secondary'}`}
                            onClick={() => setKind('against')}>Goals conceded</button>
                  </div>
                  <label><input type="checkbox" checked={showOpp} onChange={e => setShowOpp(e.target.checked)} /> show opponents</label>
                </div>
                {goals.length
                  ? <Pitch goals={goals} kind={kind} showOpp={showOpp} />
                  : <p className={styles.note}>No goals {kind === 'for' ? 'scored' : 'conceded'} in the selected games.</p>}
                <p className={styles.note}>{goals.length} goal{goals.length === 1 ? '' : 's'}; the scoring team always
                  attacks to the right. Blue: Willamette, red: opponents, yellow: the ball.</p>
              </div>

              <div className={styles.card}>
                <h2 className={styles.h2}>Per game</h2>
                <div className={styles.tableWrap}>
                  <table className={styles.table}>
                    <thead><tr><th>Game</th><th>Date</th><th>Goals</th><th>Shots</th><th>Passes</th><th>Turnovers</th></tr></thead>
                    <tbody>{rows.map(({g, s}) => (
                      <tr key={g.id}><td>{g.name}</td><td>{g.date}</td><td>{s.goalsFor} – {s.goalsAgainst}</td>
                        <td>{s.shotsFor} – {s.shotsAgainst}</td><td>{s.passesFor} – {s.passesAgainst}</td><td>{s.turnovers}</td></tr>
                    ))}</tbody>
                  </table>
                </div>
              </div>
            </section>
          </div>
        )}
      </main>
    </Layout>
  );
}
