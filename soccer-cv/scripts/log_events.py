"""Log what REALLY happens in a clip (passes, turnovers, shots, goals), in your own browser, to score the pipeline.

    python -m scripts.log_events test.mp4          (from soccer-cv/; opens http://127.0.0.1:8766)

Play the video and press a key the moment the ball is KICKED (that's the time the pipeline gives an event):
    P pass   T turnover   S shot   G goal      then   1 light shirts   2 dark shirts   (Esc = cancel)
The team is the one that passed / lost the ball / shot / scored. What counts:
    pass      a player plays the ball to a teammate, including throw-ins and free kicks to a teammate
    turnover  the other team gets the ball while it's in play: a tackle, an interception, a pass to an opponent.
              The ball going out of play is NOT a turnover: log nothing.
    In short: each time the ball goes from one player to the next, a teammate = pass, an opponent = turnover, by
    the team that had it. So a pass that's intercepted is ONLY a turnover, and a tackle that sets up a teammate is
    a turnover (by the team tackled) then a pass (by the tackler's team). That's how the pipeline counts them too.
    Timing: always the moment the ball LEAVES the player who had it (for an intercepted pass: the kick, not the
    interception), which is when the pipeline times its events. A scramble with the ball bouncing between the teams
    is ONE turnover, at its start, and only if the other team comes out of it with the ball (else nothing): the real
    answer, even if the pipeline counts each bounce, so the score shows whether it does. Likewise a pass that an
    opponent touches or deflects but that still ends up with the passer's team: log nothing (not a pass, as it
    wasn't meant for that player; not a turnover, as the ball never changed teams).
    shot / goal   an attempt on goal / a goal (log the shot too).
Space play/pause   <- -> 1 s (shift: 5 s)   , . one frame   down/up arrow slower/faster   Backspace delete the last event.
Click an event in the list to watch it again.

Everything is saved as you go to truth/<clip>_events.json, with how far into the clip you've watched: only that part
is scored. Then: python -m scripts.score_events truth/test_events.json data/test_tracks.json
"""
import argparse
import json
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

TYPES = ("pass", "turnover", "shot", "goal")


def load(path):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return {"events": [], "reviewed_until": 0.0}


def clean(body, video):
    """Keep only well-formed events, sorted by time."""
    events = []
    for e in body.get("events", []):
        if e.get("type") in TYPES and e.get("team") in (0, 1):
            events.append({"t": round(float(e["t"]), 2), "type": e["type"], "team": e["team"]})
    return {"video": Path(video).name, "teams": {"0": "light shirts", "1": "dark shirts"},
            "reviewed_until": round(float(body.get("reviewed_until", 0.0)), 2),
            "events": sorted(events, key=lambda e: e["t"])}


def make_handler(video, out, fps):
    size = Path(video).stat().st_size

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body, ctype="application/json", code=200):
            if not isinstance(body, bytes):
                body = (json.dumps(body) if isinstance(body, (dict, list)) else body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _video(self):
            """The video, in pieces when the browser asks for a range (it needs that to jump around in it)."""
            start, end = 0, size - 1
            m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start, end = int(m.group(1)), int(m.group(2) or end)
                else:
                    start = size - int(m.group(2))
                end = min(end, start + 8 * 2 ** 20 - 1, size - 1)
            with open(video, "rb") as f:
                f.seek(start)
                data = f.read(end - start + 1)
            self.send_response(206 if m else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(data)))
            if m:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                return self._send(PAGE, "text/html; charset=utf-8")
            if self.path == "/video.mp4":
                return self._video()
            if self.path == "/api/events":
                return self._send({**load(out), "fps": fps, "name": Path(video).name})
            self._send({"error": "not found"}, code=404)

        def do_POST(self):
            if self.path == "/api/events":
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                Path(out).write_text(json.dumps(clean(body, video), indent=1))
                return self._send({"ok": True})
            self._send({"error": "not found"}, code=404)

        def log_message(self, *args):
            pass

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description="log the real events of a clip in your browser")
    ap.add_argument("video")
    ap.add_argument("--out", help="default: truth/<clip>_events.json")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    if not Path(a.video).is_file():
        raise SystemExit(f"no such video: {a.video}")
    out = a.out or f"truth/{Path(a.video).stem}_events.json"
    fps = cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS) or 30.0
    for port in range(a.port, a.port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(a.video, out, fps))
            break
        except OSError:
            continue
    else:
        raise SystemExit("no free port")
    url = f"http://127.0.0.1:{port}"
    print(f"logging {a.video} at {url} -> {out}  (Ctrl+C here to stop; everything is saved as you go)")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Event Logger</title>
<style>
  :root { --bg:#16181d; --panel:#1f2229; --text:#e8e8ea; --muted:#9aa0aa; --line:#33363f; --accent:#4f8cff;
          --light:#9fd3f0; --dark:#b8344a; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.4 -apple-system, system-ui, sans-serif;
         display:flex; height:100vh; overflow:hidden; }
  #main { flex:1; display:flex; flex-direction:column; min-width:0; }
  video { flex:1; min-height:0; width:100%; background:#000; }
  #bar { display:flex; gap:16px; align-items:center; padding:8px 14px; border-top:1px solid var(--line); flex-wrap:wrap; }
  #prompt { font-weight:600; color:var(--accent); min-width:260px; }
  #side { width:300px; background:var(--panel); border-left:1px solid var(--line); display:flex; flex-direction:column; }
  #side > div { padding:12px 14px; border-bottom:1px solid var(--line); }
  h1 { font-size:15px; margin:0 0 4px; }
  .muted { color:var(--muted); font-size:12px; }
  kbd { background:#2a2f3a; border:1px solid var(--line); border-radius:4px; padding:0 5px; font-size:12px; }
  #list { flex:1; overflow:auto; padding:6px 8px !important; }
  .ev { display:flex; align-items:center; gap:8px; padding:5px 6px; border-radius:6px; cursor:pointer; }
  .ev:hover { background:#2a2f3a; }
  .dot { width:10px; height:10px; border-radius:50%; flex:none; }
  .ev .t { font-variant-numeric:tabular-nums; color:var(--muted); width:52px; }
  .ev .x { margin-left:auto; color:var(--muted); background:none; border:none; cursor:pointer; font-size:16px; }
  .sp { background:#2a2f3a; color:var(--text); border:1px solid var(--line); border-radius:6px; padding:4px 10px;
        font:inherit; cursor:pointer; }
  .sp:hover { border-color:var(--accent); }
  #seek { position:relative; padding:10px 14px 4px; }
  #scrub { width:100%; margin:0; accent-color:var(--accent); cursor:pointer; }
  #marks { position:relative; height:8px; margin:2px 7px 0; }
  #marks i { position:absolute; top:0; width:3px; height:8px; border-radius:1px; transform:translateX(-1px); }
  .counts { display:grid; grid-template-columns:1fr auto auto; gap:2px 12px; font-variant-numeric:tabular-nums; }
</style></head>
<body>
<div id="main">
  <video id="v" src="/video.mp4" preload="auto"></video>
  <div id="seek"><input type="range" id="scrub" min="0" max="1000" value="0" step="1"><div id="marks"></div></div>
  <div id="bar">
    <button class="sp" data-jump="-5">&laquo; 5 s</button>
    <button class="sp" data-jump="-1">&lsaquo; 1 s</button>
    <button class="sp" data-frame="-1">&lsaquo; frame</button>
    <button class="sp" id="play" style="min-width:90px">&#9654; Play</button>
    <button class="sp" data-frame="1">frame &rsaquo;</button>
    <button class="sp" data-jump="1">1 s &rsaquo;</button>
    <button class="sp" data-jump="5">5 s &raquo;</button>
    <span id="time">0.00 s</span>
    <button class="sp" id="slower">&darr; slower</button><span id="rate">speed 1x</span>
    <button class="sp" id="faster">&uarr; faster</button><span id="prompt"></span>
    <span class="muted" id="saved"></span>
  </div>
</div>
<div id="side">
  <div><h1 id="name">Event logger</h1>
    <div class="muted">At the moment the ball is <b>kicked</b>: <kbd>P</kbd> pass <kbd>T</kbd> turnover <kbd>S</kbd> shot
      <kbd>G</kbd> goal, then <kbd>1</kbd> light / <kbd>2</kbd> dark (the team that passed, lost it, shot, scored).
      Ball out of play is not a turnover.<br>
      <b>Each time the ball goes to the next player:</b> a teammate = pass, an opponent = turnover, by the team
      that had it. A failed pass is only a turnover; a tackle that sets up a teammate is a turnover, then a pass.
      Always press when the ball LEAVES the player who had it (the kick, not the interception). A scramble where the
      ball bounces between the teams: one turnover at its start, and only if the other team comes out with it.
      A pass deflected by an opponent that still reaches the passer's team: log nothing.<br>
      <kbd>Space</kbd> play <kbd>&larr;</kbd><kbd>&rarr;</kbd> 1 s (shift 5 s) <kbd>,</kbd><kbd>.</kbd> frame
      <kbd>&darr;</kbd><kbd>&uarr;</kbd> slower / faster <kbd>&#9003;</kbd> delete last</div></div>
  <div><div class="counts" id="counts"></div>
    <div class="muted" style="margin-top:6px">Watched up to <b id="upto">0.0</b> s (only this part is scored)</div></div>
  <div id="list"></div>
</div>
<script>
const v = document.getElementById('v');
const RATES = [0.25, 0.5, 0.75, 1, 1.5, 2];
const KEYS = {p: 'pass', t: 'turnover', s: 'shot', g: 'goal'};
const ASK = {pass: 'which team PASSED', turnover: 'which team LOST the ball', shot: 'which team SHOT', goal: 'which team SCORED'};
const BY = {pass: 'by', turnover: 'lost by', shot: 'by', goal: 'by'};
let events = [], added = [], upto = 0, fps = 30, pending = null, saveTimer = null;

function fmt(t) { return t.toFixed(2); }
function save() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    const r = await fetch('/api/events', {method: 'POST', body: JSON.stringify({events, reviewed_until: upto})});
    document.getElementById('saved').textContent = r.ok ? 'saved' : 'NOT SAVED: is the terminal still running?';
  }, 300);
}
function render() {
  events.sort((a, b) => a.t - b.t);
  const list = document.getElementById('list');
  list.innerHTML = '';
  events.forEach((e, i) => {
    const row = document.createElement('div');
    row.className = 'ev';
    row.innerHTML = `<span class="t">${fmt(e.t)}</span><span class="dot" style="background:var(--${e.team ? 'dark' : 'light'})"></span>
      <span>${e.type}</span><span class="muted">${BY[e.type]} ${e.team ? 'dark' : 'light'}</span><button class="x" title="delete">&times;</button>`;
    row.onclick = (ev) => {
      if (ev.target.classList.contains('x')) {
        events.splice(i, 1); added = added.filter(a => a !== e); render(); save(); return;
      }
      v.currentTime = Math.max(0, e.t - 2); v.play();
    };
    list.appendChild(row);
  });
  const c = {};
  for (const e of events) { c[e.type] = c[e.type] || [0, 0]; c[e.type][e.team]++; }
  document.getElementById('counts').innerHTML = '<span class="muted"></span><span class="muted">light</span><span class="muted">dark</span>' +
    ['pass', 'turnover', 'shot', 'goal'].map(k => `<span>${k === 'turnover' ? 'turnover (lost)' : k}</span><span>${(c[k] || [0, 0])[0]}</span><span>${(c[k] || [0, 0])[1]}</span>`).join('');
  document.getElementById('upto').textContent = upto.toFixed(1);
  marks();
}
function setPrompt(s) { document.getElementById('prompt').textContent = s; }
function speed(step) {
  const i = RATES.indexOf(v.playbackRate);
  v.playbackRate = RATES[Math.max(0, Math.min(RATES.length - 1, (i < 0 ? 3 : i) + step))];
  document.getElementById('rate').textContent = `speed ${v.playbackRate}x`;
}
function jump(s) { v.currentTime = Math.max(0, Math.min(v.duration || Infinity, v.currentTime + s)); }
for (const b of document.querySelectorAll('[data-jump],[data-frame]')) {
  b.onclick = () => {
    if (b.dataset.frame) { v.pause(); jump(Number(b.dataset.frame) / fps); } else jump(Number(b.dataset.jump));
    b.blur();
  };
}
const scrub = document.getElementById('scrub');
scrub.addEventListener('input', () => { if (v.duration) v.currentTime = scrub.value / 1000 * v.duration; });
scrub.addEventListener('change', () => scrub.blur());
function marks() {   // one tick per logged event under the timeline
  const box = document.getElementById('marks');
  box.innerHTML = v.duration ? events.map(e => `<i title="${e.type} ${BY[e.type]} ${e.team ? 'dark' : 'light'} at ${fmt(e.t)} s"
    style="left:${100 * e.t / v.duration}%;background:var(--${e.team ? 'dark' : 'light'})"></i>`).join('') : '';
}
v.addEventListener('loadedmetadata', marks);
const playBtn = document.getElementById('play');
playBtn.onclick = () => { v.paused ? v.play() : v.pause(); playBtn.blur(); };
v.addEventListener('play', () => { playBtn.innerHTML = '&#10074;&#10074; Pause'; });
v.addEventListener('pause', () => { playBtn.innerHTML = '&#9654; Play'; });
v.addEventListener('click', () => { v.paused ? v.play() : v.pause(); });   // clicking the video works too
for (const [id, step] of [['slower', -1], ['faster', 1]]) {
  const b = document.getElementById(id);
  b.onclick = () => { speed(step); b.blur(); };   // blur: so Space still plays/pauses afterwards
}

v.addEventListener('timeupdate', () => {
  document.getElementById('time').textContent = fmt(v.currentTime) + ' s';
  if (v.duration && document.activeElement !== scrub) scrub.value = Math.round(1000 * v.currentTime / v.duration);
  if (!v.paused && !v.seeking && v.currentTime > upto) { upto = v.currentTime; render(); save(); }
});
v.addEventListener('ended', () => { upto = v.duration; render(); save(); });

document.addEventListener('keydown', (ev) => {
  const k = ev.key.toLowerCase();
  if (pending) {
    if (k === '1' || k === '2') {
      const e = {t: pending.t, type: pending.type, team: k === '1' ? 0 : 1};
      events.push(e); added.push(e);
      setPrompt(`${pending.type} ${BY[pending.type]} ${k === '1' ? 'light' : 'dark'} at ${fmt(pending.t)} s`);
      pending = null; render(); save();
    } else if (k === 'escape') { pending = null; setPrompt('cancelled'); }
    ev.preventDefault(); return;
  }
  if (KEYS[k]) {
    pending = {t: v.currentTime, type: KEYS[k]};
    setPrompt(`${KEYS[k]} at ${fmt(pending.t)} s: ${ASK[KEYS[k]]}? 1 light / 2 dark`);
  } else if (k === ' ') { v.paused ? v.play() : v.pause(); }
  else if (k === 'arrowleft') { jump(ev.shiftKey ? -5 : -1); }
  else if (k === 'arrowright') { jump(ev.shiftKey ? 5 : 1); }
  else if (k === ',') { v.pause(); jump(-1 / fps); }
  else if (k === '.') { v.pause(); jump(1 / fps); }
  else if (k === 'arrowdown') { speed(-1); }
  else if (k === 'arrowup') { speed(1); }
  else if (k === 'backspace' && events.length) {   // the one added most recently (else the latest in the clip)
    const last = added.length ? added.pop() : events.reduce((a, b) => (a.t > b.t ? a : b));
    events.splice(events.indexOf(last), 1); setPrompt(`deleted ${last.type} at ${fmt(last.t)} s`); render(); save();
  } else return;
  ev.preventDefault();
});

fetch('/api/events').then(r => r.json()).then(d => {
  events = d.events || []; upto = d.reviewed_until || 0; fps = d.fps || 30;
  document.getElementById('name').textContent = d.name;
  render();
});
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
