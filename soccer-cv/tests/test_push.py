"""pipeline.push to Supabase, against a fake Supabase (no account or internet needed)."""
import io
import json
import urllib.error

import pytest

from pipeline import push


class FakeSupabase:
    """Stands in for urllib.request.urlopen: records requests, answers like Supabase's REST API."""

    def __init__(self, fail_events=False):
        self.calls, self.games, self.fail_events = [], [], fail_events

    def __call__(self, req, timeout=None):
        body = json.loads(req.data) if req.data else None
        self.calls.append((req.get_method(), req.full_url, dict(req.header_items()), body))
        path = req.full_url.split("/rest/v1/")[1]
        out = None
        if req.get_method() == "POST" and path == "games":
            g = dict(body[0], id=len(self.games) + 1)
            self.games.append(g)
            out = [g]
        elif req.get_method() == "POST" and path == "events" and self.fail_events:
            raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b'{"message":"bad row"}'))
        elif req.get_method() == "DELETE":
            gid = int(path.split("eq.")[1])
            out = [g for g in self.games if g["id"] == gid]
            self.games = [g for g in self.games if g["id"] != gid]
        elif req.get_method() == "GET":
            out = self.games
        return io.BytesIO(json.dumps(out).encode() if out is not None else b"")


EVENTS = [{"type": "pass", "frame": i, "t": i / 30, "team": 1, "player": 7, "data": {}} for i in range(1200)]
PAYLOAD = {"name": "vs. Linfield (H)", "date": "2026-09-05", "our_team": 1, "events": EVENTS}


def test_add_game_stores_the_game_then_its_events_in_chunks(monkeypatch):
    fake = FakeSupabase()
    monkeypatch.setattr(push.urllib.request, "urlopen", fake)
    gid = push.Supabase("https://x.supabase.co", "sb_secret_abc").add_game(PAYLOAD)
    assert gid == 1
    posts = [c for c in fake.calls if c[0] == "POST"]
    assert posts[0][3] == [{"name": "vs. Linfield (H)", "date": "2026-09-05", "our_team": 1}]
    assert [len(c[3]) for c in posts[1:]] == [500, 500, 200] and posts[1][3][0]["game_id"] == 1
    headers = {k.lower(): v for k, v in posts[0][2].items()}
    assert headers["apikey"] == "sb_secret_abc" and "authorization" not in headers   # new-style keys: apikey only


def test_a_failed_event_upload_removes_the_half_stored_game(monkeypatch):
    fake = FakeSupabase(fail_events=True)
    monkeypatch.setattr(push.urllib.request, "urlopen", fake)
    with pytest.raises(SystemExit, match="400"):
        push.Supabase("https://x.supabase.co", "sb_secret_abc").add_game(PAYLOAD)
    assert fake.games == [] and fake.calls[-1][0] == "DELETE"


def test_settings_come_from_the_env_file_and_are_required(tmp_path, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SECRET_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('# comment\nSUPABASE_URL="https://x.supabase.co/"\nSUPABASE_SECRET_KEY=sb_secret_abc\n')
    assert push.settings(env) == ("https://x.supabase.co", "sb_secret_abc")
    env.write_text("SUPABASE_URL=https://x.supabase.co/rest/v1/\nSUPABASE_SECRET_KEY=sb_secret_abc\n")
    assert push.settings(env)[0] == "https://x.supabase.co"          # the REST API address works too
    env.write_text("SUPABASE_URL=https://x.supabase.co\n")
    with pytest.raises(SystemExit, match="SUPABASE_SECRET_KEY"):
        push.settings(env)
