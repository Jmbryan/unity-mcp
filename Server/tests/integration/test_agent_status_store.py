"""Integration tests for the agent-status store (MCPC-031/033).

Covers:
- Well-formed event ingest: tail, last-activity, intent sources.
- Malformed / missing fields tolerated (never raises, never stores garbage).
- Label vs session-fallback bucketing (the MCPC-033 join key).
- Bounded ~10-event tail.
- TTL expiry of stale agents; SessionEnd ended-grace drop.

Offline: pure in-memory store, no Unity and no transport.
"""

import time

import pytest

from services.state import agent_status_store as store_mod
from services.state.agent_status_store import (
    EVENT_PRE_TOOL_USE,
    EVENT_SESSION_END,
    EVENT_SUBAGENT_START,
    EVENT_SUBAGENT_STOP,
    EVENT_USER_PROMPT_SUBMIT,
    MAX_SUMMARY_CHARS,
    MAX_TAIL_EVENTS,
    agent_status_store,
    handle_agent_status_post,
)


class _FakeRequest:
    """Minimal Starlette-Request stand-in with a JSON body (or a raising one)."""

    def __init__(self, payload=None, raise_on_json=False):
        self._payload = payload
        self._raise = raise_on_json

    async def json(self):
        if self._raise:
            raise ValueError("invalid JSON body")
        return self._payload


@pytest.fixture(autouse=True)
def _fresh_store():
    agent_status_store.reset()
    yield
    agent_status_store.reset()


def _event(label="agent-a", session="sess-1", event=EVENT_PRE_TOOL_USE, summary="ran tool", ts=None):
    return {
        "label": label,
        "session": session,
        "event": event,
        "summary": summary,
        "ts": ts if ts is not None else time.time(),
    }


class TestWellFormedIngest:
    def test_records_tail_and_last_activity(self):
        ts = time.time()
        assert agent_status_store.ingest(_event(summary="did a thing", ts=ts)) is True

        entry = agent_status_store.get_by_label("agent-a")
        assert entry is not None
        assert entry.label == "agent-a"
        assert entry.session == "sess-1"
        assert entry.last_activity_unix == ts
        tail = entry.tail_as_list()
        assert tail[-1]["event"] == EVENT_PRE_TOOL_USE
        assert tail[-1]["summary"] == "did a thing"

    def test_prompt_and_spawn_feed_intent_sources(self):
        agent_status_store.ingest(_event(event=EVENT_USER_PROMPT_SUBMIT, summary="fix the bug"))
        agent_status_store.ingest(_event(event=EVENT_SUBAGENT_START, summary="explore codebase"))

        entry = agent_status_store.get_by_label("agent-a")
        assert entry.latest_prompt_head == "fix the bug"
        assert entry.latest_spawn_description == "explore codebase"

    def test_subagent_stop_clears_spawn_description(self):
        agent_status_store.ingest(_event(event=EVENT_SUBAGENT_START, summary="explore"))
        agent_status_store.ingest(_event(event=EVENT_SUBAGENT_STOP, summary=""))

        entry = agent_status_store.get_by_label("agent-a")
        assert entry.latest_spawn_description is None

    def test_summary_clamped_to_120(self):
        long_summary = "x" * 300
        agent_status_store.ingest(_event(summary=long_summary))
        entry = agent_status_store.get_by_label("agent-a")
        assert len(entry.tail_as_list()[-1]["summary"]) == MAX_SUMMARY_CHARS


class TestMalformedTolerated:
    @pytest.mark.parametrize(
        "payload",
        [
            None,
            "not a dict",
            123,
            {},  # no label, no session
            {"event": "PreToolUse"},  # no join key
            {"label": "", "session": ""},  # blank join keys
        ],
    )
    def test_unstorable_payloads_return_false_no_raise(self, payload):
        assert agent_status_store.ingest(payload) is False

    def test_missing_fields_default_gracefully(self):
        # label present but everything else missing/garbage.
        assert agent_status_store.ingest({"label": "agent-a", "ts": "nope"}) is True
        entry = agent_status_store.get_by_label("agent-a")
        assert entry is not None
        # ts coerced to "now" (a positive float), event defaulted to "?"
        assert entry.last_activity_unix > 0
        assert entry.tail_as_list()[-1]["event"] == "?"

    def test_session_fallback_bucket_when_no_label(self):
        assert agent_status_store.ingest(
            {"session": "sess-x", "event": EVENT_PRE_TOOL_USE, "summary": "s"}
        ) is True
        # No label entry exists; the fallback bucket is keyed "session:<session>".
        assert agent_status_store.get_by_label("sess-x") is None
        snap = agent_status_store.snapshot()
        assert "session:sess-x" in snap
        assert snap["session:sess-x"].label is None


class TestBoundedTail:
    def test_tail_capped_at_max(self):
        for i in range(MAX_TAIL_EVENTS + 5):
            agent_status_store.ingest(_event(summary=f"event-{i}"))
        entry = agent_status_store.get_by_label("agent-a")
        tail = entry.tail_as_list()
        assert len(tail) == MAX_TAIL_EVENTS
        # Oldest evicted: the last MAX_TAIL_EVENTS remain.
        assert tail[0]["summary"] == f"event-{5}"
        assert tail[-1]["summary"] == f"event-{MAX_TAIL_EVENTS + 4}"


class TestExpiry:
    def test_ttl_expires_stale_agent(self, monkeypatch):
        monkeypatch.setenv("UNITY_MCP_AGENT_STATUS_TTL_S", "0.05")
        agent_status_store.ingest(_event())
        assert agent_status_store.get_by_label("agent-a") is not None
        time.sleep(0.08)
        # Next read triggers expiry sweep.
        assert agent_status_store.get_by_label("agent-a") is None

    def test_session_end_dropped_after_grace(self, monkeypatch):
        monkeypatch.setattr(store_mod, "ENDED_GRACE_SECONDS", 0.05)
        agent_status_store.ingest(_event(event=EVENT_SESSION_END, summary="bye"))
        entry = agent_status_store.get_by_label("agent-a")
        assert entry is not None and entry.ended is True
        time.sleep(0.08)
        assert agent_status_store.get_by_label("agent-a") is None


class TestRouteHandler:
    """The /agent-status route handler: always 200, never raises (MCPC-031)."""

    @pytest.mark.asyncio
    async def test_well_formed_post_stores_and_returns_200(self, monkeypatch):
        # Avoid pulling in the roster push path's transport.
        async def _noop_push(*_a, **_k):
            return False

        import services.state.session_roster as roster_mod
        monkeypatch.setattr(roster_mod.roster_publisher, "maybe_push", _noop_push)

        body, status = await handle_agent_status_post(_FakeRequest(_event()))
        assert status == 200
        assert body == {"ok": True}
        assert agent_status_store.get_by_label("agent-a") is not None

    @pytest.mark.asyncio
    async def test_malformed_json_body_returns_200_no_raise(self):
        body, status = await handle_agent_status_post(_FakeRequest(raise_on_json=True))
        assert status == 200
        assert body == {"ok": True}
        assert agent_status_store.snapshot() == {}

    @pytest.mark.asyncio
    async def test_unstorable_payload_returns_200(self):
        body, status = await handle_agent_status_post(_FakeRequest({}))
        assert status == 200
        assert body == {"ok": True}
        assert agent_status_store.snapshot() == {}
