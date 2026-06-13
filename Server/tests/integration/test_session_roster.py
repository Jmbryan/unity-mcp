"""Integration tests for the session roster build + push (MCPC-028).

Covers:
- Store merge + label join (hook data attaches to the right MCP session).
- State-enum derivation, most-specific-wins, per state (in-play, running-tests,
  waiting-parked, editing, active, idle, disconnected).
- Intent most-specific-wins (todo > spawn > prompt) and mcp-only suppression.
- mcp-only tagging for a session with no matching hook label.
- Flags surfaced on entries (looping via the middleware ring; parked-over-budget
  via the call-activity park marker).
- Roster push to a connected bridge WebSocket (fake connection captures the
  outbound session_roster message); cheap no-op when nothing changed.

Offline: pure in-memory state; the bridge WS is a fake capturing send_json.
"""

import time

import pytest

from services.state import session_roster as roster_mod
from services.state.agent_status_store import (
    EVENT_PRE_TOOL_USE,
    EVENT_SUBAGENT_START,
    EVENT_USER_PROMPT_SUBMIT,
    agent_status_store,
)
from services.state.call_activity import call_activity
from services.state.play_lease import play_lease_manager
from services.state.session_roster import (
    ROSTER_MESSAGE_TYPE,
    build_roster,
    roster_publisher,
)
from services.state.test_job_lease import test_job_lease_manager
from transport.plugin_hub import PluginHub
from transport.unity_instance_middleware import (
    UnityInstanceMiddleware,
    set_unity_instance_middleware,
)

INSTANCE = "Game@hash-x"
HASH = "hash-x"


@pytest.fixture(autouse=True)
def _fresh_state():
    agent_status_store.reset()
    call_activity.reset()
    play_lease_manager.reset()
    test_job_lease_manager.reset()
    roster_publisher.reset()
    set_unity_instance_middleware(UnityInstanceMiddleware())
    yield
    agent_status_store.reset()
    call_activity.reset()
    play_lease_manager.reset()
    test_job_lease_manager.reset()
    roster_publisher.reset()
    set_unity_instance_middleware(UnityInstanceMiddleware())


def _identity(middleware, key, label=None):
    """Assign an identity for a session key, optionally with a label."""
    identity = middleware.ensure_session_identity_for_key(key)
    if label is not None:
        identity.label = label
    return identity


def _entry_for(roster, session_key):
    return next(
        (s for s in roster["sessions"] if s["session_key"] == session_key), None
    )


class TestStoreMergeAndLabelJoin:
    def test_hook_data_joins_to_matching_label(self):
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        _identity(middleware, "sess-1", label="agent-a")

        agent_status_store.ingest(
            {"label": "agent-a", "event": EVENT_PRE_TOOL_USE, "summary": "did x", "ts": time.time()}
        )

        roster = build_roster()
        entry = _entry_for(roster, "sess-1")
        assert entry is not None
        assert entry["label"] == "agent-a"
        assert entry["source"] == "full"
        assert entry["activity_tail"][-1]["summary"] == "did x"

    def test_session_without_hook_label_is_mcp_only(self):
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        _identity(middleware, "sess-1", label="hookless")  # no store entry

        roster = build_roster()
        entry = _entry_for(roster, "sess-1")
        assert entry["source"] == "mcp-only"
        assert entry["intent"] is None  # mcp-only rows carry no intent


class TestStateEnum:
    def _setup(self, key="sess-1", label="agent-a"):
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        _identity(middleware, key, label=label)
        return middleware

    def test_in_play_wins(self):
        self._setup()
        play_lease_manager.acquire(INSTANCE, "sess-1", "AgentA")
        # Also has a parked + recent mutate; in-play still wins.
        call_activity.mark_parked("sess-1", "compiling", "AgentA")
        call_activity.mark_activity("sess-1", is_mutate=True)
        entry = _entry_for(build_roster(), "sess-1")
        assert entry["state"] == "in-play"
        assert entry["attribution"]["holds_play_lease"] is True
        assert entry["attribution"]["play_lease_owner"] == "AgentA"

    def test_running_tests(self):
        self._setup()
        test_job_lease_manager.record(INSTANCE, "job-1", "sess-1", "AgentA")
        entry = _entry_for(build_roster(), "sess-1")
        assert entry["state"] == "running-tests"

    def test_waiting_parked(self):
        self._setup()
        call_activity.mark_parked("sess-1", "domain_reload", "AgentB")
        entry = _entry_for(build_roster(), "sess-1")
        assert entry["state"] == "waiting-parked"
        assert entry["attribution"]["parked"] is True
        assert entry["attribution"]["parked_on"] == "domain_reload"

    def test_editing(self):
        self._setup()
        call_activity.mark_activity("sess-1", is_mutate=True)
        entry = _entry_for(build_roster(), "sess-1")
        assert entry["state"] == "editing"

    def test_active(self):
        self._setup()
        call_activity.mark_activity("sess-1", is_mutate=False)
        entry = _entry_for(build_roster(), "sess-1")
        assert entry["state"] == "active"

    def test_idle(self):
        self._setup()
        entry = _entry_for(build_roster(), "sess-1")
        assert entry["state"] == "idle"

    def test_disconnected_on_session_end(self):
        self._setup()
        agent_status_store.ingest({"label": "agent-a", "event": "SessionEnd", "summary": "", "ts": time.time()})
        entry = _entry_for(build_roster(), "sess-1")
        assert entry["state"] == "disconnected"


class TestIntentMostSpecificWins:
    def _setup(self):
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        _identity(middleware, "sess-1", label="agent-a")
        return middleware

    def test_prompt_head_when_only_prompt(self):
        self._setup()
        agent_status_store.ingest(
            {"label": "agent-a", "event": EVENT_USER_PROMPT_SUBMIT, "summary": "fix bug", "ts": time.time()}
        )
        assert _entry_for(build_roster(), "sess-1")["intent"] == "fix bug"

    def test_spawn_beats_prompt(self):
        self._setup()
        agent_status_store.ingest(
            {"label": "agent-a", "event": EVENT_USER_PROMPT_SUBMIT, "summary": "fix bug", "ts": time.time()}
        )
        agent_status_store.ingest(
            {"label": "agent-a", "event": EVENT_SUBAGENT_START, "summary": "explore code", "ts": time.time()}
        )
        assert _entry_for(build_roster(), "sess-1")["intent"] == "explore code"

    def test_todo_beats_spawn(self):
        self._setup()
        agent_status_store.ingest(
            {"label": "agent-a", "event": EVENT_SUBAGENT_START, "summary": "explore code", "ts": time.time()}
        )
        agent_status_store.set_todo("agent-a", "wire the route")
        assert _entry_for(build_roster(), "sess-1")["intent"] == "wire the route"


class TestFlags:
    def test_looping_flag_from_signature_ring(self):
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        _identity(middleware, "sess-1", label="agent-a")
        for _ in range(5):
            middleware.record_tool_call_signature("sess-1", "manage_editor", {"action": "play"})
        entry = _entry_for(build_roster(), "sess-1")
        assert "looping" in entry["flags"]

    def test_parked_over_budget_flag(self, monkeypatch):
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        _identity(middleware, "sess-1", label="agent-a")
        call_activity.mark_parked("sess-1", "compiling", "AgentA")
        # Force the park marker to look old (beyond the 15s budget).
        marker = call_activity._parked["sess-1"]
        marker.started_at = time.monotonic() - 20.0
        entry = _entry_for(build_roster(), "sess-1")
        assert "parked-over-budget" in entry["flags"]

    def test_no_flags_when_quiet(self):
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        _identity(middleware, "sess-1", label="agent-a")
        entry = _entry_for(build_roster(), "sess-1")
        assert entry["flags"] == []


class TestEnvelope:
    def test_envelope_has_lease_and_health(self):
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        _identity(middleware, "sess-1", label="agent-a")
        play_lease_manager.acquire(INSTANCE, "sess-1", "AgentA")

        roster = build_roster()
        assert roster["type"] == ROSTER_MESSAGE_TYPE
        assert roster["play_lease"]["owner"] == "AgentA"
        assert roster["play_lease"]["instance"] == HASH
        assert roster["health"]["ok"] is True
        assert "session_count" in roster["health"]


class _FakeWS:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


class TestPush:
    @pytest.mark.asyncio
    async def test_push_sends_roster_to_connected_bridge(self, monkeypatch):
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        _identity(middleware, "sess-1", label="agent-a")

        ws = _FakeWS()
        monkeypatch.setattr(PluginHub, "_connections", {"bridge-1": ws})
        monkeypatch.setattr(PluginHub, "is_configured", classmethod(lambda cls: True))

        sent = await roster_publisher.maybe_push(force=True)
        assert sent is True
        assert ws.sent[-1]["type"] == ROSTER_MESSAGE_TYPE
        assert any(s["session_key"] == "sess-1" for s in ws.sent[-1]["sessions"])

    @pytest.mark.asyncio
    async def test_unchanged_roster_is_noop_within_heartbeat(self, monkeypatch):
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        _identity(middleware, "sess-1", label="agent-a")

        ws = _FakeWS()
        monkeypatch.setattr(PluginHub, "_connections", {"bridge-1": ws})
        monkeypatch.setattr(PluginHub, "is_configured", classmethod(lambda cls: True))
        # Long heartbeat so only a genuine change would push.
        monkeypatch.setenv("UNITY_MCP_ROSTER_HEARTBEAT_S", "9999")

        assert await roster_publisher.maybe_push(force=True) is True
        # Nothing changed: second non-forced push is a no-op.
        assert await roster_publisher.maybe_push() is False
        assert len(ws.sent) == 1

    @pytest.mark.asyncio
    async def test_change_triggers_push(self, monkeypatch):
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        _identity(middleware, "sess-1", label="agent-a")

        ws = _FakeWS()
        monkeypatch.setattr(PluginHub, "_connections", {"bridge-1": ws})
        monkeypatch.setattr(PluginHub, "is_configured", classmethod(lambda cls: True))
        monkeypatch.setenv("UNITY_MCP_ROSTER_HEARTBEAT_S", "9999")

        await roster_publisher.maybe_push(force=True)
        # State change: acquire a play lease -> state becomes in-play.
        play_lease_manager.acquire(INSTANCE, "sess-1", "AgentA")
        assert await roster_publisher.maybe_push() is True
        assert len(ws.sent) == 2
