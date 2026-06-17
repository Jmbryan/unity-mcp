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

import asyncio
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

    def test_active_from_hook_clock_only(self):
        """A row reads 'active' on fresh hook activity even with no recent MCP call.

        Regression (Fix #7): a working subagent's hook events refresh the parent
        row's hook clock; the state must reflect that liveness instead of 'idle'.
        """
        self._setup()
        # Fresh hook event, but no MCP call recorded for this session at all.
        agent_status_store.ingest(
            {"label": "agent-a", "event": EVENT_PRE_TOOL_USE, "summary": "subagent working", "ts": time.time()}
        )
        entry = _entry_for(build_roster(), "sess-1")
        assert entry["state"] == "active"

    def test_idle_when_hook_clock_is_stale(self):
        """A stale hook clock (beyond the hook window) does not hold 'active'."""
        self._setup()
        agent_status_store.ingest(
            {
                "label": "agent-a",
                "event": EVENT_PRE_TOOL_USE,
                "summary": "old work",
                "ts": time.time() - (roster_mod.HOOK_ACTIVE_RECENCY_SECONDS + 30.0),
            }
        )
        entry = _entry_for(build_roster(), "sess-1")
        assert entry["state"] == "idle"


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


class TestStaleIdentityEviction:
    """Fix #1: identities not seen within the TTL+grace window leave the roster.

    Self-heals on any disconnect mode (including a hard kill that never fires a
    SessionEnd hook), since the row's survival hangs on liveness, not on a hook.
    """

    def test_stale_identity_dropped_from_roster(self):
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        identity = _identity(middleware, "sess-stale", label="agent-stale")

        # Fresh identity is present.
        assert _entry_for(build_roster(), "sess-stale") is not None

        # Age last_seen past the TTL + grace window (monotonic clock).
        from transport.unity_instance_middleware import (
            _identity_ttl_s,
            _IDENTITY_TTL_GRACE_SECONDS,
        )
        identity.last_seen = (
            time.monotonic() - (_identity_ttl_s() + _IDENTITY_TTL_GRACE_SECONDS + 10.0)
        )

        # The stale row is gone; the snapshot evicts it.
        assert _entry_for(build_roster(), "sess-stale") is None
        assert middleware.all_session_identities() == []

    def test_live_access_refreshes_last_seen_and_keeps_row(self):
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        identity = _identity(middleware, "sess-live", label="agent-live")

        # Push it to the brink of eviction...
        from transport.unity_instance_middleware import (
            _identity_ttl_s,
            _IDENTITY_TTL_GRACE_SECONDS,
        )
        identity.last_seen = (
            time.monotonic() - (_identity_ttl_s() + _IDENTITY_TTL_GRACE_SECONDS + 10.0)
        )
        # ...but a live request (ensure_session_identity_for_key) refreshes it.
        middleware.ensure_session_identity_for_key("sess-live")

        assert _entry_for(build_roster(), "sess-live") is not None


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

    @pytest.mark.asyncio
    async def test_empty_roster_suppressed_while_bridge_connected(self, monkeypatch):
        """Bug 2: a transient empty-sessions roster must not clobber a good one.

        A non-forced push with no live identities and a bridge connected is skipped
        without recording the fingerprint, so a subsequent genuine non-empty roster
        still counts as changed and pushes.
        """
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)

        ws = _FakeWS()
        monkeypatch.setattr(PluginHub, "_connections", {"bridge-1": ws})
        monkeypatch.setattr(PluginHub, "is_configured", classmethod(lambda cls: True))
        monkeypatch.setenv("UNITY_MCP_ROSTER_HEARTBEAT_S", "9999")

        # No identities: empty-sessions roster. Suppressed while a bridge is present.
        assert await roster_publisher.maybe_push() is False
        assert ws.sent == []

        # An identity now appears: the prior skip did not poison change detection.
        _identity(middleware, "sess-1", label="agent-a")
        assert await roster_publisher.maybe_push() is True
        assert ws.sent[-1]["type"] == ROSTER_MESSAGE_TYPE
        assert any(s["session_key"] == "sess-1" for s in ws.sent[-1]["sessions"])

    @pytest.mark.asyncio
    async def test_forced_empty_roster_still_pushes(self, monkeypatch):
        """force=True (the on-register push, deliberate empty) bypasses the guard."""
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)

        ws = _FakeWS()
        monkeypatch.setattr(PluginHub, "_connections", {"bridge-1": ws})
        monkeypatch.setattr(PluginHub, "is_configured", classmethod(lambda cls: True))

        assert await roster_publisher.maybe_push(force=True) is True
        assert ws.sent[-1]["type"] == ROSTER_MESSAGE_TYPE
        assert ws.sent[-1]["sessions"] == []


class TestRegisterPush:
    @pytest.mark.asyncio
    async def test_register_force_pushes_roster_to_new_bridge(self, monkeypatch):
        """Bug 1: a bridge registering gets the current roster within the round-trip.

        Drives _handle_register with a fake websocket and a fake registry, then
        asserts a forced roster reached the connected bridge even though nothing
        changed.
        """
        from transport.models import RegisterMessage

        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        _identity(middleware, "sess-1", label="agent-a")

        class _RegisterWS(_FakeWS):
            def __init__(self):
                super().__init__()
                self.state = type("S", (), {})()

            async def close(self, code=1000):
                pass

        class _FakeSession:
            def __init__(self, session_id):
                self.session_id = session_id

        class _FakeRegistry:
            async def register(self, session_id, *args, **kwargs):
                return _FakeSession(session_id), None

        ws = _RegisterWS()
        monkeypatch.setattr(PluginHub, "_registry", _FakeRegistry())
        monkeypatch.setattr(PluginHub, "_lock", asyncio.Lock())
        monkeypatch.setattr(PluginHub, "_connections", {})
        monkeypatch.setattr(PluginHub, "_ping_tasks", {})
        monkeypatch.setattr(PluginHub, "_last_pong", {})
        monkeypatch.setattr(PluginHub, "is_configured", classmethod(lambda cls: True))
        # Keep the per-session ping loop from actually running.
        monkeypatch.setattr(
            PluginHub,
            "_ping_loop",
            classmethod(lambda cls, sid, sock: asyncio.sleep(0)),
        )
        monkeypatch.setenv("UNITY_MCP_ROSTER_HEARTBEAT_S", "9999")

        hub = PluginHub.__new__(PluginHub)
        await hub._handle_register(
            ws,
            RegisterMessage(
                type="register",
                project_name="Game",
                project_hash="hash-x",
                unity_version="2022",
                project_path="/tmp/game",
            ),
        )

        roster_msgs = [m for m in ws.sent if m.get("type") == ROSTER_MESSAGE_TYPE]
        assert roster_msgs, "register did not force a roster push to the new bridge"
        assert any(s["session_key"] == "sess-1" for s in roster_msgs[-1]["sessions"])
