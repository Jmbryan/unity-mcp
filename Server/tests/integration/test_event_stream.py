"""Integration tests for the Phase 8 editor-edge event stream (MCPC-030).

The bridge pushes lightweight editor-edge events (play enter/exit, compile
start/finish, domain-reload done) so parked gate calls release event-driven
instead of waiting out their bounded poll. Bounded-wait deadlines and the
periodic state refresh stay the backstop — events only reduce latency, never
replace the safety net.

Covers:
- An ``event`` message parses, relaxes the cached snapshot's blocking flags,
  and sets the instance edge signal.
- The gate park loop wakes early on a pushed event instead of waiting out a
  long poll sleep; the signal auto-resets so it can't latch.
- An unknown inbound message type is still ignored gracefully (old/new
  tolerance both directions).
- Missing-event-never-wedges: with no event ever pushed, the deadline /
  poll backstop still releases the park.
"""

import asyncio
import time

import pytest

from core.config import config
from models.models import MCPResponse
from services.state import edit_ledger
from services.state import operation_gate
from services.state.editor_state_cache import editor_state_cache
from services.state.operation_gate import CLASS_MUTATE, gate_for_class, gate_tool_call
from transport.models import EDITOR_EDGE_EVENTS, EventMessage
from transport.plugin_hub import PluginHub

from .test_helpers import DummyContext

# Ensure manage_gameobject is in the server registry for classification.
import services.tools.manage_gameobject  # noqa: F401


COMPILING_STATE = {"compilation": {"is_compiling": True}}
IDLE_STATE = {"compilation": {}, "editor": {}, "tests": {}}
PLAY_CHANGING_STATE = {
    "compilation": {},
    "editor": {"play_mode": {"is_changing": True}},
    "tests": {},
}


class GateContext(DummyContext):
    def __init__(self, **meta):
        super().__init__(**meta)
        self.progress_reports = []

    async def report_progress(self, progress, total=None, message=None):
        self.progress_reports.append((progress, total, message))


class FakeEditorState:
    """Injectable editor-state source; last state repeats forever."""

    def __init__(self, *states):
        self.states = list(states)
        self.fetch_count = 0

    async def __call__(self, ctx):
        index = min(self.fetch_count, len(self.states) - 1)
        self.fetch_count += 1
        state = self.states[index]
        if isinstance(state, Exception):
            raise state
        return MCPResponse(success=True, message="ok", data=state)


@pytest.fixture(autouse=True)
def _fresh_gate_state(monkeypatch):
    editor_state_cache.reset(ttl_s=0.01)
    monkeypatch.setattr(config, "reload_retry_ms", 50, raising=False)
    monkeypatch.setattr(operation_gate, "_registry_last_refresh", 0.0)
    operation_gate._stale_warn_last.clear()
    yield
    editor_state_cache.reset(ttl_s=0.5)


def _inject_state(monkeypatch, fake: FakeEditorState) -> None:
    import services.resources.editor_state as editor_state_mod

    monkeypatch.setattr(editor_state_mod, "get_editor_state", fake)


def _no_fence(monkeypatch) -> None:
    async def _no_root(unity_instance):
        return None

    monkeypatch.setattr(edit_ledger, "resolve_project_root", _no_root)


# ----------------------------------------------------------------------
# Event model + on_receive handling
# ----------------------------------------------------------------------
class TestEventMessageModel:
    def test_event_message_parses_full_payload(self):
        msg = EventMessage(**{
            "type": "event",
            "event": "compile_finished",
            "instance": "Game@hash-x",
            "project_hash": "hash-x",
            "ts": 1730000000.5,
            "payload": {"errors": 0},
        })
        assert msg.event == "compile_finished"
        assert msg.instance == "Game@hash-x"
        assert msg.project_hash == "hash-x"
        assert msg.ts == 1730000000.5
        assert msg.payload == {"errors": 0}

    def test_event_message_tolerates_missing_optional_fields(self):
        # Only `event` is required; everything else defaults so an older or
        # truncated bridge push still parses.
        msg = EventMessage(**{"type": "event", "event": "entered_play"})
        assert msg.event == "entered_play"
        assert msg.instance is None
        assert msg.project_hash is None
        assert msg.ts is None
        assert msg.payload == {}

    def test_event_name_set_is_the_documented_contract(self):
        assert EDITOR_EDGE_EVENTS == {
            "entered_play",
            "exited_play",
            "compile_started",
            "compile_finished",
            "domain_reload_done",
        }


class TestOnReceiveEventBranch:
    @pytest.mark.asyncio
    async def test_event_updates_cache_and_sets_signal(self, monkeypatch):
        # Seed the cache with a compiling snapshot for the instance.
        editor_state_cache.reset(ttl_s=60.0)
        fake = FakeEditorState(dict(COMPILING_STATE))
        _inject_state(monkeypatch, fake)
        ctx = GateContext()
        state = await editor_state_cache.get(ctx, "Game@hash-x")
        assert state["compilation"]["is_compiling"] is True

        # A compile_finished event relaxes the cached flag and fires the
        # signal so a waiter wakes immediately.
        hub = PluginHub.__new__(PluginHub)
        hub._handle_event(EventMessage(
            type="event", event="compile_finished", instance="Game@hash-x",
            project_hash="hash-x", ts=time.time(),
        ))

        cached = editor_state_cache._entries["Game@hash-x"].state
        assert cached["compilation"]["is_compiling"] is False
        # Signal is set: a wait returns True without blocking.
        fired = await editor_state_cache.wait_for_edge_event("Game@hash-x", 1.0)
        assert fired is True

    @pytest.mark.asyncio
    async def test_play_edge_relaxes_play_changing_flag(self, monkeypatch):
        editor_state_cache.reset(ttl_s=60.0)
        fake = FakeEditorState(
            {"compilation": {}, "editor": {"play_mode": {"is_changing": True}}, "tests": {}})
        _inject_state(monkeypatch, fake)
        await editor_state_cache.get(GateContext(), "Game@hash-x")

        hub = PluginHub.__new__(PluginHub)
        hub._handle_event(EventMessage(
            type="event", event="entered_play", instance="Game@hash-x"))

        cached = editor_state_cache._entries["Game@hash-x"].state
        assert cached["editor"]["play_mode"]["is_changing"] is False

    def test_unknown_event_name_is_ignored(self):
        # An unrecognized event name fires no signal and raises nothing.
        hub = PluginHub.__new__(PluginHub)
        hub._handle_event(EventMessage(type="event", event="warp_core_breach"))
        # No signal recorded for the default key.
        assert "default" not in editor_state_cache._event_signals or (
            not editor_state_cache._event_signals["default"].is_set()
        )

    @pytest.mark.asyncio
    async def test_on_receive_ignores_unknown_message_type(self):
        # Old-server / new-bridge tolerance: an entirely unknown inbound type
        # is swallowed by on_receive's else branch without raising. We drive
        # on_receive directly with a fake websocket; it should return cleanly.
        hub = PluginHub.__new__(PluginHub)

        class _FakeWS:
            pass

        # Unknown type: hits the else branch (debug-logged, ignored).
        await hub.on_receive(_FakeWS(), {"type": "totally_new_message", "x": 1})
        # Non-dict payload: warned and ignored.
        await hub.on_receive(_FakeWS(), ["not", "a", "dict"])


# ----------------------------------------------------------------------
# Park-loop early wakeup (MCPC-030)
# ----------------------------------------------------------------------
class TestParkLoopEarlyWakeup:
    @pytest.mark.asyncio
    async def test_park_wakes_early_on_pushed_event(self, monkeypatch):
        # First poll sees compiling, second sees idle. Force a *long* poll
        # sleep so the only way to release quickly is the pushed event.
        monkeypatch.setattr(operation_gate, "_poll_sleep_s", lambda: 5.0)
        monkeypatch.setenv("UNITY_MCP_GATE_PARK_MAX_WAIT_S", "20")
        fake = FakeEditorState(dict(COMPILING_STATE), dict(IDLE_STATE))
        _inject_state(monkeypatch, fake)
        _no_fence(monkeypatch)

        async def _push_event_soon():
            await asyncio.sleep(0.1)
            hub = PluginHub.__new__(PluginHub)
            hub._handle_event(EventMessage(
                type="event", event="compile_finished", instance="Game@hash-x"))

        ctx = GateContext()
        started = time.monotonic()
        _, result = await asyncio.gather(
            gate_for_class(ctx, CLASS_MUTATE, "manage_gameobject", "Game@hash-x"),
            _push_event_soon(),
        )
        elapsed = time.monotonic() - started

        assert result is None  # released
        # Released right after the ~0.1s push, far short of the 5s poll sleep.
        assert elapsed < 2.0

    @pytest.mark.asyncio
    async def test_signal_auto_resets_and_does_not_latch(self, monkeypatch):
        # Fire an event, consume it, then confirm a subsequent wait does not
        # spuriously return True (the signal cleared on consumption).
        editor_state_cache.apply_edge_event("Game@hash-x", "compile_finished")
        assert await editor_state_cache.wait_for_edge_event("Game@hash-x", 1.0) is True
        # Auto-reset: no fresh event, so this wait times out (False), proving
        # the signal didn't latch.
        started = time.monotonic()
        assert await editor_state_cache.wait_for_edge_event("Game@hash-x", 0.2) is False
        assert time.monotonic() - started >= 0.15


# ----------------------------------------------------------------------
# Backstop: missing event never wedges (MCPC-030)
# ----------------------------------------------------------------------
class TestMissingEventBackstop:
    @pytest.mark.asyncio
    async def test_poll_releases_without_any_event(self, monkeypatch):
        # No event is ever pushed. The bounded poll alone must still release
        # the park when the state flips to idle.
        fake = FakeEditorState(
            dict(COMPILING_STATE), dict(COMPILING_STATE), dict(IDLE_STATE))
        _inject_state(monkeypatch, fake)
        _no_fence(monkeypatch)

        ctx = GateContext()
        result = await gate_for_class(
            ctx, CLASS_MUTATE, "manage_gameobject", "Game@hash-x")

        assert result is None  # released via polling, no event needed
        assert fake.fetch_count >= 3

    @pytest.mark.asyncio
    async def test_deadline_backstop_still_fires_without_event(self, monkeypatch):
        # Editor stays compiling forever and no event is pushed: the absolute
        # park deadline must still convert to a structured busy result.
        monkeypatch.setenv("UNITY_MCP_GATE_PARK_MAX_WAIT_S", "0.2")
        fake = FakeEditorState(dict(COMPILING_STATE))
        _inject_state(monkeypatch, fake)
        _no_fence(monkeypatch)

        ctx = GateContext()
        result = await gate_tool_call(ctx, "manage_gameobject", {"action": "create"})

        assert result is not None
        assert result["success"] is False
        assert result["hint"] == "retry"
        assert result["data"]["reason"] == "compiling"
