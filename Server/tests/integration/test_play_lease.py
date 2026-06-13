"""Integration tests for the play-session lease (MCPC-012..018).

Covers:
- Implicit acquire when a gated call successfully enters play mode; no
  explicit request/release API exists (MCPC-012).
- Non-owner stop/pause/play and compile-class calls during an active lease
  receive a structured busy naming the owner (MCPC-013).
- Play-scoped tools (drivers, camera control) are owner-only while leased;
  reads always pass (MCPC-014, MCPC-015).
- Play entered with no MCP cause is leased to owner "user" (MCPC-016).
- Liveness never wedges: inactivity TTL expiry frees the lease, and
  validate-on-block self-clears it when the editor already left play mode
  (MCPC-017).
- Lease survives a domain-reload reconnect (keyed to project identity, not
  the WebSocket session) and fails open after the disconnect grace window;
  the RunState mirror is written fail-open (MCPC-018).
"""

import asyncio
import json
import os
import time
import types

import pytest

from core.config import config
from models.models import MCPResponse, ToolDefinitionModel
from services.state import edit_ledger
from services.state import operation_gate
from services.state import play_lease
from services.state.editor_state_cache import editor_state_cache
from services.state.operation_gate import (
    CLASS_EXCLUSIVE,
    gate_for_class,
    gate_tool_call,
)
from services.state.play_lease import play_lease_manager
from transport.plugin_hub import PluginHub
from transport.plugin_registry import PluginRegistry
from transport.unity_instance_middleware import (
    UnityInstanceMiddleware,
    set_unity_instance_middleware,
)

from .test_helpers import DummyContext

# Ensure these tools are present in the server registry for classification.
import services.tools.manage_camera  # noqa: F401
import services.tools.manage_editor  # noqa: F401
import services.tools.read_console  # noqa: F401
import services.tools.refresh_unity  # noqa: F401


INSTANCE = "Game@hash-x"
IDLE_STATE = {"compilation": {}, "editor": {}, "tests": {}}
PLAYING_STATE = {
    "compilation": {},
    "editor": {"play_mode": {"is_playing": True, "is_paused": False, "is_changing": False}},
    "tests": {},
}
NOT_PLAYING_STATE = {
    "compilation": {},
    "editor": {"play_mode": {"is_playing": False, "is_paused": False, "is_changing": False}},
    "tests": {},
}


class FakeEditorState:
    """Injectable editor-state source (monkeypatches get_editor_state)."""

    def __init__(self, *states):
        # Last state repeats forever.
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
def _fresh_lease_state(monkeypatch):
    """Fresh lease/cache/identity state and fast pacing for every test."""
    play_lease_manager.reset()
    editor_state_cache.reset(ttl_s=0.01)
    monkeypatch.setattr(config, "reload_retry_ms", 50, raising=False)
    set_unity_instance_middleware(UnityInstanceMiddleware())
    monkeypatch.setattr(operation_gate, "_registry_last_refresh", 0.0)
    yield
    play_lease_manager.reset()
    editor_state_cache.reset(ttl_s=0.5)
    set_unity_instance_middleware(UnityInstanceMiddleware())


@pytest.fixture(autouse=True)
def _reset_plugin_hub():
    old_registry = PluginHub._registry
    old_connections = PluginHub._connections.copy()
    old_pending = PluginHub._pending.copy()
    old_lock = PluginHub._lock
    old_loop = PluginHub._loop
    yield
    PluginHub._registry = old_registry
    PluginHub._connections = old_connections
    PluginHub._pending = old_pending
    PluginHub._lock = old_lock
    PluginHub._loop = old_loop


def _inject_state(monkeypatch, fake: FakeEditorState) -> None:
    import services.resources.editor_state as editor_state_mod

    monkeypatch.setattr(editor_state_mod, "get_editor_state", fake)


def _no_fence(monkeypatch) -> None:
    async def _no_root(unity_instance):
        return None

    monkeypatch.setattr(edit_ledger, "resolve_project_root", _no_root)


def _fake_probe(monkeypatch, playing: bool):
    """Monkeypatch PluginHub.send_command with a raw-bridge-shaped probe."""
    calls = []

    async def fake_send(session_id, command_type, params):
        calls.append((session_id, command_type))
        return {
            "status": "success",
            "result": {
                "editor": {
                    "play_mode": {"is_playing": playing, "is_changing": False}
                }
            },
        }

    monkeypatch.setattr(PluginHub, "send_command", fake_send)
    return calls


async def _register_instance(tmp_path=None, session_id="guid-1"):
    registry = PluginRegistry()
    PluginHub.configure(registry, asyncio.get_running_loop())
    await registry.register(
        session_id, "Game", "hash-x", "6000.0",
        project_path=str(tmp_path) if tmp_path is not None else None,
    )
    return registry


async def _pinned_context():
    ctx = DummyContext()
    await ctx.set_state("unity_instance", INSTANCE)
    return ctx


def _acquire_for(ctx, project_root=None):
    """Acquire the instance lease for a context's session key."""
    return play_lease_manager.acquire(
        INSTANCE, ctx.session_id, "AgentA", project_root=project_root)


def _mirror_path(project_root):
    return os.path.join(
        str(project_root), "Library", "MCPForUnity", "RunState", "play_lease.json")


def _read_mirror(project_root):
    with open(_mirror_path(project_root), "r", encoding="utf-8") as f:
        return json.load(f)


class _MiddlewareContext:
    def __init__(self, ctx, tool_name, arguments):
        self.fastmcp_context = ctx
        self.message = types.SimpleNamespace(name=tool_name, arguments=arguments)


async def _middleware_call(middleware, ctx, tool_name, arguments, tool_result):
    """Run one tool call through the middleware with a canned tool result."""
    context = _MiddlewareContext(ctx, tool_name, dict(arguments))

    async def call_next(_context):
        return tool_result

    return await middleware.on_call_tool(context, call_next)


# ----------------------------------------------------------------------
# Implicit acquisition (MCPC-012)
# ----------------------------------------------------------------------
class TestImplicitAcquire:
    @pytest.mark.asyncio
    async def test_successful_play_call_acquires_lease(self, monkeypatch, tmp_path):
        """The session whose gated call enters play mode owns the lease."""
        await _register_instance(tmp_path)
        _inject_state(monkeypatch, FakeEditorState(IDLE_STATE))

        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        ctx_a = DummyContext()
        await middleware.set_active_instance(ctx_a, INSTANCE)

        result = await _middleware_call(
            middleware, ctx_a, "manage_editor", {"action": "play"},
            {"success": True, "message": "Entered play mode."},
        )

        assert result == {"success": True, "message": "Entered play mode."}
        lease = play_lease_manager.get_active_lease(INSTANCE)
        assert lease is not None
        assert lease.owner_key == ctx_a.session_id
        identity = await middleware.get_session_identity(ctx_a)
        assert lease.owner_display == identity.display_name
        # MCPC-018: lease state mirrored into the project's RunState dir.
        mirror = _read_mirror(tmp_path)
        assert mirror["active"] is True
        assert mirror["owner"] == identity.display_name

    @pytest.mark.asyncio
    async def test_failed_play_call_acquires_nothing(self, monkeypatch):
        _inject_state(monkeypatch, FakeEditorState(IDLE_STATE))
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        ctx_a = DummyContext()
        await middleware.set_active_instance(ctx_a, INSTANCE)

        await _middleware_call(
            middleware, ctx_a, "manage_editor", {"action": "play"},
            {"success": False, "error": "Play mode is not allowed right now."},
        )

        assert play_lease_manager.get_active_lease(INSTANCE) is None

    @pytest.mark.asyncio
    async def test_play_intent_attributes_lease_across_reload(self, monkeypatch):
        """A play call whose result is lost to the play-enter domain reload
        still gets the lease via its recorded intent once the editor-state
        stream shows play active (no misattribution to 'user')."""
        _inject_state(monkeypatch, FakeEditorState(IDLE_STATE))
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        ctx_a = DummyContext()
        await middleware.set_active_instance(ctx_a, INSTANCE)

        # The retry-shaped failure the transport returns when the WebSocket
        # drops mid-call: the intent must survive it.
        await _middleware_call(
            middleware, ctx_a, "manage_editor", {"action": "play"},
            {"success": False, "error": "plugin disconnected", "hint": "retry"},
        )
        assert play_lease_manager.get_active_lease(INSTANCE) is None

        # After the reload, a fresh snapshot shows play active.
        editor_state_cache.reset(ttl_s=0.0)
        _inject_state(monkeypatch, FakeEditorState(PLAYING_STATE))
        await editor_state_cache.get(DummyContext(), INSTANCE)

        lease = play_lease_manager.get_active_lease(INSTANCE)
        assert lease is not None
        assert lease.owner_key == ctx_a.session_id

    @pytest.mark.asyncio
    async def test_owner_stop_releases_lease(self, monkeypatch, tmp_path):
        await _register_instance(tmp_path)
        _inject_state(monkeypatch, FakeEditorState(PLAYING_STATE))
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        ctx_a = DummyContext()
        await middleware.set_active_instance(ctx_a, INSTANCE)
        _acquire_for(ctx_a, project_root=str(tmp_path))

        result = await _middleware_call(
            middleware, ctx_a, "manage_editor", {"action": "stop"},
            {"success": True, "message": "Exited play mode."},
        )

        assert result["success"] is True
        assert play_lease_manager.get_active_lease(INSTANCE) is None
        mirror = _read_mirror(tmp_path)
        assert mirror["active"] is False
        assert mirror["released_reason"] == "owner_stopped"


# ----------------------------------------------------------------------
# Non-owner enforcement (MCPC-013/014/015)
# ----------------------------------------------------------------------
class TestNonOwnerEnforcement:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["stop", "pause", "play"])
    async def test_non_owner_editor_control_busy_names_owner(
        self, monkeypatch, action,
    ):
        await _register_instance()
        _inject_state(monkeypatch, FakeEditorState(PLAYING_STATE))
        probe_calls = _fake_probe(monkeypatch, playing=True)

        ctx_a = await _pinned_context()
        ctx_b = await _pinned_context()
        _acquire_for(ctx_a)

        result = await gate_tool_call(ctx_b, "manage_editor", {"action": action})

        assert result is not None
        assert result["success"] is False
        assert result["hint"] == "retry"
        assert result["data"]["reason"] == "play_lease"
        assert result["data"]["blocked_by"] == "AgentA"
        assert "AgentA" in result["error"]
        assert probe_calls  # validate-on-block probed before refusing
        # The owner's lease is intact.
        lease = play_lease_manager.get_active_lease(INSTANCE)
        assert lease is not None and lease.owner_key == ctx_a.session_id

    @pytest.mark.asyncio
    async def test_non_owner_compile_class_call_busy_names_owner(self, monkeypatch):
        await _register_instance()
        _inject_state(monkeypatch, FakeEditorState(PLAYING_STATE))
        _fake_probe(monkeypatch, playing=True)
        _no_fence(monkeypatch)

        ctx_a = await _pinned_context()
        ctx_b = await _pinned_context()
        _acquire_for(ctx_a)

        result = await gate_for_class(ctx_b, CLASS_EXCLUSIVE, "refresh_unity", INSTANCE)

        assert result is not None
        assert result["data"]["reason"] == "play_lease"
        assert result["data"]["blocked_by"] == "AgentA"

    @pytest.mark.asyncio
    async def test_non_owner_play_scoped_driver_blocked_via_bridge_class(
        self, monkeypatch,
    ):
        """Bridge-declared play-scoped tools (drivers) are owner-only."""
        registry = await _register_instance()
        await registry.register_tools_for_session("guid-1", [
            ToolDefinitionModel(name="synthetic_input", concurrency_class="play-scoped"),
        ])
        _inject_state(monkeypatch, FakeEditorState(PLAYING_STATE))
        _fake_probe(monkeypatch, playing=True)

        ctx_a = await _pinned_context()
        ctx_b = await _pinned_context()
        _acquire_for(ctx_a)

        result = await gate_tool_call(ctx_b, "synthetic_input", {"action": "key_press"})

        assert result is not None
        assert result["data"]["reason"] == "play_lease"
        assert result["data"]["blocked_by"] == "AgentA"

    @pytest.mark.asyncio
    async def test_non_owner_camera_control_blocked_screenshot_passes(
        self, monkeypatch,
    ):
        """MCPC-015: camera control is play-scoped; screenshots stay reads."""
        await _register_instance()
        _inject_state(monkeypatch, FakeEditorState(PLAYING_STATE))
        _fake_probe(monkeypatch, playing=True)

        ctx_a = await _pinned_context()
        ctx_b = await _pinned_context()
        _acquire_for(ctx_a)

        blocked = await gate_tool_call(ctx_b, "manage_camera", {"action": "move"})
        passed = await gate_tool_call(ctx_b, "manage_camera", {"action": "screenshot"})

        assert blocked is not None
        assert blocked["data"]["reason"] == "play_lease"
        assert passed is None

    @pytest.mark.asyncio
    async def test_reads_always_pass_during_lease(self, monkeypatch):
        await _register_instance()
        fake = FakeEditorState(PLAYING_STATE)
        _inject_state(monkeypatch, fake)
        probe_calls = _fake_probe(monkeypatch, playing=True)

        ctx_a = await _pinned_context()
        ctx_b = await _pinned_context()
        _acquire_for(ctx_a)

        result = await gate_tool_call(ctx_b, "read_console", {"action": "get"})

        assert result is None
        assert fake.fetch_count == 0  # reads never touch the gate machinery
        assert not probe_calls

    @pytest.mark.asyncio
    async def test_non_owner_plain_mutation_passes(self, monkeypatch):
        """Observing/mutating outside play scope is permitted by design."""
        await _register_instance()
        _inject_state(monkeypatch, FakeEditorState(PLAYING_STATE))
        probe_calls = _fake_probe(monkeypatch, playing=True)

        ctx_a = await _pinned_context()
        ctx_b = await _pinned_context()
        _acquire_for(ctx_a)

        result = await gate_tool_call(ctx_b, "manage_editor", {"action": "add_tag"})

        assert result is None  # mutate class: not owner-only
        assert not probe_calls

    @pytest.mark.asyncio
    async def test_owner_calls_pass_and_renew(self, monkeypatch):
        await _register_instance()
        _inject_state(monkeypatch, FakeEditorState(IDLE_STATE))

        ctx_a = await _pinned_context()
        lease = _acquire_for(ctx_a)
        stale = time.monotonic() - 60.0
        monkeypatch.setattr(lease, "last_activity", stale)

        result = await gate_tool_call(ctx_a, "manage_editor", {"action": "pause"})

        assert result is None
        assert lease.last_activity > stale  # inactivity TTL renewed


# ----------------------------------------------------------------------
# Human play (MCPC-016)
# ----------------------------------------------------------------------
class TestHumanPlay:
    @pytest.mark.asyncio
    async def test_play_with_no_mcp_cause_is_leased_to_user(
        self, monkeypatch, tmp_path,
    ):
        await _register_instance(tmp_path)
        _inject_state(monkeypatch, FakeEditorState(PLAYING_STATE))

        # The shared cache observes a play transition with no gated call.
        await editor_state_cache.get(DummyContext(), INSTANCE)

        lease = play_lease_manager.get_active_lease(INSTANCE)
        assert lease is not None
        assert lease.owner_key is None
        assert lease.owner_display == "user"
        mirror = _read_mirror(tmp_path)
        assert mirror["owner"] == "user"

        # Identical protections: every MCP session is a non-owner.
        _fake_probe(monkeypatch, playing=True)
        ctx_b = await _pinned_context()
        result = await gate_tool_call(ctx_b, "manage_editor", {"action": "stop"})

        assert result is not None
        assert result["data"]["reason"] == "play_lease"
        assert result["data"]["blocked_by"] == "user"

    @pytest.mark.asyncio
    async def test_natural_play_exit_clears_lease(self, monkeypatch):
        await _register_instance()
        ctx_a = await _pinned_context()
        _acquire_for(ctx_a)

        editor_state_cache.reset(ttl_s=0.0)
        _inject_state(monkeypatch, FakeEditorState(NOT_PLAYING_STATE))
        await editor_state_cache.get(DummyContext(), INSTANCE)

        assert play_lease_manager.get_active_lease(INSTANCE) is None


# ----------------------------------------------------------------------
# Liveness (MCPC-017)
# ----------------------------------------------------------------------
class TestLiveness:
    @pytest.mark.asyncio
    async def test_ttl_expiry_frees_lease(self, monkeypatch):
        await _register_instance()
        _inject_state(monkeypatch, FakeEditorState(IDLE_STATE))
        probe_calls = _fake_probe(monkeypatch, playing=True)

        ctx_a = await _pinned_context()
        ctx_b = await _pinned_context()
        lease = _acquire_for(ctx_a)
        monkeypatch.setattr(lease, "last_activity", time.monotonic() - 9999.0)

        result = await gate_tool_call(ctx_b, "manage_editor", {"action": "stop"})

        assert result is None  # expired lease never blocks
        assert play_lease_manager.get_active_lease(INSTANCE) is None
        assert not probe_calls  # freed before any probe was needed

    @pytest.mark.asyncio
    async def test_validate_on_block_self_clears_stale_lease(self, monkeypatch):
        """A refused call probes the editor; play already ended, so the
        lease self-clears and the call passes."""
        await _register_instance()
        # The cached snapshot is stale (still says playing) — the fresh
        # fast-fail probe is what reveals play has ended.
        _inject_state(monkeypatch, FakeEditorState(PLAYING_STATE))
        probe_calls = _fake_probe(monkeypatch, playing=False)

        ctx_a = await _pinned_context()
        ctx_b = await _pinned_context()
        _acquire_for(ctx_a)

        result = await gate_tool_call(ctx_b, "manage_editor", {"action": "stop"})

        assert result is None
        assert probe_calls
        assert play_lease_manager.get_active_lease(INSTANCE) is None

    @pytest.mark.asyncio
    async def test_inconclusive_probe_keeps_block(self, monkeypatch):
        """A failed probe never silently steals the owner's session."""
        await _register_instance()
        _inject_state(monkeypatch, FakeEditorState(PLAYING_STATE))

        async def failing_send(session_id, command_type, params):
            return {"success": False, "error": "Unity did not respond", "hint": "retry"}

        monkeypatch.setattr(PluginHub, "send_command", failing_send)

        ctx_a = await _pinned_context()
        ctx_b = await _pinned_context()
        _acquire_for(ctx_a)

        result = await gate_tool_call(ctx_b, "manage_editor", {"action": "stop"})

        assert result is not None
        assert result["data"]["reason"] == "play_lease"
        assert play_lease_manager.get_active_lease(INSTANCE) is not None


# ----------------------------------------------------------------------
# Lifecycle: reconnect survival and fail-open (MCPC-018)
# ----------------------------------------------------------------------
class TestLifecycle:
    @pytest.mark.asyncio
    async def test_lease_survives_domain_reload_reconnect(self):
        """The lease is keyed to project identity, not the WebSocket session:
        an unregister followed by a same-instance re-registration (domain
        reload) keeps the lease alive."""
        registry = await _register_instance(session_id="guid-1")
        ctx_a = await _pinned_context()
        lease = _acquire_for(ctx_a)

        await registry.unregister("guid-1")
        assert lease.disconnected_at is not None
        # Still inside the reconnect grace window: the lease holds.
        assert play_lease_manager.get_active_lease(INSTANCE) is lease

        await registry.register("guid-2", "Game", "hash-x", "6000.0")
        assert lease.disconnected_at is None
        assert play_lease_manager.get_active_lease(INSTANCE) is lease

    @pytest.mark.asyncio
    async def test_lease_survives_same_instance_eviction(self):
        """A reconnect race (new registration evicting the old session)
        never clears the lease."""
        registry = await _register_instance(session_id="guid-1")
        ctx_a = await _pinned_context()
        lease = _acquire_for(ctx_a)

        _, evicted = await registry.register("guid-2", "Game", "hash-x", "6000.0")
        assert evicted == "guid-1"
        assert play_lease_manager.get_active_lease(INSTANCE) is lease
        assert lease.disconnected_at is None

    @pytest.mark.asyncio
    async def test_disconnect_fails_open_after_grace(self, monkeypatch):
        """An instance that never reconnects frees its lease at the grace
        boundary — fail open, never a deadlock."""
        registry = await _register_instance(session_id="guid-1")
        _inject_state(monkeypatch, FakeEditorState(IDLE_STATE))
        ctx_a = await _pinned_context()
        ctx_b = await _pinned_context()
        lease = _acquire_for(ctx_a)

        await registry.unregister("guid-1")
        monkeypatch.setattr(
            lease, "disconnected_at",
            time.monotonic() - (play_lease.DISCONNECT_GRACE_SECONDS + 1.0))

        result = await gate_tool_call(ctx_b, "manage_editor", {"action": "stop"})

        assert result is None
        assert play_lease_manager.get_active_lease(INSTANCE) is None


# ----------------------------------------------------------------------
# RunState mirror (MCPC-018)
# ----------------------------------------------------------------------
class TestRunStateMirror:
    def test_acquire_and_release_write_mirror(self, tmp_path):
        play_lease_manager.acquire(
            INSTANCE, "session-a", "AgentA", project_root=str(tmp_path))

        mirror = _read_mirror(tmp_path)
        assert mirror["active"] is True
        assert mirror["owner"] == "AgentA"
        assert mirror["owner_session_key"] == "session-a"
        assert mirror["instance"] == "hash-x"
        assert mirror["schema"] == "unity-mcp/play_lease@1"

        play_lease_manager.release(INSTANCE, reason="play_exited")
        mirror = _read_mirror(tmp_path)
        assert mirror["active"] is False
        assert mirror["released_reason"] == "play_exited"

    def test_unwritable_project_root_fails_open(self, tmp_path):
        # A file where the project directory should be: makedirs will fail.
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("occupied")

        lease = play_lease_manager.acquire(
            INSTANCE, "session-a", "AgentA", project_root=str(blocker))

        assert lease is not None
        assert play_lease_manager.get_active_lease(INSTANCE) is lease

    def test_missing_project_root_fails_open(self):
        lease = play_lease_manager.acquire(
            INSTANCE, "session-a", "AgentA", project_root=None)

        assert lease is not None
        assert play_lease_manager.get_active_lease(INSTANCE) is lease
