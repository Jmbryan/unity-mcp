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
    from services.state.test_job_lease import test_job_lease_manager

    play_lease_manager.reset()
    test_job_lease_manager.reset()
    editor_state_cache.reset(ttl_s=0.01)
    monkeypatch.setattr(config, "reload_retry_ms", 50, raising=False)
    set_unity_instance_middleware(UnityInstanceMiddleware())
    monkeypatch.setattr(operation_gate, "_registry_last_refresh", 0.0)
    yield
    play_lease_manager.reset()
    test_job_lease_manager.reset()
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
# Wrapper side-door attribution (MCPC-009)
# ----------------------------------------------------------------------
CHANGING_STATE = {
    "compilation": {},
    "editor": {"play_mode": {"is_playing": False, "is_changing": True}},
    "tests": {},
}


class TestWrapperPlayAttribution:
    """A play-enter dispatched through a wrapper tool (batch_execute /
    execute_custom_tool) records a play intent for the CALLING session, so the
    lease the play transition produces is attributed to the agent — never the
    fall-through "user" owner (MCPC-009)."""

    @pytest.mark.asyncio
    async def test_post_hoc_attributes_settled_play_edge_to_caller(self, monkeypatch):
        """A play edge that has already SETTLED (is_playing True, is_changing
        False) when the synchronous wrapper call returns still attributes: the
        intent is recorded, then the subsequent editor-state observation
        consumes it instead of defaulting to 'user'."""
        from services.state.operation_gate import (
            record_exclusive_edge_after_arbitrary_code,
        )

        await _register_instance()
        editor_state_cache.reset(ttl_s=0.0)
        _inject_state(monkeypatch, FakeEditorState(PLAYING_STATE))

        ctx_a = await _pinned_context()
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        await middleware.set_active_instance(ctx_a, INSTANCE)

        # The post-hoc edge recording (as called by the wrappers) records the
        # intent BEFORE the forced 0-age fetch that drives observe_editor_state.
        await record_exclusive_edge_after_arbitrary_code(ctx_a, INSTANCE)

        lease = play_lease_manager.get_active_lease(INSTANCE)
        assert lease is not None
        assert lease.owner_key == ctx_a.session_id  # the caller, not "user"
        assert lease.owner_display != "user"

    @pytest.mark.asyncio
    async def test_non_play_edge_does_not_leave_stale_intent(self, monkeypatch):
        """An execute path that turns out NOT to be a play edge must not leave a
        speculative intent behind that would steal a later 'user' play-enter."""
        from services.state.operation_gate import (
            record_exclusive_edge_after_arbitrary_code,
        )

        await _register_instance()
        editor_state_cache.reset(ttl_s=0.0)
        _inject_state(monkeypatch, FakeEditorState(IDLE_STATE))

        ctx_a = await _pinned_context()
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        await middleware.set_active_instance(ctx_a, INSTANCE)

        await record_exclusive_edge_after_arbitrary_code(ctx_a, INSTANCE)
        assert play_lease_manager.get_active_lease(INSTANCE) is None

        # A later human play-enter (no MCP cause) must still resolve to "user".
        editor_state_cache.reset(ttl_s=0.0)
        _inject_state(monkeypatch, FakeEditorState(PLAYING_STATE))
        await editor_state_cache.get(DummyContext(), INSTANCE)

        lease = play_lease_manager.get_active_lease(INSTANCE)
        assert lease is not None
        assert lease.owner_key is None
        assert lease.owner_display == "user"

    @pytest.mark.asyncio
    async def test_batch_execute_inner_play_attributes_to_caller(self, monkeypatch):
        """End-to-end through batch_execute: an inner manage_editor action=play
        leases to the calling session, not 'user'."""
        import services.tools.batch_execute as batch_mod

        await _register_instance()
        editor_state_cache.reset(ttl_s=0.0)
        _inject_state(monkeypatch, FakeEditorState(PLAYING_STATE))

        ctx_a = await _pinned_context()
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        await middleware.set_active_instance(ctx_a, INSTANCE)

        async def fake_send(send_fn, unity_instance, command, payload):
            return {"success": True, "message": "Entered play mode."}

        monkeypatch.setattr(batch_mod, "send_with_unity_instance", fake_send)
        # No editor-state limit lookup fan-out.
        monkeypatch.setattr(batch_mod, "_cached_max_commands", 25, raising=False)

        result = await batch_mod.batch_execute(
            ctx_a,
            commands=[{"tool": "manage_editor", "params": {"action": "play"}}],
        )

        assert result["success"] is True
        lease = play_lease_manager.get_active_lease(INSTANCE)
        assert lease is not None
        assert lease.owner_key == ctx_a.session_id
        assert lease.owner_display != "user"

    @pytest.mark.asyncio
    async def test_refused_batch_play_enter_leaves_no_stale_intent(self, monkeypatch):
        """A gate-refused batch never dispatches, so its speculative intent
        must not survive to steal a later, unrelated play-enter."""
        import services.tools.batch_execute as batch_mod

        monkeypatch.setenv("UNITY_MCP_GATE_PARK_MAX_WAIT_S", "0.2")
        await _register_instance()
        editor_state_cache.reset(ttl_s=0.01)
        _inject_state(monkeypatch, FakeEditorState({
            "compilation": {"is_compiling": True}, "editor": {}, "tests": {},
        }))

        ctx_a = await _pinned_context()
        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        await middleware.set_active_instance(ctx_a, INSTANCE)
        monkeypatch.setattr(batch_mod, "_cached_max_commands", 25, raising=False)

        result = await batch_mod.batch_execute(
            ctx_a,
            commands=[{"tool": "manage_editor", "params": {"action": "play"}}],
        )

        assert result["success"] is False
        assert play_lease_manager._intents == {}

        # A later human play-enter (no MCP cause) resolves to "user".
        editor_state_cache.reset(ttl_s=0.0)
        _inject_state(monkeypatch, FakeEditorState(PLAYING_STATE))
        await editor_state_cache.get(DummyContext(), INSTANCE)
        lease = play_lease_manager.get_active_lease(INSTANCE)
        assert lease is not None
        assert lease.owner_display == "user"


class TestPluginToolPlayAttribution:
    """A bridge-registered plugin tool declaring a play-owning class dispatches
    through the per-tool middleware, not the custom-tool wrapper. Its play
    transition is invisible to argument inspection, so the dispatch records an
    intent for the CALLING session — otherwise the lease falls through to
    "user" and the driver's own play-scoped calls (its cleanup stop included)
    are refused as non-owner."""

    @staticmethod
    async def _register_driver(name="test_driver", concurrency_class="play-scoped"):
        registry = await _register_instance()
        await registry.register_tools_for_session("guid-1", [
            ToolDefinitionModel(name=name, concurrency_class=concurrency_class),
        ])
        return registry

    @pytest.mark.asyncio
    async def test_dispatched_play_scoped_plugin_tool_leases_to_caller(
        self, monkeypatch,
    ):
        """The editor is idle when the driver is gated and playing when it
        returns: the settled edge attributes to the caller, not 'user'."""
        await self._register_driver()
        editor_state_cache.reset(ttl_s=0.0)
        _inject_state(monkeypatch, FakeEditorState(IDLE_STATE, PLAYING_STATE))

        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        ctx_a = DummyContext()
        await middleware.set_active_instance(ctx_a, INSTANCE)

        result = await _middleware_call(
            middleware, ctx_a, "test_driver", {"action": "run_script"},
            {"success": True, "message": "Test script completed."},
        )

        assert result["success"] is True
        lease = play_lease_manager.get_active_lease(INSTANCE)
        assert lease is not None
        assert lease.owner_key == ctx_a.session_id
        assert lease.owner_display != "user"

    @pytest.mark.asyncio
    async def test_owner_can_stop_the_play_session_it_started(self, monkeypatch):
        """The self-deadlock this closes: after the driver's own play session
        is leased to it, the session's cleanup stop is not refused."""
        await self._register_driver()
        editor_state_cache.reset(ttl_s=0.0)
        _inject_state(monkeypatch, FakeEditorState(IDLE_STATE, PLAYING_STATE))

        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        ctx_a = DummyContext()
        await middleware.set_active_instance(ctx_a, INSTANCE)

        await _middleware_call(
            middleware, ctx_a, "test_driver", {"action": "run_script"},
            {"success": True, "message": "Test script completed."},
        )

        assert await gate_tool_call(ctx_a, "manage_editor", {"action": "stop"}) is None

    @pytest.mark.asyncio
    async def test_refused_plugin_call_leaves_no_stale_intent(self, monkeypatch):
        """A gate-refused plugin call never dispatches, so its speculative
        intent must not survive to steal a later, unrelated play-enter."""
        from fastmcp.exceptions import ToolError

        monkeypatch.setenv("UNITY_MCP_GATE_PARK_MAX_WAIT_S", "0.2")
        await self._register_driver()
        editor_state_cache.reset(ttl_s=0.01)
        _inject_state(monkeypatch, FakeEditorState({
            "compilation": {"is_compiling": True}, "editor": {}, "tests": {},
        }))

        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        ctx_a = DummyContext()
        await middleware.set_active_instance(ctx_a, INSTANCE)

        with pytest.raises(ToolError):
            await _middleware_call(
                middleware, ctx_a, "test_driver", {"action": "run_script"},
                {"success": True, "message": "Test script completed."},
            )

        assert play_lease_manager._intents == {}

        # A later human play-enter (no MCP cause) still resolves to "user".
        editor_state_cache.reset(ttl_s=0.0)
        _inject_state(monkeypatch, FakeEditorState(PLAYING_STATE))
        await editor_state_cache.get(DummyContext(), INSTANCE)
        lease = play_lease_manager.get_active_lease(INSTANCE)
        assert lease is not None
        assert lease.owner_display == "user"

    @pytest.mark.asyncio
    async def test_plain_plugin_tool_records_no_intent(self, monkeypatch):
        """Only play-owning classes attribute: a mutate-class plugin tool
        leaves a concurrent human play-enter attributed to 'user'."""
        await self._register_driver(name="gameplay_query", concurrency_class="mutate")
        editor_state_cache.reset(ttl_s=0.0)
        _inject_state(monkeypatch, FakeEditorState(IDLE_STATE, PLAYING_STATE))

        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        ctx_a = DummyContext()
        await middleware.set_active_instance(ctx_a, INSTANCE)

        await _middleware_call(
            middleware, ctx_a, "gameplay_query", {"action": "tags"},
            {"success": True, "message": "ok"},
        )
        await editor_state_cache.get(DummyContext(), INSTANCE)

        lease = play_lease_manager.get_active_lease(INSTANCE)
        assert lease is not None
        assert lease.owner_key is None
        assert lease.owner_display == "user"


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


# ----------------------------------------------------------------------
# Force-release HTTP control (human dashboard override)
# ----------------------------------------------------------------------
class _FakeRequest:
    """Minimal Starlette-Request stand-in with a JSON body (or a raising one)."""

    def __init__(self, payload=None, raise_on_json=False):
        self._payload = payload
        self._raise = raise_on_json

    async def json(self):
        if self._raise:
            raise ValueError("invalid JSON body")
        return self._payload


class TestForceRelease:
    """POST /lease/release force-clears a lease regardless of owner; always
    200, fails open, never raises."""

    @pytest.mark.asyncio
    async def test_force_release_clears_active_lease(self, tmp_path):
        play_lease_manager.acquire(
            INSTANCE, "session-a", "AgentA", project_root=str(tmp_path))
        assert play_lease_manager.get_active_lease(INSTANCE) is not None

        body, status = await play_lease.handle_lease_release_post(
            _FakeRequest({"instance": "hash-x"}))

        assert status == 200
        assert body == {"released": True, "instance": "hash-x"}
        assert play_lease_manager.get_active_lease(INSTANCE) is None
        # MCPC-018: mirror flipped to inactive with the override reason.
        mirror = _read_mirror(tmp_path)
        assert mirror["active"] is False
        assert mirror["released_reason"] == "force_release"

    @pytest.mark.asyncio
    async def test_force_release_ignores_owner(self):
        """A non-owner override still clears the lease (deliberate human act)."""
        play_lease_manager.acquire(INSTANCE, "session-owner", "AgentA")
        assert play_lease_manager.get_active_lease(INSTANCE) is not None

        body, status = await play_lease.handle_lease_release_post(
            _FakeRequest({"instance": "hash-x"}))

        assert status == 200
        assert body["released"] is True
        assert play_lease_manager.get_active_lease(INSTANCE) is None

    @pytest.mark.asyncio
    async def test_project_hash_alias_key_accepted(self):
        play_lease_manager.acquire(INSTANCE, "session-a", "AgentA")

        body, status = await play_lease.handle_lease_release_post(
            _FakeRequest({"project_hash": "hash-x"}))

        assert status == 200
        assert body == {"released": True, "instance": "hash-x"}
        assert play_lease_manager.get_active_lease(INSTANCE) is None

    @pytest.mark.asyncio
    async def test_force_release_absent_lease_is_noop(self):
        assert play_lease_manager.get_active_lease(INSTANCE) is None

        body, status = await play_lease.handle_lease_release_post(
            _FakeRequest({"instance": "hash-x"}))

        assert status == 200
        assert body == {"released": False, "instance": "hash-x"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload",
        [None, "not a dict", 123, {}, {"instance": ""}, {"instance": None}],
    )
    async def test_missing_or_malformed_body_tolerated(self, payload):
        play_lease_manager.acquire(INSTANCE, "session-a", "AgentA")

        body, status = await play_lease.handle_lease_release_post(
            _FakeRequest(payload))

        assert status == 200
        assert body["released"] is False
        # Nothing was cleared: the lease survives a bodyless/garbage call.
        assert play_lease_manager.get_active_lease(INSTANCE) is not None

    @pytest.mark.asyncio
    async def test_unreadable_json_body_tolerated(self):
        play_lease_manager.acquire(INSTANCE, "session-a", "AgentA")

        body, status = await play_lease.handle_lease_release_post(
            _FakeRequest(raise_on_json=True))

        assert status == 200
        assert body["released"] is False
        assert play_lease_manager.get_active_lease(INSTANCE) is not None

    @pytest.mark.asyncio
    async def test_fail_open_on_internal_error(self, monkeypatch):
        """An internal manager failure still yields released:False, 200."""
        def _boom(*_a, **_k):
            raise RuntimeError("manager exploded")

        monkeypatch.setattr(play_lease_manager, "force_release", _boom)

        body, status = await play_lease.handle_lease_release_post(
            _FakeRequest({"instance": "hash-x"}))

        assert status == 200
        assert body == {"released": False, "instance": "hash-x"}


# ----------------------------------------------------------------------
# TTL expiry during play never flips the owner (mis-attribution fix 2)
# ----------------------------------------------------------------------
EXIT_TRANSITION_STATE = {
    "compilation": {},
    "editor": {"play_mode": {"is_playing": True, "is_paused": False, "is_changing": True}},
    "tests": {},
}
TESTS_PLAYING_STATE = {
    "compilation": {},
    "editor": {"play_mode": {"is_playing": True, "is_paused": False, "is_changing": False}},
    "tests": {"is_running": True},
}


class TestTtlExpiryDuringPlay:
    @pytest.mark.asyncio
    async def test_expired_lease_renewed_in_place_while_playing(self, monkeypatch):
        """A snapshot showing play still active renews a TTL-lapsed lease for
        its owner instead of releasing it and re-acquiring as 'user'."""
        _inject_state(monkeypatch, FakeEditorState(PLAYING_STATE))
        ctx_a = await _pinned_context()
        lease = _acquire_for(ctx_a)
        stale = time.monotonic() - 9999.0
        monkeypatch.setattr(lease, "last_activity", stale)

        editor_state_cache.reset(ttl_s=0.0)
        await editor_state_cache.get(DummyContext(), INSTANCE)

        renewed = play_lease_manager.get_active_lease(INSTANCE)
        assert renewed is lease
        assert renewed.owner_key == ctx_a.session_id
        assert renewed.owner_display == "AgentA"
        assert renewed.last_activity > stale

    @pytest.mark.asyncio
    async def test_owner_kept_across_expiry_then_non_owner_still_refused(
        self, monkeypatch,
    ):
        """After the in-place renewal a non-owner's play-scoped call is refused
        naming the real owner — never 'user'."""
        await _register_instance()
        _inject_state(monkeypatch, FakeEditorState(PLAYING_STATE))
        _fake_probe(monkeypatch, playing=True)

        ctx_a = await _pinned_context()
        ctx_b = await _pinned_context()
        lease = _acquire_for(ctx_a)
        monkeypatch.setattr(lease, "last_activity", time.monotonic() - 9999.0)

        result = await gate_tool_call(ctx_b, "manage_editor", {"action": "stop"})

        assert result is not None
        assert result["data"]["reason"] == "play_lease"
        assert result["data"]["blocked_by"] == "AgentA"

    @pytest.mark.asyncio
    async def test_expired_lease_still_frees_when_not_playing(self, monkeypatch):
        """Liveness unchanged outside play: an abandoned lease with no play
        session behind it expires as before."""
        _inject_state(monkeypatch, FakeEditorState(NOT_PLAYING_STATE))
        ctx_a = await _pinned_context()
        lease = _acquire_for(ctx_a)
        monkeypatch.setattr(lease, "last_activity", time.monotonic() - 9999.0)

        assert play_lease_manager.get_active_lease(INSTANCE) is None


# ----------------------------------------------------------------------
# Correcting an inferred-"user" lease (mis-attribution fix 3)
# ----------------------------------------------------------------------
class TestUserLeaseCorrection:
    @pytest.mark.asyncio
    async def test_acquire_with_real_owner_corrects_user_lease(self):
        """A proven MCP cause (successful play call) claims an inferred-'user'
        lease in place instead of being ignored."""
        play_lease_manager.acquire(INSTANCE, None, "user")

        lease = play_lease_manager.acquire(INSTANCE, "sess-a", "AgentA")

        assert lease.owner_key == "sess-a"
        assert lease.owner_display == "AgentA"
        assert play_lease_manager.get_active_lease(INSTANCE) is lease

    @pytest.mark.asyncio
    async def test_acquire_never_steals_between_real_owners(self):
        play_lease_manager.acquire(INSTANCE, "sess-a", "AgentA")

        lease = play_lease_manager.acquire(INSTANCE, "sess-b", "AgentB")

        assert lease.owner_key == "sess-a"
        assert lease.owner_display == "AgentA"

    @pytest.mark.asyncio
    async def test_late_intent_reattributes_user_lease_on_observation(self):
        """A pending intent from a real session corrects an inferred-'user'
        lease on the next play-active observation."""
        play_lease_manager.acquire(INSTANCE, None, "user")
        play_lease_manager.record_play_intent(INSTANCE, "sess-a", "AgentA")

        await play_lease_manager.observe_editor_state(INSTANCE, PLAYING_STATE)

        lease = play_lease_manager.get_active_lease(INSTANCE)
        assert lease is not None
        assert lease.owner_key == "sess-a"
        assert lease.owner_display == "AgentA"

    @pytest.mark.asyncio
    async def test_observation_never_reattributes_a_real_owner(self):
        """A stray intent must not steal a lease that already has a real owner."""
        play_lease_manager.acquire(INSTANCE, "sess-a", "AgentA")
        play_lease_manager.record_play_intent(INSTANCE, "sess-b", "AgentB")

        await play_lease_manager.observe_editor_state(INSTANCE, PLAYING_STATE)

        lease = play_lease_manager.get_active_lease(INSTANCE)
        assert lease.owner_key == "sess-a"


# ----------------------------------------------------------------------
# Force-release suppression (mis-attribution fix 3, force-release leg)
# ----------------------------------------------------------------------
class TestForceReleaseSuppression:
    @pytest.mark.asyncio
    async def test_force_release_not_undone_by_next_observation(self):
        """After a human force-release, the still-running play session must
        not immediately re-lease to 'user'."""
        play_lease_manager.acquire(INSTANCE, "sess-a", "AgentA")
        assert play_lease_manager.force_release("hash-x") is True

        await play_lease_manager.observe_editor_state(INSTANCE, PLAYING_STATE)

        assert play_lease_manager.get_active_lease(INSTANCE) is None

    @pytest.mark.asyncio
    async def test_suppression_ends_when_play_exits(self):
        """Play exit closes the override; the NEXT play session attributes
        normally (a human enter is 'user' again)."""
        play_lease_manager.acquire(INSTANCE, "sess-a", "AgentA")
        play_lease_manager.force_release("hash-x")

        await play_lease_manager.observe_editor_state(INSTANCE, NOT_PLAYING_STATE)
        await play_lease_manager.observe_editor_state(INSTANCE, PLAYING_STATE)

        lease = play_lease_manager.get_active_lease(INSTANCE)
        assert lease is not None
        assert lease.owner_key is None
        assert lease.owner_display == "user"

    @pytest.mark.asyncio
    async def test_new_intent_overrides_suppression(self):
        """A real MCP cause arriving after the force-release attributes the
        (still running) play session to that session."""
        play_lease_manager.acquire(INSTANCE, "sess-a", "AgentA")
        play_lease_manager.force_release("hash-x")
        play_lease_manager.record_play_intent(INSTANCE, "sess-b", "AgentB")

        await play_lease_manager.observe_editor_state(INSTANCE, PLAYING_STATE)

        lease = play_lease_manager.get_active_lease(INSTANCE)
        assert lease is not None
        assert lease.owner_key == "sess-b"

    @pytest.mark.asyncio
    async def test_explicit_acquire_overrides_suppression(self):
        """A successful play call's acquire supersedes the override."""
        play_lease_manager.acquire(INSTANCE, "sess-a", "AgentA")
        play_lease_manager.force_release("hash-x")

        lease = play_lease_manager.acquire(INSTANCE, "sess-b", "AgentB")
        assert lease.owner_key == "sess-b"
        # The override is spent: later observations renew, not suppress.
        await play_lease_manager.observe_editor_state(INSTANCE, PLAYING_STATE)
        assert play_lease_manager.get_active_lease(INSTANCE) is lease


# ----------------------------------------------------------------------
# Transition windows never mint a lease (mis-attribution fix 4)
# ----------------------------------------------------------------------
class TestTransitionWindow:
    @pytest.mark.asyncio
    async def test_exit_transition_snapshot_does_not_acquire_user_lease(self):
        """The play-exit window (is_playing still true, is_changing true)
        right after the owner's stop must not manufacture a 'user' lease."""
        await play_lease_manager.observe_editor_state(
            INSTANCE, EXIT_TRANSITION_STATE)

        assert play_lease_manager.get_active_lease(INSTANCE) is None

    @pytest.mark.asyncio
    async def test_transition_snapshot_preserves_intent_for_settled_one(self):
        """An intent is not consumed by a transitional snapshot; the settled
        snapshot that follows attributes to it."""
        play_lease_manager.record_play_intent(INSTANCE, "sess-a", "AgentA")

        await play_lease_manager.observe_editor_state(
            INSTANCE, EXIT_TRANSITION_STATE)
        assert play_lease_manager.get_active_lease(INSTANCE) is None

        await play_lease_manager.observe_editor_state(INSTANCE, PLAYING_STATE)
        lease = play_lease_manager.get_active_lease(INSTANCE)
        assert lease is not None
        assert lease.owner_key == "sess-a"

    @pytest.mark.asyncio
    async def test_transition_snapshot_still_renews_existing_lease(self):
        ctx_a = await _pinned_context()
        lease = _acquire_for(ctx_a)
        stale = time.monotonic() - 60.0
        lease.last_activity = stale

        await play_lease_manager.observe_editor_state(
            INSTANCE, EXIT_TRANSITION_STATE)

        assert lease.last_activity > stale
        assert play_lease_manager.get_active_lease(INSTANCE) is lease


# ----------------------------------------------------------------------
# PlayMode test runs attribute to the run's owner (mis-attribution fix 1)
# ----------------------------------------------------------------------
class TestPlayModeTestRunAttribution:
    @pytest.mark.asyncio
    async def test_play_during_test_run_leases_to_job_owner(self):
        """With no intent (e.g. expired across the play-enter reload), a play
        session observed while tests run attributes to the test job's owner."""
        from services.state.test_job_lease import test_job_lease_manager

        test_job_lease_manager.record(INSTANCE, "job-1", "sess-t", "TestOwner")

        await play_lease_manager.observe_editor_state(
            INSTANCE, TESTS_PLAYING_STATE)

        lease = play_lease_manager.get_active_lease(INSTANCE)
        assert lease is not None
        assert lease.owner_key == "sess-t"
        assert lease.owner_display == "TestOwner"

    @pytest.mark.asyncio
    async def test_unowned_test_run_still_falls_through_to_user(self):
        """A UI-initiated run (no MCP job ownership) stays a 'user' play
        session."""
        await play_lease_manager.observe_editor_state(
            INSTANCE, TESTS_PLAYING_STATE)

        lease = play_lease_manager.get_active_lease(INSTANCE)
        assert lease is not None
        assert lease.owner_key is None
        assert lease.owner_display == "user"

    @pytest.mark.asyncio
    async def test_intent_wins_over_job_owner(self):
        from services.state.test_job_lease import test_job_lease_manager

        test_job_lease_manager.record(INSTANCE, "job-1", "sess-t", "TestOwner")
        play_lease_manager.record_play_intent(INSTANCE, "sess-a", "AgentA")

        await play_lease_manager.observe_editor_state(
            INSTANCE, TESTS_PLAYING_STATE)

        lease = play_lease_manager.get_active_lease(INSTANCE)
        assert lease.owner_key == "sess-a"
