"""Integration tests for the server-side operation-class gate.

Covers:
- Reads never gated, in any editor state (MCPC-006).
- Mutate/exclusive park while the editor is busy, then release on a state
  flip; past the budget they convert to a structured busy result (MCPC-007).
- Test runs park only compile-class (exclusive) operations (MCPC-008).
- Wrapper unwrapping: batch max-severity classification and custom-tool
  definition classification (MCPC-009).
- One shared cached editor-state source; cache miss fails open (MCPC-010).
- Compile fence: another session's hot ledger entries park compile-class
  calls; own edits never park own compile; entries expire; missing ledger
  fails open (MCPC-025).
- Unknown classes (server and bridge) gate as mutate.
- Busy shaping carries the blocking owner's display name.
"""

import asyncio
import time
import types

import pytest

from core.config import config
from models.models import MCPResponse, ToolDefinitionModel
from services.state import edit_ledger
from services.state import operation_gate
from services.state.editor_state_cache import editor_state_cache
from services.state.operation_gate import (
    CLASS_EXCLUSIVE,
    CLASS_MUTATE,
    CLASS_PLAY_SCOPED,
    CLASS_READ,
    escalate_class,
    gate_for_class,
    gate_tool_call,
    resolve_tool_class,
)
from transport.plugin_hub import PluginHub
from transport.plugin_registry import PluginRegistry
from transport.unity_instance_middleware import (
    UnityInstanceMiddleware,
    get_unity_instance_middleware,
    set_unity_instance_middleware,
)

from .test_helpers import DummyContext

# Ensure these tools are present in the server registry for classification.
import services.tools.batch_execute  # noqa: F401
import services.tools.find_gameobjects  # noqa: F401
import services.tools.manage_editor  # noqa: F401
import services.tools.manage_gameobject  # noqa: F401
import services.tools.read_console  # noqa: F401
import services.tools.refresh_unity  # noqa: F401
import services.tools.run_tests  # noqa: F401


IDLE_STATE = {"compilation": {}, "editor": {}, "tests": {}}
COMPILING_STATE = {"compilation": {"is_compiling": True}}
TESTING_STATE = {"tests": {"is_running": True, "started_by": "Aurora"}}


class GateContext(DummyContext):
    """DummyContext + progress-heartbeat recording."""

    def __init__(self, **meta):
        super().__init__(**meta)
        self.progress_reports = []

    async def report_progress(self, progress, total=None, message=None):
        self.progress_reports.append((progress, total, message))


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
def _fresh_gate_state(monkeypatch):
    """Fast pacing, fresh cache and middleware identity for every test."""
    editor_state_cache.reset(ttl_s=0.01)
    monkeypatch.setattr(config, "reload_retry_ms", 50, raising=False)
    set_unity_instance_middleware(UnityInstanceMiddleware())
    # Force a registry-class refresh on first lookup in each test.
    monkeypatch.setattr(operation_gate, "_registry_last_refresh", 0.0)
    yield
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


# ----------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------
class TestClassification:
    @pytest.mark.asyncio
    async def test_server_registry_classes(self):
        assert await resolve_tool_class("read_console", {}) == CLASS_READ
        assert await resolve_tool_class("manage_gameobject", {}) == CLASS_MUTATE
        assert await resolve_tool_class("refresh_unity", {}) == CLASS_EXCLUSIVE

    @pytest.mark.asyncio
    async def test_argument_override_manage_editor(self):
        read_class = await resolve_tool_class(
            "manage_editor", {"action": "telemetry_status"})
        play_class = await resolve_tool_class("manage_editor", {"action": "play"})
        compile_class = await resolve_tool_class(
            "manage_editor", {"action": "deploy_package"})
        default_class = await resolve_tool_class("manage_editor", {"action": "add_tag"})

        assert read_class == CLASS_READ
        assert play_class == CLASS_PLAY_SCOPED
        assert compile_class == CLASS_EXCLUSIVE
        assert default_class == CLASS_MUTATE

    @pytest.mark.asyncio
    async def test_unknown_tool_is_mutate(self):
        assert await resolve_tool_class("no_such_tool_anywhere", {}) == CLASS_MUTATE

    @pytest.mark.asyncio
    async def test_bridge_declared_class_resolves_through_plugin_hub(self):
        registry = PluginRegistry()
        PluginHub.configure(registry, asyncio.get_running_loop())
        await registry.register("guid-1", "Game", "hash-x", "6000.0")
        await registry.register_tools_for_session("guid-1", [
            ToolDefinitionModel(name="warden_qa", concurrency_class="exclusive"),
            ToolDefinitionModel(name="bizarre_tool", concurrency_class="bizarre"),
            ToolDefinitionModel(name="legacy_tool"),
        ])

        assert await resolve_tool_class(
            "warden_qa", {}, "Game@hash-x") == CLASS_EXCLUSIVE
        # Unknown bridge class value gates as mutate.
        assert await resolve_tool_class(
            "bizarre_tool", {}, "Game@hash-x") == CLASS_MUTATE
        # Older bridge payloads default to mutate.
        assert await resolve_tool_class(
            "legacy_tool", {}, "Game@hash-x") == CLASS_MUTATE

    def test_escalate_class_orders_severity(self):
        assert escalate_class(CLASS_READ, CLASS_MUTATE) == CLASS_MUTATE
        assert escalate_class(CLASS_MUTATE, CLASS_READ) == CLASS_MUTATE
        assert escalate_class(CLASS_MUTATE, CLASS_PLAY_SCOPED) == CLASS_PLAY_SCOPED
        assert escalate_class(CLASS_PLAY_SCOPED, CLASS_EXCLUSIVE) == CLASS_EXCLUSIVE

    def test_register_tools_payload_carries_concurrency_class(self):
        from transport.models import RegisterToolsMessage

        message = RegisterToolsMessage(**{
            "type": "register_tools",
            "tools": [
                {"name": "warden_qa", "concurrency_class": "exclusive"},
                {"name": "older_tool"},
            ],
        })
        assert message.tools[0].concurrency_class == "exclusive"
        assert message.tools[1].concurrency_class == "mutate"

    def test_decorator_rejects_unknown_class_and_never_leaks_kwarg(self):
        from services.registry import get_registered_tools, mcp_for_unity_tool

        with pytest.raises(ValueError, match="Unknown concurrency_class"):
            @mcp_for_unity_tool(concurrency_class="nope")
            async def _bad_tool(ctx):
                pass

        @mcp_for_unity_tool(concurrency_class="read")
        async def _gate_probe_tool(ctx):
            pass

        info = next(
            t for t in get_registered_tools() if t["name"] == "_gate_probe_tool")
        assert info["concurrency_class"] == "read"
        assert "concurrency_class" not in info["kwargs"]


# ----------------------------------------------------------------------
# Gate behavior
# ----------------------------------------------------------------------
class TestGateBehavior:
    @pytest.mark.asyncio
    async def test_reads_never_gated_and_never_poll_state(self, monkeypatch):
        fake = FakeEditorState(COMPILING_STATE)
        _inject_state(monkeypatch, fake)

        ctx = GateContext()
        result = await gate_tool_call(ctx, "read_console", {"action": "get"})

        assert result is None
        assert fake.fetch_count == 0  # reads pass without touching the cache

    @pytest.mark.asyncio
    async def test_mutate_parks_then_releases_on_state_flip(self, monkeypatch):
        fake = FakeEditorState(COMPILING_STATE, COMPILING_STATE, IDLE_STATE)
        _inject_state(monkeypatch, fake)

        ctx = GateContext()
        result = await gate_tool_call(ctx, "manage_gameobject", {"action": "create"})

        assert result is None  # released once the state flipped to idle
        assert fake.fetch_count >= 3
        assert ctx.progress_reports  # heartbeats were emitted while parked
        assert any("compiling" in (msg or "") for _, _, msg in ctx.progress_reports)

    @pytest.mark.asyncio
    async def test_mutate_converts_to_busy_past_budget(self, monkeypatch):
        monkeypatch.setenv("UNITY_MCP_GATE_PARK_MAX_WAIT_S", "0.2")
        fake = FakeEditorState(COMPILING_STATE)
        _inject_state(monkeypatch, fake)

        ctx = GateContext()
        result = await gate_tool_call(ctx, "manage_gameobject", {"action": "create"})

        assert result is not None
        assert result["success"] is False
        assert result["hint"] == "retry"
        assert result["data"]["reason"] == "compiling"
        assert result["data"]["retry_after_ms"] == 1000

    @pytest.mark.asyncio
    async def test_mutations_pass_during_test_runs(self, monkeypatch):
        fake = FakeEditorState(TESTING_STATE)
        _inject_state(monkeypatch, fake)

        ctx = GateContext()
        result = await gate_tool_call(ctx, "manage_gameobject", {"action": "create"})

        assert result is None  # MCPC-008: plain mutations pass during tests

    @pytest.mark.asyncio
    async def test_exclusive_parks_during_test_runs_with_owner(self, monkeypatch):
        monkeypatch.setenv("UNITY_MCP_GATE_PARK_MAX_WAIT_S", "0.2")
        fake = FakeEditorState(TESTING_STATE)
        _inject_state(monkeypatch, fake)
        _no_fence(monkeypatch)

        ctx = GateContext()
        result = await gate_tool_call(ctx, "refresh_unity", {})

        assert result is not None
        assert result["success"] is False
        assert result["data"]["reason"] == "running_tests"
        # Busy shaping names the blocking owner.
        assert result["data"]["blocked_by"] == "Aurora"
        assert "Aurora" in result["error"]

    @pytest.mark.asyncio
    async def test_unknown_class_gates_as_mutate(self, monkeypatch):
        monkeypatch.setenv("UNITY_MCP_GATE_PARK_MAX_WAIT_S", "0.2")
        fake = FakeEditorState(COMPILING_STATE)
        _inject_state(monkeypatch, fake)

        ctx = GateContext()
        result = await gate_tool_call(ctx, "no_such_tool_anywhere", {})

        assert result is not None
        assert result["data"]["reason"] == "compiling"

    @pytest.mark.asyncio
    async def test_cache_miss_fails_open(self, monkeypatch):
        fake = FakeEditorState(RuntimeError("bridge unreachable"))
        _inject_state(monkeypatch, fake)

        ctx = GateContext()
        result = await gate_tool_call(ctx, "manage_gameobject", {"action": "create"})

        assert result is None  # no snapshot -> gate fails open
        assert fake.fetch_count >= 1

    @pytest.mark.asyncio
    async def test_compile_owner_attribution_from_exclusive_edge(self, monkeypatch):
        monkeypatch.setenv("UNITY_MCP_GATE_PARK_MAX_WAIT_S", "0.2")
        fake = FakeEditorState(COMPILING_STATE)
        _inject_state(monkeypatch, fake)

        editor_state_cache.record_exclusive_edge(None, "Basalt", kind="compile")
        ctx = GateContext()
        result = await gate_tool_call(ctx, "manage_gameobject", {"action": "create"})

        assert result is not None
        assert result["data"]["blocked_by"] == "Basalt"


# ----------------------------------------------------------------------
# Shared editor-state cache (MCPC-010)
# ----------------------------------------------------------------------
class TestSharedStateCache:
    @pytest.mark.asyncio
    async def test_fetches_are_coalesced_within_ttl(self, monkeypatch):
        editor_state_cache.reset(ttl_s=60.0)
        fake = FakeEditorState(IDLE_STATE)
        _inject_state(monkeypatch, fake)

        ctx = GateContext()
        for _ in range(5):
            state = await editor_state_cache.get(ctx, "Game@hash-x")
            assert state == IDLE_STATE

        assert fake.fetch_count == 1

    @pytest.mark.asyncio
    async def test_stale_snapshot_survives_one_failed_refresh(self, monkeypatch):
        editor_state_cache.reset(ttl_s=0.0)
        fake = FakeEditorState(IDLE_STATE, RuntimeError("boom"))
        _inject_state(monkeypatch, fake)

        ctx = GateContext()
        first = await editor_state_cache.get(ctx, None)
        second = await editor_state_cache.get(ctx, None)

        assert first == IDLE_STATE
        assert second == IDLE_STATE  # stale-but-recent snapshot still answers

    @pytest.mark.asyncio
    async def test_exclusive_edge_records_expire(self, monkeypatch):
        editor_state_cache.record_exclusive_edge("inst", "Cobalt", kind="compile")
        assert editor_state_cache.get_exclusive_edge_owner("inst") == "Cobalt"

        edge = editor_state_cache._edges["inst"]
        monkeypatch.setattr(edge, "recorded_at", time.monotonic() - 120.0)
        assert editor_state_cache.get_exclusive_edge_owner("inst") is None


# ----------------------------------------------------------------------
# Compile fence (MCPC-025)
# ----------------------------------------------------------------------
class TestCompileFence:
    def _fence_root(self, monkeypatch, tmp_path):
        root = str(tmp_path)

        async def _root(unity_instance):
            return root

        monkeypatch.setattr(edit_ledger, "resolve_project_root", _root)
        return root

    @pytest.mark.asyncio
    async def test_other_session_hot_edit_parks_compile(self, monkeypatch, tmp_path):
        monkeypatch.setenv("UNITY_MCP_GATE_PARK_MAX_WAIT_S", "0.2")
        root = self._fence_root(monkeypatch, tmp_path)
        fake = FakeEditorState(IDLE_STATE)
        _inject_state(monkeypatch, fake)
        edit_ledger.append_entry(root, "OtherAgent", "Assets/Scripts/Foo.cs")

        ctx = GateContext()
        result = await gate_for_class(ctx, CLASS_EXCLUSIVE, "refresh_unity", "Game@hash-x")

        assert result is not None
        assert result["data"]["reason"] == "edit_fence"
        assert result["data"]["blocked_by"] == "OtherAgent"

    @pytest.mark.asyncio
    async def test_own_hot_edits_never_park_own_compile(self, monkeypatch, tmp_path):
        root = self._fence_root(monkeypatch, tmp_path)
        fake = FakeEditorState(IDLE_STATE)
        _inject_state(monkeypatch, fake)

        ctx = GateContext()
        identity = await get_unity_instance_middleware().get_session_identity(ctx)
        edit_ledger.append_entry(root, identity.display_name, "Assets/Scripts/Foo.cs")

        result = await gate_for_class(ctx, CLASS_EXCLUSIVE, "refresh_unity", "Game@hash-x")

        assert result is None

    @pytest.mark.asyncio
    async def test_fence_releases_when_entries_expire(self, monkeypatch, tmp_path):
        root = self._fence_root(monkeypatch, tmp_path)
        fake = FakeEditorState(IDLE_STATE)
        _inject_state(monkeypatch, fake)
        # Hot window is ~12s; an entry from a minute ago has quiesced.
        stale_ts = time.time() - 60.0
        edit_ledger.append_entry(
            root, "OtherAgent", "Assets/Scripts/Foo.cs", timestamp=stale_ts)
        ledger_file = edit_ledger.ledger_path(root)
        import os as _os
        _os.utime(ledger_file, (stale_ts, stale_ts))

        ctx = GateContext()
        result = await gate_for_class(ctx, CLASS_EXCLUSIVE, "refresh_unity", "Game@hash-x")

        assert result is None

    @pytest.mark.asyncio
    async def test_fence_fails_open_when_ledger_missing(self, monkeypatch, tmp_path):
        self._fence_root(monkeypatch, tmp_path)  # root exists, no ledger file
        fake = FakeEditorState(IDLE_STATE)
        _inject_state(monkeypatch, fake)

        ctx = GateContext()
        result = await gate_for_class(ctx, CLASS_EXCLUSIVE, "refresh_unity", "Game@hash-x")

        assert result is None

    @pytest.mark.asyncio
    async def test_mtime_fallback_fences_unparseable_ledger(self, monkeypatch, tmp_path):
        monkeypatch.setenv("UNITY_MCP_GATE_PARK_MAX_WAIT_S", "0.2")
        root = self._fence_root(monkeypatch, tmp_path)
        fake = FakeEditorState(IDLE_STATE)
        _inject_state(monkeypatch, fake)

        import os as _os
        ledger_file = edit_ledger.ledger_path(root)
        _os.makedirs(_os.path.dirname(ledger_file), exist_ok=True)
        with open(ledger_file, "w", encoding="utf-8") as f:
            f.write("not json at all\n")

        ctx = GateContext()
        result = await gate_for_class(ctx, CLASS_EXCLUSIVE, "refresh_unity", "Game@hash-x")

        assert result is not None
        assert result["data"]["reason"] == "edit_fence"
        assert "blocked_by" not in result["data"]  # fence without attribution

    @pytest.mark.asyncio
    async def test_fence_only_applies_to_exclusive(self, monkeypatch, tmp_path):
        root = self._fence_root(monkeypatch, tmp_path)
        fake = FakeEditorState(IDLE_STATE)
        _inject_state(monkeypatch, fake)
        edit_ledger.append_entry(root, "OtherAgent", "Assets/Scripts/Foo.cs")

        ctx = GateContext()
        result = await gate_for_class(ctx, CLASS_MUTATE, "manage_gameobject", "Game@hash-x")

        assert result is None  # mutations are not fenced


# ----------------------------------------------------------------------
# Wrapper unwrapping (MCPC-009)
# ----------------------------------------------------------------------
class TestBatchUnwrap:
    def _setup_batch(self, monkeypatch, fake):
        import services.tools.batch_execute as batch_mod

        _inject_state(monkeypatch, fake)
        batch_mod.invalidate_cached_max_commands()
        sent = {}

        async def fake_send(send_fn, unity_instance, command_type, params, **kwargs):
            sent["command_type"] = command_type
            sent["params"] = params
            return {"success": True, "data": {"results": []}}

        monkeypatch.setattr(batch_mod, "send_with_unity_instance", fake_send)
        return batch_mod, sent

    @pytest.mark.asyncio
    async def test_batch_of_reads_passes_while_editor_busy(self, monkeypatch):
        fake = FakeEditorState(COMPILING_STATE)
        batch_mod, sent = self._setup_batch(monkeypatch, fake)

        result = await batch_mod.batch_execute(GateContext(), commands=[
            {"tool": "read_console", "params": {"action": "get"}},
            {"tool": "find_gameobjects", "params": {"query": "Player"}},
        ])

        assert sent["command_type"] == "batch_execute"
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_batch_with_mutation_blocks_while_editor_busy(self, monkeypatch):
        monkeypatch.setenv("UNITY_MCP_GATE_PARK_MAX_WAIT_S", "0.2")
        fake = FakeEditorState(COMPILING_STATE)
        batch_mod, sent = self._setup_batch(monkeypatch, fake)

        result = await batch_mod.batch_execute(GateContext(), commands=[
            {"tool": "read_console", "params": {"action": "get"}},
            {"tool": "manage_gameobject", "params": {"action": "create"}},
        ])

        assert "command_type" not in sent  # never reached Unity
        assert result["success"] is False
        assert result["data"]["reason"] == "compiling"

    @pytest.mark.asyncio
    async def test_batch_with_exclusive_parks_during_tests(self, monkeypatch):
        monkeypatch.setenv("UNITY_MCP_GATE_PARK_MAX_WAIT_S", "0.2")
        fake = FakeEditorState(TESTING_STATE)
        batch_mod, sent = self._setup_batch(monkeypatch, fake)
        _no_fence(monkeypatch)

        result = await batch_mod.batch_execute(GateContext(), commands=[
            {"tool": "manage_editor", "params": {"action": "deploy_package"}},
        ])

        assert "command_type" not in sent
        assert result["success"] is False
        assert result["data"]["reason"] == "running_tests"
        assert result["data"]["blocked_by"] == "Aurora"


class TestCustomToolUnwrap:
    @pytest.fixture
    def custom_tool_service(self):
        from services.custom_tool_service import CustomToolService

        previous = CustomToolService._instance

        class _FakeMCP:
            def custom_route(self, *args, **kwargs):
                def deco(fn):
                    return fn
                return deco

        service = CustomToolService(_FakeMCP())
        yield service
        CustomToolService._instance = previous

    def _setup_custom(self, monkeypatch, service, definition):
        import services.tools.execute_custom_tool as tool_mod
        from services.custom_tool_service import CustomToolService

        service._register_project_tools(
            "proj-1", [definition], project_hash="hash-x")
        monkeypatch.setattr(
            tool_mod, "resolve_project_id_for_unity_instance", lambda _: "proj-1")

        executed = {}

        async def fake_execute_tool(self, project_id, tool_name, unity_instance,
                                    params=None, user_id=None):
            executed["tool_name"] = tool_name
            return MCPResponse(success=True, message="ran")

        monkeypatch.setattr(CustomToolService, "execute_tool", fake_execute_tool)
        return tool_mod, executed

    @pytest.mark.asyncio
    async def test_exclusive_custom_tool_blocks_during_tests(
        self, monkeypatch, custom_tool_service,
    ):
        monkeypatch.setenv("UNITY_MCP_GATE_PARK_MAX_WAIT_S", "0.2")
        _inject_state(monkeypatch, FakeEditorState(TESTING_STATE))
        _no_fence(monkeypatch)
        tool_mod, executed = self._setup_custom(
            monkeypatch, custom_tool_service,
            ToolDefinitionModel(name="warden_qa", concurrency_class="exclusive"),
        )

        ctx = GateContext()
        await ctx.set_state("unity_instance", "Game@hash-x")
        result = await tool_mod.execute_custom_tool(ctx, "warden_qa", {})

        assert "tool_name" not in executed  # gated before dispatch
        assert result.success is False
        assert result.hint == "retry"
        assert result.data["reason"] == "running_tests"
        assert result.data["blocked_by"] == "Aurora"

    @pytest.mark.asyncio
    async def test_read_custom_tool_passes_during_tests(
        self, monkeypatch, custom_tool_service,
    ):
        _inject_state(monkeypatch, FakeEditorState(TESTING_STATE))
        tool_mod, executed = self._setup_custom(
            monkeypatch, custom_tool_service,
            ToolDefinitionModel(name="scene_query", concurrency_class="read"),
        )

        ctx = GateContext()
        await ctx.set_state("unity_instance", "Game@hash-x")
        result = await tool_mod.execute_custom_tool(ctx, "scene_query", {})

        assert executed["tool_name"] == "scene_query"
        assert result.success is True


# ----------------------------------------------------------------------
# Post-hoc exclusive ownership (MCPC-011)
# ----------------------------------------------------------------------
class TestArbitraryCodeOwnership:
    @pytest.mark.asyncio
    async def test_execute_code_compile_edge_assigns_ownership(self, monkeypatch):
        import services.tools.execute_code as code_mod

        # Idle before the call; compiling right after it.
        _inject_state(monkeypatch, FakeEditorState(COMPILING_STATE))

        async def fake_send(send_fn, unity_instance, command_type, params, **kwargs):
            return {"success": True, "data": {"result": "ok"}}

        monkeypatch.setattr(code_mod, "send_with_unity_instance", fake_send)

        ctx = GateContext()
        identity = await get_unity_instance_middleware().get_session_identity(ctx)
        result = await code_mod.execute_code(
            ctx, action="execute", code="UnityEditor.AssetDatabase.Refresh();")

        assert result["success"] is True
        assert editor_state_cache.get_exclusive_edge_owner(None) == identity.display_name

    @pytest.mark.asyncio
    async def test_execute_code_without_edge_records_nothing(self, monkeypatch):
        import services.tools.execute_code as code_mod

        _inject_state(monkeypatch, FakeEditorState(IDLE_STATE))

        async def fake_send(send_fn, unity_instance, command_type, params, **kwargs):
            return {"success": True, "data": {"result": "ok"}}

        monkeypatch.setattr(code_mod, "send_with_unity_instance", fake_send)

        ctx = GateContext()
        await code_mod.execute_code(ctx, action="execute", code="return 1;")

        assert editor_state_cache.get_exclusive_edge_owner(None) is None


# ----------------------------------------------------------------------
# Middleware wiring
# ----------------------------------------------------------------------
class _MiddlewareContext:
    def __init__(self, ctx, tool_name, arguments):
        self.fastmcp_context = ctx
        self.message = types.SimpleNamespace(name=tool_name, arguments=arguments)


class TestMiddlewareGate:
    @pytest.mark.asyncio
    async def test_busy_call_returns_tool_result_without_calling_next(self, monkeypatch):
        monkeypatch.setenv("UNITY_MCP_GATE_PARK_MAX_WAIT_S", "0.2")
        _inject_state(monkeypatch, FakeEditorState(COMPILING_STATE))

        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        ctx = GateContext()
        await middleware.set_active_instance(ctx, "ProjectA@hash-a")
        context = _MiddlewareContext(ctx, "manage_gameobject", {"action": "create"})

        called = {}

        async def call_next(_context):
            called["next"] = True
            return "tool-ran"

        result = await middleware.on_call_tool(context, call_next)

        assert "next" not in called
        payload = result.structured_content
        assert payload["success"] is False
        assert payload["hint"] == "retry"
        assert payload["data"]["reason"] == "compiling"

    @pytest.mark.asyncio
    async def test_read_call_passes_straight_through(self, monkeypatch):
        _inject_state(monkeypatch, FakeEditorState(COMPILING_STATE))

        middleware = UnityInstanceMiddleware()
        set_unity_instance_middleware(middleware)
        ctx = GateContext()
        await middleware.set_active_instance(ctx, "ProjectA@hash-a")
        context = _MiddlewareContext(ctx, "read_console", {"action": "get"})

        async def call_next(_context):
            return "tool-ran"

        result = await middleware.on_call_tool(context, call_next)

        assert result == "tool-ran"
