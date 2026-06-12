"""Integration tests for per-session identity and routing.

Covers:
- Session-key derivation keys per-session state by MCP session identity
  (regression: active-instance pins must never collapse onto a shared key).
- Sticky friendly-name/color identity per session key.
- X-Agent-Label header caching on first sight.
- Default routing derives the primary instance from registered project
  identity (project_hash), never the per-registration launch GUID.
"""

import asyncio
import sys
import types

import pytest

from core.config import config
from transport.plugin_hub import PluginHub
from transport.plugin_registry import PluginRegistry
from transport.unity_instance_middleware import UnityInstanceMiddleware

from .test_helpers import DummyContext


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


class _MiddlewareContext:
    def __init__(self, ctx):
        self.fastmcp_context = ctx


def _stub_http_headers(monkeypatch, headers: dict[str, str]) -> None:
    """Install a fastmcp.server.dependencies stub exposing get_http_headers."""
    dependencies = types.ModuleType("fastmcp.server.dependencies")
    dependencies.get_http_headers = lambda include_all=False, include=None: dict(headers)
    monkeypatch.setitem(sys.modules, "fastmcp.server.dependencies", dependencies)


class TestSessionKeyIsolation:
    @pytest.mark.asyncio
    async def test_two_contexts_produce_distinct_session_keys(self):
        """Regression: two MCP sessions must never share a session key."""
        middleware = UnityInstanceMiddleware()

        ctx1 = DummyContext()
        ctx2 = DummyContext()

        key1 = await middleware.get_session_key(ctx1)
        key2 = await middleware.get_session_key(ctx2)

        assert key1 == ctx1.session_id
        assert key2 == ctx2.session_id
        assert key1 != key2
        assert key1 != "global"
        assert key2 != "global"

    @pytest.mark.asyncio
    async def test_instance_pins_are_independent_per_session(self):
        """Regression: one session re-pinning must not re-route another session."""
        middleware = UnityInstanceMiddleware()

        ctx1 = DummyContext()
        ctx2 = DummyContext()

        await middleware.set_active_instance(ctx1, "ProjectA@hash-a")
        await middleware.set_active_instance(ctx2, "ProjectB@hash-b")

        assert await middleware.get_active_instance(ctx1) == "ProjectA@hash-a"
        assert await middleware.get_active_instance(ctx2) == "ProjectB@hash-b"

        # Session 2 re-pins; session 1's pin must be untouched.
        await middleware.set_active_instance(ctx2, "ProjectC@hash-c")
        assert await middleware.get_active_instance(ctx1) == "ProjectA@hash-a"
        assert await middleware.get_active_instance(ctx2) == "ProjectC@hash-c"


class TestStickySessionIdentity:
    @pytest.mark.asyncio
    async def test_identity_is_sticky_across_calls_for_same_session_key(self):
        middleware = UnityInstanceMiddleware()
        ctx = DummyContext()

        first = await middleware.get_session_identity(ctx)
        second = await middleware.get_session_identity(ctx)

        assert first is second
        assert first.name == second.name
        assert first.color == second.color
        assert first.name  # non-empty friendly name
        assert first.color.startswith("#")

    @pytest.mark.asyncio
    async def test_distinct_sessions_get_distinct_names(self):
        middleware = UnityInstanceMiddleware()
        ctx1 = DummyContext()
        ctx2 = DummyContext()

        identity1 = await middleware.get_session_identity(ctx1)
        identity2 = await middleware.get_session_identity(ctx2)

        assert identity1.name != identity2.name

    @pytest.mark.asyncio
    async def test_unlabeled_session_displays_auto_name(self):
        middleware = UnityInstanceMiddleware()
        ctx = DummyContext()

        identity = await middleware.get_session_identity(ctx)

        assert identity.label is None
        assert identity.display_name == identity.name


class TestAgentLabelCaching:
    @pytest.mark.asyncio
    async def test_label_header_cached_on_first_sight(self, monkeypatch):
        middleware = UnityInstanceMiddleware()
        ctx = DummyContext()
        # Pre-pin an instance so _inject_unity_instance skips auto-selection.
        await middleware.set_active_instance(ctx, "ProjectA@hash-a")

        _stub_http_headers(monkeypatch, {"x-agent-label": "evertower-qa"})
        await middleware._inject_unity_instance(_MiddlewareContext(ctx))

        identity = await middleware.get_session_identity(ctx)
        assert identity.label == "evertower-qa"
        assert identity.display_name == "evertower-qa"
        # Auto name still assigned underneath the label.
        assert identity.name and identity.name != "evertower-qa"

        # A later request with a different header value must not change it.
        _stub_http_headers(monkeypatch, {"x-agent-label": "something-else"})
        await middleware._inject_unity_instance(_MiddlewareContext(ctx))
        identity = await middleware.get_session_identity(ctx)
        assert identity.label == "evertower-qa"

    @pytest.mark.asyncio
    async def test_missing_or_empty_label_header_is_tolerated(self, monkeypatch):
        middleware = UnityInstanceMiddleware()
        ctx = DummyContext()
        await middleware.set_active_instance(ctx, "ProjectA@hash-a")

        # No header at all.
        _stub_http_headers(monkeypatch, {})
        await middleware._inject_unity_instance(_MiddlewareContext(ctx))
        identity = await middleware.get_session_identity(ctx)
        assert identity.label is None
        assert identity.display_name == identity.name

        # Whitespace-only header value is treated as absent.
        _stub_http_headers(monkeypatch, {"x-agent-label": "   "})
        await middleware._inject_unity_instance(_MiddlewareContext(ctx))
        identity = await middleware.get_session_identity(ctx)
        assert identity.label is None

        # Label can still attach later, once a real value shows up.
        _stub_http_headers(monkeypatch, {"x-agent-label": "late-label"})
        await middleware._inject_unity_instance(_MiddlewareContext(ctx))
        identity = await middleware.get_session_identity(ctx)
        assert identity.label == "late-label"


class TestAutoselectFromProjectIdentity:
    @pytest.mark.asyncio
    async def test_autoselect_derives_primary_instance_from_project_hash(self, monkeypatch):
        """Unpinned sessions route to the instance named by registered project identity."""
        monkeypatch.setattr(config, "transport_mode", "http")

        registry = PluginRegistry()
        PluginHub.configure(registry, asyncio.get_running_loop())
        launch_token = "11111111-2222-3333-4444-555555555555"
        await registry.register(launch_token, "EverTower", "hash-et", "6000.0")

        middleware = UnityInstanceMiddleware()
        ctx1 = DummyContext()
        ctx2 = DummyContext()

        selected1 = await middleware._maybe_autoselect_instance(ctx1)
        selected2 = await middleware._maybe_autoselect_instance(ctx2)

        # Both unpinned sessions route to the same primary instance, and the
        # identity is project-derived — the launch token never appears.
        assert selected1 == "EverTower@hash-et"
        assert selected2 == "EverTower@hash-et"
        assert launch_token not in selected1
        assert await middleware.get_active_instance(ctx1) == "EverTower@hash-et"
        assert await middleware.get_active_instance(ctx2) == "EverTower@hash-et"

    @pytest.mark.asyncio
    async def test_autoselect_dedupes_duplicate_registrations_by_hash(self, monkeypatch):
        """Duplicate session GUIDs for one project still resolve to one primary instance."""
        from transport.models import SessionDetails, SessionList

        monkeypatch.setattr(config, "transport_mode", "http")

        registry = PluginRegistry()
        PluginHub.configure(registry, asyncio.get_running_loop())

        details = dict(unity_version="6000.0", connected_at="2026-06-12T00:00:00+00:00")

        async def fake_get_sessions(user_id=None):
            # Two per-launch GUID registrations, one registered project.
            return SessionList(sessions={
                "guid-1": SessionDetails(project="EverTower", hash="hash-et", **details),
                "guid-2": SessionDetails(project="EverTower", hash="hash-et", **details),
            })

        monkeypatch.setattr(PluginHub, "get_sessions", fake_get_sessions)

        middleware = UnityInstanceMiddleware()
        ctx = DummyContext()

        selected = await middleware._maybe_autoselect_instance(ctx)

        assert selected == "EverTower@hash-et"
        assert await middleware.get_active_instance(ctx) == "EverTower@hash-et"

    @pytest.mark.asyncio
    async def test_autoselect_declines_with_multiple_distinct_projects(self, monkeypatch):
        """With two distinct registered projects there is no primary instance."""
        monkeypatch.setattr(config, "transport_mode", "http")

        registry = PluginRegistry()
        PluginHub.configure(registry, asyncio.get_running_loop())
        await registry.register("guid-1", "EverTower", "hash-et", "6000.0")
        await registry.register("guid-2", "OtherGame", "hash-og", "6000.0")

        middleware = UnityInstanceMiddleware()
        ctx = DummyContext()

        selected = await middleware._maybe_autoselect_instance(ctx)

        assert selected is None
        assert await middleware.get_active_instance(ctx) is None
