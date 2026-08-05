"""Integration tests for the server-side preflight guard.

Covers the compile-gate fixes:
- ``requires_no_tests`` is evaluated BEFORE any side-effectful refresh, so
  preflight can never fire a refresh into the very test run it is rejecting.
- ``refresh_if_dirty`` never refreshes while a test run or play session is
  active (the bridge defers compiles during those spans; a refresh would
  import held script writes and recompile into the run).
- The idle + dirty path still refreshes.

preflight normally no-ops under pytest (its callers stub transports); these
tests disable that seam explicitly to exercise the real logic against an
injected editor-state source.
"""

import pytest

from models import MCPResponse

import services.tools.preflight as preflight_mod
from services.tools.preflight import preflight

from .test_helpers import DummyContext


def _state(
    *,
    tests_running=False,
    dirty=False,
    playing=False,
    changing=False,
    compiling=False,
):
    return MCPResponse(
        success=True,
        message="ok",
        data={
            "compilation": {"is_compiling": compiling},
            "editor": {
                "play_mode": {
                    "is_playing": playing,
                    "is_paused": False,
                    "is_changing": changing,
                }
            },
            "tests": {"is_running": tests_running},
            "assets": {"external_changes_dirty": dirty},
        },
    )


@pytest.fixture(autouse=True)
def _real_preflight(monkeypatch):
    """Disable the in-pytest no-op so the guard logic actually runs."""
    monkeypatch.setattr(preflight_mod, "_in_pytest", lambda: False)


def _inject(monkeypatch, state_response, refresh_calls):
    import services.resources.editor_state as editor_state_mod
    import services.tools.refresh_unity as refresh_mod

    async def fake_get_editor_state(ctx):
        return state_response

    async def fake_refresh_unity(ctx, **kwargs):
        refresh_calls.append(kwargs)
        return MCPResponse(success=True, message="refreshed")

    monkeypatch.setattr(editor_state_mod, "get_editor_state", fake_get_editor_state)
    monkeypatch.setattr(refresh_mod, "refresh_unity", fake_refresh_unity)


class TestRequiresNoTestsOrdering:
    @pytest.mark.asyncio
    async def test_busy_returned_without_firing_refresh(self, monkeypatch):
        """Dirty project + running tests: the tests_running busy must come
        back with NO refresh dispatched (the old order refreshed first)."""
        refresh_calls = []
        _inject(monkeypatch, _state(tests_running=True, dirty=True), refresh_calls)

        result = await preflight(
            DummyContext(),
            requires_no_tests=True,
            wait_for_no_compile=True,
            refresh_if_dirty=True,
        )

        assert isinstance(result, MCPResponse)
        assert result.success is False
        assert result.data["reason"] == "tests_running"
        assert refresh_calls == []


class TestRefreshIfDirtyGuard:
    @pytest.mark.asyncio
    async def test_no_refresh_while_tests_running_even_without_requires(
        self, monkeypatch,
    ):
        """Callers that don't require exclusivity (find_gameobjects et al)
        must still never refresh into a running test span."""
        refresh_calls = []
        _inject(monkeypatch, _state(tests_running=True, dirty=True), refresh_calls)

        result = await preflight(DummyContext(), refresh_if_dirty=True)

        assert result is None  # the tool itself proceeds
        assert refresh_calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("play_kwargs", [
        {"playing": True},
        {"changing": True},
    ])
    async def test_no_refresh_during_play_span(self, monkeypatch, play_kwargs):
        refresh_calls = []
        _inject(monkeypatch, _state(dirty=True, **play_kwargs), refresh_calls)

        result = await preflight(DummyContext(), refresh_if_dirty=True)

        assert result is None
        assert refresh_calls == []

    @pytest.mark.asyncio
    async def test_idle_dirty_project_still_refreshes(self, monkeypatch):
        refresh_calls = []
        _inject(monkeypatch, _state(dirty=True), refresh_calls)

        result = await preflight(DummyContext(), refresh_if_dirty=True)

        assert result is None
        assert len(refresh_calls) == 1
        assert refresh_calls[0]["mode"] == "if_dirty"

    @pytest.mark.asyncio
    async def test_clean_project_does_not_refresh(self, monkeypatch):
        refresh_calls = []
        _inject(monkeypatch, _state(dirty=False), refresh_calls)

        result = await preflight(DummyContext(), refresh_if_dirty=True)

        assert result is None
        assert refresh_calls == []
