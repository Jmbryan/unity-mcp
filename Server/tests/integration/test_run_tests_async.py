import pytest

from services.state.editor_state_cache import editor_state_cache
from services.state.play_lease import play_lease_manager
from services.state.test_job_lease import test_job_lease_manager
from transport.unity_instance_middleware import (
    UnityInstanceMiddleware,
    set_unity_instance_middleware,
)

from .test_helpers import DummyContext


@pytest.fixture(autouse=True)
def _fresh_ownership_state():
    """Fresh test-job ownership + identity state for every test.

    Successful starts record the caller as the job owner (MCPC-022); without a
    reset, ownership recorded by earlier tests leaks into the clear_stuck tests
    and the ownership gate refuses the clear as a non-owner call. The lease and
    cache resets keep the play-intent and tests-pending markers recorded around
    dispatch from leaking between tests.
    """
    test_job_lease_manager.reset()
    play_lease_manager.reset()
    editor_state_cache.reset(ttl_s=0.5)
    set_unity_instance_middleware(UnityInstanceMiddleware())
    yield
    test_job_lease_manager.reset()
    play_lease_manager.reset()
    editor_state_cache.reset(ttl_s=0.5)
    set_unity_instance_middleware(UnityInstanceMiddleware())


@pytest.mark.asyncio
async def test_run_tests_async_forwards_params(monkeypatch):
    from services.tools.run_tests import run_tests

    captured = {}

    async def fake_send_with_unity_instance(send_fn, unity_instance, command_type, params, **kwargs):
        captured["command_type"] = command_type
        captured["params"] = params
        return {"success": True, "data": {"job_id": "abc123", "status": "running", "mode": "EditMode"}}

    import services.tools.run_tests as mod
    monkeypatch.setattr(
        mod.unity_transport, "send_with_unity_instance", fake_send_with_unity_instance)

    resp = await run_tests(
        DummyContext(),
        mode="EditMode",
        test_names="MyNamespace.MyTests.TestA",
        include_details=True,
    )
    assert captured["command_type"] == "run_tests"
    assert captured["params"]["mode"] == "EditMode"
    assert captured["params"]["testNames"] == ["MyNamespace.MyTests.TestA"]
    assert captured["params"]["includeDetails"] is True
    # Bridge contract: RunTests.cs stores startedBy as the job's StartedBy and
    # EditorStateCache republishes it as tests.started_by.
    started_by = captured["params"]["startedBy"]
    assert isinstance(started_by, str) and started_by
    assert resp.success is True
    assert resp.data is not None
    assert resp.data.job_id == "abc123"


@pytest.mark.asyncio
async def test_run_tests_forwards_init_timeout(monkeypatch):
    from services.tools.run_tests import run_tests

    captured = {}

    async def fake_send_with_unity_instance(send_fn, unity_instance, command_type, params, **kwargs):
        captured["params"] = params
        return {"success": True, "data": {"job_id": "abc123", "status": "running", "mode": "PlayMode"}}

    import services.tools.run_tests as mod
    monkeypatch.setattr(
        mod.unity_transport, "send_with_unity_instance", fake_send_with_unity_instance)

    resp = await run_tests(
        DummyContext(),
        mode="PlayMode",
        init_timeout=120000,
    )
    assert captured["params"]["initTimeout"] == 120000
    assert resp.success is True


@pytest.mark.asyncio
async def test_run_tests_omits_init_timeout_when_none(monkeypatch):
    from services.tools.run_tests import run_tests

    captured = {}

    async def fake_send_with_unity_instance(send_fn, unity_instance, command_type, params, **kwargs):
        captured["params"] = params
        return {"success": True, "data": {"job_id": "abc123", "status": "running", "mode": "EditMode"}}

    import services.tools.run_tests as mod
    monkeypatch.setattr(
        mod.unity_transport, "send_with_unity_instance", fake_send_with_unity_instance)

    resp = await run_tests(DummyContext(), mode="EditMode")
    assert "initTimeout" not in captured["params"]
    assert resp.success is True


@pytest.mark.asyncio
async def test_run_tests_rejects_negative_init_timeout():
    from services.tools.run_tests import run_tests

    resp = await run_tests(DummyContext(), mode="EditMode", init_timeout=-1)
    assert resp.success is False
    assert "init_timeout" in resp.error


@pytest.mark.asyncio
async def test_run_tests_rejects_zero_init_timeout():
    from services.tools.run_tests import run_tests

    resp = await run_tests(DummyContext(), mode="EditMode", init_timeout=0)
    assert resp.success is False
    assert "init_timeout" in resp.error


@pytest.mark.asyncio
async def test_run_tests_clear_stuck_forwards_only_the_flag(monkeypatch):
    from services.tools.run_tests import run_tests

    captured = {}

    async def fake_send_with_unity_instance(send_fn, unity_instance, command_type, params, **kwargs):
        captured["command_type"] = command_type
        captured["params"] = params
        return {"success": True, "message": "Stuck job cleared.", "data": {"cleared": True}}

    import services.tools.run_tests as mod
    monkeypatch.setattr(
        mod.unity_transport, "send_with_unity_instance", fake_send_with_unity_instance)

    resp = await run_tests(DummyContext(), clear_stuck=True)

    # C# reads @params["clear_stuck"] verbatim (RunTests.cs:23), so the key must stay snake_case.
    assert captured["command_type"] == "run_tests"
    assert captured["params"] == {"clear_stuck": True}
    assert resp.success is True
    assert resp.data == {"cleared": True}


@pytest.mark.asyncio
async def test_run_tests_clear_stuck_bypasses_preflight(monkeypatch):
    """#1272: preflight(requires_no_tests=True) would reject the call that clears the job blocking it."""
    from services.tools.run_tests import run_tests

    async def fake_send_with_unity_instance(send_fn, unity_instance, command_type, params, **kwargs):
        return {"success": True, "message": "Stuck job cleared.", "data": {"cleared": True}}

    async def exploding_preflight(*args, **kwargs):
        raise AssertionError("clear_stuck must short-circuit before preflight")

    import services.tools.run_tests as mod
    monkeypatch.setattr(
        mod.unity_transport, "send_with_unity_instance", fake_send_with_unity_instance)
    monkeypatch.setattr(mod, "preflight", exploding_preflight)

    resp = await run_tests(DummyContext(), clear_stuck=True)
    assert resp.success is True


@pytest.mark.asyncio
async def test_run_tests_clear_stuck_ignores_invalid_init_timeout(monkeypatch):
    """Recovery must be unconditional: an unrelated bad arg must not block clearing."""
    from services.tools.run_tests import run_tests

    async def fake_send_with_unity_instance(send_fn, unity_instance, command_type, params, **kwargs):
        return {"success": True, "message": "Stuck job cleared.", "data": {"cleared": True}}

    import services.tools.run_tests as mod
    monkeypatch.setattr(
        mod.unity_transport, "send_with_unity_instance", fake_send_with_unity_instance)

    resp = await run_tests(DummyContext(), clear_stuck=True, init_timeout=0)
    assert resp.success is True


@pytest.mark.asyncio
async def test_run_tests_without_clear_stuck_still_preflights(monkeypatch):
    from services.tools.run_tests import run_tests

    calls = []

    async def fake_send_with_unity_instance(send_fn, unity_instance, command_type, params, **kwargs):
        return {"success": True, "data": {"job_id": "abc123", "status": "running", "mode": "EditMode"}}

    async def recording_preflight(*args, **kwargs):
        calls.append(kwargs)
        return None

    import services.tools.run_tests as mod
    monkeypatch.setattr(
        mod.unity_transport, "send_with_unity_instance", fake_send_with_unity_instance)
    monkeypatch.setattr(mod, "preflight", recording_preflight)

    resp = await run_tests(DummyContext(), mode="EditMode")
    assert len(calls) == 1
    assert calls[0]["requires_no_tests"] is True
    assert resp.success is True


PLAYING_TESTS_STATE = {
    "compilation": {},
    "editor": {"play_mode": {"is_playing": True, "is_paused": False, "is_changing": False}},
    "tests": {"is_running": True},
}


@pytest.mark.asyncio
async def test_playmode_run_records_play_intent_for_caller(monkeypatch):
    """A PlayMode dispatch records a play intent so the resulting play-mode
    lease attributes to the calling session, never to 'user'."""
    from services.tools.run_tests import run_tests

    async def fake_send_with_unity_instance(send_fn, unity_instance, command_type, params, **kwargs):
        return {"success": True, "data": {"job_id": "abc123", "status": "running", "mode": "PlayMode"}}

    import services.tools.run_tests as mod
    monkeypatch.setattr(
        mod.unity_transport, "send_with_unity_instance", fake_send_with_unity_instance)

    ctx = DummyContext()
    resp = await run_tests(ctx, mode="PlayMode")
    assert resp.success is True

    # The observation of the run's play session resolves through the intent
    # (or the recorded job ownership) to the caller.
    await play_lease_manager.observe_editor_state(None, PLAYING_TESTS_STATE)
    lease = play_lease_manager.get_active_lease(None)
    assert lease is not None
    assert lease.owner_key == ctx.session_id
    assert lease.owner_display != "user"


@pytest.mark.asyncio
async def test_playmode_lease_releases_when_play_session_ends(monkeypatch):
    """The lease acquired for a PlayMode run clears on the run's play exit."""
    from services.tools.run_tests import run_tests

    async def fake_send_with_unity_instance(send_fn, unity_instance, command_type, params, **kwargs):
        return {"success": True, "data": {"job_id": "abc123", "status": "running", "mode": "PlayMode"}}

    import services.tools.run_tests as mod
    monkeypatch.setattr(
        mod.unity_transport, "send_with_unity_instance", fake_send_with_unity_instance)

    ctx = DummyContext()
    await run_tests(ctx, mode="PlayMode")
    await play_lease_manager.observe_editor_state(None, PLAYING_TESTS_STATE)
    assert play_lease_manager.get_active_lease(None) is not None

    await play_lease_manager.observe_editor_state(None, {
        "compilation": {},
        "editor": {"play_mode": {"is_playing": False, "is_paused": False, "is_changing": False}},
        "tests": {},
    })
    assert play_lease_manager.get_active_lease(None) is None


@pytest.mark.asyncio
async def test_editmode_run_records_no_play_intent(monkeypatch):
    from services.tools.run_tests import run_tests

    async def fake_send_with_unity_instance(send_fn, unity_instance, command_type, params, **kwargs):
        return {"success": True, "data": {"job_id": "abc123", "status": "running", "mode": "EditMode"}}

    import services.tools.run_tests as mod
    monkeypatch.setattr(
        mod.unity_transport, "send_with_unity_instance", fake_send_with_unity_instance)

    await run_tests(DummyContext(), mode="EditMode")

    assert play_lease_manager._intents == {}


@pytest.mark.asyncio
async def test_dispatch_pins_tests_pending_marker(monkeypatch):
    """A successful start leaves the optimistic marker pinned (cleared later
    by the confirming snapshot or its TTL), so compile-risk calls park in the
    dispatch->snapshot race window."""
    from services.tools.run_tests import run_tests

    async def fake_send_with_unity_instance(send_fn, unity_instance, command_type, params, **kwargs):
        # The marker must already be pinned at dispatch time.
        assert editor_state_cache.tests_pending(unity_instance) is not None
        return {"success": True, "data": {"job_id": "abc123", "status": "running", "mode": "EditMode"}}

    import services.tools.run_tests as mod
    monkeypatch.setattr(
        mod.unity_transport, "send_with_unity_instance", fake_send_with_unity_instance)

    resp = await run_tests(DummyContext(), mode="EditMode")
    assert resp.success is True
    assert editor_state_cache.tests_pending(None) is not None


@pytest.mark.asyncio
async def test_failed_start_clears_tests_pending_marker(monkeypatch):
    from services.tools.run_tests import run_tests

    async def fake_send_with_unity_instance(send_fn, unity_instance, command_type, params, **kwargs):
        return {"success": False, "error": "Editor refused the run."}

    import services.tools.run_tests as mod
    monkeypatch.setattr(
        mod.unity_transport, "send_with_unity_instance", fake_send_with_unity_instance)

    resp = await run_tests(DummyContext(), mode="EditMode")
    assert resp.success is False
    assert editor_state_cache.tests_pending(None) is None


@pytest.mark.asyncio
async def test_failed_playmode_start_clears_play_intent(monkeypatch):
    from services.tools.run_tests import run_tests

    async def fake_send_with_unity_instance(send_fn, unity_instance, command_type, params, **kwargs):
        return {"success": False, "error": "Editor refused the run."}

    import services.tools.run_tests as mod
    monkeypatch.setattr(
        mod.unity_transport, "send_with_unity_instance", fake_send_with_unity_instance)

    resp = await run_tests(DummyContext(), mode="PlayMode")
    assert resp.success is False
    assert play_lease_manager._intents == {}


@pytest.mark.asyncio
async def test_retry_hinted_playmode_failure_keeps_intent(monkeypatch):
    """The play-enter reload can eat a start that actually succeeded: a
    retry-hinted failure must keep the intent alive for attribution."""
    from services.tools.run_tests import run_tests

    async def fake_send_with_unity_instance(send_fn, unity_instance, command_type, params, **kwargs):
        return {"success": False, "error": "plugin disconnected", "hint": "retry"}

    import services.tools.run_tests as mod
    monkeypatch.setattr(
        mod.unity_transport, "send_with_unity_instance", fake_send_with_unity_instance)

    ctx = DummyContext()
    await run_tests(ctx, mode="PlayMode")

    await play_lease_manager.observe_editor_state(None, PLAYING_TESTS_STATE)
    lease = play_lease_manager.get_active_lease(None)
    assert lease is not None
    assert lease.owner_key == ctx.session_id


@pytest.mark.asyncio
async def test_get_test_job_forwards_job_id(monkeypatch):
    from services.tools.run_tests import get_test_job

    captured = {}

    async def fake_send_with_unity_instance(send_fn, unity_instance, command_type, params, **kwargs):
        captured["command_type"] = command_type
        captured["params"] = params
        return {"success": True, "data": {"job_id": params["job_id"], "status": "running", "mode": "EditMode"}}

    import services.tools.run_tests as mod
    monkeypatch.setattr(
        mod.unity_transport, "send_with_unity_instance", fake_send_with_unity_instance)

    resp = await get_test_job(DummyContext(), job_id="job-1")
    assert captured["command_type"] == "get_test_job"
    assert captured["params"]["job_id"] == "job-1"
    assert resp.success is True
    assert resp.data is not None
    assert resp.data.job_id == "job-1"
