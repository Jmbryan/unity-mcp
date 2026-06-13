"""Integration tests for async test-job ownership (MCPC-022).

Covers:
- Owner recorded when run_tests successfully starts a job (MCPC-022).
- A non-owner abort/clear (run_tests clear_stuck) receives a structured busy
  result naming the owner; the owner's clear is allowed.
- Status reads (get_test_job) are never gated and never refuse.
- Liveness fails open: TTL expiry frees ownership; an unknown/expired job
  allows the clear; a terminal status observed via get_test_job frees
  ownership so a later clear is unguarded.

Offline: no Unity editor; the transport send is faked (fake_send) following
the run-tests-async / play-lease test patterns. Two DummyContexts model an
owner session and a non-owner session.
"""

import time

import pytest

from services.state import test_job_lease
from services.state.test_job_lease import (
    enforce_clear_ownership,
    instance_key,
    note_status_observed,
    record_started_job,
    test_job_lease_manager,
)
from transport.unity_instance_middleware import (
    UnityInstanceMiddleware,
    set_unity_instance_middleware,
)

from .test_helpers import DummyContext

INSTANCE = "Game@hash-x"


@pytest.fixture(autouse=True)
def _fresh_ownership_state():
    """Fresh ownership + identity state for every test."""
    test_job_lease_manager.reset()
    set_unity_instance_middleware(UnityInstanceMiddleware())
    yield
    test_job_lease_manager.reset()
    set_unity_instance_middleware(UnityInstanceMiddleware())


def _fake_send(captured, response):
    """A send_with_unity_instance stub capturing command/params."""

    async def send_with_unity_instance(send_fn, unity_instance, command_type, params, **kwargs):
        captured.append((command_type, dict(params)))
        return response

    return send_with_unity_instance


async def _pinned_context():
    ctx = DummyContext()
    await ctx.set_state("unity_instance", INSTANCE)
    return ctx


# ----------------------------------------------------------------------
# Owner recorded on a successful start (MCPC-022)
# ----------------------------------------------------------------------
class TestOwnerRecording:
    @pytest.mark.asyncio
    async def test_successful_start_records_owner(self, monkeypatch):
        from services.tools import run_tests as mod

        captured = []
        monkeypatch.setattr(
            mod.unity_transport,
            "send_with_unity_instance",
            _fake_send(captured, {
                "success": True,
                "data": {"job_id": "job-1", "status": "running", "mode": "EditMode"},
            }),
        )
        # preflight does a transport probe; stub it to pass cleanly.
        async def _ok_preflight(ctx, **kwargs):
            return None

        monkeypatch.setattr(mod, "preflight", _ok_preflight)

        ctx_a = await _pinned_context()
        resp = await mod.run_tests(ctx_a, mode="EditMode")

        assert resp.success is True
        owned = test_job_lease_manager.get_active(INSTANCE)
        assert owned is not None
        assert owned.job_id == "job-1"
        assert owned.owner_key == ctx_a.session_id
        from transport.unity_instance_middleware import get_unity_instance_middleware

        identity = await get_unity_instance_middleware().get_session_identity(ctx_a)
        assert owned.owner_display == identity.display_name

    @pytest.mark.asyncio
    async def test_failed_start_records_nothing(self, monkeypatch):
        from services.tools import run_tests as mod

        captured = []
        monkeypatch.setattr(
            mod.unity_transport,
            "send_with_unity_instance",
            _fake_send(captured, {"success": False, "error": "tests_running"}),
        )

        async def _ok_preflight(ctx, **kwargs):
            return None

        monkeypatch.setattr(mod, "preflight", _ok_preflight)

        ctx_a = await _pinned_context()
        resp = await mod.run_tests(ctx_a, mode="EditMode")

        assert resp.success is False
        assert test_job_lease_manager.get_active(INSTANCE) is None


# ----------------------------------------------------------------------
# Abort/clear guard (MCPC-022)
# ----------------------------------------------------------------------
class TestClearGuard:
    @pytest.mark.asyncio
    async def test_non_owner_clear_is_busy_with_owner(self, monkeypatch):
        from services.tools import run_tests as mod

        captured = []
        monkeypatch.setattr(
            mod.unity_transport,
            "send_with_unity_instance",
            _fake_send(captured, {"success": True, "data": {"cleared": True}}),
        )

        ctx_a = await _pinned_context()
        ctx_b = await _pinned_context()
        # Record A as the owner directly (a successful start happened earlier).
        await record_started_job(ctx_a, INSTANCE, "job-1")
        owner = test_job_lease_manager.get_active(INSTANCE).owner_display

        resp = await mod.run_tests(ctx_b, clear_stuck=True)

        assert resp.success is False
        assert resp.hint == "retry"
        assert resp.data["reason"] == "test_job"
        assert resp.data["blocked_by"] == owner
        assert "job-1" in resp.error
        # Refused before any transport call: A's job is untouched.
        assert not captured
        assert test_job_lease_manager.get_active(INSTANCE) is not None

    @pytest.mark.asyncio
    async def test_owner_clear_is_allowed(self, monkeypatch):
        from services.tools import run_tests as mod

        captured = []
        monkeypatch.setattr(
            mod.unity_transport,
            "send_with_unity_instance",
            _fake_send(captured, {"success": True, "data": {"cleared": True}}),
        )

        ctx_a = await _pinned_context()
        await record_started_job(ctx_a, INSTANCE, "job-1")

        resp = await mod.run_tests(ctx_a, clear_stuck=True)

        assert resp.success is True
        assert captured and captured[0][0] == "run_tests"
        assert captured[0][1] == {"clear_stuck": True}
        # Ownership freed after the clear.
        assert test_job_lease_manager.get_active(INSTANCE) is None

    @pytest.mark.asyncio
    async def test_clear_with_no_owned_job_fails_open(self, monkeypatch):
        from services.tools import run_tests as mod

        captured = []
        monkeypatch.setattr(
            mod.unity_transport,
            "send_with_unity_instance",
            _fake_send(captured, {"success": True, "data": {"cleared": False}}),
        )

        ctx_b = await _pinned_context()
        # No ownership recorded at all -> unknown job -> fail open (allow).
        resp = await mod.run_tests(ctx_b, clear_stuck=True)

        assert resp.success is True
        assert captured and captured[0][1] == {"clear_stuck": True}


# ----------------------------------------------------------------------
# Status reads never gated (MCPC-022)
# ----------------------------------------------------------------------
class TestStatusReadsNeverGated:
    @pytest.mark.asyncio
    async def test_non_owner_status_poll_passes(self, monkeypatch):
        from services.tools import run_tests as mod

        captured = []
        monkeypatch.setattr(
            mod.unity_transport,
            "send_with_unity_instance",
            _fake_send(captured, {
                "success": True,
                "data": {"job_id": "job-1", "status": "running", "mode": "EditMode"},
            }),
        )

        ctx_a = await _pinned_context()
        ctx_b = await _pinned_context()
        await record_started_job(ctx_a, INSTANCE, "job-1")

        # A non-owner polling status is never refused.
        resp = await mod.get_test_job(ctx_b, job_id="job-1")
        assert resp.success is True
        assert resp.data.status == "running"
        # Owner unchanged by a non-owner read.
        owned = test_job_lease_manager.get_active(INSTANCE)
        assert owned is not None and owned.owner_key == ctx_a.session_id

    @pytest.mark.asyncio
    async def test_owner_poll_renews_ttl(self, monkeypatch):
        ctx_a = await _pinned_context()
        await record_started_job(ctx_a, INSTANCE, "job-1")
        owned = test_job_lease_manager.get_active(INSTANCE)
        owned.last_activity = time.monotonic() - 100.0

        note_status_observed(INSTANCE, "job-1", "running")

        assert test_job_lease_manager.get_active(INSTANCE).last_activity > (
            time.monotonic() - 5.0
        )

    @pytest.mark.asyncio
    async def test_terminal_status_frees_ownership(self, monkeypatch):
        from services.tools import run_tests as mod

        captured = []
        monkeypatch.setattr(
            mod.unity_transport,
            "send_with_unity_instance",
            _fake_send(captured, {
                "success": True,
                "data": {"job_id": "job-1", "status": "succeeded", "mode": "EditMode"},
            }),
        )

        ctx_a = await _pinned_context()
        await record_started_job(ctx_a, INSTANCE, "job-1")

        resp = await mod.get_test_job(ctx_a, job_id="job-1")
        assert resp.success is True
        # Observing completion frees ownership -> a later clear is unguarded.
        assert test_job_lease_manager.get_active(INSTANCE) is None


# ----------------------------------------------------------------------
# Liveness / fail-open (MCPC-022)
# ----------------------------------------------------------------------
class TestLiveness:
    @pytest.mark.asyncio
    async def test_ttl_expiry_frees_ownership(self):
        ctx_a = await _pinned_context()
        await record_started_job(ctx_a, INSTANCE, "job-1")
        owned = test_job_lease_manager._owned[instance_key(INSTANCE)]
        owned.last_activity = time.monotonic() - 99999.0

        # Expired entry is not returned and is purged.
        assert test_job_lease_manager.get_active(INSTANCE) is None

    @pytest.mark.asyncio
    async def test_expired_job_clear_fails_open(self):
        ctx_a = await _pinned_context()
        ctx_b = await _pinned_context()
        await record_started_job(ctx_a, INSTANCE, "job-1")
        owned = test_job_lease_manager._owned[instance_key(INSTANCE)]
        owned.last_activity = time.monotonic() - 99999.0

        # A non-owner clear against an expired entry is allowed (fail open).
        busy = await enforce_clear_ownership(ctx_b, INSTANCE)
        assert busy is None

    @pytest.mark.asyncio
    async def test_terminal_status_release_then_clear_unguarded(self):
        ctx_a = await _pinned_context()
        ctx_b = await _pinned_context()
        await record_started_job(ctx_a, INSTANCE, "job-1")
        note_status_observed(INSTANCE, "job-1", "failed")

        busy = await enforce_clear_ownership(ctx_b, INSTANCE)
        assert busy is None

    @pytest.mark.asyncio
    async def test_touch_ignores_mismatched_job_id(self):
        ctx_a = await _pinned_context()
        await record_started_job(ctx_a, INSTANCE, "job-1")
        owned = test_job_lease_manager._owned[instance_key(INSTANCE)]
        stale = time.monotonic() - 100.0
        owned.last_activity = stale

        # A poll of a different/older job id never resurrects the entry's TTL.
        note_status_observed(INSTANCE, "job-OTHER", "running")
        assert test_job_lease_manager._owned[instance_key(INSTANCE)].last_activity == stale
