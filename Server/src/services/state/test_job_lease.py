"""Async test-job ownership (MCPC-022).

When a session successfully starts an async Unity test run, the initiating
session is recorded as the job's owner in a server-memory store keyed by the
Unity instance. The store carries the owning MCP session key plus its display
name for attribution, and an inactivity-renewed TTL so a session that walks
away never wedges the job forever.

The guard surface is small. There is no generic server-side test cancel: the
only call that aborts/clears a *running* test job is ``run_tests`` invoked with
``clear_stuck`` set — a bridge action that force-clears the editor's current
job. While a job is owned, a non-owner's clear/abort receives a structured
busy result naming the owner (MCPC-022); the owner clears freely, and
read-only status polling (``get_test_job``) is never gated.

Liveness fails open everywhere, lease-style:

- ownership clears on a successful start of a *new* job, on owner clear, and on
  observing a terminal job status (succeeded/failed/cancelled);
- an inactivity-renewed TTL expires a forgotten entry (the bridge's own
  stale-job self-healing then unblocks anything behind it);
- an unknown job, an expired entry, or a missing owner all *allow* the call —
  the guard never blocks a clear it cannot positively attribute to another live
  session.

State lives in server memory and dies with the process (fail open on restart),
mirroring the play-lease precedent without a RunState mirror: a test job is a
transient, self-healing thing whose ownership need not survive a server bounce.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from threading import RLock

logger = logging.getLogger(__name__)

# Inactivity-renewed ownership TTL (seconds). Renewed whenever the owner polls
# or otherwise touches the job; only fires when the owner has gone quiet. A
# test run can legitimately run for many minutes, so the default is generous.
DEFAULT_TEST_JOB_TTL_SECONDS = 1800.0
MAX_TEST_JOB_TTL_SECONDS = 7200.0

# Terminal job statuses observed in get_test_job results; reaching any of them
# clears ownership (the job is no longer abortable).
TERMINAL_STATUSES: frozenset[str] = frozenset({"succeeded", "failed", "cancelled"})


def instance_key(unity_instance: str | None) -> str:
    """Ownership key for an instance reference: bare project_hash when present.

    Mirrors play_lease.instance_key — accepts ``Name@hash``, a bare hash, or
    None (keys as ``default``).
    """
    if not unity_instance:
        return "default"
    if "@" in unity_instance:
        _, _, suffix = unity_instance.rpartition("@")
        return suffix or "default"
    return unity_instance


def _test_job_ttl_s() -> float:
    raw = os.environ.get("UNITY_MCP_TEST_JOB_TTL_S")
    if raw is None:
        return DEFAULT_TEST_JOB_TTL_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid UNITY_MCP_TEST_JOB_TTL_S=%r, using default %.1f",
            raw,
            DEFAULT_TEST_JOB_TTL_SECONDS,
        )
        return DEFAULT_TEST_JOB_TTL_SECONDS
    return max(1.0, min(value, MAX_TEST_JOB_TTL_SECONDS))


@dataclass
class TestJobOwnership:
    instance_key: str
    job_id: str
    owner_key: str  # MCP session key of the initiating session
    owner_display: str
    started_at: float  # time.monotonic()
    last_activity: float  # time.monotonic()


class TestJobLeaseManager:
    """Process-local async test-job ownership, keyed by project_hash."""

    def __init__(self) -> None:
        self._owned: dict[str, TestJobOwnership] = {}
        self._lock = RLock()

    def reset(self) -> None:
        """Clear all ownership state (test seam)."""
        with self._lock:
            self._owned.clear()

    def get_active(self, unity_instance: str | None) -> TestJobOwnership | None:
        """The unexpired ownership for an instance, applying TTL expiry."""
        key = instance_key(unity_instance)
        with self._lock:
            owned = self._owned.get(key)
            if owned is None:
                return None
            if (time.monotonic() - owned.last_activity) > _test_job_ttl_s():
                self._owned.pop(key, None)
                logger.info(
                    "Test-job ownership for instance %s ('%s', job %s) expired (ttl)",
                    key,
                    owned.owner_display,
                    owned.job_id,
                )
                return None
            return owned

    def record(
        self,
        unity_instance: str | None,
        job_id: str,
        owner_key: str,
        owner_display: str,
    ) -> TestJobOwnership:
        """Record (replacing any prior) ownership for a freshly started job."""
        key = instance_key(unity_instance)
        now = time.monotonic()
        with self._lock:
            owned = TestJobOwnership(
                instance_key=key,
                job_id=job_id,
                owner_key=owner_key,
                owner_display=owner_display or owner_key,
                started_at=now,
                last_activity=now,
            )
            self._owned[key] = owned
            logger.info(
                "Test-job ownership recorded for instance %s: job %s owned by '%s'",
                key,
                job_id,
                owned.owner_display,
            )
            return owned

    def touch(self, unity_instance: str | None, job_id: str | None = None) -> None:
        """Renew the ownership's inactivity TTL.

        When ``job_id`` is given, only the matching job's ownership is renewed
        (a poll of a stale/other job id never resurrects a forgotten entry).
        """
        key = instance_key(unity_instance)
        with self._lock:
            owned = self._owned.get(key)
            if owned is None:
                return
            if job_id is not None and owned.job_id != job_id:
                return
            owned.last_activity = time.monotonic()

    def release(
        self,
        unity_instance: str | None,
        job_id: str | None = None,
        reason: str = "released",
    ) -> None:
        """Clear ownership for an instance (optionally only for one job id)."""
        key = instance_key(unity_instance)
        with self._lock:
            owned = self._owned.get(key)
            if owned is None:
                return
            if job_id is not None and owned.job_id != job_id:
                return
            self._owned.pop(key, None)
            logger.info(
                "Test-job ownership for instance %s ('%s', job %s) released: %s",
                key,
                owned.owner_display,
                owned.job_id,
                reason,
            )


# Global singleton (process-local) — same pattern as play_lease_manager.
test_job_lease_manager = TestJobLeaseManager()


# ----------------------------------------------------------------------
# Session identity helper (shared shape with play_lease)
# ----------------------------------------------------------------------
async def _session_key_and_display(ctx) -> tuple[str | None, str | None]:
    try:
        from transport.unity_instance_middleware import get_unity_instance_middleware

        middleware = get_unity_instance_middleware()
        key = await middleware.get_session_key(ctx)
        identity = middleware.ensure_session_identity_for_key(key)
        return key, identity.display_name
    except Exception as exc:
        logger.debug("test_job_lease: session identity lookup failed (fail-open): %r", exc)
        return None, None


# ----------------------------------------------------------------------
# Recording (wired into run_tests' successful start / status observation)
# ----------------------------------------------------------------------
async def record_started_job(
    ctx,
    unity_instance: str | None,
    job_id: str | None,
) -> None:
    """Record the calling session as the owner of a freshly started job.

    Never raises. A missing job id or unresolvable identity simply records
    nothing (the guard then fails open for that job).
    """
    try:
        if not job_id:
            return
        session_key, display_name = await _session_key_and_display(ctx)
        if not session_key:
            return
        test_job_lease_manager.record(
            unity_instance, job_id, session_key, display_name or session_key
        )
    except Exception as exc:
        logger.debug("test_job_lease: start recording skipped (fail-open): %r", exc)


def note_status_observed(
    unity_instance: str | None,
    job_id: str | None,
    status: str | None,
) -> None:
    """Renew on a live poll of the owned job; clear on a terminal status.

    Status reads are never gated, but observing the owned job complete is the
    cheapest place to free ownership so a later non-owner clear is unguarded.
    Never raises.
    """
    try:
        if not job_id:
            return
        normalized = status.strip().lower() if isinstance(status, str) else ""
        if normalized in TERMINAL_STATUSES:
            test_job_lease_manager.release(
                unity_instance, job_id, reason=f"status_{normalized}"
            )
        else:
            test_job_lease_manager.touch(unity_instance, job_id)
    except Exception as exc:
        logger.debug("test_job_lease: status observation skipped (fail-open): %r", exc)


# ----------------------------------------------------------------------
# Guard (owner-only abort/clear of a running test job)
# ----------------------------------------------------------------------
def _busy_response(owned: TestJobOwnership) -> dict:
    from transport.plugin_hub import PluginHub

    owner = owned.owner_display
    message = (
        f"Test-job abort/clear refused: instance's running test job (id "
        f"{owned.job_id}) is owned by {owner}. Aborting another session's test "
        "run is owner-only; let the owner stop it, or wait for it to finish."
    )
    return PluginHub._unavailable_retry_response(
        "test_job", owner=owner, message=message, retry_after_ms=2000
    )


async def enforce_clear_ownership(
    ctx,
    unity_instance: str | None,
) -> dict | None:
    """Owner-only enforcement for a test-job abort/clear call.

    Returns ``None`` when the call may proceed (no owned job, the caller is the
    owner, or ownership cannot be positively attributed), or a structured busy
    payload naming the owner. Every internal failure fails open.
    """
    try:
        owned = test_job_lease_manager.get_active(unity_instance)
        if owned is None:
            return None  # unknown/expired job — fail open

        session_key, _ = await _session_key_and_display(ctx)
        if session_key is not None and session_key == owned.owner_key:
            # Owner clears freely; renew while we are here.
            test_job_lease_manager.touch(unity_instance, owned.job_id)
            return None
        if not owned.owner_key:
            return None  # missing owner — fail open

        return _busy_response(owned)
    except Exception as exc:
        logger.debug("test_job_lease: enforcement failed open: %r", exc)
        return None
