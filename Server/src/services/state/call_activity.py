"""Per-session in-flight call activity (parked + long-op tracking).

Records, per MCP session key, the call the session is currently parked on (if
any) and the start time of its current in-flight gated call. Feeds two roster
surfaces:

- the ``parked`` attribution flag (MCPC-028): a call currently parked in the
  gate, plus *what* it is parked on (the blocking reason);
- the ``long-op`` / ``parked-over-budget`` rabbit-hole flags (MCPC-032): a single
  call running or parked beyond a threshold.

Pure observation: the gate records park start/clear and call start/end; the
roster reads. Everything is process-local, fails open, and ages out — a record
left behind by a crashed call simply reflects a stale long-op until its session
is forgotten. There is no cleanup step.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import RLock

logger = logging.getLogger(__name__)

# A parked marker older than this is assumed stale (the call returned without
# clearing) and ignored by readers. The gate's park budget is well under 20s,
# so anything older than this never reflects a genuine in-progress park.
PARK_STALE_SECONDS = 30.0


@dataclass
class _ParkMarker:
    reason: str
    owner: str | None
    started_at: float  # time.monotonic()
    started_at_unix: float  # time.time()


@dataclass
class _CallMarker:
    tool_name: str
    started_at: float  # time.monotonic()
    started_at_unix: float


class CallActivityTracker:
    """Process-local in-flight call + park markers, keyed by MCP session key."""

    def __init__(self) -> None:
        self._parked: dict[str, _ParkMarker] = {}
        self._calls: dict[str, _CallMarker] = {}
        # Last-any and last-mutate MCP activity per session (time.monotonic()),
        # for the roster's `active` / `editing` state derivation.
        self._last_any: dict[str, float] = {}
        self._last_mutate: dict[str, float] = {}
        self._lock = RLock()

    def reset(self) -> None:
        with self._lock:
            self._parked.clear()
            self._calls.clear()
            self._last_any.clear()
            self._last_mutate.clear()

    # ------------------------------------------------------------------
    # Activity recency (state derivation: active / editing)
    # ------------------------------------------------------------------
    def mark_activity(self, key: str | None, is_mutate: bool) -> None:
        if not key:
            return
        now = time.monotonic()
        with self._lock:
            self._last_any[key] = now
            if is_mutate:
                self._last_mutate[key] = now

    def activity_ages(self, key: str | None) -> tuple[float | None, float | None]:
        """(seconds since last any-call, seconds since last mutate) or Nones."""
        if not key:
            return (None, None)
        now = time.monotonic()
        with self._lock:
            last_any = self._last_any.get(key)
            last_mutate = self._last_mutate.get(key)
        any_age = (now - last_any) if last_any is not None else None
        mutate_age = (now - last_mutate) if last_mutate is not None else None
        return (any_age, mutate_age)

    # ------------------------------------------------------------------
    # Park markers (set/cleared by the gate's park loop)
    # ------------------------------------------------------------------
    def mark_parked(self, key: str | None, reason: str, owner: str | None) -> None:
        if not key:
            return
        with self._lock:
            existing = self._parked.get(key)
            # Preserve the original park start across heartbeats so
            # parked-duration / over-budget reflects the true wait.
            if existing is not None and existing.reason == reason:
                existing.owner = owner
                return
            self._parked[key] = _ParkMarker(
                reason=reason,
                owner=owner,
                started_at=time.monotonic(),
                started_at_unix=time.time(),
            )

    def clear_parked(self, key: str | None) -> None:
        if not key:
            return
        with self._lock:
            self._parked.pop(key, None)

    def get_parked(self, key: str | None) -> dict | None:
        """The active park marker for a session (reason, owner, age), or None."""
        if not key:
            return None
        now = time.monotonic()
        with self._lock:
            marker = self._parked.get(key)
            if marker is None:
                return None
            age = now - marker.started_at
            if age > PARK_STALE_SECONDS:
                self._parked.pop(key, None)
                return None
            return {
                "reason": marker.reason,
                "owner": marker.owner,
                "age_seconds": age,
                "since_unix": marker.started_at_unix,
            }

    # ------------------------------------------------------------------
    # In-flight call markers (start/end around dispatch)
    # ------------------------------------------------------------------
    def mark_call_start(self, key: str | None, tool_name: str | None) -> None:
        if not key or not tool_name:
            return
        with self._lock:
            self._calls[key] = _CallMarker(
                tool_name=tool_name,
                started_at=time.monotonic(),
                started_at_unix=time.time(),
            )

    def mark_call_end(self, key: str | None) -> None:
        if not key:
            return
        with self._lock:
            self._calls.pop(key, None)

    def get_call(self, key: str | None) -> dict | None:
        """The in-flight call for a session (tool, age), or None."""
        if not key:
            return None
        now = time.monotonic()
        with self._lock:
            marker = self._calls.get(key)
            if marker is None:
                return None
            return {
                "tool_name": marker.tool_name,
                "age_seconds": now - marker.started_at,
                "since_unix": marker.started_at_unix,
            }


# Global singleton (process-local) — same pattern as the other state stores.
call_activity = CallActivityTracker()
