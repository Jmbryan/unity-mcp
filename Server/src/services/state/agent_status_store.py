"""In-memory agent-status store fed by harness hook events (MCPC-031/033).

A per-harness hook posts fire-and-forget status events to the ``/agent-status``
custom route. Each event is summarized locally by the hook (<=120 chars, never
raw tool payloads) and carries a launch-scoped ``label`` — the MCPC-033 join
key that links hook events to a session's MCP traffic (the middleware caches the
same ``X-Agent-Label`` per session key as ``SessionIdentity.label``).

This store keeps, per agent label:

- a bounded recent-event activity tail (~10 events),
- the last-activity timestamp,
- the head of the latest submitted prompt (``UserPromptSubmit``) and the latest
  subagent spawn description (``SubagentStart``) for intent derivation,
- the latest in-progress todo line, if a harness ever provides one.

A fallback bucket (keyed by ``session``) holds events that arrive with no label.
Entries TTL-expire so a dead harness ages out. Every path fails open: a
malformed event is dropped, never raised; an unknown event type is still
recorded into the activity tail.

The store is pure ingest + read. The roster builder (session_roster) joins this
hook data with middleware/gate/lease state by label; intent derivation and flag
derivation read from here but live in their own modules.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# Hook event names (MCPC-031). PreToolUse is match-all; PostToolUse is never
# sent by design (halves per-call process spawns on Windows).
EVENT_PRE_TOOL_USE = "PreToolUse"
EVENT_SUBAGENT_START = "SubagentStart"
EVENT_SUBAGENT_STOP = "SubagentStop"
EVENT_USER_PROMPT_SUBMIT = "UserPromptSubmit"
EVENT_STOP = "Stop"
EVENT_SESSION_START = "SessionStart"
EVENT_SESSION_END = "SessionEnd"

KNOWN_EVENTS: frozenset[str] = frozenset(
    {
        EVENT_PRE_TOOL_USE,
        EVENT_SUBAGENT_START,
        EVENT_SUBAGENT_STOP,
        EVENT_USER_PROMPT_SUBMIT,
        EVENT_STOP,
        EVENT_SESSION_START,
        EVENT_SESSION_END,
    }
)

# Bounded activity tail length per agent (MCPC-028 "~10-event activity tail").
MAX_TAIL_EVENTS = 10

# Per-event summary cap. The hook is contracted to keep summaries <=120 chars,
# but the store re-clamps defensively (never trusts the client).
MAX_SUMMARY_CHARS = 120

# Agent-entry inactivity TTL (seconds): an agent with no event in this window
# is forgotten. SessionEnd marks the entry ended immediately but the entry is
# only dropped at the (shorter) ended-grace boundary so a final roster push can
# still show "disconnected".
DEFAULT_AGENT_TTL_SECONDS = 900.0
MAX_AGENT_TTL_SECONDS = 86400.0

# Grace after SessionEnd before the entry is dropped entirely.
ENDED_GRACE_SECONDS = 30.0


def _agent_ttl_s() -> float:
    raw = os.environ.get("UNITY_MCP_AGENT_STATUS_TTL_S")
    if raw is None:
        return DEFAULT_AGENT_TTL_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid UNITY_MCP_AGENT_STATUS_TTL_S=%r, using default %.1f",
            raw,
            DEFAULT_AGENT_TTL_SECONDS,
        )
        return DEFAULT_AGENT_TTL_SECONDS
    return max(0.01, min(value, MAX_AGENT_TTL_SECONDS))


def _clamp_summary(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if len(text) > MAX_SUMMARY_CHARS:
        return text[:MAX_SUMMARY_CHARS]
    return text


@dataclass
class AgentEvent:
    event: str
    summary: str
    ts: float  # client-reported unix timestamp (best effort)
    received_at: float  # server time.time() on ingest

    def as_dict(self) -> dict[str, Any]:
        return {"event": self.event, "summary": self.summary, "ts": self.ts}


@dataclass
class AgentStatus:
    """Hook-derived status for one agent, keyed by label (or session fallback)."""

    key: str  # label, or "session:<session>" for the fallback bucket
    label: str | None
    session: str | None
    tail: deque[AgentEvent] = field(default_factory=lambda: deque(maxlen=MAX_TAIL_EVENTS))
    last_activity_unix: float = 0.0
    last_received_at: float = 0.0  # time.monotonic() for TTL
    latest_prompt_head: str | None = None
    latest_spawn_description: str | None = None
    latest_todo: str | None = None
    ended: bool = False
    ended_at: float | None = None  # time.monotonic()

    def tail_as_list(self) -> list[dict[str, Any]]:
        return [event.as_dict() for event in self.tail]


class AgentStatusStore:
    """Process-local hook-event store keyed primarily by agent label."""

    def __init__(self) -> None:
        self._by_key: dict[str, AgentStatus] = {}
        self._lock = RLock()

    def reset(self) -> None:
        """Clear all status state (test seam)."""
        with self._lock:
            self._by_key.clear()

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    def ingest(self, payload: Any) -> bool:
        """Record one hook event. Returns True if anything was stored.

        Tolerates malformed/missing fields without raising. The exact contract
        is ``{label, session, event, summary, ts}`` but every field is treated
        as optional and defensively coerced.
        """
        try:
            if not isinstance(payload, dict):
                return False

            label = payload.get("label")
            label = label.strip() if isinstance(label, str) and label.strip() else None
            session = payload.get("session")
            session = session.strip() if isinstance(session, str) and session.strip() else None

            event = payload.get("event")
            event = event.strip() if isinstance(event, str) else ""
            summary = _clamp_summary(payload.get("summary"))

            ts = payload.get("ts")
            if isinstance(ts, (int, float)) and ts > 0:
                ts = float(ts)
            else:
                ts = time.time()

            # Need at least a join key to bucket the event.
            if label is not None:
                key = label
            elif session is not None:
                key = f"session:{session}"
            else:
                # No label and no session: nothing to attribute it to.
                return False

            now_mono = time.monotonic()
            with self._lock:
                entry = self._by_key.get(key)
                if entry is None:
                    entry = AgentStatus(key=key, label=label, session=session)
                    self._by_key[key] = entry
                else:
                    # Late-arriving identity fields backfill the entry.
                    if label is not None and entry.label is None:
                        entry.label = label
                    if session is not None:
                        entry.session = session

                entry.tail.append(
                    AgentEvent(event=event or "?", summary=summary, ts=ts, received_at=time.time())
                )
                entry.last_activity_unix = ts
                entry.last_received_at = now_mono

                if event == EVENT_USER_PROMPT_SUBMIT and summary:
                    entry.latest_prompt_head = summary
                elif event == EVENT_SUBAGENT_START and summary:
                    entry.latest_spawn_description = summary
                elif event == EVENT_SUBAGENT_STOP:
                    # A finished subagent no longer describes current intent.
                    entry.latest_spawn_description = None
                elif event in (EVENT_SESSION_END, EVENT_STOP):
                    if event == EVENT_SESSION_END:
                        entry.ended = True
                        entry.ended_at = now_mono
                elif event == EVENT_SESSION_START:
                    entry.ended = False
                    entry.ended_at = None
            return True
        except Exception as exc:  # fail open: ingest must never 500
            logger.debug("agent_status_store: ingest failed open: %r", exc)
            return False

    def set_todo(self, label_or_session_key: str, todo: str | None) -> None:
        """Record an in-progress todo line for intent (optional source).

        Not part of the hook contract today, but the store accepts it so a
        future harness signal can feed the most-specific intent layer.
        """
        try:
            with self._lock:
                entry = self._by_key.get(label_or_session_key)
                if entry is not None:
                    entry.latest_todo = _clamp_summary(todo) or None
        except Exception as exc:
            logger.debug("agent_status_store: set_todo failed open: %r", exc)

    # ------------------------------------------------------------------
    # Read / expiry
    # ------------------------------------------------------------------
    def _expire_locked(self, now_mono: float) -> None:
        ttl = _agent_ttl_s()
        for key in list(self._by_key):
            entry = self._by_key[key]
            if entry.ended and entry.ended_at is not None:
                if (now_mono - entry.ended_at) > ENDED_GRACE_SECONDS:
                    del self._by_key[key]
                    continue
            if (now_mono - entry.last_received_at) > ttl:
                del self._by_key[key]

    def get_by_label(self, label: str | None) -> AgentStatus | None:
        """Return the live entry for a label, applying TTL expiry."""
        if not label:
            return None
        now_mono = time.monotonic()
        with self._lock:
            self._expire_locked(now_mono)
            return self._by_key.get(label)

    def snapshot(self) -> dict[str, AgentStatus]:
        """A shallow copy of all live entries (TTL-expired first)."""
        now_mono = time.monotonic()
        with self._lock:
            self._expire_locked(now_mono)
            return dict(self._by_key)


# Global singleton (process-local) — same pattern as the other state stores.
agent_status_store = AgentStatusStore()


async def handle_agent_status_post(request: Any) -> tuple[dict[str, Any], int]:
    """Core handler for the ``POST /agent-status`` route (MCPC-031).

    Reads the JSON body (tolerating malformed/absent bodies), ingests it into
    the store, and opportunistically nudges a roster push when something was
    stored. Returns ``(body, status_code)`` — always ``({"ok": True}, 200)``
    so the fire-and-forget hook is never blocked or errored.

    Lives here (not inline in the route closure) so it is unit-testable without
    standing up FastMCP.
    """
    payload: Any = None
    try:
        payload = await request.json()
    except Exception:
        payload = None

    stored = False
    try:
        stored = agent_status_store.ingest(payload)
    except Exception as exc:
        logger.debug("agent_status: handler ingest failed open: %r", exc)

    if stored:
        try:
            from services.state.session_roster import roster_publisher

            await roster_publisher.maybe_push()
        except Exception as exc:
            logger.debug("agent_status: roster nudge failed open: %r", exc)

    return {"ok": True}, 200
