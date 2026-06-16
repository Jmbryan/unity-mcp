"""Session roster build + push to the bridge (MCPC-028).

The server publishes a roster of all known agent sessions to the connected
bridge over the existing plugin WebSocket as an outbound ``session_roster``
message. The dashboard renders the roster's banner (play lease), health row, and
per-agent rows from this one message.

Each roster entry merges three data sources, joined by session identity:

- middleware ``SessionIdentity`` — the authoritative set of MCP sessions, each
  with name / label / color and the session key the leases attribute to;
- the agent-status store (hook events, keyed by ``label``) — activity tail,
  intent sources, last-activity timestamp;
- coordination state — play lease, test-job lease, the operation gate's parked /
  in-flight call markers, and the recent-tool-call ring for the looping flag.

State enum (most-specific-wins): ``in-play`` (owns the play lease) >
``running-tests`` (owns the test job) > ``waiting-parked`` (a call parked in the
gate) > ``editing`` (recent mutate) > ``active`` (recent any call) > ``idle`` >
``disconnected`` (hook reported SessionEnd).

Intent (most-specific-wins): latest in-progress todo > active subagent spawn
description > head of the submitted prompt. ``mcp-only`` rows (a hookless
harness — no matching label in the store) carry identity / activity / lease only
and NO intent, and are tagged ``source: "mcp-only"`` so the UI marks them.

Pushing: a periodic heartbeat (~2-3s) plus an on-demand push when a meaningful
change is detected. A no-op when nothing changed and the heartbeat interval has
not elapsed. Every path fails open: a build error skips that push, never raises.
"""

from __future__ import annotations

import logging
import os
import time
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

ROSTER_MESSAGE_TYPE = "session_roster"
ROSTER_SCHEMA = "unity-mcp/session_roster@1"

# Heartbeat cadence (seconds): the dashboard stays fresh even when nothing
# changes (ages tick, flags fire on elapsed time). MCPC-028 asks for ~2-3s.
DEFAULT_HEARTBEAT_SECONDS = 2.5

# Window in which an MCP session counts as `active` (recent any-call) for the
# state enum; mutate within this window is `editing`.
ACTIVE_RECENCY_SECONDS = 20.0

# Window in which a hook event (PreToolUse from this agent or any of its
# subagents — AGENT_LABEL is process-tree inherited, so a subagent's events land
# on the parent row) keeps the row `active`. Hook events are sparser than MCP
# calls (PreToolUse only, no PostToolUse), so this window is wider than the MCP
# one. The hook cannot distinguish mutate from read, so it only feeds `active`,
# never `editing`.
HOOK_ACTIVE_RECENCY_SECONDS = 45.0

# Window for the intent-stale flag's "activity continues" signal.
INTENT_ACTIVITY_RECENCY_SECONDS = 120.0

# States that are themselves activity for the intent-stale "activity_recent"
# signal (a parked or running call is ongoing work).
_ACTIVE_STATES = frozenset(
    {"in-play", "running-tests", "waiting-parked", "editing", "active"}
)


def _heartbeat_s() -> float:
    raw = os.environ.get("UNITY_MCP_ROSTER_HEARTBEAT_S")
    if raw is None:
        return DEFAULT_HEARTBEAT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_HEARTBEAT_SECONDS
    return max(0.5, value)


def _now_unix() -> float:
    return time.time()


# ----------------------------------------------------------------------
# Intent-change tracking (for the intent-stale flag)
# ----------------------------------------------------------------------
class _IntentTracker:
    """Remembers when each session's intent line last changed."""

    def __init__(self) -> None:
        self._last: dict[str, tuple[str, float]] = {}  # key -> (intent, monotonic)
        self._lock = RLock()

    def reset(self) -> None:
        with self._lock:
            self._last.clear()

    def unchanged_seconds(self, key: str, intent: str | None) -> float | None:
        """Seconds the intent has stayed the same; None when no intent."""
        if not intent:
            with self._lock:
                self._last.pop(key, None)
            return None
        now = time.monotonic()
        with self._lock:
            prior = self._last.get(key)
            if prior is None or prior[0] != intent:
                self._last[key] = (intent, now)
                return 0.0
            return now - prior[1]


_intent_tracker = _IntentTracker()


# ----------------------------------------------------------------------
# Roster build
# ----------------------------------------------------------------------
def _derive_state(
    *,
    holds_play_lease: bool,
    owns_test_job: bool,
    parked: dict | None,
    any_age: float | None,
    mutate_age: float | None,
    hook_age: float | None,
    ended: bool,
) -> str:
    if holds_play_lease:
        return "in-play"
    if owns_test_job:
        return "running-tests"
    if parked is not None:
        return "waiting-parked"
    if mutate_age is not None and mutate_age <= ACTIVE_RECENCY_SECONDS:
        return "editing"
    # `active` if EITHER recent MCP traffic OR recent hook activity. Hook events
    # (including a working subagent's) prove the agent is busy even when its own
    # MCP-call clock has gone quiet — without this an agent whose subagents are
    # doing all the work would read 'idle'. Hook recency only feeds `active`, not
    # `editing` (the hook can't tell mutate from read).
    if any_age is not None and any_age <= ACTIVE_RECENCY_SECONDS:
        return "active"
    if hook_age is not None and hook_age <= HOOK_ACTIVE_RECENCY_SECONDS:
        return "active"
    if ended:
        return "disconnected"
    return "idle"


def _derive_intent(agent_status, source_is_full: bool) -> str | None:
    """Most-specific-wins intent line; mcp-only rows have no intent."""
    if not source_is_full or agent_status is None:
        return None
    if agent_status.latest_todo:
        return agent_status.latest_todo
    if agent_status.latest_spawn_description:
        return agent_status.latest_spawn_description
    if agent_status.latest_prompt_head:
        return agent_status.latest_prompt_head
    return None


def build_roster() -> dict[str, Any]:
    """Assemble the full roster envelope. Pure; never raises (fails open)."""
    try:
        return _build_roster_inner()
    except Exception as exc:
        logger.debug("session_roster: build failed open: %r", exc)
        return {
            "type": ROSTER_MESSAGE_TYPE,
            "schema": ROSTER_SCHEMA,
            "generated_at_unix": _now_unix(),
            "sessions": [],
            "play_lease": None,
            "health": {"ok": False, "error": "roster_build_failed"},
        }


def _build_roster_inner() -> dict[str, Any]:
    from services.state.agent_status_store import agent_status_store
    from services.state.call_activity import call_activity
    from services.state.play_lease import play_lease_manager
    from services.state.test_job_lease import test_job_lease_manager
    from services.state.rabbit_hole_flags import FlagInputs, derive_flags
    from transport.plugin_hub import PluginHub
    from transport.unity_instance_middleware import get_unity_instance_middleware

    middleware = get_unity_instance_middleware()
    identities = middleware.all_session_identities()
    store_snapshot = agent_status_store.snapshot()

    # Index hook entries by label for the join; track which got consumed so a
    # hookless harness (no matching MCP identity) still surfaces if needed.
    store_by_label = {
        entry.label: entry
        for entry in store_snapshot.values()
        if entry.label
    }

    # Collect known play leases / test jobs across instances so we can attribute
    # them to a session key. Both are keyed by project_hash; both carry the
    # owning MCP session key (None for the "user" play owner).
    play_leases = list(getattr(play_lease_manager, "_leases", {}).values())
    test_jobs = list(getattr(test_job_lease_manager, "_owned", {}).values())

    play_owner_keys = {
        lease.owner_key for lease in play_leases if lease.owner_key is not None
    }
    test_owner_keys = {job.owner_key for job in test_jobs if job.owner_key}

    sessions_payload: list[dict[str, Any]] = []
    now_unix = _now_unix()
    now_mono = time.monotonic()

    for identity in identities:
        key = identity.key
        label = identity.label
        agent_status = store_by_label.get(label) if label else None
        source_is_full = agent_status is not None

        holds_play_lease = key in play_owner_keys
        owns_test_job = key in test_owner_keys
        parked = call_activity.get_parked(key)
        in_flight = call_activity.get_call(key)
        any_age, mutate_age = call_activity.activity_ages(key)

        ended = bool(agent_status.ended) if agent_status is not None else False

        # Hook activity recency (full-source rows only): the hook clock is
        # refreshed on every hook event, including a working subagent's, so it
        # proves liveness even when this row's own MCP-call clock is quiet.
        hook_age: float | None = None
        if (
            agent_status is not None
            and not ended
            and agent_status.last_activity_unix > 0
        ):
            hook_age = max(0.0, now_unix - agent_status.last_activity_unix)

        state = _derive_state(
            holds_play_lease=holds_play_lease,
            owns_test_job=owns_test_job,
            parked=parked,
            any_age=any_age,
            mutate_age=mutate_age,
            hook_age=hook_age,
            ended=ended,
        )

        intent = _derive_intent(agent_status, source_is_full)

        # Last activity: prefer the hook last-activity (wall clock); else derive
        # from the most recent MCP call age.
        if agent_status is not None and agent_status.last_activity_unix > 0:
            last_activity_unix = agent_status.last_activity_unix
        elif any_age is not None:
            last_activity_unix = now_unix - any_age
        else:
            last_activity_unix = None
        last_activity_age = (
            (now_unix - last_activity_unix) if last_activity_unix is not None else None
        )

        tail = agent_status.tail_as_list() if agent_status is not None else []

        # Attribution flags.
        attribution: dict[str, Any] = {"holds_play_lease": holds_play_lease}
        if holds_play_lease:
            owner_lease = next(
                (lease for lease in play_leases if lease.owner_key == key), None
            )
            if owner_lease is not None:
                attribution["play_lease_owner"] = owner_lease.owner_display
        if parked is not None:
            attribution["parked"] = True
            attribution["parked_on"] = parked.get("reason")
            if parked.get("owner"):
                attribution["parked_owner"] = parked.get("owner")

        # Rabbit-hole flags.
        _, loop_count = middleware.max_repeat_count_in_window(key)
        play_lease_idle = None
        if holds_play_lease:
            owner_lease = next(
                (lease for lease in play_leases if lease.owner_key == key), None
            )
            if owner_lease is not None:
                play_lease_idle = max(
                    0.0, now_mono - owner_lease.last_activity
                )
        activity_recent = (
            state in _ACTIVE_STATES
            or (any_age is not None and any_age <= INTENT_ACTIVITY_RECENCY_SECONDS)
        )
        intent_unchanged = _intent_tracker.unchanged_seconds(key, intent)
        flags = derive_flags(
            FlagInputs(
                loop_repeat_count=loop_count,
                holds_play_lease=holds_play_lease,
                play_lease_idle_seconds=play_lease_idle,
                in_flight_call_age_seconds=(
                    in_flight.get("age_seconds") if in_flight else None
                ),
                parked_age_seconds=(parked.get("age_seconds") if parked else None),
                intent_unchanged_seconds=intent_unchanged,
                activity_recent=activity_recent,
            )
        )

        sessions_payload.append(
            {
                "session_key": key,
                "name": identity.name,
                "label": label,
                "color": identity.color,
                "state": state,
                "intent": intent,
                "last_activity_unix": last_activity_unix,
                "last_activity_age_seconds": last_activity_age,
                "activity_tail": tail,
                "attribution": attribution,
                "flags": flags,
                "source": "full" if source_is_full else "mcp-only",
            }
        )

    # Play-lease envelope summary (banner): the first/most relevant active lease.
    play_lease_summary = None
    if play_leases:
        lease = play_leases[0]
        play_lease_summary = {
            "owner": lease.owner_display,
            "owner_session_key": lease.owner_key,
            "instance": lease.instance_key,
            "acquired_at_unix": lease.acquired_at_unix,
            "since_seconds": max(0.0, now_mono - lease.acquired_at),
        }

    health = {
        "ok": True,
        "bridge_configured": PluginHub.is_configured(),
        "connected_instances": len(getattr(PluginHub, "_connections", {})),
        "session_count": len(sessions_payload),
    }

    return {
        "type": ROSTER_MESSAGE_TYPE,
        "schema": ROSTER_SCHEMA,
        "generated_at_unix": now_unix,
        "sessions": sessions_payload,
        "play_lease": play_lease_summary,
        "health": health,
    }


# ----------------------------------------------------------------------
# Change detection + push
# ----------------------------------------------------------------------
def _roster_fingerprint(roster: dict[str, Any]) -> str:
    """A cheap fingerprint of the roster's meaningful (non-time) content."""
    parts: list[str] = []
    for session in roster.get("sessions", []):
        parts.append(
            "|".join(
                str(session.get(field))
                for field in ("session_key", "state", "intent", "source")
            )
            + "#"
            + ",".join(session.get("flags", []))
            + "#"
            + ",".join(
                f"{k}={v}" for k, v in sorted(session.get("attribution", {}).items())
            )
        )
    lease = roster.get("play_lease") or {}
    parts.append(f"lease={lease.get('owner')}@{lease.get('instance')}")
    parts.append(f"n={(roster.get('health') or {}).get('session_count', 0)}")
    return "||".join(parts)


class RosterPublisher:
    """Builds and pushes the roster on change + heartbeat. Process-local."""

    def __init__(self) -> None:
        self._last_fingerprint: str | None = None
        self._last_push_mono: float = 0.0
        self._lock = RLock()

    def reset(self) -> None:
        with self._lock:
            self._last_fingerprint = None
            self._last_push_mono = 0.0
        _intent_tracker.reset()

    async def maybe_push(self, force: bool = False) -> bool:
        """Build the roster and push it if changed or the heartbeat elapsed.

        Returns True if a push was sent. Fails open: any error is swallowed and
        reported as "not sent".
        """
        try:
            roster = build_roster()
            fingerprint = _roster_fingerprint(roster)
            now = time.monotonic()
            with self._lock:
                changed = fingerprint != self._last_fingerprint
                heartbeat_due = (now - self._last_push_mono) >= _heartbeat_s()
                if not force and not changed and not heartbeat_due:
                    return False
                self._last_fingerprint = fingerprint
                self._last_push_mono = now
            return await _push_roster(roster)
        except Exception as exc:
            logger.debug("session_roster: maybe_push failed open: %r", exc)
            return False


roster_publisher = RosterPublisher()


async def _push_roster(roster: dict[str, Any]) -> bool:
    """Send the roster to every connected bridge WebSocket. Fails open."""
    from transport.plugin_hub import PluginHub

    if not PluginHub.is_configured():
        return False
    connections = dict(getattr(PluginHub, "_connections", {}))
    if not connections:
        return False
    sent_any = False
    for session_id, websocket in connections.items():
        try:
            await websocket.send_json(roster)
            sent_any = True
        except Exception as exc:
            logger.debug(
                "session_roster: push to bridge session %s failed: %r",
                session_id,
                exc,
            )
    return sent_any


async def roster_heartbeat_loop() -> None:
    """Periodic roster push loop (started from the server lifespan).

    Pushes on the heartbeat cadence; the change-detection path inside
    ``maybe_push`` makes off-beat changes cheap no-ops. Runs until cancelled.
    """
    interval = _heartbeat_s()
    logger.info("session_roster: heartbeat loop started (%.1fs)", interval)
    try:
        while True:
            await _async_sleep(interval)
            await roster_publisher.maybe_push()
    except Exception as exc:
        if _is_cancel(exc):
            raise
        logger.debug("session_roster: heartbeat loop error: %r", exc)


async def _async_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def _is_cancel(exc: BaseException) -> bool:
    import asyncio

    return isinstance(exc, asyncio.CancelledError)
