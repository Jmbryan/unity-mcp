"""Play-session lease (MCPC-012..018).

At most one play lease exists per Unity instance, keyed by registered
project identity (project_hash) — never the per-connection WebSocket
session, so a domain-reload reconnect keeps the lease alive. The lease is
acquired implicitly by the session whose gated tool call successfully
enters play mode; no agent ever requests or releases a lease explicitly.
A play session entered with no MCP cause (the human pressed Play, observed
as a play transition in the shared editor-state snapshot with no
corresponding gated call) is leased to owner ``"user"`` with identical
protections (MCPC-016).

While a lease is active:

- play-scoped calls (play/pause/stop, synthetic input, UI/test/QA drivers,
  camera control) execute only for the lease owner (MCPC-014, MCPC-015);
- exclusive (compile-class) calls from non-owners receive a structured busy
  result naming the owner (MCPC-013);
- reads and plain mutations pass for every session.

Liveness never wedges (MCPC-017): the lease carries an inactivity-renewed
TTL (renewed by the owner's gated calls and by editor-state observations of
play still running), and refusing a non-owner first validates the lease
with a fast-fail editor-state probe — if play already ended, the lease
self-clears and the call passes (validate-on-block).

Lease state lives in server memory mirrored to
``<project>/Library/MCPForUnity/RunState/play_lease.json`` (MCPC-018) and
fails open everywhere: mirror IO never fails a call, a server restart
forgets all leases, and an instance disconnect expires its lease after a
short reconnect grace window — worst case one unguarded window, never a
deadlock.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from threading import RLock
from typing import Any

logger = logging.getLogger(__name__)

# Concurrency classes that are owner-only while a lease is active. String
# literals match the gate's declared classes (services.registry
# CONCURRENCY_CLASSES); kept literal here to avoid an import cycle with
# operation_gate.
_OWNER_ONLY_CLASSES: frozenset[str] = frozenset({"play-scoped", "exclusive"})

# Inactivity-renewed lease TTL (seconds). Renewed by owner gated calls and
# by editor-state observations showing play still active, so it only fires
# when both the owner and the editor-state stream have gone quiet.
DEFAULT_PLAY_LEASE_TTL_SECONDS = 300.0
MAX_PLAY_LEASE_TTL_SECONDS = 3600.0

# How long a play intent (a gated call about to enter play mode) stays
# valid for lease attribution. Spans the play-enter domain reload.
INTENT_TTL_SECONDS = 30.0

# Grace window after the owning instance disconnects before the lease
# expires. A domain-reload reconnect re-registers the same project_hash
# well inside this window; a genuinely closed editor frees the lease at the
# boundary (fail open).
DISCONNECT_GRACE_SECONDS = 30.0

# Owner display for play sessions with no MCP cause (MCPC-016).
USER_OWNER_DISPLAY = "user"

# RunState mirror location relative to the Unity project root. Same
# directory convention as the bridge's PID files and the shared edit ledger.
MIRROR_RELATIVE_PATH = "Library/MCPForUnity/RunState/play_lease.json"


def _play_lease_ttl_s() -> float:
    raw = os.environ.get("UNITY_MCP_PLAY_LEASE_TTL_S")
    if raw is None:
        return DEFAULT_PLAY_LEASE_TTL_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid UNITY_MCP_PLAY_LEASE_TTL_S=%r, using default %.1f",
            raw,
            DEFAULT_PLAY_LEASE_TTL_SECONDS,
        )
        return DEFAULT_PLAY_LEASE_TTL_SECONDS
    return max(1.0, min(value, MAX_PLAY_LEASE_TTL_SECONDS))


def instance_key(unity_instance: str | None) -> str:
    """Lease key for an instance reference: bare project_hash when present.

    Accepts ``Name@hash``, a bare hash, or None (keys as ``default``, like
    the editor-state cache).
    """
    if not unity_instance:
        return "default"
    if "@" in unity_instance:
        _, _, suffix = unity_instance.rpartition("@")
        return suffix or "default"
    return unity_instance


@dataclass
class PlayLease:
    instance_key: str
    owner_key: str | None  # MCP session key; None for the "user" owner
    owner_display: str
    acquired_at: float  # time.monotonic()
    last_activity: float  # time.monotonic()
    acquired_at_unix: float  # time.time()
    last_activity_unix: float  # time.time()
    project_root: str | None = None  # resolved once for the RunState mirror
    disconnected_at: float | None = None  # time.monotonic(); reconnect grace


@dataclass
class _PlayIntent:
    session_key: str
    display_name: str
    recorded_at: float  # time.monotonic()


class PlayLeaseManager:
    """Process-local play leases, keyed by project_hash."""

    def __init__(self) -> None:
        self._leases: dict[str, PlayLease] = {}
        self._intents: dict[str, _PlayIntent] = {}
        # Instances whose fall-through "user" acquisition is suppressed after a
        # human force-release: the override must stick until play actually
        # exits or a real MCP cause (intent / explicit acquire) appears.
        self._user_suppressed: dict[str, float] = {}
        self._lock = RLock()

    def reset(self) -> None:
        """Clear all lease state (test seam)."""
        with self._lock:
            self._leases.clear()
            self._intents.clear()
            self._user_suppressed.clear()

    # ------------------------------------------------------------------
    # Core lease operations
    # ------------------------------------------------------------------
    def get_active_lease(self, unity_instance: str | None) -> PlayLease | None:
        """The unexpired lease for an instance, applying TTL and grace expiry."""
        key = instance_key(unity_instance)
        with self._lock:
            lease = self._leases.get(key)
            if lease is None:
                return None
            now = time.monotonic()
            if (now - lease.last_activity) > _play_lease_ttl_s():
                self._release_locked(key, lease, reason="ttl_expired")
                return None
            if (
                lease.disconnected_at is not None
                and (now - lease.disconnected_at) > DISCONNECT_GRACE_SECONDS
            ):
                self._release_locked(key, lease, reason="instance_disconnected")
                return None
            return lease

    def peek_lease(self, unity_instance: str | None) -> PlayLease | None:
        """The instance's lease, applying only disconnect-grace expiry (not TTL).

        Used by the observation path: a snapshot proving play is still active
        renews the lease regardless of how stale its inactivity clock is, so
        TTL expiry can never flip a mid-play lease to the fall-through "user"
        owner. Disconnect grace still applies — an editor that vanished frees
        its lease at the boundary.
        """
        key = instance_key(unity_instance)
        with self._lock:
            lease = self._leases.get(key)
            if lease is None:
                return None
            if (
                lease.disconnected_at is not None
                and (time.monotonic() - lease.disconnected_at)
                > DISCONNECT_GRACE_SECONDS
            ):
                self._release_locked(key, lease, reason="instance_disconnected")
                return None
            return lease

    def active_leases(self) -> list[PlayLease]:
        """Expiry-aware snapshot of all leases (pure read; never releases).

        Applies the same TTL and disconnect-grace filters as
        :meth:`get_active_lease` without mutating state, so a roster/banner
        view cannot release a lease out from under the enforcement and
        observation paths.
        """
        now = time.monotonic()
        ttl = _play_lease_ttl_s()
        with self._lock:
            leases = list(self._leases.values())
        surviving: list[PlayLease] = []
        for lease in leases:
            if (now - lease.last_activity) > ttl:
                continue
            if (
                lease.disconnected_at is not None
                and (now - lease.disconnected_at) > DISCONNECT_GRACE_SECONDS
            ):
                continue
            surviving.append(lease)
        return surviving

    def acquire(
        self,
        unity_instance: str | None,
        owner_key: str | None,
        owner_display: str,
        project_root: str | None = None,
    ) -> PlayLease:
        """Acquire (or renew, for the same owner) the instance's lease.

        Never steals between two real MCP owners: when another session's lease
        is still active, that lease is returned untouched. An *inferred*
        ``"user"`` lease (``owner_key is None`` — the fall-through guess made
        when no MCP cause was known) is not protected the same way: a caller
        with a proven session identity corrects it in place, so a single
        mis-attribution never outlives the arrival of the real owner.
        """
        key = instance_key(unity_instance)
        with self._lock:
            existing = self.get_active_lease(unity_instance)
            if existing is not None:
                if existing.owner_key is not None and existing.owner_key == owner_key:
                    self.touch(unity_instance)
                elif existing.owner_key is None and owner_key is not None:
                    self._reattribute_locked(
                        existing, owner_key, owner_display, project_root
                    )
                    self.touch(unity_instance)
                return existing
            now_mono = time.monotonic()
            now_wall = time.time()
            lease = PlayLease(
                instance_key=key,
                owner_key=owner_key,
                owner_display=owner_display or USER_OWNER_DISPLAY,
                acquired_at=now_mono,
                last_activity=now_mono,
                acquired_at_unix=now_wall,
                last_activity_unix=now_wall,
                project_root=project_root,
            )
            self._leases[key] = lease
            self._intents.pop(key, None)
            if owner_key is not None:
                # A real MCP cause supersedes a pending force-release override.
                self._user_suppressed.pop(key, None)
            logger.info(
                "Play lease acquired for instance %s by '%s'", key, lease.owner_display
            )
            self._write_mirror(lease, active=True)
            return lease

    def _reattribute_locked(
        self,
        lease: PlayLease,
        owner_key: str,
        owner_display: str,
        project_root: str | None = None,
    ) -> None:
        """Rewrite an inferred-"user" lease's owner in place (caller holds lock)."""
        previous = lease.owner_display
        lease.owner_key = owner_key
        lease.owner_display = owner_display or owner_key
        if project_root and not lease.project_root:
            lease.project_root = project_root
        self._user_suppressed.pop(lease.instance_key, None)
        logger.info(
            "Play lease for instance %s re-attributed from '%s' to '%s'",
            lease.instance_key,
            previous,
            lease.owner_display,
        )
        self._write_mirror(lease, active=True)

    def release(self, unity_instance: str | None, reason: str = "released") -> None:
        key = instance_key(unity_instance)
        with self._lock:
            lease = self._leases.get(key)
            if lease is not None:
                self._release_locked(key, lease, reason=reason)

    def force_release(
        self, instance_or_hash: str | None, reason: str = "force_release"
    ) -> bool:
        """Clear a lease for an instance regardless of owner (human override).

        Unlike :meth:`get_active_lease`, this inspects the raw lease map so a
        wedged lease whose TTL/grace would otherwise gate its visibility is
        still cleared. Also discards any pending play intent for the instance
        and suppresses fall-through ``"user"`` re-acquisition until play mode
        actually exits or a real MCP cause arrives — without that, the next
        editor-state observation of the still-running play session would
        immediately re-manufacture the lease the human just cleared.
        Returns True when a lease was present and cleared, False when there was
        nothing to clear. Fails open: never raises.
        """
        key = instance_key(instance_or_hash)
        with self._lock:
            self._intents.pop(key, None)
            self._user_suppressed[key] = time.monotonic()
            lease = self._leases.get(key)
            if lease is None:
                return False
            self._release_locked(key, lease, reason=reason)
            return True

    def _release_locked(self, key: str, lease: PlayLease, reason: str) -> None:
        self._leases.pop(key, None)
        logger.info(
            "Play lease for instance %s ('%s') released: %s",
            key,
            lease.owner_display,
            reason,
        )
        self._write_mirror(lease, active=False, reason=reason)

    def touch(self, unity_instance: str | None) -> None:
        """Renew the lease's inactivity TTL."""
        key = instance_key(unity_instance)
        with self._lock:
            lease = self._leases.get(key)
            if lease is not None:
                lease.last_activity = time.monotonic()
                lease.last_activity_unix = time.time()

    # ------------------------------------------------------------------
    # Play intents (MCP cause attribution across the play-enter reload)
    # ------------------------------------------------------------------
    def record_play_intent(
        self,
        unity_instance: str | None,
        session_key: str,
        display_name: str,
    ) -> None:
        """Record that a session's gated call is about to enter play mode."""
        if not session_key:
            return
        with self._lock:
            self._intents[instance_key(unity_instance)] = _PlayIntent(
                session_key=session_key,
                display_name=display_name or session_key,
                recorded_at=time.monotonic(),
            )

    def clear_play_intent(self, unity_instance: str | None, session_key: str) -> None:
        """Drop a session's pending intent (its play call definitively failed)."""
        key = instance_key(unity_instance)
        with self._lock:
            intent = self._intents.get(key)
            if intent is not None and intent.session_key == session_key:
                self._intents.pop(key, None)

    def _take_intent(self, key: str) -> _PlayIntent | None:
        with self._lock:
            intent = self._intents.pop(key, None)
        if intent is None:
            return None
        if (time.monotonic() - intent.recorded_at) > INTENT_TTL_SECONDS:
            return None
        return intent

    # ------------------------------------------------------------------
    # Instance lifecycle (registration / disconnect)
    # ------------------------------------------------------------------
    def mark_instance_disconnected(self, project_hash: str | None) -> None:
        """Start the reconnect grace window for an instance's lease."""
        key = instance_key(project_hash)
        with self._lock:
            lease = self._leases.get(key)
            if lease is not None and lease.disconnected_at is None:
                lease.disconnected_at = time.monotonic()

    def mark_instance_registered(self, project_hash: str | None) -> None:
        """The instance (re)registered: cancel any pending grace expiry."""
        key = instance_key(project_hash)
        with self._lock:
            lease = self._leases.get(key)
            if lease is not None:
                lease.disconnected_at = None

    # ------------------------------------------------------------------
    # Editor-state observation (MCPC-016 + natural play exit)
    # ------------------------------------------------------------------
    async def observe_editor_state(
        self,
        unity_instance: str | None,
        state: dict[str, Any] | None,
    ) -> None:
        """React to a fresh shared editor-state snapshot.

        Play active with a lease: renew it (bypassing TTL expiry — a snapshot
        proving play is still running is renewal, so an inactivity lapse can
        never flip a mid-play lease to "user"); an inferred-"user" lease is
        re-attributed when a pending intent names the real cause. Play active
        with no lease and the transition settled: assign ownership from the
        pending play intent, else the active test job's owner (a PlayMode test
        run is an MCP cause), else ``"user"`` (the human pressed Play) unless
        a force-release suppressed the fall-through. A transitional snapshot
        (``is_changing``) never mints a lease — exit/enter windows carry no
        new ownership information. Play explicitly ended: clear the lease.
        """
        try:
            if not isinstance(state, dict):
                return
            play_mode = (state.get("editor") or {}).get("play_mode") or {}
            is_playing = play_mode.get("is_playing")
            is_changing = play_mode.get("is_changing")
            key = instance_key(unity_instance)

            if is_playing is True:
                lease = self.peek_lease(unity_instance)
                if lease is not None:
                    if lease.owner_key is None:
                        intent = self._take_intent(key)
                        if intent is not None:
                            with self._lock:
                                self._reattribute_locked(
                                    lease, intent.session_key, intent.display_name
                                )
                    self.touch(unity_instance)
                    return
                if is_changing is True:
                    # Enter/exit transition window: renewing above is fine,
                    # but never infer NEW ownership from a transitional
                    # snapshot (the pending intent is left for the settled
                    # one).
                    return
                intent = self._take_intent(key)
                if intent is not None:
                    owner_key: str | None = intent.session_key
                    owner_display = intent.display_name
                else:
                    job_owner = _test_job_owner(unity_instance, state)
                    if job_owner is not None:
                        owner_key, owner_display = job_owner
                    else:
                        with self._lock:
                            suppressed = key in self._user_suppressed
                        if suppressed:
                            return
                        owner_key = None
                        owner_display = USER_OWNER_DISPLAY
                project_root = await _resolve_project_root(unity_instance)
                self.acquire(
                    unity_instance, owner_key, owner_display, project_root=project_root
                )
            elif is_playing is False and is_changing is not True:
                with self._lock:
                    # Play has exited: a force-release override has run its
                    # course; the next play session attributes normally.
                    self._user_suppressed.pop(key, None)
                if self.get_active_lease(unity_instance) is not None:
                    self.release(unity_instance, reason="play_exited")
        except Exception as exc:
            logger.debug("play_lease: editor-state observation skipped: %r", exc)

    # ------------------------------------------------------------------
    # RunState mirror (MCPC-018)
    # ------------------------------------------------------------------
    def _write_mirror(
        self,
        lease: PlayLease,
        active: bool,
        reason: str | None = None,
    ) -> None:
        """Mirror lease state into the project's RunState directory.

        Fails open: a missing or unwritable project directory never fails
        the lease operation it mirrors.
        """
        if not lease.project_root:
            return
        try:
            payload: dict[str, Any] = {
                "schema": "unity-mcp/play_lease@1",
                "active": active,
                "instance": lease.instance_key,
                "owner": lease.owner_display,
                "owner_session_key": lease.owner_key,
                "acquired_at_unix": lease.acquired_at_unix,
                "last_activity_unix": lease.last_activity_unix,
                "ttl_seconds": _play_lease_ttl_s(),
                "updated_at_unix": time.time(),
            }
            if reason:
                payload["released_reason"] = reason
            path = os.path.normpath(
                os.path.join(lease.project_root, MIRROR_RELATIVE_PATH)
            ).replace("\\", "/")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.debug("play_lease: RunState mirror write failed (fail-open): %r", exc)


# Global singleton (simple, process-local) — same pattern as
# editor_state_cache.
play_lease_manager = PlayLeaseManager()


# ----------------------------------------------------------------------
# Force-release HTTP control (human dashboard override)
# ----------------------------------------------------------------------
async def handle_lease_release_post(request) -> tuple[dict[str, Any], int]:
    """Force-clear a wedged play lease regardless of owner.

    Accepts a JSON body ``{"instance": "<project_hash or instance id>"}``;
    ``project_hash`` is tolerated as an alias key. A missing or unreadable
    body clears nothing. The lease for the resolved instance is force-cleared
    in the manager and its RunState mirror updated to inactive with
    ``released_reason="force_release"``.

    Always returns ``({"released": bool, "instance": <resolved>}, 200)``. Fails
    open: any internal error yields ``released: False`` with a 200 — never a
    500, never raises.
    """
    instance: Any = None
    try:
        try:
            body = await request.json()
        except Exception:
            body = None
        if isinstance(body, dict):
            instance = body.get("instance")
            if instance is None:
                instance = body.get("project_hash")
        if not isinstance(instance, str) or not instance:
            return {"released": False, "instance": instance}, 200
        released = play_lease_manager.force_release(instance, reason="force_release")
        return {"released": bool(released), "instance": instance}, 200
    except Exception as exc:
        logger.debug("play_lease: force-release failed open: %r", exc)
        return {"released": False, "instance": instance}, 200


# ----------------------------------------------------------------------
# Helpers shared by the module-level async API
# ----------------------------------------------------------------------
def _test_job_owner(
    unity_instance: str | None,
    state: dict[str, Any],
) -> tuple[str, str] | None:
    """(owner_key, owner_display) of the active test job while tests run.

    A play session observed during a snapshot whose ``tests.is_running`` is
    true belongs to whoever started the run (a PlayMode test run enters play
    mode), not to the human — the test-job ownership store carries that
    session even when the shorter-lived play intent has expired across the
    run's domain reload. Fails open (None) on any lookup error.
    """
    try:
        tests = state.get("tests") or {}
        if tests.get("is_running") is not True:
            return None
        from services.state.test_job_lease import test_job_lease_manager

        owned = test_job_lease_manager.get_active(unity_instance)
        if owned is not None and owned.owner_key:
            return owned.owner_key, owned.owner_display
    except Exception as exc:
        logger.debug("play_lease: test-job owner lookup failed (fail-open): %r", exc)
    return None


async def _resolve_project_root(unity_instance: str | None) -> str | None:
    """Project root for the RunState mirror; None fails open (no mirror)."""
    try:
        from services.state import edit_ledger

        return await edit_ledger.resolve_project_root(unity_instance)
    except Exception as exc:
        logger.debug("play_lease: project-root resolution failed (fail-open): %r", exc)
        return None


async def _session_key_and_display(ctx) -> tuple[str | None, str | None]:
    try:
        from transport.unity_instance_middleware import get_unity_instance_middleware

        middleware = get_unity_instance_middleware()
        key = await middleware.get_session_key(ctx)
        identity = middleware.ensure_session_identity_for_key(key)
        return key, identity.display_name
    except Exception as exc:
        logger.debug("play_lease: session identity lookup failed (fail-open): %r", exc)
        return None, None


async def _ctx_unity_instance(ctx) -> str | None:
    try:
        get_state = getattr(ctx, "get_state", None)
        if callable(get_state):
            value = await get_state("unity_instance")
            if isinstance(value, str) and value:
                return value
    except Exception:
        pass
    return None


def _manage_editor_action(tool_name: str | None, arguments: Any) -> str | None:
    if tool_name != "manage_editor" or not isinstance(arguments, dict):
        return None
    action = arguments.get("action")
    if isinstance(action, str):
        return action.strip().lower()
    return None


def _result_success(result: Any) -> bool | None:
    """True/False when the tool result's success is determinable, else None."""
    payload = getattr(result, "structured_content", None)
    if not isinstance(payload, dict):
        payload = result if isinstance(result, dict) else None
    if payload is None:
        return None
    success = payload.get("success")
    if isinstance(success, bool):
        return success
    return None


def _result_retry_hinted(result: Any) -> bool:
    payload = getattr(result, "structured_content", None)
    if not isinstance(payload, dict):
        payload = result if isinstance(result, dict) else None
    if payload is None:
        return False
    return payload.get("hint") == "retry"


async def _probe_play_active(unity_instance: str | None) -> bool | None:
    """Fast-fail editor-state probe for validate-on-block.

    Returns True (still in play mode), False (definitively not playing), or
    None when the probe is inconclusive (no transport, fast-fail timeout,
    unparseable payload) — inconclusive keeps the lease, the TTL backstops.
    """
    try:
        from models.unity_response import normalize_unity_response
        from transport.plugin_hub import PluginHub

        registry = PluginHub._registry
        if registry is None or not PluginHub.is_configured():
            return None
        key = instance_key(unity_instance)
        session_id: str | None = None
        if key != "default":
            session_id = await registry.get_session_id_by_hash(key)
        else:
            try:
                sessions = await registry.list_sessions()
            except Exception:
                sessions = {}
            if len(sessions) == 1:
                session_id = next(iter(sessions.keys()))
        if not session_id:
            # Mid-reload window or editor gone; the disconnect grace and TTL
            # decide, not this probe.
            return None
        raw = await PluginHub.send_command(session_id, "get_editor_state", {})
        normalized = normalize_unity_response(raw)
        if not isinstance(normalized, dict) or normalized.get("success") is not True:
            return None
        data = normalized.get("data")
        if not isinstance(data, dict):
            return None
        play_mode = (data.get("editor") or {}).get("play_mode") or {}
        if play_mode.get("is_changing") is True:
            return True
        is_playing = play_mode.get("is_playing")
        if isinstance(is_playing, bool):
            return is_playing
        return None
    except Exception as exc:
        logger.debug("play_lease: validate-on-block probe failed (fail-open): %r", exc)
        return None


def _busy_response(lease: PlayLease, tool_name: str) -> dict[str, Any]:
    from transport.plugin_hub import PluginHub

    owner = lease.owner_display
    message = (
        f"'{tool_name}' refused: an active play session is leased to {owner}. "
        "Play-scoped and compile-class calls are owner-only while play mode "
        "is active; retry after the play session ends."
    )
    return PluginHub._unavailable_retry_response(
        "play_lease", owner=owner, message=message, retry_after_ms=2000
    )


# ----------------------------------------------------------------------
# Gate enforcement (MCPC-013/014/015/017)
# ----------------------------------------------------------------------
async def enforce_lease(
    ctx,
    concurrency_class: str,
    tool_name: str,
    unity_instance: str | None,
) -> dict[str, Any] | None:
    """Owner-only enforcement for one gated call.

    Returns None when the call may proceed (no active lease, caller is the
    owner, the class is not owner-only, or the lease self-cleared via
    validate-on-block), or a structured busy payload naming the owner.
    Every internal failure fails open.
    """
    try:
        lease = play_lease_manager.get_active_lease(unity_instance)
        if lease is None:
            return None

        session_key, _ = await _session_key_and_display(ctx)
        if (
            lease.owner_key is not None
            and session_key is not None
            and session_key == lease.owner_key
        ):
            play_lease_manager.touch(unity_instance)
            return None

        if concurrency_class not in _OWNER_ONLY_CLASSES:
            return None

        # Validate-on-block (MCPC-017): a refused call first probes the
        # editor; if play already ended, the lease self-clears and the call
        # passes.
        still_playing = await _probe_play_active(unity_instance)
        if still_playing is False:
            play_lease_manager.release(unity_instance, reason="validate_on_block")
            return None

        return _busy_response(lease, tool_name)
    except Exception as exc:
        logger.debug("play_lease: enforcement failed open for '%s': %r", tool_name, exc)
        return None


# ----------------------------------------------------------------------
# Implicit acquisition (MCPC-012) — wired around gated tool dispatch
# ----------------------------------------------------------------------
async def note_play_call_dispatch(ctx, tool_name: str | None, arguments: Any) -> None:
    """Record a play intent when a gated play-enter call is dispatched.

    The intent attributes the upcoming play transition to this session even
    when the call's own result is lost to the play-enter domain reload.
    Never raises.
    """
    try:
        if _manage_editor_action(tool_name, arguments) != "play":
            return
        session_key, display_name = await _session_key_and_display(ctx)
        if not session_key:
            return
        unity_instance = await _ctx_unity_instance(ctx)
        play_lease_manager.record_play_intent(
            unity_instance, session_key, display_name or session_key
        )
    except Exception as exc:
        logger.debug("play_lease: intent recording skipped: %r", exc)


async def record_play_intent_for_session(ctx, unity_instance: str | None) -> None:
    """Attribute an observed play transition to the calling session.

    Used for calls that cannot be classified by inspection (arbitrary-code
    execution, wrapper-dispatched play-enters) when a play edge is detected
    after the fact. Never raises.
    """
    try:
        session_key, display_name = await _session_key_and_display(ctx)
        if not session_key:
            return
        play_lease_manager.record_play_intent(
            unity_instance, session_key, display_name or session_key
        )
    except Exception as exc:
        logger.debug("play_lease: post-hoc intent recording skipped: %r", exc)


async def clear_play_intent_for_session(ctx, unity_instance: str | None) -> None:
    """Drop the calling session's pending play intent.

    Used to undo a speculatively recorded intent once inspection shows the
    call was not a play edge after all. Never raises.
    """
    try:
        session_key, _ = await _session_key_and_display(ctx)
        if not session_key:
            return
        play_lease_manager.clear_play_intent(unity_instance, session_key)
    except Exception as exc:
        logger.debug("play_lease: intent clear skipped: %r", exc)


async def observe_play_call_result(
    ctx,
    tool_name: str | None,
    arguments: Any,
    result: Any,
) -> None:
    """Observe a play/pause/stop call's outcome and update the lease.

    A successful play acquires (or renews) the lease for the calling
    session; a successful stop by the owner releases it; a definitive play
    failure drops the pending intent (retry-hinted transport failures keep
    it — the play-enter reload eats the result of a call that succeeded).
    Never raises.
    """
    try:
        action = _manage_editor_action(tool_name, arguments)
        if action not in ("play", "pause", "stop"):
            return
        session_key, display_name = await _session_key_and_display(ctx)
        if not session_key:
            return
        unity_instance = await _ctx_unity_instance(ctx)
        success = _result_success(result)

        if action == "play":
            if success is True:
                project_root = await _resolve_project_root(unity_instance)
                play_lease_manager.acquire(
                    unity_instance,
                    session_key,
                    display_name or session_key,
                    project_root=project_root,
                )
            elif success is False and not _result_retry_hinted(result):
                play_lease_manager.clear_play_intent(unity_instance, session_key)
            return

        lease = play_lease_manager.get_active_lease(unity_instance)
        if lease is None or lease.owner_key != session_key:
            return
        if action == "stop" and success is True:
            play_lease_manager.release(unity_instance, reason="owner_stopped")
        elif action == "pause" and success is True:
            play_lease_manager.touch(unity_instance)
    except Exception as exc:
        logger.debug("play_lease: result observation skipped: %r", exc)
