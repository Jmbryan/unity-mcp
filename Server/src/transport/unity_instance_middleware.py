"""
Middleware for managing Unity instance selection per session.

This middleware intercepts all tool calls and injects the active Unity instance
into the request-scoped state, allowing tools to access it via ctx.get_state("unity_instance").
"""
from collections import deque
from dataclasses import dataclass, field
from threading import RLock
import hashlib
import json
import logging
import os
import time

from fastmcp.server.middleware import Middleware, MiddlewareContext

from core.config import config
from core.constants import AGENT_LABEL_HEADER
from services.registry import get_registered_tools
from transport.plugin_hub import PluginHub

logger = logging.getLogger("mcp-for-unity-server")
# Separate logger that propagates to root -> stderr so diagnostics show in console
_diag = logging.getLogger("transport.unity_instance_middleware")

# Store a global reference to the middleware instance so tools can interact
# with it to set or clear the active unity instance.
_unity_instance_middleware = None
_middleware_lock = RLock()


def get_unity_instance_middleware() -> 'UnityInstanceMiddleware':
    """Get the global Unity instance middleware."""
    global _unity_instance_middleware
    if _unity_instance_middleware is None:
        with _middleware_lock:
            if _unity_instance_middleware is None:
                # Auto-initialize if not set (lazy singleton) to handle import order or test cases
                _unity_instance_middleware = UnityInstanceMiddleware()

    return _unity_instance_middleware


def set_unity_instance_middleware(middleware: 'UnityInstanceMiddleware') -> None:
    """Replace the global middleware instance.

    This is a test seam: production code uses ``get_unity_instance_middleware()``
    which lazy-initialises the singleton.  Tests call this function to inject a
    mock or pre-configured middleware before exercising tool/resource code.
    """
    global _unity_instance_middleware
    _unity_instance_middleware = middleware


# Pools for auto-assigned per-session identity. Names cycle with a numeric
# suffix once exhausted; colors simply cycle.
_FRIENDLY_NAMES: tuple[str, ...] = (
    "Aurora", "Basalt", "Cobalt", "Dune", "Ember", "Fjord", "Garnet",
    "Harbor", "Indigo", "Juniper", "Kestrel", "Lumen", "Maple", "Nimbus",
    "Onyx", "Pumice", "Quartz", "Rowan", "Sable", "Tundra", "Umber",
    "Vesper", "Willow", "Xenon", "Yarrow", "Zephyr",
)
_IDENTITY_COLORS: tuple[str, ...] = (
    "#E06C75", "#61AFEF", "#98C379", "#E5C07B",
    "#C678DD", "#56B6C2", "#D19A66", "#7F9F7F",
)

# Loop-detection window (seconds) for the `looping` rabbit-hole flag (MCPC-032:
# "same tool + args-hash >=5x in 10 min"). Env-overridable; the ring buffer is
# bounded so a busy session never grows it without limit.
_DEFAULT_LOOP_WINDOW_SECONDS = 600.0
_LOOP_RING_MAXLEN = 200


def _loop_window_s() -> float:
    raw = os.environ.get("UNITY_MCP_LOOP_WINDOW_S")
    if raw is None:
        return _DEFAULT_LOOP_WINDOW_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_LOOP_WINDOW_SECONDS
    return max(1.0, value)


# Session-identity inactivity TTL (seconds): an identity not seen on any request
# within this window is evicted so a hard-killed console (which never fires a
# SessionEnd hook) stops occupying a dashboard row. Mirrors the agent-status
# store's TTL/grace pair (DEFAULT_AGENT_TTL_SECONDS / ENDED_GRACE_SECONDS). The
# grace window keeps the row visible briefly as 'disconnected' before it drops.
_DEFAULT_IDENTITY_TTL_SECONDS = 900.0
_IDENTITY_TTL_GRACE_SECONDS = 30.0
_MAX_IDENTITY_TTL_SECONDS = 86400.0


def _identity_ttl_s() -> float:
    raw = os.environ.get("UNITY_MCP_IDENTITY_TTL_S")
    if raw is None:
        return _DEFAULT_IDENTITY_TTL_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_IDENTITY_TTL_SECONDS
    return max(0.01, min(value, _MAX_IDENTITY_TTL_SECONDS))


@dataclass
class SessionIdentity:
    """Sticky identity for one MCP session.

    Assigned on first sight of a session key. The optional ``label`` is the
    human-meaningful agent label supplied via the ``X-Agent-Label`` header;
    labeled sessions display it, unlabeled sessions fall back to the
    auto-assigned friendly name.

    ``last_seen`` is a monotonic timestamp refreshed on every access that proves
    the session is live (a request passing through the middleware). It drives
    TTL eviction so a hard-killed console's dashboard row self-heals without any
    SessionEnd hook — the row reaches 'disconnected' during the grace window and
    then disappears once the TTL elapses.
    """

    key: str
    name: str
    color: str
    label: str | None = None
    last_seen: float = field(default_factory=time.monotonic)

    @property
    def display_name(self) -> str:
        return self.label or self.name


class UnityInstanceMiddleware(Middleware):
    """
    Middleware that manages per-session Unity instance selection.

    Stores active instance per session_id and injects it into request state
    for all tool and resource calls.
    """

    def __init__(self):
        super().__init__()
        self._active_by_key: dict[str, str] = {}
        # Sticky per-session identity (name/color/label), keyed like
        # _active_by_key. Never evicted: lives for the server process lifetime.
        self._identity_by_key: dict[str, SessionIdentity] = {}
        self._identity_counter = 0
        # Recent tool-call signatures per session key, for the rabbit-hole
        # `looping` flag (MCPC-032): (signature, monotonic_ts) pairs, bounded.
        self._recent_tool_calls: dict[str, deque[tuple[str, float]]] = {}
        self._lock = RLock()
        self._metadata_lock = RLock()
        self._unity_managed_tool_names: set[str] = set()
        self._tool_alias_to_unity_target: dict[str, str] = {}
        self._server_only_tool_names: set[str] = set()
        self._tool_visibility_signature: tuple[tuple[str, str], ...] = ()
        self._last_tool_visibility_refresh = 0.0
        self._tool_visibility_refresh_interval_seconds = 0.5
        self._has_logged_empty_registry_warning = False

    async def get_session_key(self, ctx) -> str:
        """
        Derive a stable key for the calling session.

        Prefers the direct ``ctx.session_id`` property: over streamable HTTP
        it is the only populated identity — unique per client session — while
        ``ctx.request_context.session_id`` and ``client_id`` are None. All
        per-session state must key through this method so concurrent agent
        sessions never collapse onto a shared key.

        Order: session_id, then client_id, then user_id (remote-hosted
        isolation). The literal 'global' fallback is reserved for stdio,
        where a single client owns the process.
        """
        try:
            session_id = getattr(ctx, "session_id", None)
        except Exception:
            # fastmcp raises RuntimeError outside a request context.
            session_id = None
        if isinstance(session_id, str) and session_id:
            return session_id

        client_id = getattr(ctx, "client_id", None)
        if isinstance(client_id, str) and client_id:
            return client_id

        # In remote-hosted mode, use user_id so different users get isolated instance selections
        user_id = await ctx.get_state("user_id")
        if isinstance(user_id, str) and user_id:
            return f"user:{user_id}"

        transport = (config.transport_mode or "stdio").lower()
        if transport != "http":
            # stdio: one client per process, a shared key is correct.
            return "global"

        # An HTTP session with no identity should not occur (streamable HTTP
        # always carries a session id). Keep such sessions isolated from the
        # stdio 'global' key rather than silently sharing state with it.
        logger.warning(
            "HTTP session presented no session identity; keying state as 'http-unkeyed'"
        )
        return "http-unkeyed"

    async def set_active_instance(self, ctx, instance_id: str) -> None:
        """Store the active instance for this session."""
        key = await self.get_session_key(ctx)
        with self._lock:
            self._active_by_key[key] = instance_id

    async def get_active_instance(self, ctx) -> str | None:
        """Retrieve the active instance for this session."""
        key = await self.get_session_key(ctx)
        with self._lock:
            return self._active_by_key.get(key)

    async def clear_active_instance(self, ctx) -> None:
        """Clear the stored instance for this session."""
        key = await self.get_session_key(ctx)
        with self._lock:
            self._active_by_key.pop(key, None)

    def ensure_session_identity_for_key(self, key: str) -> SessionIdentity:
        """Return the sticky identity for a session key, assigning one on first sight.

        Refreshes the identity's ``last_seen`` monotonic stamp — this method sits
        on the live-request path (every tool/resource call reaches it via
        ``get_session_identity``), so the refresh is the liveness signal the TTL
        eviction reads.
        """
        now = time.monotonic()
        with self._lock:
            identity = self._identity_by_key.get(key)
            if identity is None:
                index = self._identity_counter
                self._identity_counter += 1
                name = _FRIENDLY_NAMES[index % len(_FRIENDLY_NAMES)]
                cycle = index // len(_FRIENDLY_NAMES)
                if cycle:
                    name = f"{name}-{cycle + 1}"
                color = _IDENTITY_COLORS[index % len(_IDENTITY_COLORS)]
                identity = SessionIdentity(key=key, name=name, color=color, last_seen=now)
                self._identity_by_key[key] = identity
                logger.info("Assigned session identity '%s' (%s) to key %s",
                            name, color, key)
            else:
                identity.last_seen = now
            return identity

    async def get_session_identity(self, ctx) -> SessionIdentity:
        """Return the sticky identity for the calling session."""
        key = await self.get_session_key(ctx)
        return self.ensure_session_identity_for_key(key)

    def _expire_identities_locked(self, now_mono: float) -> None:
        """Drop identities not seen within the TTL+grace window. Caller holds lock.

        The TTL gates the disappearance; the grace adds a short tail past the TTL
        so a session that just went stale still surfaces (as 'disconnected' once
        the agent-status entry reports ended, else 'idle') for one more window
        before the row is removed.
        """
        ttl = _identity_ttl_s() + _IDENTITY_TTL_GRACE_SECONDS
        for key in list(self._identity_by_key):
            if (now_mono - self._identity_by_key[key].last_seen) > ttl:
                identity = self._identity_by_key.pop(key)
                self._active_by_key.pop(key, None)
                self._recent_tool_calls.pop(key, None)
                logger.info("Evicted stale session identity '%s' (key %s)",
                            identity.name, key)

    def all_session_identities(self) -> list[SessionIdentity]:
        """Snapshot of every live session identity (roster source).

        Stale identities (no request within the TTL+grace window) are evicted
        here so the roster self-heals on any disconnect mode — including a hard
        kill that never fires a SessionEnd hook.
        """
        now = time.monotonic()
        with self._lock:
            self._expire_identities_locked(now)
            return list(self._identity_by_key.values())

    # ------------------------------------------------------------------
    # Recent tool-call signatures (rabbit-hole `looping` flag, MCPC-032)
    # ------------------------------------------------------------------
    @staticmethod
    def _tool_call_signature(tool_name: str, arguments: Any) -> str:
        """Stable hash of tool name + canonicalized arguments."""
        try:
            args_repr = json.dumps(arguments, sort_keys=True, default=str)
        except Exception:
            args_repr = repr(arguments)
        digest = hashlib.sha1(args_repr.encode("utf-8", "replace")).hexdigest()[:12]
        return f"{tool_name}:{digest}"

    def record_tool_call_signature(
        self, key: str, tool_name: str | None, arguments: Any
    ) -> None:
        """Append a tool call's signature to the session's recent ring buffer.

        Bounded by count and pruned to the loop-detection window so the buffer
        cannot grow without bound for a long-lived session. Fails open.
        """
        if not key or not tool_name:
            return
        try:
            signature = self._tool_call_signature(tool_name, arguments)
            now = time.monotonic()
            window = _loop_window_s()
            with self._lock:
                ring = self._recent_tool_calls.get(key)
                if ring is None:
                    ring = deque(maxlen=_LOOP_RING_MAXLEN)
                    self._recent_tool_calls[key] = ring
                ring.append((signature, now))
                # Prune entries older than the window from the left.
                while ring and (now - ring[0][1]) > window:
                    ring.popleft()
        except Exception as exc:
            _diag.debug(
                "recording tool-call signature failed open (%s)",
                type(exc).__name__,
            )

    def max_repeat_count_in_window(self, key: str) -> tuple[str | None, int]:
        """Most-repeated recent signature for a session and its count.

        Returns ``(signature, count)`` over calls within the loop window;
        ``(None, 0)`` when the session has no tracked calls. Fails open to
        ``(None, 0)``.
        """
        try:
            now = time.monotonic()
            window = _loop_window_s()
            with self._lock:
                ring = self._recent_tool_calls.get(key)
                if not ring:
                    return (None, 0)
                counts: dict[str, int] = {}
                for signature, ts in ring:
                    if (now - ts) <= window:
                        counts[signature] = counts.get(signature, 0) + 1
            if not counts:
                return (None, 0)
            top = max(counts.items(), key=lambda item: item[1])
            return top
        except Exception:
            return (None, 0)

    @staticmethod
    def _read_agent_label_header() -> str | None:
        """Read the agent label from the current HTTP request, if any.

        Returns None for stdio transport, requests without the header, or
        empty/whitespace header values.
        """
        try:
            from fastmcp.server.dependencies import get_http_headers
            headers = get_http_headers(include_all=True)
        except Exception:
            return None
        raw = headers.get(AGENT_LABEL_HEADER.lower())
        if isinstance(raw, str):
            raw = raw.strip()
            if raw:
                return raw
        return None

    async def _capture_agent_label(self, ctx) -> None:
        """Ensure the session has a sticky identity and cache its agent label.

        The label is cached per session key on first sight; later requests
        (with or without the header) never change it.
        """
        identity = await self.get_session_identity(ctx)
        if identity.label is not None:
            return
        label = self._read_agent_label_header()
        if not label:
            return
        with self._lock:
            if identity.label is None:
                identity.label = label
                logger.info("Session %s (%s) labeled '%s'",
                            identity.key, identity.name, label)

    async def _discover_instances(self, ctx) -> list:
        """
        Return running Unity instances across both HTTP (PluginHub) and stdio transports.

        Returns a list of objects with .id (Name@hash) and .hash attributes.
        """
        from types import SimpleNamespace
        transport = (config.transport_mode or "stdio").lower()
        results: list = []

        if PluginHub.is_configured():
            try:
                user_id = None
                get_state_fn = getattr(ctx, "get_state", None)
                if callable(get_state_fn) and config.http_remote_hosted:
                    user_id = await get_state_fn("user_id")
                sessions_data = await PluginHub.get_sessions(user_id=user_id)
                sessions = sessions_data.sessions or {}
                # Instance identity is the registered project identity
                # (project_hash) — never the per-registration session id,
                # which is a random per-launch GUID. Dedupe by hash so a
                # reconnect race never presents one project as two instances.
                seen_hashes: set[str] = set()
                for session_info in sessions.values():
                    project = getattr(session_info, "project", None) or "Unknown"
                    hash_value = getattr(session_info, "hash", None)
                    if hash_value and hash_value not in seen_hashes:
                        seen_hashes.add(hash_value)
                        results.append(SimpleNamespace(
                            id=f"{project}@{hash_value}",
                            hash=hash_value,
                            name=project,
                        ))
            except Exception as exc:
                if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                    raise
                logger.debug("PluginHub instance discovery failed (%s)", type(exc).__name__, exc_info=True)

        if not results and transport != "http":
            try:
                from transport.legacy.unity_connection import get_unity_connection_pool
                pool = get_unity_connection_pool()
                results = pool.discover_all_instances(force_refresh=True)
            except Exception as exc:
                if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                    raise
                logger.debug("Stdio instance discovery failed (%s)", type(exc).__name__, exc_info=True)

        return results

    async def _resolve_instance_value(self, value: str, ctx) -> str:
        """
        Resolve a unity_instance string to a validated instance identifier.

        Accepts:
          - Bare port number like "6401" (stdio only) -> resolved Name@hash
          - "Name@hash" exact match
          - Hash prefix (unique prefix match against running instances)

        Raises ValueError with a user-friendly message on failure.
        """
        value = value.strip()
        if not value:
            raise ValueError("unity_instance value must not be empty.")

        transport = (config.transport_mode or "stdio").lower()

        # Port number (stdio only) — resolve to Name@hash via status file lookup
        if value.isdigit():
            if transport == "http":
                raise ValueError(
                    f"Port-based targeting ('{value}') is not supported in HTTP transport mode. "
                    "Use Name@hash or a hash prefix. Read mcpforunity://instances for available instances."
                )
            port_int = int(value)
            instances = await self._discover_instances(ctx)
            for inst in instances:
                if getattr(inst, "port", None) == port_int:
                    return inst.id
            available = ", ".join(
                f"{getattr(i, 'id', '?')} (port {getattr(i, 'port', '?')})"
                for i in instances
            ) or "none"
            raise ValueError(
                f"No Unity instance found on port {value}. Available: {available}."
            )

        instances = await self._discover_instances(ctx)
        ids = {
            getattr(inst, "id", None): inst
            for inst in instances
            if getattr(inst, "id", None)
        }

        # Exact Name@hash match
        if "@" in value:
            if value in ids:
                return value
            available = ", ".join(ids) or "none"
            raise ValueError(
                f"Instance '{value}' not found. Available: {available}. "
                "Read mcpforunity://instances for current sessions."
            )

        # Hash prefix match
        lookup = value.lower()
        matches = [
            inst for inst in instances
            if getattr(inst, "hash", "") and getattr(inst, "hash", "").lower().startswith(lookup)
        ]
        if len(matches) == 1:
            return matches[0].id
        if len(matches) > 1:
            ambiguous = ", ".join(getattr(m, "id", "?") for m in matches)
            raise ValueError(
                f"Hash prefix '{value}' is ambiguous ({ambiguous}). "
                "Provide the full Name@hash from mcpforunity://instances."
            )
        available = ", ".join(ids) or "none"
        raise ValueError(
            f"No running Unity instance matches '{value}'. Available: {available}. "
            "Read mcpforunity://instances for current sessions."
        )

    async def _maybe_autoselect_instance(self, ctx) -> str | None:
        """
        Auto-select the primary Unity instance when no active instance is set.

        The primary instance is derived from registered project identity (the
        project_hash the bridge sends at registration) — never from the
        per-registration session GUID, which changes on every reconnect.
        A session with no explicit pin routes to the sole registered project;
        with multiple distinct projects there is no primary and the caller
        must select explicitly.

        Note: This method both *discovers* and *persists* the selection via
        `set_active_instance` as a side-effect, since callers expect the selection
        to stick for subsequent tool/resource calls in the same session.
        """
        try:
            transport = (config.transport_mode or "stdio").lower()
            # This implicit behavior works well for solo-users, but is dangerous for multi-user setups
            if transport == "http" and config.http_remote_hosted:
                return None
            if PluginHub.is_configured():
                try:
                    sessions_data = await PluginHub.get_sessions()
                    sessions = sessions_data.sessions or {}
                    # Key by project_hash so duplicate registrations for the
                    # same project (e.g. a domain-reload reconnect race) still
                    # resolve to one primary instance.
                    ids_by_hash: dict[str, str] = {}
                    for session_info in sessions.values():
                        project = getattr(
                            session_info, "project", None) or "Unknown"
                        hash_value = getattr(session_info, "hash", None)
                        if hash_value and hash_value not in ids_by_hash:
                            ids_by_hash[hash_value] = f"{project}@{hash_value}"
                    ids = list(ids_by_hash.values())
                    if len(ids) == 1:
                        chosen = ids[0]
                        await self.set_active_instance(ctx, chosen)
                        logger.info(
                            "Auto-selected sole Unity instance via PluginHub: %s",
                            chosen,
                        )
                        return chosen
                    if len(ids) > 1:
                        logger.info(
                            "Multiple Unity instances found (%d). Pass unity_instance on any tool call "
                            "or call set_active_instance to choose one. Available: %s",
                            len(ids), ", ".join(ids),
                        )
                except (ConnectionError, ValueError, KeyError, TimeoutError, AttributeError) as exc:
                    logger.debug(
                        "PluginHub auto-select probe failed (%s); falling back to stdio",
                        type(exc).__name__,
                        exc_info=True,
                    )
                except Exception as exc:
                    if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                        raise
                    logger.debug(
                        "PluginHub auto-select probe failed with unexpected error (%s); falling back to stdio",
                        type(exc).__name__,
                        exc_info=True,
                    )

            if transport != "http":
                try:
                    # Import here to avoid circular imports in legacy transport paths.
                    from transport.legacy.unity_connection import get_unity_connection_pool

                    pool = get_unity_connection_pool()
                    instances = pool.discover_all_instances(force_refresh=True)
                    ids = [getattr(inst, "id", None) for inst in instances]
                    ids = [inst_id for inst_id in ids if inst_id]
                    if len(ids) == 1:
                        chosen = ids[0]
                        await self.set_active_instance(ctx, chosen)
                        logger.info(
                            "Auto-selected sole Unity instance via stdio discovery: %s",
                            chosen,
                        )
                        return chosen
                    if len(ids) > 1:
                        logger.info(
                            "Multiple Unity instances found (%d). Pass unity_instance on any tool call "
                            "or call set_active_instance to choose one. Available: %s",
                            len(ids), ", ".join(ids),
                        )
                except (ConnectionError, ValueError, KeyError, TimeoutError, AttributeError) as exc:
                    logger.debug(
                        "Stdio auto-select probe failed (%s)",
                        type(exc).__name__,
                        exc_info=True,
                    )
                except Exception as exc:
                    if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                        raise
                    logger.debug(
                        "Stdio auto-select probe failed with unexpected error (%s)",
                        type(exc).__name__,
                        exc_info=True,
                    )
        except Exception as exc:
            if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                raise
            logger.debug(
                "Auto-select path encountered an unexpected error (%s)",
                type(exc).__name__,
                exc_info=True,
            )

        return None

    async def _resolve_user_id(self) -> str | None:
        """Extract user_id from the current HTTP request's API key."""
        if not config.http_remote_hosted:
            return None
        # Lazy import to avoid circular dependencies (same pattern as _maybe_autoselect_instance).
        from transport.unity_transport import _resolve_user_id_from_request
        return await _resolve_user_id_from_request()

    async def _inject_unity_instance(self, context: MiddlewareContext) -> None:
        """Inject active Unity instance and user_id into context if available."""
        ctx = context.fastmcp_context

        # Resolve user_id from the HTTP request's API key header
        user_id = await self._resolve_user_id()
        if config.http_remote_hosted and user_id is None:
            raise RuntimeError(
                "API key authentication required. Provide a valid X-API-Key header."
            )
        if user_id:
            await ctx.set_state("user_id", user_id)

        # Ensure the session has a sticky identity and cache its optional
        # X-Agent-Label header on first sight.
        await self._capture_agent_label(ctx)

        # Per-call routing: check if this tool call explicitly specifies unity_instance.
        # context.message.arguments is a mutable dict on CallToolRequestParams; resource
        # reads use ReadResourceRequestParams which has no .arguments, so this is a no-op for them.
        # We pop the key here so Pydantic's type_adapter.validate_python() never sees it.
        active_instance: str | None = None
        msg_args = getattr(getattr(context, "message", None), "arguments", None)
        if isinstance(msg_args, dict) and "unity_instance" in msg_args:
            raw = msg_args.pop("unity_instance")
            if raw is not None:
                raw_str = str(raw).strip()
                if raw_str:
                    # Raises ValueError with a user-friendly message on invalid input.
                    active_instance = await self._resolve_instance_value(raw_str, ctx)
                    logger.debug("Per-call unity_instance resolved to: %s", active_instance)

        if not active_instance:
            active_instance = await self.get_active_instance(ctx)
        if not active_instance:
            active_instance = await self._maybe_autoselect_instance(ctx)
        if active_instance:
            # If using HTTP transport (PluginHub configured), validate session
            # But for stdio transport (no PluginHub needed or maybe partially configured),
            # we should be careful not to clear instance just because PluginHub can't resolve it.
            # The 'active_instance' (Name@hash) might be valid for stdio even if PluginHub fails.

            session_id: str | None = None
            # Only validate via PluginHub if we are actually using HTTP transport.
            # For stdio transport, skip PluginHub entirely - we only need the instance ID.
            from transport.unity_transport import _is_http_transport
            if _is_http_transport() and PluginHub.is_configured():
                try:
                    # resolving session_id might fail if the plugin disconnected
                    # We only need session_id for HTTP transport routing.
                    # For stdio, we just need the instance ID.
                    # Pass user_id for remote-hosted mode session isolation
                    session_id = await PluginHub._resolve_session_id(active_instance, user_id=user_id)
                except (ConnectionError, ValueError, KeyError, TimeoutError) as exc:
                    # If resolution fails, it means the Unity instance is not reachable via HTTP/WS.
                    # If we are in stdio mode, this might still be fine if the user is just setting state?
                    # But usually if PluginHub is configured, we expect it to work.
                    # Let's LOG the error but NOT clear the instance immediately to avoid flickering,
                    # or at least debug why it's failing.
                    logger.debug(
                        "PluginHub session resolution failed for %s: %s; leaving active_instance unchanged",
                        active_instance,
                        exc,
                        exc_info=True,
                    )
                except Exception as exc:
                    # Re-raise unexpected system exceptions to avoid swallowing critical failures
                    if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                        raise
                    logger.error(
                        "Unexpected error during PluginHub session resolution for %s: %s",
                        active_instance,
                        exc,
                        exc_info=True
                    )

            await ctx.set_state("unity_instance", active_instance)
            if session_id is not None:
                await ctx.set_state("unity_session_id", session_id)

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        """Inject active Unity instance, then apply the operation-class gate.

        The gate parks mutate/exclusive/play-scoped calls while the editor is
        busy and converts to a structured busy result past the park budget.
        Reads always pass; resource reads and tool listing are never gated.

        Around the dispatch, play-enter/exit calls feed the play lease: a
        dispatched play call records an intent (so the lease attributes
        correctly even when the play-enter domain reload eats the result),
        and observed play/pause/stop outcomes acquire, renew, or release the
        lease implicitly. A dispatched play-owning plugin tool gets the same
        treatment through :meth:`_note_plugin_play_dispatch` — its play
        transition is invisible to argument inspection.

        A busy verdict is raised as a ToolError rather than returned. A
        short-circuited call never reaches ``Tool.convert_result``, so a
        returned envelope skips the ``x-fastmcp-wrap-result`` ``{"result": ...}``
        wrap that tools with a non-object output schema declare (``run_tests``,
        ``get_test_job``, ``refresh_unity``), and the SDK rejects the call with
        "Output validation error: 'result' is a required property". Raising
        bypasses output validation for every tool regardless of its schema.
        """
        await self._inject_unity_instance(context)
        tool_name, arguments = self._tool_call_shape(context)
        plugin_play = await self._note_plugin_play_dispatch(context, tool_name)
        busy = await self._gate_tool_call(context)
        if busy is not None:
            from fastmcp.exceptions import ToolError

            if plugin_play:
                # The refused call never dispatches: drop the speculative
                # intent so it cannot misattribute the next play transition.
                await self._drop_plugin_play_intent(context)
            message = busy.get("error") or "Editor is busy; retry shortly."
            data = busy.get("data")
            if isinstance(data, dict):
                detail = ", ".join(
                    f"{key}={data[key]}"
                    for key in ("reason", "blocked_by", "retry_after_ms")
                    if data.get(key) is not None
                )
                if detail:
                    message = f"{message} ({detail})"
            raise ToolError(message)
        session_key: str | None = None
        if tool_name is not None:
            try:
                session_key = await self.get_session_key(context.fastmcp_context)
                self.record_tool_call_signature(session_key, tool_name, arguments)
            except Exception as exc:
                _diag.debug(
                    "tool-call signature tracking failed open (%s)",
                    type(exc).__name__,
                )
            self._mark_call_start(session_key, tool_name)
        await self._note_play_dispatch(context, tool_name, arguments)
        try:
            result = await call_next(context)
        finally:
            self._mark_call_end(session_key)
        await self._observe_play_result(context, tool_name, arguments, result)
        if plugin_play:
            await self._settle_plugin_play_edge(context)
        return result

    @staticmethod
    def _mark_call_start(session_key: str | None, tool_name: str | None) -> None:
        try:
            from services.state.call_activity import call_activity

            call_activity.mark_call_start(session_key, tool_name)
        except Exception:
            pass

    @staticmethod
    def _mark_call_end(session_key: str | None) -> None:
        try:
            from services.state.call_activity import call_activity

            call_activity.mark_call_end(session_key)
        except Exception:
            pass

    @staticmethod
    def _tool_call_shape(context: MiddlewareContext) -> tuple[str | None, dict | None]:
        message = getattr(context, "message", None)
        tool_name = getattr(message, "name", None)
        if not isinstance(tool_name, str) or not tool_name:
            tool_name = None
        arguments = getattr(message, "arguments", None)
        if not isinstance(arguments, dict):
            arguments = None
        return tool_name, arguments

    async def _note_play_dispatch(self, context, tool_name, arguments) -> None:
        """Record a play intent for a dispatched play-enter call. Fails open."""
        if tool_name is None:
            return
        try:
            from services.state.play_lease import note_play_call_dispatch

            await note_play_call_dispatch(
                context.fastmcp_context, tool_name, arguments)
        except Exception as exc:
            if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                raise
            _diag.debug(
                "play-lease intent recording failed open (%s)",
                type(exc).__name__,
                exc_info=True,
            )

    @staticmethod
    async def _gate_state(ctx) -> tuple[str | None, str | None]:
        """(unity_instance, user_id) from context state; either may be None."""
        unity_instance = None
        user_id = None
        get_state = getattr(ctx, "get_state", None)
        if callable(get_state):
            try:
                unity_instance = await get_state("unity_instance")
            except Exception:
                unity_instance = None
            try:
                user_id = await get_state("user_id")
            except Exception:
                user_id = None
        return unity_instance, user_id

    async def _note_plugin_play_dispatch(self, context, tool_name) -> bool:
        """Record a play intent for a play-owning plugin tool. Fails open.

        Plugin tools dispatch straight through this middleware rather than the
        custom-tool wrapper, so a driver that enters play mode leaves the
        transition with no MCP cause and the lease falls through to "user" —
        which then refuses the automation's own play-scoped calls, its cleanup
        stop included. Returns True when an intent was recorded, so the caller
        drops it if the gate refuses and settles the edge after dispatch.
        """
        if tool_name is None:
            return False
        try:
            ctx = context.fastmcp_context
            unity_instance, user_id = await self._gate_state(ctx)

            from services.state.operation_gate import plugin_tool_owns_play

            if not await plugin_tool_owns_play(tool_name, unity_instance, user_id):
                return False

            from services.state.play_lease import (
                play_lease_manager,
                record_play_intent_for_session,
            )

            lease = play_lease_manager.get_active_lease(unity_instance)
            if lease is not None and lease.owner_key == await self.get_session_key(ctx):
                # Already this session's play session: the gate renews it and
                # there is nothing to attribute.
                return False
            await record_play_intent_for_session(ctx, unity_instance)
            return True
        except Exception as exc:
            if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                raise
            _diag.debug(
                "plugin play-intent recording failed open (%s)",
                type(exc).__name__,
                exc_info=True,
            )
            return False

    async def _drop_plugin_play_intent(self, context) -> None:
        """Drop the pending plugin play intent of this session. Fails open."""
        try:
            ctx = context.fastmcp_context
            unity_instance, _ = await self._gate_state(ctx)

            from services.state.play_lease import clear_play_intent_for_session

            await clear_play_intent_for_session(ctx, unity_instance)
        except Exception as exc:
            if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                raise
            _diag.debug(
                "plugin play-intent clear failed open (%s)",
                type(exc).__name__,
                exc_info=True,
            )

    async def _settle_plugin_play_edge(self, context) -> None:
        """Attribute a play edge that only settled after the plugin dispatch.

        Runs whatever the call returned: the same pass drops the pre-recorded
        intent when the snapshot shows no play edge, so a plugin call that
        never entered play leaves nothing behind. Fails open.
        """
        try:
            ctx = context.fastmcp_context
            unity_instance, _ = await self._gate_state(ctx)

            from services.state.operation_gate import (
                record_exclusive_edge_after_arbitrary_code,
            )

            await record_exclusive_edge_after_arbitrary_code(ctx, unity_instance)
        except Exception as exc:
            if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                raise
            _diag.debug(
                "plugin play-edge attribution failed open (%s)",
                type(exc).__name__,
                exc_info=True,
            )

    async def _observe_play_result(self, context, tool_name, arguments, result) -> None:
        """Feed a play/pause/stop call's outcome to the play lease. Fails open."""
        if tool_name is None:
            return
        try:
            from services.state.play_lease import observe_play_call_result

            await observe_play_call_result(
                context.fastmcp_context, tool_name, arguments, result)
        except Exception as exc:
            if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                raise
            _diag.debug(
                "play-lease result observation failed open (%s)",
                type(exc).__name__,
                exc_info=True,
            )

    async def _gate_tool_call(self, context: MiddlewareContext) -> dict | None:
        """Run the operation-class gate for one tool call. Fails open."""
        try:
            message = getattr(context, "message", None)
            tool_name = getattr(message, "name", None)
            if not isinstance(tool_name, str) or not tool_name:
                return None
            arguments = getattr(message, "arguments", None)

            from services.state.operation_gate import gate_tool_call

            return await gate_tool_call(context.fastmcp_context, tool_name, arguments)
        except Exception as exc:
            if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                raise
            _diag.debug(
                "operation gate failed open for tool call (%s)",
                type(exc).__name__,
                exc_info=True,
            )
            return None

    async def on_read_resource(self, context: MiddlewareContext, call_next):
        """Inject active Unity instance into resource context if available."""
        await self._inject_unity_instance(context)
        return await call_next(context)

    async def on_list_tools(self, context: MiddlewareContext, call_next):
        """Filter MCP tool listing to the Unity-enabled set when session data is available."""
        try:
            await self._inject_unity_instance(context)
        except Exception as exc:
            # Re-raise authentication errors so callers get a proper auth failure
            if isinstance(exc, RuntimeError) and "authentication" in str(exc).lower():
                raise
            _diag.warning(
                "on_list_tools: _inject_unity_instance failed (%s: %s), continuing without instance",
                type(exc).__name__, exc,
            )

        tools = await call_next(context)

        tool_names_from_fastmcp = sorted(getattr(t, "name", "?") for t in tools)
        _diag.debug(
            "on_list_tools: FastMCP returned %d tools: %s",
            len(tools), tool_names_from_fastmcp,
        )

        if not self._should_filter_tool_listing():
            _diag.debug("on_list_tools: skipping middleware filter (not HTTP or PluginHub not configured)")
            return tools

        self._refresh_tool_visibility_metadata_from_registry()
        enabled_tool_names = await self._resolve_enabled_tool_names_for_context(context)
        if enabled_tool_names is None:
            _diag.debug("on_list_tools: no Unity session data, returning %d tools from FastMCP as-is", len(tools))
            return tools

        filtered = []
        for tool in tools:
            tool_name = getattr(tool, "name", None)
            if self._is_tool_visible(tool_name, enabled_tool_names):
                filtered.append(tool)

        _diag.debug(
            "on_list_tools: filtered %d/%d tools visible (Unity register_tools). "
            "enabled_names=%s",
            len(filtered), len(tools), sorted(enabled_tool_names),
        )
        return filtered

    def _should_filter_tool_listing(self) -> bool:
        transport = (config.transport_mode or "stdio").lower()
        return transport == "http" and PluginHub.is_configured()

    async def _resolve_enabled_tool_names_for_context(
        self,
        context: MiddlewareContext,
    ) -> set[str] | None:
        ctx = context.fastmcp_context
        user_id = (await ctx.get_state("user_id")) if config.http_remote_hosted else None
        active_instance = await ctx.get_state("unity_instance")
        project_hashes = self._resolve_candidate_project_hashes(active_instance)
        try:
            sessions_data = await PluginHub.get_sessions(user_id=user_id)
            sessions = sessions_data.sessions if sessions_data else {}
        except Exception as exc:
            logger.debug(
                "Failed to fetch sessions for tool filtering (user_id=%s, %s)",
                user_id,
                type(exc).__name__,
                exc_info=True,
            )
            return None

        session_hashes = {
            getattr(session, "hash", None)
            for session in sessions.values()
            if getattr(session, "hash", None)
        }

        if project_hashes:
            active_hash = project_hashes[0]
            # Stale active_instance should not hide all Unity-managed tools.
            if active_hash not in session_hashes:
                return None
        else:
            if not sessions:
                return None

            if len(sessions) == 1:
                only_session = next(iter(sessions.values()))
                only_hash = getattr(only_session, "hash", None)
                if only_hash:
                    project_hashes = [only_hash]
            else:
                # Multiple sessions without explicit selection: use a union so we don't
                # hide tools that are valid in at least one visible Unity instance.
                project_hashes = [hash_value for hash_value in session_hashes if hash_value]

        if not project_hashes:
            return None

        enabled_tool_names: set[str] = set()
        resolved_any_project = False
        for project_hash in project_hashes:
            try:
                registered_tools = await PluginHub.get_tools_for_project(project_hash, user_id=user_id)
                # Only mark as resolved if tools are actually registered.
                # An empty list means register_tools hasn't been sent yet.
                if registered_tools:
                    resolved_any_project = True
            except Exception as exc:
                logger.debug(
                    "Failed to fetch tools for project hash %s (user_id=%s, %s)",
                    project_hash,
                    user_id,
                    type(exc).__name__,
                    exc_info=True,
                )
                continue

            for tool in registered_tools:
                tool_name = getattr(tool, "name", None)
                if isinstance(tool_name, str) and tool_name:
                    enabled_tool_names.add(tool_name)

        if not resolved_any_project:
            return None

        return enabled_tool_names

    def _refresh_tool_visibility_metadata_from_registry(self) -> None:
        now = time.monotonic()
        if now - self._last_tool_visibility_refresh < self._tool_visibility_refresh_interval_seconds:
            return

        with self._metadata_lock:
            now = time.monotonic()
            if now - self._last_tool_visibility_refresh < self._tool_visibility_refresh_interval_seconds:
                return

            try:
                registry_tools = get_registered_tools()
            except Exception:
                logger.warning(
                    "Failed to refresh tool visibility metadata from registry; keeping previous metadata.",
                    exc_info=True,
                )
                self._last_tool_visibility_refresh = now
                return

            if not registry_tools and not self._has_logged_empty_registry_warning:
                logger.warning(
                    "Tool registry is empty during tool-list filtering; treating tools as unknown/visible."
                )
                self._has_logged_empty_registry_warning = True
            elif registry_tools:
                self._has_logged_empty_registry_warning = False

            unity_managed_tool_names: set[str] = set()
            tool_alias_to_unity_target: dict[str, str] = {}
            server_only_tool_names: set[str] = set()
            signature_entries: list[tuple[str, str]] = []

            for tool_info in registry_tools:
                tool_name = tool_info.get("name")
                if not isinstance(tool_name, str) or not tool_name:
                    continue

                unity_target = tool_info.get("unity_target", tool_name)
                if unity_target is None:
                    server_only_tool_names.add(tool_name)
                    signature_entries.append((tool_name, "<server-only>"))
                    continue

                if not isinstance(unity_target, str) or not unity_target:
                    logger.debug(
                        "Skipping tool visibility metadata with invalid unity_target: %s",
                        tool_info,
                    )
                    continue

                if unity_target == tool_name:
                    unity_managed_tool_names.add(tool_name)
                    signature_entries.append((tool_name, unity_target))
                    continue

                tool_alias_to_unity_target[tool_name] = unity_target
                unity_managed_tool_names.add(unity_target)
                signature_entries.append((tool_name, unity_target))

            signature = tuple(sorted(signature_entries, key=lambda item: item[0]))
            if signature == self._tool_visibility_signature:
                self._last_tool_visibility_refresh = now
                return

            self._unity_managed_tool_names = unity_managed_tool_names
            self._tool_alias_to_unity_target = tool_alias_to_unity_target
            self._server_only_tool_names = server_only_tool_names
            self._tool_visibility_signature = signature
            self._last_tool_visibility_refresh = now

    @staticmethod
    def _resolve_candidate_project_hashes(active_instance: str | None) -> list[str]:
        if not active_instance:
            return []

        if "@" in active_instance:
            _, _, suffix = active_instance.rpartition("@")
            return [suffix] if suffix else []

        return [active_instance]

    def _is_tool_visible(self, tool_name: str | None, enabled_tool_names: set[str]) -> bool:
        if not isinstance(tool_name, str) or not tool_name:
            return True

        if tool_name in self._server_only_tool_names:
            return True

        if tool_name in enabled_tool_names:
            return True

        unity_target = self._tool_alias_to_unity_target.get(tool_name)
        if unity_target:
            return unity_target in enabled_tool_names

        # Keep unknown tools visible for forward compatibility.
        if tool_name not in self._unity_managed_tool_names:
            return True

        return False
