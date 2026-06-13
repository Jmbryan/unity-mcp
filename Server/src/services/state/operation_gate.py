"""Server-side operation-class gate.

Every tool call carries a declared concurrency class — ``read``, ``mutate``,
``exclusive``, or ``play-scoped`` (``wrapper`` is server-internal: the
effective class is computed inside the wrapper's own body). The gate enforces:

- ``read`` always passes, in every editor state, for every session.
- ``mutate`` / ``exclusive`` / ``play-scoped`` transparently park (with
  progress heartbeats) while the editor is compiling, domain-reloading, or
  transitioning into/out of play mode, then execute. Past the park budget
  they convert to a structured busy result with reason and retry hint.
- During test runs only ``exclusive`` (compile-class) operations park; plain
  mutations pass.
- ``exclusive`` operations additionally respect the compile fence: they park
  while another session's edit-ledger entries are hot (edited within the
  quiescence window), proceeding when that session quiesces. A session's own
  edits never park its own compile.

Classification resolves in order: per-argument override table (for tools whose
class depends on arguments), the server tool registry, the bridge-registered
tool definition, then ``mutate`` for anything unknown.

Gate decisions read one shared, cached editor-state source
(``editor_state_cache``) — never per-call polling fan-out — and every failure
path fails open: a cache miss, ledger error, or internal exception lets the
call proceed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from services.registry import (
    CONCURRENCY_CLASSES,
    DEFAULT_CONCURRENCY_CLASS,
    get_registered_tools,
)
from services.state import edit_ledger
from services.state.editor_state_cache import editor_state_cache

logger = logging.getLogger(__name__)

CLASS_READ = "read"
CLASS_MUTATE = "mutate"
CLASS_EXCLUSIVE = "exclusive"
CLASS_PLAY_SCOPED = "play-scoped"
CLASS_WRAPPER = "wrapper"

# Severity ordering for wrapper unwrapping (max-severity wins).
_CLASS_SEVERITY: dict[str, int] = {
    CLASS_READ: 0,
    CLASS_MUTATE: 1,
    CLASS_PLAY_SCOPED: 2,
    CLASS_EXCLUSIVE: 3,
}

# Compile-fence quiescence window (seconds). Deliberately much shorter than
# the edit ledger's 120s attribution window: the fence only needs to span an
# active edit burst, while attribution must outlive compile + reload latency.
DEFAULT_FENCE_WINDOW_SECONDS = 12.0

# Park budget default (seconds). Must stay safely under MCP client call
# timeouts (~30s+); clamped like UNITY_MCP_SESSION_RESOLVE_MAX_WAIT_S.
DEFAULT_PARK_BUDGET_SECONDS = 15.0
MAX_PARK_BUDGET_SECONDS = 20.0

# Heartbeat cadence while parked.
_HEARTBEAT_INTERVAL_SECONDS = 1.0

# Per-argument classification overrides for tools whose class depends on the
# requested action (MCPC-005's argument-inspecting table). Tool name ->
# {action value -> class}; unlisted actions fall back to the declared class.
ACTION_CLASS_OVERRIDES: dict[str, dict[str, str]] = {
    "manage_editor": {
        "telemetry_status": CLASS_READ,
        "telemetry_ping": CLASS_READ,
        "play": CLASS_PLAY_SCOPED,
        "pause": CLASS_PLAY_SCOPED,
        "stop": CLASS_PLAY_SCOPED,
        # Package deploy/restore trigger a recompile.
        "deploy_package": CLASS_EXCLUSIVE,
        "restore_package": CLASS_EXCLUSIVE,
    },
    "manage_scene": {
        "get_hierarchy": CLASS_READ,
        "get_active": CLASS_READ,
        "get_build_settings": CLASS_READ,
        "get_loaded_scenes": CLASS_READ,
        "validate": CLASS_READ,
    },
    "manage_script": {
        "read": CLASS_READ,
    },
    "manage_asset": {
        "search": CLASS_READ,
        "get_info": CLASS_READ,
        "get_components": CLASS_READ,
    },
    "manage_prefabs": {
        "get_info": CLASS_READ,
        "get_hierarchy": CLASS_READ,
    },
    "manage_camera": {
        # Point-in-time screenshots are reads; camera *control* stays
        # play-scoped (the declared class).
        "screenshot": CLASS_READ,
        "get_info": CLASS_READ,
    },
}


def normalize_class(value: Any) -> str:
    """Coerce any declared class value to a known class; unknown -> mutate."""
    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in CONCURRENCY_CLASSES:
            return candidate
    return DEFAULT_CONCURRENCY_CLASS


def escalate_class(current: str, new: str) -> str:
    """Max-severity combination for wrapper unwrapping."""
    # A nested wrapper cannot be unwrapped again; treat it as a mutation.
    current_rank = _CLASS_SEVERITY.get(current, _CLASS_SEVERITY[CLASS_MUTATE])
    new_rank = _CLASS_SEVERITY.get(new, _CLASS_SEVERITY[CLASS_MUTATE])
    return current if current_rank >= new_rank else new


# ----------------------------------------------------------------------
# Classification lookup
# ----------------------------------------------------------------------
# Throttled cached refresh of the server registry's name -> class map, same
# pattern as the middleware's tool-visibility metadata refresh.
_REGISTRY_REFRESH_INTERVAL_SECONDS = 0.5
_registry_class_by_name: dict[str, str] = {}
_registry_last_refresh = 0.0


def _server_declared_class(tool_name: str) -> str | None:
    global _registry_last_refresh
    now = time.monotonic()
    if now - _registry_last_refresh >= _REGISTRY_REFRESH_INTERVAL_SECONDS:
        _registry_last_refresh = now
        try:
            refreshed: dict[str, str] = {}
            for tool_info in get_registered_tools():
                name = tool_info.get("name")
                declared = tool_info.get("concurrency_class")
                if isinstance(name, str) and name and isinstance(declared, str):
                    refreshed[name] = declared
            _registry_class_by_name.clear()
            _registry_class_by_name.update(refreshed)
        except Exception as exc:
            logger.debug(
                "operation_gate: registry class refresh failed; keeping previous map: %r",
                exc,
            )
    return _registry_class_by_name.get(tool_name)


async def _bridge_declared_class(
    tool_name: str,
    unity_instance: str | None,
    user_id: str | None,
) -> str | None:
    """Concurrency class from the bridge's register_tools payload, if any."""
    try:
        from transport.plugin_hub import PluginHub

        if not PluginHub.is_configured() or not unity_instance:
            return None
        target_hash = unity_instance
        if "@" in target_hash:
            _, _, target_hash = target_hash.rpartition("@")
        if not target_hash:
            return None
        definition = await PluginHub.get_tool_definition(
            target_hash, tool_name, user_id=user_id
        )
        if definition is None:
            return None
        declared = getattr(definition, "concurrency_class", None)
        return declared if isinstance(declared, str) and declared else None
    except Exception as exc:
        logger.debug(
            "operation_gate: bridge class lookup failed for '%s' (fail-open): %r",
            tool_name,
            exc,
        )
        return None


async def resolve_tool_class(
    tool_name: str,
    arguments: Any,
    unity_instance: str | None = None,
    user_id: str | None = None,
) -> str:
    """Resolve a tool call's concurrency class. Unknown tools are mutate."""
    overrides = ACTION_CLASS_OVERRIDES.get(tool_name)
    if overrides and isinstance(arguments, dict):
        action = arguments.get("action")
        if isinstance(action, str):
            override = overrides.get(action.strip().lower())
            if override is not None:
                return override

    declared = _server_declared_class(tool_name)
    if declared is not None:
        return declared if declared in CONCURRENCY_CLASSES else DEFAULT_CONCURRENCY_CLASS

    bridge_declared = await _bridge_declared_class(tool_name, unity_instance, user_id)
    if bridge_declared is not None:
        return normalize_class(bridge_declared)

    return DEFAULT_CONCURRENCY_CLASS


# ----------------------------------------------------------------------
# Budgets and pacing
# ----------------------------------------------------------------------
def _park_budget_s() -> float:
    raw = os.environ.get("UNITY_MCP_GATE_PARK_MAX_WAIT_S")
    if raw is None:
        return DEFAULT_PARK_BUDGET_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid UNITY_MCP_GATE_PARK_MAX_WAIT_S=%r, using default %.1f",
            raw,
            DEFAULT_PARK_BUDGET_SECONDS,
        )
        return DEFAULT_PARK_BUDGET_SECONDS
    return max(0.0, min(value, MAX_PARK_BUDGET_SECONDS))


def _fence_window_s() -> float:
    raw = os.environ.get("UNITY_MCP_COMPILE_FENCE_WINDOW_S")
    if raw is None:
        return DEFAULT_FENCE_WINDOW_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid UNITY_MCP_COMPILE_FENCE_WINDOW_S=%r, using default %.1f",
            raw,
            DEFAULT_FENCE_WINDOW_SECONDS,
        )
        return DEFAULT_FENCE_WINDOW_SECONDS
    return max(0.0, min(value, 30.0))


def _poll_sleep_s() -> float:
    try:
        from core.config import config

        retry_ms = float(getattr(config, "reload_retry_ms", 250))
    except Exception:
        retry_ms = 250.0
    return max(0.05, min(0.25, retry_ms / 1000.0))


# ----------------------------------------------------------------------
# Blocking checks
# ----------------------------------------------------------------------
def _editor_block(
    state: dict[str, Any] | None,
    concurrency_class: str,
    unity_instance: str | None,
) -> tuple[str, str | None] | None:
    """(reason, owner) when the editor state blocks this class, else None.

    A missing snapshot (cache miss) never blocks — the gate fails open.
    """
    if not isinstance(state, dict):
        return None

    compilation = state.get("compilation") or {}
    editor = state.get("editor") or {}
    play_mode = editor.get("play_mode") or {}
    tests = state.get("tests") or {}

    edge_owner = editor_state_cache.get_exclusive_edge_owner(unity_instance)

    if compilation.get("is_compiling") is True:
        return ("compiling", edge_owner)
    if compilation.get("is_domain_reload_pending") is True:
        return ("domain_reload", edge_owner)
    if play_mode.get("is_changing") is True:
        return ("play_mode_transition", edge_owner)
    # Test runs park only compile-class (exclusive) operations: script state
    # on disk cannot affect an in-flight run until a recompile picks it up.
    if tests.get("is_running") is True and concurrency_class == CLASS_EXCLUSIVE:
        owner = tests.get("started_by")
        return ("running_tests", owner if isinstance(owner, str) and owner else edge_owner)
    return None


async def _fence_block(ctx, unity_instance: str | None) -> tuple[str, str | None] | None:
    """Compile fence: (reason, owner) while another session's edits are hot.

    Own edits never park own compile (label linkage). When the ledger has no
    attributable rows but the ledger file itself was written within the
    window (a writer without parseable rows), fall back to mtime quiescence —
    fence without attribution. A missing ledger fails open.
    """
    project_root = await edit_ledger.resolve_project_root(unity_instance)
    if not project_root:
        return None
    path = edit_ledger.ledger_path(project_root)
    if not os.path.exists(path):
        return None

    window = _fence_window_s()
    if window <= 0:
        return None
    now = time.time()
    own_labels = await edit_ledger.get_own_labels(ctx)
    hot = edit_ledger.hot_entries_by_file(
        project_root, now=now, window_seconds=window
    )

    others = [
        entry
        for entry in hot.values()
        if (entry.get("label") or "") not in own_labels
    ]
    if others:
        newest = max(others, key=lambda entry: entry["timestamp"])
        label = newest.get("label")
        return ("edit_fence", label if isinstance(label, str) and label else None)

    if not hot:
        # mtime-quiescence fallback: someone wrote the ledger recently but no
        # row is hot/parseable — fence without attribution.
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return None
        if (now - mtime) <= window:
            return ("edit_fence", None)

    return None


# ----------------------------------------------------------------------
# Gate core
# ----------------------------------------------------------------------
def _busy_response(reason: str, owner: str | None, tool_name: str) -> dict[str, Any]:
    from transport.plugin_hub import PluginHub

    if reason == "edit_fence":
        detail = "another session's recent edits have not quiesced"
    else:
        detail = f"editor is busy ({reason})"
    owner_part = f" — owned by {owner}" if owner else ""
    message = (
        f"'{tool_name}' parked past its wait budget: {detail}{owner_part}. "
        "Retry shortly."
    )
    return PluginHub._unavailable_retry_response(
        reason, owner=owner, message=message, retry_after_ms=1000
    )


async def _report_progress(ctx, elapsed: float, budget: float, reason: str, owner: str | None) -> None:
    """Best-effort park heartbeat via MCP progress reporting."""
    try:
        report = getattr(ctx, "report_progress", None)
        if not callable(report):
            return
        owner_part = f" (owned by {owner})" if owner else ""
        await report(
            progress=min(elapsed, budget),
            total=budget,
            message=f"parked: {reason}{owner_part}",
        )
    except Exception:
        pass


async def _session_display_name(ctx) -> str | None:
    try:
        from transport.unity_instance_middleware import get_unity_instance_middleware

        identity = await get_unity_instance_middleware().get_session_identity(ctx)
        return identity.display_name
    except Exception:
        return None


async def gate_for_class(
    ctx,
    concurrency_class: str,
    tool_name: str,
    unity_instance: str | None,
) -> dict[str, Any] | None:
    """Park an operation of the given class until the editor admits it.

    Returns ``None`` when the call may proceed, or a structured busy payload
    (success=False, hint=retry, owner attribution) once the park budget is
    exhausted. Reads pass immediately; every internal failure fails open.
    """
    concurrency_class = (
        concurrency_class
        if concurrency_class in CONCURRENCY_CLASSES
        else normalize_class(concurrency_class)
    )
    if concurrency_class in (CLASS_READ, CLASS_WRAPPER):
        return None

    budget = _park_budget_s()
    sleep_s = _poll_sleep_s()
    started = time.monotonic()
    deadline = started + budget
    last_heartbeat = 0.0

    while True:
        block: tuple[str, str | None] | None = None
        try:
            state = await editor_state_cache.get(ctx, unity_instance)
        except Exception as exc:
            logger.debug("operation_gate: state read failed (fail-open): %r", exc)
            state = None
        try:
            block = _editor_block(state, concurrency_class, unity_instance)
        except Exception as exc:
            logger.debug("operation_gate: block check failed (fail-open): %r", exc)
            block = None
        if block is None and concurrency_class == CLASS_EXCLUSIVE:
            try:
                block = await _fence_block(ctx, unity_instance)
            except Exception as exc:
                logger.debug("operation_gate: fence check failed (fail-open): %r", exc)
                block = None

        if block is None:
            if concurrency_class == CLASS_EXCLUSIVE:
                # The caller is about to drive an exclusive transition; record
                # it as the owner so calls parked behind it see attribution.
                owner = await _session_display_name(ctx)
                if owner:
                    editor_state_cache.record_exclusive_edge(
                        unity_instance, owner, kind="exclusive_op"
                    )
            return None

        reason, owner = block
        now = time.monotonic()
        if now >= deadline:
            return _busy_response(reason, owner, tool_name)
        if now - last_heartbeat >= _HEARTBEAT_INTERVAL_SECONDS:
            last_heartbeat = now
            await _report_progress(ctx, now - started, budget, reason, owner)
        await asyncio.sleep(sleep_s)


async def gate_tool_call(ctx, tool_name: str, arguments: Any) -> dict[str, Any] | None:
    """Classify and gate one tool call. Returns None to proceed, busy payload otherwise."""
    try:
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

        concurrency_class = await resolve_tool_class(
            tool_name, arguments, unity_instance, user_id
        )
        return await gate_for_class(ctx, concurrency_class, tool_name, unity_instance)
    except Exception as exc:
        logger.debug("operation_gate: gate failed open for '%s': %r", tool_name, exc)
        return None


async def record_exclusive_edge_after_arbitrary_code(ctx, unity_instance: str | None) -> None:
    """Post-hoc ownership for arbitrary-code execution.

    Arbitrary code cannot be classified by inspection. After such a call,
    probe the shared editor-state snapshot; if an exclusive transition is in
    progress (compile, domain reload, play-mode change), assign its ownership
    to the calling session so parked calls see attribution. Never raises.
    """
    try:
        state = await editor_state_cache.get(ctx, unity_instance, max_age_s=0.0)
        if not isinstance(state, dict):
            return
        compilation = state.get("compilation") or {}
        play_mode = (state.get("editor") or {}).get("play_mode") or {}
        kind: str | None = None
        if compilation.get("is_compiling") is True or compilation.get(
            "is_domain_reload_pending"
        ) is True:
            kind = "compile"
        elif play_mode.get("is_changing") is True:
            kind = "play_transition"
        if kind is None:
            return
        owner = await _session_display_name(ctx)
        if owner:
            editor_state_cache.record_exclusive_edge(unity_instance, owner, kind=kind)
    except Exception as exc:
        logger.debug(
            "operation_gate: post-hoc exclusive-edge recording skipped: %r", exc
        )
