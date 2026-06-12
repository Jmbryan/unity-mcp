"""Shared multi-agent edit ledger: recording and compile-error attribution.

The ledger is a JSONL file at ``<project>/Library/MCPForUnity/RunState/edit_ledger.jsonl``
shared between this server (script edits made through MCP tools) and per-harness
post-edit hooks (raw file edits that never transit MCP). One JSON object per line:

    {"label": str, "file": absolute path with forward slashes, "timestamp": unix seconds float}

Contract rules:
- Writes are single-line ``O_APPEND`` appends; the file is never rewritten.
- Entries expire by timestamp comparison at read time — there is no cleanup pass.
- Everything fails open: an IO error never fails an edit, and a missing or
  unreadable ledger behaves as empty.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

# Ledger location relative to the Unity project root. Shared contract with the
# per-harness post-edit hooks — do not rename without updating the hooks.
LEDGER_RELATIVE_PATH = "Library/MCPForUnity/RunState/edit_ledger.jsonl"

# Attribution hot window (seconds). Entries older than this are expired at read
# time. Deliberately wider than the compile-fence quiescence window (~10-15s):
# compile errors are read after compile + domain-reload latency, so attribution
# must outlive the burst of edits that caused them.
HOT_WINDOW_SECONDS = 120.0

# Result key carrying per-file attribution annotations on tool results.
ATTRIBUTION_KEY = "compile_error_attribution"

# Classification values for per-file attribution.
CLASSIFICATION_OWN = "own"
CLASSIFICATION_OTHER_SESSION = "other_session_hot"
CLASSIFICATION_UNATTRIBUTED = "unattributed"

# Compile-error file reference as Unity prints it to the console, e.g.
# "Assets/Scripts/Foo.cs(12,34): error CS0103: ...". Anchored on the
# Assets/Packages/drive-root prefix so surrounding prose is never captured.
_COMPILE_ERROR_RE = re.compile(
    r"(?P<file>(?:[A-Za-z]:[/\\]|Assets[/\\]|Packages[/\\])[^(:\r\n]*?\.cs)"
    r"\(\d+,\d+\):\s*(?:error|fatal)",
    re.IGNORECASE,
)


def normalize_path(path: str) -> str:
    """Normalize a path to forward slashes with ../ and ./ collapsed."""
    return os.path.normpath(str(path)).replace("\\", "/")


def _match_key(path: str) -> str:
    """Comparison key for ledger lookups (case-insensitive across writers)."""
    return normalize_path(path).casefold()


def ledger_path(project_root: str) -> str:
    """Absolute ledger file path for a Unity project root."""
    return normalize_path(os.path.join(project_root, LEDGER_RELATIVE_PATH))


def append_entry(
    project_root: str,
    label: str,
    file_path: str,
    timestamp: float | None = None,
) -> bool:
    """Append one edit row to the ledger as a single O_APPEND write.

    Fails open: returns False on any error, never raises — recording must
    never fail the edit it describes.
    """
    try:
        if not project_root or not file_path:
            return False
        abs_file = str(file_path)
        if not os.path.isabs(abs_file):
            abs_file = os.path.join(project_root, abs_file)
        entry = {
            "label": str(label or ""),
            "file": normalize_path(abs_file),
            "timestamp": float(timestamp if timestamp is not None else time.time()),
        }
        line = json.dumps(entry, ensure_ascii=False)
        path = ledger_path(project_root)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        try:
            os.write(fd, (line + "\n").encode("utf-8"))
        finally:
            os.close(fd)
        return True
    except Exception as exc:
        logger.debug("edit_ledger: append failed (fail-open): %r", exc)
        return False


def read_entries(project_root: str) -> list[dict[str, Any]]:
    """All parseable ledger rows, oldest first.

    Fails open: a missing or unreadable ledger behaves as empty, and
    malformed lines (e.g. a torn write from another process) are skipped.
    """
    try:
        with open(ledger_path(project_root), "r", encoding="utf-8", errors="replace") as f:
            raw_lines = f.readlines()
    except Exception:
        return []

    entries: list[dict[str, Any]] = []
    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        file_value = obj.get("file")
        if not isinstance(file_value, str) or not file_value:
            continue
        try:
            obj["timestamp"] = float(obj.get("timestamp"))
        except (TypeError, ValueError):
            continue
        entries.append(obj)
    return entries


def hot_entries_by_file(
    project_root: str,
    now: float | None = None,
    window_seconds: float = HOT_WINDOW_SECONDS,
) -> dict[str, dict[str, Any]]:
    """Latest still-hot entry per file, keyed by case-folded normalized path.

    Expiry happens here, by timestamp comparison at read time — the ledger
    file itself is never modified.
    """
    now = time.time() if now is None else float(now)
    latest: dict[str, dict[str, Any]] = {}
    for entry in read_entries(project_root):
        key = _match_key(entry["file"])
        prev = latest.get(key)
        if prev is None or entry["timestamp"] >= prev["timestamp"]:
            latest[key] = entry
    return {
        key: entry
        for key, entry in latest.items()
        if (now - entry["timestamp"]) <= window_seconds
    }


def _classify_against(
    file_path: str,
    hot_by_key: dict[str, dict[str, Any]],
    project_root: str,
    own_labels: set[str],
) -> dict[str, Any]:
    raw = str(file_path)
    abs_file = raw if os.path.isabs(raw) else os.path.join(project_root, raw)
    entry = hot_by_key.get(_match_key(abs_file))
    if entry is None:
        return {
            "file": raw,
            "classification": CLASSIFICATION_UNATTRIBUTED,
            "last_editor": None,
            "guidance": "unattributed (likely human)",
        }
    label = str(entry.get("label") or "")
    if label and label in own_labels:
        return {
            "file": raw,
            "classification": CLASSIFICATION_OWN,
            "last_editor": label,
            "guidance": None,
        }
    editor = label or "another session"
    return {
        "file": raw,
        "classification": CLASSIFICATION_OTHER_SESSION,
        "last_editor": label or None,
        "guidance": (
            f"not your change — do not fix, retry shortly (last edited by {editor})"
        ),
    }


def classify_file(
    file_path: str,
    project_root: str,
    own_labels: set[str],
    now: float | None = None,
    window_seconds: float = HOT_WINDOW_SECONDS,
) -> dict[str, Any]:
    """Classify one file against the ledger: own / other-session-hot / unattributed."""
    hot = hot_entries_by_file(project_root, now=now, window_seconds=window_seconds)
    return _classify_against(file_path, hot, project_root, own_labels)


def build_annotations(
    files: list[str],
    project_root: str,
    own_labels: set[str],
    now: float | None = None,
    window_seconds: float = HOT_WINDOW_SECONDS,
) -> list[dict[str, Any]]:
    """Per-file attribution annotations for compile errors.

    Files hot-edited by the reading session itself are omitted — an agent's
    own change needs no warning.
    """
    hot = hot_entries_by_file(project_root, now=now, window_seconds=window_seconds)
    annotations: list[dict[str, Any]] = []
    for file_path in files:
        info = _classify_against(file_path, hot, project_root, own_labels)
        if info["classification"] == CLASSIFICATION_OWN:
            continue
        annotations.append(info)
    return annotations


def extract_compile_error_files(payload: Any, max_depth: int = 8) -> list[str]:
    """Distinct .cs file references from compile-error text anywhere in a payload.

    Walks dicts/lists/strings so it works for any tool-result shape that
    carries Unity console text (``lines``, ``items``, plain strings, ...).
    """
    found: list[str] = []
    seen: set[str] = set()

    def visit(node: Any, depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(node, str):
            for match in _COMPILE_ERROR_RE.finditer(node):
                file_ref = match.group("file")
                key = _match_key(file_ref)
                if key not in seen:
                    seen.add(key)
                    found.append(file_ref)
        elif isinstance(node, dict):
            for value in node.values():
                visit(value, depth + 1)
        elif isinstance(node, (list, tuple)):
            for value in node:
                visit(value, depth + 1)

    visit(payload, 0)
    return found


async def resolve_project_root(unity_instance: str | None) -> str | None:
    """Project root for an instance, from the registered plugin session.

    Fails open: returns None when the instance, registry, or session path is
    unavailable (e.g. stdio transport, or the bridge has not registered).
    """
    if not unity_instance:
        return None
    try:
        from transport.plugin_hub import PluginHub

        registry = PluginHub._registry
        if not registry:
            return None
        target_hash = unity_instance
        if "@" in target_hash:
            _, _, target_hash = target_hash.rpartition("@")
        if not target_hash:
            return None
        session_id = await registry.get_session_id_by_hash(target_hash)
        if not session_id:
            return None
        session = await registry.get_session(session_id)
        if not session:
            return None
        return session.project_path or None
    except Exception as exc:
        logger.debug("edit_ledger: project-root resolution failed (fail-open): %r", exc)
        return None


async def _get_session_identity(ctx) -> Any | None:
    try:
        from transport.unity_instance_middleware import get_unity_instance_middleware

        return await get_unity_instance_middleware().get_session_identity(ctx)
    except Exception as exc:
        logger.debug("edit_ledger: session-identity lookup failed (fail-open): %r", exc)
        return None


async def get_own_labels(ctx) -> set[str]:
    """Labels under which the calling session's own edits may appear in the ledger.

    Server-side rows record the display name (label or friendly name); harness
    hooks record the launch label — both must match as "own".
    """
    identity = await _get_session_identity(ctx)
    if identity is None:
        return set()
    return {value for value in (identity.label, identity.name) if value}


async def record_mcp_edit(
    ctx,
    unity_instance: str | None,
    directory: str | None,
    name: str | None,
) -> None:
    """Record a successful MCP script edit into the shared ledger.

    Mirrors what a harness post-edit hook writes for raw file edits, so the
    compile fence and attribution see all edit paths. Never raises.
    """
    try:
        script_name = (name or "").strip().replace("\\", "/")
        if not script_name:
            return
        if not script_name.lower().endswith(".cs"):
            script_name += ".cs"
        folder = (directory or "").strip().replace("\\", "/").strip("/")
        relative = f"{folder}/{script_name}" if folder else script_name

        project_root = await resolve_project_root(unity_instance)
        if not project_root:
            return
        identity = await _get_session_identity(ctx)
        label = identity.display_name if identity is not None else ""
        append_entry(project_root, label, relative)
    except Exception as exc:
        logger.debug("edit_ledger: record_mcp_edit failed (fail-open): %r", exc)


async def annotate_compile_errors_for_context(
    ctx,
    unity_instance: str | None,
    result: Any,
) -> None:
    """Annotate a tool result carrying compile errors with per-file attribution.

    Adds ``compile_error_attribution`` (a list of per-file annotations) to the
    result's data when errors reference files hot-edited by another session
    ("not your change") or absent from the ledger ("unattributed"). The
    reading session's own files get no annotation. Never raises.
    """
    try:
        if not isinstance(result, dict):
            return
        data = result.get("data")
        files = extract_compile_error_files(data if data is not None else result)
        if not files:
            return
        project_root = await resolve_project_root(unity_instance)
        if not project_root:
            return
        own_labels = await get_own_labels(ctx)
        annotations = build_annotations(files, project_root, own_labels)
        if not annotations:
            return
        if isinstance(data, dict):
            data[ATTRIBUTION_KEY] = annotations
        else:
            result[ATTRIBUTION_KEY] = annotations
    except Exception as exc:
        logger.debug("edit_ledger: compile-error attribution skipped (fail-open): %r", exc)
