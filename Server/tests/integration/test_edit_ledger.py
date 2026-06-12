"""Tests for the shared edit ledger: recording and compile-error attribution.

Covers:
- Ledger append on the MCP tool-edit path (script edits land in the shared
  JSONL ledger with the session's display label).
- Attribution classification: other-session hot / expired / unledgered / own.
- Read-time expiry (no cleanup pass, the file is never rewritten).
- Fail-open behavior: missing directories and unwritable roots never raise.
- read_console annotation of compile errors with per-file attribution.
"""

import asyncio
import json
import os
import time

import pytest

from services.state import edit_ledger
from transport.plugin_hub import PluginHub
from transport.plugin_registry import PluginRegistry

from .test_helpers import DummyContext, setup_script_tools


@pytest.fixture(autouse=True)
def _reset_plugin_hub():
    old_registry = PluginHub._registry
    old_connections = PluginHub._connections.copy()
    old_pending = PluginHub._pending.copy()
    old_lock = PluginHub._lock
    old_loop = PluginHub._loop

    yield

    PluginHub._registry = old_registry
    PluginHub._connections = old_connections
    PluginHub._pending = old_pending
    PluginHub._lock = old_lock
    PluginHub._loop = old_loop


@pytest.fixture()
def fresh_middleware():
    """Install a fresh middleware singleton so identity names are deterministic."""
    import transport.unity_instance_middleware as mw_mod

    old = mw_mod._unity_instance_middleware
    middleware = mw_mod.UnityInstanceMiddleware()
    mw_mod.set_unity_instance_middleware(middleware)
    yield middleware
    mw_mod.set_unity_instance_middleware(old)


async def _register_project(tmp_path, project_hash="hash-et"):
    registry = PluginRegistry()
    PluginHub.configure(registry, asyncio.get_running_loop())
    await registry.register(
        "guid-1", "Proj", project_hash, "6000.0", project_path=str(tmp_path),
    )
    return registry


async def _context_pinned_to(instance_id):
    ctx = DummyContext()
    await ctx.set_state("unity_instance", instance_id)
    return ctx


def _ledger_rows(project_root):
    path = edit_ledger.ledger_path(str(project_root))
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Ledger primitives ────────────────────────────────────────────────


class TestLedgerAppendAndRead:
    def test_append_writes_pinned_contract_row(self, tmp_path):
        before = time.time()
        assert edit_ledger.append_entry(
            str(tmp_path), "agent-a", "Assets/Scripts/Foo.cs") is True
        after = time.time()

        rows = _ledger_rows(tmp_path)
        assert len(rows) == 1
        row = rows[0]
        assert set(row) == {"label", "file", "timestamp"}
        assert row["label"] == "agent-a"
        assert row["file"] == str(tmp_path).replace("\\", "/") + "/Assets/Scripts/Foo.cs"
        assert "\\" not in row["file"]
        assert os.path.isabs(row["file"])
        assert before <= row["timestamp"] <= after

    def test_append_fails_open_when_run_state_dir_cannot_be_created(self, tmp_path):
        # Make Library a file so the RunState directory cannot be created.
        (tmp_path / "Library").write_text("not a directory")
        assert edit_ledger.append_entry(
            str(tmp_path), "agent-a", "Assets/Foo.cs") is False

    def test_missing_ledger_reads_as_empty(self, tmp_path):
        assert edit_ledger.read_entries(str(tmp_path)) == []
        assert edit_ledger.hot_entries_by_file(str(tmp_path)) == {}

    def test_missing_project_root_reads_as_empty(self, tmp_path):
        gone = str(tmp_path / "does" / "not" / "exist")
        assert edit_ledger.read_entries(gone) == []
        assert edit_ledger.hot_entries_by_file(gone) == {}

    def test_malformed_lines_are_skipped(self, tmp_path):
        edit_ledger.append_entry(str(tmp_path), "agent-a", "Assets/Foo.cs")
        path = edit_ledger.ledger_path(str(tmp_path))
        with open(path, "a", encoding="utf-8") as f:
            f.write("{torn write\n")
            f.write("[1, 2, 3]\n")
            f.write('{"label": "x", "file": "y.cs", "timestamp": "not-a-number"}\n')
        edit_ledger.append_entry(str(tmp_path), "agent-b", "Assets/Bar.cs")

        entries = edit_ledger.read_entries(str(tmp_path))
        assert [e["label"] for e in entries] == ["agent-a", "agent-b"]


class TestReadTimeExpiry:
    def test_entries_expire_by_timestamp_at_read_time(self, tmp_path):
        now = time.time()
        stale = now - edit_ledger.HOT_WINDOW_SECONDS - 1
        edit_ledger.append_entry(
            str(tmp_path), "agent-a", "Assets/Old.cs", timestamp=stale)
        edit_ledger.append_entry(
            str(tmp_path), "agent-b", "Assets/New.cs", timestamp=now)

        hot = edit_ledger.hot_entries_by_file(str(tmp_path), now=now)
        labels = {e["label"] for e in hot.values()}
        assert labels == {"agent-b"}
        # No cleanup pass: the expired row is still physically in the file.
        assert len(_ledger_rows(tmp_path)) == 2

    def test_latest_entry_per_file_wins(self, tmp_path):
        now = time.time()
        edit_ledger.append_entry(
            str(tmp_path), "agent-a", "Assets/Foo.cs", timestamp=now - 5)
        edit_ledger.append_entry(
            str(tmp_path), "agent-b", "Assets/Foo.cs", timestamp=now - 1)

        hot = edit_ledger.hot_entries_by_file(str(tmp_path), now=now)
        assert len(hot) == 1
        assert next(iter(hot.values()))["label"] == "agent-b"


# ── Attribution classification ───────────────────────────────────────


class TestClassification:
    def test_other_session_hot_edit_warns_do_not_fix(self, tmp_path):
        now = time.time()
        edit_ledger.append_entry(
            str(tmp_path), "codex-1", "Assets/Scripts/Other.cs", timestamp=now - 2)

        info = edit_ledger.classify_file(
            "Assets/Scripts/Other.cs", str(tmp_path), {"claude-1"}, now=now)

        assert info["classification"] == edit_ledger.CLASSIFICATION_OTHER_SESSION
        assert info["last_editor"] == "codex-1"
        assert "not your change" in info["guidance"]
        assert "do not fix" in info["guidance"]
        assert "codex-1" in info["guidance"]

    def test_expired_entry_classifies_unattributed(self, tmp_path):
        now = time.time()
        stale = now - edit_ledger.HOT_WINDOW_SECONDS - 1
        edit_ledger.append_entry(
            str(tmp_path), "codex-1", "Assets/Scripts/Other.cs", timestamp=stale)

        info = edit_ledger.classify_file(
            "Assets/Scripts/Other.cs", str(tmp_path), {"claude-1"}, now=now)

        assert info["classification"] == edit_ledger.CLASSIFICATION_UNATTRIBUTED
        assert info["guidance"] == "unattributed (likely human)"

    def test_unledgered_file_classifies_unattributed(self, tmp_path):
        info = edit_ledger.classify_file(
            "Assets/Scripts/Human.cs", str(tmp_path), {"claude-1"})

        assert info["classification"] == edit_ledger.CLASSIFICATION_UNATTRIBUTED
        assert info["last_editor"] is None
        assert info["guidance"] == "unattributed (likely human)"

    def test_own_edit_classifies_own_and_is_omitted_from_annotations(self, tmp_path):
        now = time.time()
        edit_ledger.append_entry(
            str(tmp_path), "claude-1", "Assets/Scripts/Mine.cs", timestamp=now - 2)
        edit_ledger.append_entry(
            str(tmp_path), "codex-1", "Assets/Scripts/Other.cs", timestamp=now - 2)

        own = edit_ledger.classify_file(
            "Assets/Scripts/Mine.cs", str(tmp_path), {"claude-1"}, now=now)
        assert own["classification"] == edit_ledger.CLASSIFICATION_OWN

        annotations = edit_ledger.build_annotations(
            ["Assets/Scripts/Mine.cs", "Assets/Scripts/Other.cs"],
            str(tmp_path), {"claude-1"}, now=now)
        assert [a["file"] for a in annotations] == ["Assets/Scripts/Other.cs"]

    def test_path_matching_tolerates_separator_and_case_differences(self, tmp_path):
        now = time.time()
        edit_ledger.append_entry(
            str(tmp_path), "codex-1", "Assets\\Scripts\\Other.cs", timestamp=now - 2)

        info = edit_ledger.classify_file(
            "assets/scripts/other.cs", str(tmp_path), set(), now=now)
        assert info["classification"] == edit_ledger.CLASSIFICATION_OTHER_SESSION

    def test_classification_fails_open_without_ledger_dir(self, tmp_path):
        gone = str(tmp_path / "missing-project")
        annotations = edit_ledger.build_annotations(
            ["Assets/Scripts/Foo.cs"], gone, {"claude-1"})
        assert len(annotations) == 1
        assert annotations[0]["classification"] == edit_ledger.CLASSIFICATION_UNATTRIBUTED


class TestCompileErrorExtraction:
    def test_extracts_files_from_unity_console_shapes(self):
        payload = {
            "lines": [
                {"level": "error",
                 "message": "Assets/Scripts/Foo.cs(12,34): error CS0103: name does not exist"},
                {"level": "error",
                 "message": "Assets/Scripts/Foo.cs(40,1): error CS1002: ; expected"},
                {"level": "warning",
                 "message": "Assets/Scripts/Warn.cs(1,1): warning CS0414: unused"},
                {"level": "log", "message": "plain log line"},
            ],
            "items": [
                "Packages/com.example.pkg/Runtime/Bar.cs(3,7): error CS0246: type not found",
            ],
        }
        files = edit_ledger.extract_compile_error_files(payload)
        assert files == [
            "Assets/Scripts/Foo.cs",
            "Packages/com.example.pkg/Runtime/Bar.cs",
        ]

    def test_no_match_on_non_compile_errors(self):
        payload = {"lines": [
            {"message": "NullReferenceException: Object reference not set"},
            {"message": "error CS9999 mentioned without a file"},
        ]}
        assert edit_ledger.extract_compile_error_files(payload) == []


# ── MCPC-024: tool-edit path appends to the ledger ───────────────────


class TestRecordingOnToolEditPath:
    @pytest.mark.asyncio
    async def test_apply_text_edits_appends_ledger_row(
            self, tmp_path, monkeypatch, fresh_middleware):
        await _register_project(tmp_path, "hash-et")
        ctx = await _context_pinned_to("Proj@hash-et")

        async def fake_send(_cmd, _params, **_kwargs):
            return {"success": True}

        import transport.legacy.unity_connection
        monkeypatch.setattr(
            transport.legacy.unity_connection,
            "async_send_command_with_retry",
            fake_send,
        )

        tools = setup_script_tools()
        before = time.time()
        resp = await tools["apply_text_edits"](
            ctx,
            uri="Assets/Scripts/F.cs",
            edits=[{"startLine": 1, "startCol": 1,
                    "endLine": 1, "endCol": 1, "newText": "//x"}],
            precondition_sha256="sha",
        )
        assert resp["success"] is True

        rows = _ledger_rows(tmp_path)
        assert len(rows) == 1
        row = rows[0]
        identity = await fresh_middleware.get_session_identity(ctx)
        assert row["label"] == identity.display_name
        assert row["file"] == str(tmp_path).replace("\\", "/") + "/Assets/Scripts/F.cs"
        assert before <= row["timestamp"] <= time.time()

    @pytest.mark.asyncio
    async def test_delete_script_appends_ledger_row(
            self, tmp_path, monkeypatch, fresh_middleware):
        await _register_project(tmp_path, "hash-et")
        ctx = await _context_pinned_to("Proj@hash-et")

        async def fake_send(_cmd, _params, **_kwargs):
            return {"success": True}

        import transport.legacy.unity_connection
        monkeypatch.setattr(
            transport.legacy.unity_connection,
            "async_send_command_with_retry",
            fake_send,
        )

        tools = setup_script_tools()
        resp = await tools["delete_script"](ctx, uri="Assets/Scripts/Dead.cs")
        assert resp["success"] is True

        rows = _ledger_rows(tmp_path)
        assert len(rows) == 1
        assert rows[0]["file"].endswith("/Assets/Scripts/Dead.cs")

    @pytest.mark.asyncio
    async def test_failed_edit_records_nothing(
            self, tmp_path, monkeypatch, fresh_middleware):
        await _register_project(tmp_path, "hash-et")
        ctx = await _context_pinned_to("Proj@hash-et")

        async def fake_send(_cmd, _params, **_kwargs):
            return {"success": False, "message": "sha mismatch"}

        import transport.legacy.unity_connection
        monkeypatch.setattr(
            transport.legacy.unity_connection,
            "async_send_command_with_retry",
            fake_send,
        )

        tools = setup_script_tools()
        resp = await tools["apply_text_edits"](
            ctx,
            uri="Assets/Scripts/F.cs",
            edits=[{"startLine": 1, "startCol": 1,
                    "endLine": 1, "endCol": 1, "newText": "//x"}],
            precondition_sha256="sha",
        )
        assert resp["success"] is False
        assert _ledger_rows(tmp_path) == []

    @pytest.mark.asyncio
    async def test_non_script_mutation_records_nothing(
            self, tmp_path, monkeypatch, fresh_middleware):
        await _register_project(tmp_path, "hash-et")
        ctx = await _context_pinned_to("Proj@hash-et")

        async def fake_send(_cmd, _params, **_kwargs):
            return {"success": True}

        import transport.legacy.unity_connection
        monkeypatch.setattr(
            transport.legacy.unity_connection,
            "async_send_command_with_retry",
            fake_send,
        )

        from services.tools.refresh_unity import send_mutation
        resp = await send_mutation(
            ctx, "Proj@hash-et", "manage_ui",
            {"action": "create", "name": "Panel", "path": "Assets/UI"})
        assert resp["success"] is True
        assert _ledger_rows(tmp_path) == []

    @pytest.mark.asyncio
    async def test_edit_succeeds_when_ledger_unwritable(
            self, tmp_path, monkeypatch, fresh_middleware):
        """IO failure on the ledger must never fail the edit (fail-open)."""
        (tmp_path / "Library").write_text("not a directory")
        await _register_project(tmp_path, "hash-et")
        ctx = await _context_pinned_to("Proj@hash-et")

        async def fake_send(_cmd, _params, **_kwargs):
            return {"success": True}

        import transport.legacy.unity_connection
        monkeypatch.setattr(
            transport.legacy.unity_connection,
            "async_send_command_with_retry",
            fake_send,
        )

        tools = setup_script_tools()
        resp = await tools["apply_text_edits"](
            ctx,
            uri="Assets/Scripts/F.cs",
            edits=[{"startLine": 1, "startCol": 1,
                    "endLine": 1, "endCol": 1, "newText": "//x"}],
            precondition_sha256="sha",
        )
        assert resp["success"] is True

    @pytest.mark.asyncio
    async def test_no_registered_project_records_nothing(
            self, monkeypatch, fresh_middleware):
        """No registered plugin session (e.g. stdio) → recording silently skips."""
        PluginHub._registry = None
        ctx = await _context_pinned_to("Proj@hash-et")

        async def fake_send(_cmd, _params, **_kwargs):
            return {"success": True}

        import transport.legacy.unity_connection
        monkeypatch.setattr(
            transport.legacy.unity_connection,
            "async_send_command_with_retry",
            fake_send,
        )

        tools = setup_script_tools()
        resp = await tools["apply_text_edits"](
            ctx,
            uri="Assets/Scripts/F.cs",
            edits=[{"startLine": 1, "startCol": 1,
                    "endLine": 1, "endCol": 1, "newText": "//x"}],
            precondition_sha256="sha",
        )
        assert resp["success"] is True


# ── MCPC-026: read_console annotates compile errors ──────────────────


def _setup_console_tool():
    from .test_helpers import DummyMCP

    mcp = DummyMCP()
    import services.tools.read_console  # noqa: F401 — trigger registration
    from services.registry import get_registered_tools
    for tool_info in get_registered_tools():
        if tool_info["name"] == "read_console":
            mcp.tools["read_console"] = tool_info["func"]
    return mcp.tools["read_console"]


class TestReadConsoleAttribution:
    @pytest.mark.asyncio
    async def test_compile_errors_annotated_per_file(
            self, tmp_path, monkeypatch, fresh_middleware):
        await _register_project(tmp_path, "hash-et")
        ctx = await _context_pinned_to("Proj@hash-et")
        identity = await fresh_middleware.get_session_identity(ctx)

        now = time.time()
        edit_ledger.append_entry(
            str(tmp_path), identity.display_name,
            "Assets/Scripts/Mine.cs", timestamp=now - 2)
        edit_ledger.append_entry(
            str(tmp_path), "codex-qa",
            "Assets/Scripts/Other.cs", timestamp=now - 2)

        async def fake_send(_cmd, _params, **_kwargs):
            return {
                "success": True,
                "data": {"lines": [
                    {"level": "error",
                     "message": "Assets/Scripts/Other.cs(10,5): error CS0103: nope"},
                    {"level": "error",
                     "message": "Assets/Scripts/Mine.cs(4,2): error CS1002: ; expected"},
                    {"level": "error",
                     "message": "Assets/Scripts/Human.cs(1,1): error CS0246: type not found"},
                ]},
            }

        import services.tools.read_console as read_console_mod
        monkeypatch.setattr(
            read_console_mod, "async_send_command_with_retry", fake_send)

        read_console = _setup_console_tool()
        resp = await read_console(ctx, action="get", types=["error"])
        assert resp["success"] is True

        annotations = resp["data"][edit_ledger.ATTRIBUTION_KEY]
        by_file = {a["file"]: a for a in annotations}
        assert set(by_file) == {"Assets/Scripts/Other.cs", "Assets/Scripts/Human.cs"}

        other = by_file["Assets/Scripts/Other.cs"]
        assert other["classification"] == edit_ledger.CLASSIFICATION_OTHER_SESSION
        assert other["last_editor"] == "codex-qa"
        assert "not your change" in other["guidance"]
        assert "codex-qa" in other["guidance"]

        human = by_file["Assets/Scripts/Human.cs"]
        assert human["classification"] == edit_ledger.CLASSIFICATION_UNATTRIBUTED
        assert human["guidance"] == "unattributed (likely human)"

    @pytest.mark.asyncio
    async def test_no_compile_errors_means_no_annotation_key(
            self, tmp_path, monkeypatch, fresh_middleware):
        await _register_project(tmp_path, "hash-et")
        ctx = await _context_pinned_to("Proj@hash-et")

        async def fake_send(_cmd, _params, **_kwargs):
            return {"success": True, "data": {"lines": [
                {"level": "log", "message": "all good"},
            ]}}

        import services.tools.read_console as read_console_mod
        monkeypatch.setattr(
            read_console_mod, "async_send_command_with_retry", fake_send)

        read_console = _setup_console_tool()
        resp = await read_console(ctx, action="get")
        assert resp["success"] is True
        assert edit_ledger.ATTRIBUTION_KEY not in resp["data"]

    @pytest.mark.asyncio
    async def test_annotation_fails_open_without_registered_project(
            self, monkeypatch, fresh_middleware):
        PluginHub._registry = None
        ctx = await _context_pinned_to("Proj@hash-et")

        async def fake_send(_cmd, _params, **_kwargs):
            return {"success": True, "data": {"lines": [
                {"level": "error",
                 "message": "Assets/Scripts/Foo.cs(1,1): error CS0246: type not found"},
            ]}}

        import services.tools.read_console as read_console_mod
        monkeypatch.setattr(
            read_console_mod, "async_send_command_with_retry", fake_send)

        read_console = _setup_console_tool()
        resp = await read_console(ctx, action="get")
        assert resp["success"] is True
        assert edit_ledger.ATTRIBUTION_KEY not in resp["data"]

    @pytest.mark.asyncio
    async def test_all_own_errors_get_no_annotation(
            self, tmp_path, monkeypatch, fresh_middleware):
        await _register_project(tmp_path, "hash-et")
        ctx = await _context_pinned_to("Proj@hash-et")
        identity = await fresh_middleware.get_session_identity(ctx)
        edit_ledger.append_entry(
            str(tmp_path), identity.display_name, "Assets/Scripts/Mine.cs")

        async def fake_send(_cmd, _params, **_kwargs):
            return {"success": True, "data": {"lines": [
                {"level": "error",
                 "message": "Assets/Scripts/Mine.cs(4,2): error CS1002: ; expected"},
            ]}}

        import services.tools.read_console as read_console_mod
        monkeypatch.setattr(
            read_console_mod, "async_send_command_with_retry", fake_send)

        read_console = _setup_console_tool()
        resp = await read_console(ctx, action="get")
        assert resp["success"] is True
        assert edit_ledger.ATTRIBUTION_KEY not in resp["data"]
