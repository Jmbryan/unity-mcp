"""Async Unity Test Runner jobs: start + poll."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Annotated, Any, Literal

from fastmcp import Context
from mcp.types import ToolAnnotations
from pydantic import BaseModel

from models import MCPResponse
from services.registry import mcp_for_unity_tool
from services.tools import get_unity_instance_from_context
from services.tools.preflight import preflight
import transport.unity_transport as unity_transport
from transport.legacy.unity_connection import async_send_command_with_retry
from transport.plugin_hub import PluginHub
from utils.focus_nudge import nudge_unity_focus, should_nudge, reset_nudge_backoff

logger = logging.getLogger(__name__)

# Strong references to background fire-and-forget tasks to prevent premature GC.
_background_tasks: set[asyncio.Task] = set()


def _note_test_job_status(
    unity_instance: str | None,
    job_id: str | None,
    status: str | None,
) -> None:
    """Feed an observed test-job status to the ownership store (fail-open).

    Renews the owner's inactivity TTL on a live poll and frees ownership when
    the job reaches a terminal status, so a later non-owner clear is unguarded.
    """
    try:
        from services.state.test_job_lease import note_status_observed

        note_status_observed(unity_instance, job_id, status)
    except Exception as exc:
        logger.debug("run_tests: test-job status observation skipped: %r", exc)


async def _session_display_name(ctx) -> str | None:
    """Calling session's display name for attribution (fail-open)."""
    try:
        from transport.unity_instance_middleware import get_unity_instance_middleware

        identity = await get_unity_instance_middleware().get_session_identity(ctx)
        return identity.display_name
    except Exception:
        return None


async def _get_unity_project_path(unity_instance: str | None) -> str | None:
    """Get the project root path for a Unity instance (for focus nudging).

    Args:
        unity_instance: Unity instance hash or "Name@hash" format or None

    Returns:
        Project root path (e.g., "/Users/name/project"), or falls back to project_name if path unavailable
    """
    if not unity_instance:
        return None

    try:
        registry = PluginHub._registry
        if not registry:
            return None

        # Parse Name@hash format if present (middleware stores instances as "Name@hash")
        target_hash = unity_instance
        if "@" in target_hash:
            _, _, target_hash = target_hash.rpartition("@")
        if not target_hash:
            return None

        # Get session by hash
        session_id = await registry.get_session_id_by_hash(target_hash)
        if not session_id:
            return None

        session = await registry.get_session(session_id)
        if not session:
            return None

    except Exception as e:
        # Re-raise cancellation errors so task cancellation propagates
        if isinstance(e, asyncio.CancelledError):
            raise
        logger.debug(f"Could not get Unity project path: {e}")
        return None
    else:
        # Return full path if available, otherwise fall back to project name
        if session.project_path:
            return session.project_path
        return session.project_name if session.project_name else None


class RunTestsSummary(BaseModel):
    total: int
    passed: int
    failed: int
    skipped: int
    durationSeconds: float
    resultState: str


class RunTestsTestResult(BaseModel):
    name: str
    fullName: str
    state: str
    durationSeconds: float
    message: str | None = None
    stackTrace: str | None = None
    output: str | None = None


class RunTestsResult(BaseModel):
    mode: str
    summary: RunTestsSummary
    results: list[RunTestsTestResult] | None = None


class RunTestsStartData(BaseModel):
    job_id: str
    status: str
    mode: str | None = None
    include_details: bool | None = None
    include_failed_tests: bool | None = None


class RunTestsStartResponse(MCPResponse):
    data: RunTestsStartData | None = None


class TestJobFailure(BaseModel):
    full_name: str | None = None
    message: str | None = None


class TestJobProgress(BaseModel):
    completed: int | None = None
    total: int | None = None
    current_test_full_name: str | None = None
    current_test_started_unix_ms: int | None = None
    last_finished_test_full_name: str | None = None
    last_finished_unix_ms: int | None = None
    stuck_suspected: bool | None = None
    editor_is_focused: bool | None = None
    blocked_reason: str | None = None
    failures_so_far: list[TestJobFailure] | None = None
    failures_capped: bool | None = None


class GetTestJobData(BaseModel):
    job_id: str
    status: str
    mode: str | None = None
    started_unix_ms: int | None = None
    finished_unix_ms: int | None = None
    last_update_unix_ms: int | None = None
    progress: TestJobProgress | None = None
    error: str | None = None
    result: RunTestsResult | None = None


class GetTestJobResponse(MCPResponse):
    data: GetTestJobData | None = None


@mcp_for_unity_tool(
    group="testing",
    description="Starts a Unity test run asynchronously and returns a job_id immediately. Poll with get_test_job for progress.",
    annotations=ToolAnnotations(
        title="Run Tests",
        destructiveHint=True,
    ),
    concurrency_class="exclusive",
)
async def run_tests(
    ctx: Context,
    mode: Annotated[Literal["EditMode", "PlayMode"],
                    "Unity test mode to run"] = "EditMode",
    test_names: Annotated[list[str] | str,
                          "Full names of specific tests to run"] | None = None,
    group_names: Annotated[list[str] | str,
                           "Same as test_names, except it allows for Regex"] | None = None,
    category_names: Annotated[list[str] | str,
                              "NUnit category names to filter by"] | None = None,
    assembly_names: Annotated[list[str] | str,
                              "Assembly names to filter tests by"] | None = None,
    include_failed_tests: Annotated[bool,
                                    "Include details for failed/skipped tests only (default: false)"] = False,
    include_details: Annotated[bool,
                               "Include details for all tests (default: false)"] = False,
    init_timeout: Annotated[int | None,
                            "Initialization timeout in milliseconds. PlayMode tests may need longer "
                            "due to domain reload (default: 15000). Recommended: 120000 for PlayMode."] = None,
    clear_stuck: Annotated[bool,
                           "Force-clear the editor's current (stuck or orphaned) test job instead of "
                           "starting a run. Owner-only while another session owns the running job."] = False,
) -> RunTestsStartResponse | MCPResponse:
    unity_instance = await get_unity_instance_from_context(ctx)

    # Abort/clear surface (MCPC-022): clearing the editor's running job is the
    # only call that cancels another session's test run. It is owner-only — a
    # non-owner receives a structured busy result naming the owner; an unknown
    # or expired job fails open. Runs before both the init_timeout check and
    # preflight on purpose: neither is relevant to clearing, and
    # requires_no_tests would reject the very call that exists to clear the
    # orphaned job blocking it.
    if clear_stuck:
        from services.state.test_job_lease import (
            enforce_clear_ownership,
            test_job_lease_manager,
        )

        busy = await enforce_clear_ownership(ctx, unity_instance)
        if busy is not None:
            return MCPResponse(**busy)

        response = await unity_transport.send_with_unity_instance(
            async_send_command_with_retry,
            unity_instance,
            "run_tests",
            {"clear_stuck": True},
        )
        # The job (if any) is gone; drop ownership regardless of outcome.
        test_job_lease_manager.release(unity_instance, reason="cleared")
        if isinstance(response, dict):
            return MCPResponse(**response)
        return MCPResponse(success=False, error=str(response))

    if init_timeout is not None and init_timeout <= 0:
        return MCPResponse(success=False, error="init_timeout must be a positive integer (milliseconds) or None")

    gate = await preflight(ctx, requires_no_tests=True, wait_for_no_compile=True, refresh_if_dirty=True)
    if isinstance(gate, MCPResponse):
        return gate

    def _coerce_string_list(value) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return [value] if value.strip() else None
        if isinstance(value, list):
            result = [str(v).strip() for v in value if v and str(v).strip()]
            return result if result else None
        return None

    params: dict[str, Any] = {"mode": mode}
    if (t := _coerce_string_list(test_names)):
        params["testNames"] = t
    if (g := _coerce_string_list(group_names)):
        params["groupNames"] = g
    if (c := _coerce_string_list(category_names)):
        params["categoryNames"] = c
    if (a := _coerce_string_list(assembly_names)):
        params["assemblyNames"] = a
    if include_failed_tests:
        params["includeFailedTests"] = True
    if include_details:
        params["includeDetails"] = True
    if init_timeout is not None and init_timeout > 0:
        params["initTimeout"] = init_timeout
    caller_display_name = await _session_display_name(ctx)
    if caller_display_name:
        # Bridge contract: TestJobManager stores this as the job's StartedBy,
        # published back as tests.started_by for busy attribution.
        params["startedBy"] = caller_display_name

    from services.state.editor_state_cache import editor_state_cache
    from services.state.play_lease import (
        clear_play_intent_for_session,
        record_play_intent_for_session,
    )

    # A PlayMode run enters play mode: record a play intent for this session
    # (the same mechanism manage_editor action=play uses) so the resulting
    # play-mode lease attributes to the caller, never the fall-through "user"
    # owner. The intent's TTL may lapse across a long play-enter reload; the
    # test-job ownership recorded below is the durable backstop the play
    # lease's observation path also consults.
    if mode == "PlayMode":
        await record_play_intent_for_session(ctx, unity_instance)

    # Pin the optimistic tests-pending marker BEFORE dispatch: compile-risk
    # calls park immediately instead of racing the 1-2s window until the
    # bridge snapshot publishes tests.is_running. Cleared on a definitive
    # start failure below, by the first confirming snapshot, or by its TTL.
    editor_state_cache.mark_tests_pending(unity_instance, caller_display_name)

    response = await unity_transport.send_with_unity_instance(
        async_send_command_with_retry,
        unity_instance,
        "run_tests",
        params,
    )

    def _start_failed(resp: Any) -> bool:
        return isinstance(resp, dict) and resp.get("success", True) is False

    if _start_failed(response) or not isinstance(response, dict):
        editor_state_cache.clear_tests_pending(unity_instance)
        # Keep the intent for retry-hinted transport failures (the play-enter
        # reload can eat a result that actually succeeded); drop it on a
        # definitive refusal.
        if mode == "PlayMode" and _start_failed(response) and response.get("hint") != "retry":
            await clear_play_intent_for_session(ctx, unity_instance)

    if isinstance(response, dict):
        if not response.get("success", True):
            return MCPResponse(**response)
        start = RunTestsStartResponse(**response)
        # Record the initiating session as the owner of this job (MCPC-022).
        if start.success and start.data is not None and start.data.job_id:
            from services.state.test_job_lease import record_started_job

            await record_started_job(ctx, unity_instance, start.data.job_id)
        return start
    return MCPResponse(success=False, error=str(response))


@mcp_for_unity_tool(
    group="testing",
    description="Polls an async Unity test job by job_id.",
    annotations=ToolAnnotations(
        title="Get Test Job",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    concurrency_class="read",
)
async def get_test_job(
    ctx: Context,
    job_id: Annotated[str, "Job id returned by run_tests"],
    include_failed_tests: Annotated[bool,
                                    "Include details for failed/skipped tests only (default: false)"] = False,
    include_details: Annotated[bool,
                               "Include details for all tests (default: false)"] = False,
    wait_timeout: Annotated[int | None,
                            "If set, wait up to this many seconds for tests to complete before returning. "
                            "Reduces polling frequency and avoids client-side loop detection. "
                            "Recommended: 30-60 seconds. Returns immediately if tests complete sooner."] = None,
) -> GetTestJobResponse | MCPResponse:
    unity_instance = await get_unity_instance_from_context(ctx)

    params: dict[str, Any] = {"job_id": job_id}
    if include_failed_tests:
        params["includeFailedTests"] = True
    if include_details:
        params["includeDetails"] = True

    async def _fetch_status() -> dict[str, Any]:
        return await unity_transport.send_with_unity_instance(
            async_send_command_with_retry,
            unity_instance,
            "get_test_job",
            params,
        )

    # If wait_timeout is specified, poll server-side until complete or timeout
    if wait_timeout and wait_timeout > 0:
        deadline = asyncio.get_event_loop().time() + wait_timeout
        poll_interval = 2.0  # Poll Unity every 2 seconds
        prev_last_update_unix_ms = None

        # Get project path once for focus nudging (multi-instance support)
        project_path = await _get_unity_project_path(unity_instance)

        while True:
            response = await _fetch_status()

            if not isinstance(response, dict):
                return MCPResponse(success=False, error=str(response))

            if not response.get("success", True):
                return MCPResponse(**response)

            # Check if tests are done
            data = response.get("data", {})
            status = data.get("status", "")
            # Renew owner activity on a live poll; free ownership on completion.
            _note_test_job_status(unity_instance, job_id, status)
            if status in ("succeeded", "failed", "cancelled"):
                return GetTestJobResponse(**response)

            # Detect progress and reset exponential backoff
            last_update_unix_ms = data.get("last_update_unix_ms")
            if prev_last_update_unix_ms is not None and last_update_unix_ms != prev_last_update_unix_ms:
                # Progress detected - reset exponential backoff for next potential stall
                reset_nudge_backoff()
                logger.debug(f"Test job {job_id} made progress - reset nudge backoff")
            prev_last_update_unix_ms = last_update_unix_ms

            # Check if Unity needs a focus nudge to make progress
            # This handles OS-level throttling (e.g., macOS App Nap) that can
            # stall PlayMode tests when Unity is in the background.
            # Uses exponential backoff: 1s, 2s, 4s, 8s, 10s max between nudges.
            progress = data.get("progress") or {}
            editor_is_focused = progress.get("editor_is_focused", True)
            current_time_ms = int(time.time() * 1000)

            if should_nudge(
                status=status,
                editor_is_focused=editor_is_focused,
                last_update_unix_ms=last_update_unix_ms,
                current_time_ms=current_time_ms,
                # Use default stall_threshold_ms (3s)
            ):
                logger.info(f"Test job {job_id} appears stalled (unfocused Unity), attempting nudge...")
                # Lazily resolve project path if not yet available (registry may have become ready)
                if project_path is None:
                    project_path = await _get_unity_project_path(unity_instance)
                # Pass project path for multi-instance support
                nudged = await nudge_unity_focus(unity_project_path=project_path)
                if nudged:
                    logger.info(f"Test job {job_id} nudge completed")

            # Check timeout
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                # Timeout reached, return current status
                return GetTestJobResponse(**response)

            # Wait before next poll (but don't exceed remaining time)
            await asyncio.sleep(min(poll_interval, remaining))
    
    # No wait_timeout - return immediately (original behavior)
    response = await _fetch_status()
    if not isinstance(response, dict):
        return MCPResponse(success=False, error=str(response))
    if not response.get("success", True):
        return MCPResponse(**response)

    # Fire-and-forget nudge check: even without wait_timeout, clients may poll
    # externally. Check if Unity needs a nudge on every call so stalls get
    # detected regardless of polling style.
    data = response.get("data", {})
    status = data.get("status", "")
    _note_test_job_status(unity_instance, job_id, status)
    if status == "running":
        progress = data.get("progress") or {}
        editor_is_focused = progress.get("editor_is_focused", True)
        last_update_unix_ms = data.get("last_update_unix_ms")
        current_time_ms = int(time.time() * 1000)
        if should_nudge(
            status=status,
            editor_is_focused=editor_is_focused,
            last_update_unix_ms=last_update_unix_ms,
            current_time_ms=current_time_ms,
        ):
            logger.info(f"Test job {job_id} appears stalled (unfocused Unity), scheduling background nudge...")
            project_path = await _get_unity_project_path(unity_instance)
            task = asyncio.create_task(nudge_unity_focus(unity_project_path=project_path))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

    return GetTestJobResponse(**response)
