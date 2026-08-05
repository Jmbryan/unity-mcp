using System;
using System.Threading;
using System.Threading.Tasks;
using MCPForUnity.Editor.Helpers;
using MCPForUnity.Editor.Services;
using Newtonsoft.Json.Linq;
using UnityEditor;

namespace MCPForUnity.Editor.Tools
{
    /// <summary>
    /// Explicitly refreshes Unity's asset database and optionally requests a script compilation.
    /// This is side-effectful and should be treated as a tool.
    /// </summary>
    [McpForUnityTool("refresh_unity", AutoRegister = false, ConcurrencyClass = "exclusive")]
    public static class RefreshUnity
    {
        private const int DefaultWaitTimeoutSeconds = 60;

        public static async Task<object> HandleCommand(JObject @params)
        {
            string mode = @params?["mode"]?.ToString() ?? "if_dirty";
            string scope = @params?["scope"]?.ToString() ?? "all";
            string compile = @params?["compile"]?.ToString() ?? "none";
            bool waitForReady = ParamCoercion.CoerceBool(@params?["wait_for_ready"], false);

            bool refreshTriggered = false;
            bool compileRequested = false;
            bool compileDeferred = false;

            try
            {
                // Best-effort semantics: if_dirty currently behaves like force unless future dirty signals are added.
                bool shouldRefresh = string.Equals(mode, "force", StringComparison.OrdinalIgnoreCase)
                                     || string.Equals(mode, "if_dirty", StringComparison.OrdinalIgnoreCase);

                if (shouldRefresh)
                {
                    if (string.Equals(scope, "scripts", StringComparison.OrdinalIgnoreCase))
                    {
                        // For scripts, requesting compilation is usually the meaningful action.
                        // We avoid a heavyweight full refresh by default.
                    }
                    else
                    {
                        AssetDatabase.Refresh(ImportAssetOptions.ForceUpdate | ImportAssetOptions.ForceSynchronousImport);
                        refreshTriggered = true;
                    }
                }

                if (string.Equals(compile, "request", StringComparison.OrdinalIgnoreCase))
                {
                    // Defer-don't-error: route through the funnel. Fires immediately when idle; while
                    // play mode or a test run is active it records a pending request that flushes on
                    // return to idle. Replaces the former tests_running hard error.
                    compileDeferred = DeferredCompileService.RequestCompile("refresh_unity");
                    compileRequested = true;
                }

                if (string.Equals(scope, "all", StringComparison.OrdinalIgnoreCase) && !refreshTriggered)
                {
                    // If the caller asked for "all" and we skipped refresh above (e.g., scripts-only path),
                    // do a lightweight refresh now. Use ForceSynchronousImport to ensure the refresh
                    // completes before returning, preventing stalls when Unity is backgrounded.
                    AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                    refreshTriggered = true;
                }
            }
            catch (Exception ex)
            {
                return new ErrorResponse($"refresh_failed: {ex.Message}");
            }

            // Unity 6+ fix: Skip wait_for_ready when compile was requested.
            // The EditorApplication.update polling in WaitForUnityReadyAsync doesn't survive
            // domain reloads properly in Unity 6+, causing infinite compilation loops.
            // When compilation is requested, return immediately and let client poll editor_state.
            // Earlier Unity versions retain the original behavior.
#if UNITY_6000_0_OR_NEWER
            bool shouldWaitForReady = waitForReady && !compileRequested;
#else
            // Never block on readiness when the compile was deferred — nothing will run until the
            // blocking play/test span ends, so the wait would only time out.
            bool shouldWaitForReady = waitForReady && !compileDeferred;
#endif
            if (shouldWaitForReady)
            {
                try
                {
                    await WaitForUnityReadyAsync(
                        TimeSpan.FromSeconds(DefaultWaitTimeoutSeconds)).ConfigureAwait(true);
                }
                catch (TimeoutException)
                {
                    return new ErrorResponse("refresh_timeout_waiting_for_ready", new
                    {
                        refresh_triggered = refreshTriggered,
                        compile_requested = compileRequested,
                        resulting_state = "unknown",
                    });
                }
                catch (Exception ex)
                {
                    return new ErrorResponse($"refresh_wait_failed: {ex.Message}");
                }
            }

            string resultingState = EditorStateCache.GetActualIsCompiling()
                ? "compiling"
                : (EditorApplication.isUpdating ? "asset_import" : "idle");

            return new SuccessResponse("Refresh requested.", new
            {
                refresh_triggered = refreshTriggered,
                compile_requested = compileRequested,
                compile_deferred = compileDeferred,
                resulting_state = resultingState,
                hint = compileDeferred
                    ? "Compile deferred while play mode / a test run is active; it will flush automatically on return to idle. Poll the mcpforunity://editor/state resource until data.compilation.deferred_compile_pending clears."
                    : (shouldWaitForReady
                        ? "Unity refresh completed; editor should be ready."
                        : "If Unity enters compilation/domain reload, poll the mcpforunity://editor/state resource until data.advice.ready_for_tools is true.")
            });
        }

        private static Task WaitForUnityReadyAsync(TimeSpan timeout)
        {
            var tcs = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            var start = DateTime.UtcNow;

            void Tick()
            {
                try
                {
                    if (tcs.Task.IsCompleted)
                    {
                        EditorApplication.update -= Tick;
                        return;
                    }

                    if ((DateTime.UtcNow - start) > timeout)
                    {
                        EditorApplication.update -= Tick;
                        tcs.TrySetException(new TimeoutException());
                        return;
                    }

                    if (!EditorStateCache.GetActualIsCompiling()
                        && !EditorApplication.isUpdating
                        && !TestRunStatus.IsRunning
                        && !EditorApplication.isPlayingOrWillChangePlaymode)
                    {
                        EditorApplication.update -= Tick;
                        tcs.TrySetResult(true);
                    }
                }
                catch (Exception ex)
                {
                    EditorApplication.update -= Tick;
                    tcs.TrySetException(ex);
                }
            }

            EditorApplication.update += Tick;
            // Nudge Unity to pump once in case update is throttled.
            try { EditorApplication.QueuePlayerLoopUpdate(); } catch { }
            return tcs.Task;
        }
    }
}
