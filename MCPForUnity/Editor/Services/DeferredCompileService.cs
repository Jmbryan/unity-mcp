using System;
using MCPForUnity.Editor.Helpers;
using UnityEditor;
using UnityEditor.Compilation;
using UnityEditor.TestTools.TestRunner.Api;
using UnityEngine;

namespace MCPForUnity.Editor.Services
{
    /// <summary>
    /// Single defer-aware funnel for every bridge compile path (MCPC-019..021).
    ///
    /// While play mode or a test run is active, script-compilation requests are held instead of
    /// fired immediately, and Unity's asset auto-refresh is disallowed so an external change cannot
    /// slip a compile under the running session. The pending request and the auto-refresh suppression
    /// depth are persisted to <see cref="SessionState"/> so they survive the domain reload that play
    /// entry / test runs trigger. On return to idle (play exit, test-run end) the held compile flushes.
    ///
    /// Importing a script asset is itself a compile trigger — <see cref="AssetDatabase.ImportAsset"/>
    /// on a <c>.cs</c> file queues it into the compilation pipeline regardless of
    /// <see cref="AssetDatabase.DisallowAutoRefresh"/>. The disk write has already landed (the edit
    /// tool wrote the file before requesting a refresh), so while a defer span is active the import is
    /// held too: pending paths accumulate in <see cref="SessionState"/> and are imported together with
    /// the compile request on flush. Deferring only the <see cref="CompilationPipeline.RequestScriptCompilation()"/>
    /// edge while still importing would leak a compile into play mode / a test run.
    /// </summary>
    [InitializeOnLoad]
    internal static class DeferredCompileService
    {
        // Pending-compile persistence (MCPC-019).
        private const string SessionKey_PendingCompile = "MCPForUnity.DeferredCompile.Pending";
        private const string SessionKey_PendingReason = "MCPForUnity.DeferredCompile.Reason";

        // Pending script imports held while deferring (MCPC-019). Newline-delimited Assets-relative
        // paths; imported together with the compile flush on return to idle. Persisted so a domain
        // reload mid-defer (play entry / test start) does not drop the queued imports.
        private const string SessionKey_PendingImports = "MCPForUnity.DeferredCompile.PendingImports";

        // Auto-refresh suppression depth (MCPC-021). Refcounted so nested play/test spans balance.
        private const string SessionKey_AutoRefreshDepth = "MCPForUnity.DeferredCompile.AutoRefreshDepth";

        private static TestRunnerApi _api;
        private static bool _autoRefreshSuppressedThisLoad;

        static DeferredCompileService()
        {
            try
            {
                EditorApplication.playModeStateChanged += OnPlayModeStateChanged;

                _api = ScriptableObject.CreateInstance<TestRunnerApi>();
                _api.hideFlags = HideFlags.HideAndDontSave;
                _api.RegisterCallbacks(new TestCallbacks());

                // Rebalance after a domain reload or crash: AssetDatabase auto-refresh suppression is
                // process-local and is dropped by the reload, but our recorded depth survives in
                // SessionState. Reconcile the live editor state to the persisted intent.
                ReconcileAutoRefreshOnLoad();

                // If we reloaded into an idle editor with a pending compile (e.g. play just exited),
                // flush it now. If still blocked, the play/test end callbacks will flush later.
                if (!IsDeferActive)
                {
                    FlushIfPending("after_reload");
                }
            }
            catch (Exception e)
            {
                McpLog.Warn($"[DeferredCompileService] Failed to initialise: {e}");
            }
        }

        /// <summary>True while a compile must be held: play mode (or its transition) or a test run.</summary>
        internal static bool IsDeferActive =>
            EditorApplication.isPlayingOrWillChangePlaymode
            || EditorApplication.isPlaying
            || TestRunStatus.IsRunning;

        /// <summary>Whether a compile request is currently held pending return to idle (MCPC-020).</summary>
        internal static bool HasPendingCompile => SessionState.GetBool(SessionKey_PendingCompile, false);

        /// <summary>Reason recorded for the held compile, surfaced in the editor-state snapshot (MCPC-020).</summary>
        internal static string PendingReason =>
            HasPendingCompile ? SessionState.GetString(SessionKey_PendingReason, "deferred") : null;

        /// <summary>
        /// Number of script imports currently held pending flush. Counts newline-delimited paths in
        /// the queued-imports buffer; zero when nothing is queued (a held compile may still exist with
        /// no queued import — e.g. a bare <see cref="RequestCompile"/> deferral).
        /// </summary>
        internal static int PendingImportCount
        {
            get
            {
                var queued = SessionState.GetString(SessionKey_PendingImports, string.Empty);
                if (string.IsNullOrEmpty(queued)) return 0;

                int count = 0;
                foreach (var p in queued.Split('\n'))
                {
                    if (!string.IsNullOrEmpty(p)) count++;
                }
                return count;
            }
        }

        /// <summary>
        /// Request a script compilation through the defer funnel. Fires immediately when idle;
        /// records the request and returns when play mode or a test run is active.
        /// Returns true if the compile was deferred, false if it fired immediately.
        /// </summary>
        internal static bool RequestCompile(string reason)
        {
            if (IsDeferActive)
            {
                SetPending(true, reason);
                McpLog.Info($"[DeferredCompileService] Compile deferred ({reason}); will flush on return to idle.", always: false);
                return true;
            }

            SetPending(false, null);
            CompilationPipeline.RequestScriptCompilation();
            return false;
        }

        /// <summary>
        /// Import a script asset and route its compile through the defer funnel. When idle, the asset
        /// is imported now (which compiles) and a compilation request is issued. While play mode or a
        /// test run is active the import is held — importing a script is itself a compile trigger, so
        /// firing it now would leak a compile under the running session — and the path is queued to be
        /// imported on flush. The disk write has already landed before this call, so holding the import
        /// loses nothing.
        /// </summary>
        internal static void ImportAndRequestCompile(string assetsRelativePath, bool synchronous = true)
        {
            if (IsDeferActive)
            {
                EnqueuePendingImport(assetsRelativePath);
                SetPending(true, $"import:{assetsRelativePath}");
                McpLog.Info($"[DeferredCompileService] Import+compile deferred ({assetsRelativePath}); will flush on return to idle.", always: false);
                return;
            }

            ImportNow(assetsRelativePath, synchronous);
            SetPending(false, null);
            CompilationPipeline.RequestScriptCompilation();
        }

        private static void ImportNow(string assetsRelativePath, bool synchronous)
        {
            var opts = ImportAssetOptions.ForceUpdate;
            if (synchronous) opts |= ImportAssetOptions.ForceSynchronousImport;
            AssetDatabase.ImportAsset(assetsRelativePath, opts);
        }

        #region Auto-refresh suppression (MCPC-021)

        /// <summary>
        /// Begin an auto-refresh suppression span for play mode / test runs. Refcounted; each call
        /// must be balanced by <see cref="EndAutoRefreshSuppression"/>. Survives domain reload via
        /// the persisted depth + on-load reconciliation.
        /// </summary>
        internal static void BeginAutoRefreshSuppression()
        {
            int depth = GetAutoRefreshDepth();
            if (depth == 0)
            {
                ApplyDisallowAutoRefresh();
            }
            SetAutoRefreshDepth(depth + 1);
        }

        /// <summary>End an auto-refresh suppression span. Restores auto-refresh when depth returns to zero.</summary>
        internal static void EndAutoRefreshSuppression()
        {
            int depth = GetAutoRefreshDepth();
            if (depth <= 0)
            {
                // Already balanced (e.g. reconciled away after a reload); nothing to do.
                SetAutoRefreshDepth(0);
                return;
            }

            depth--;
            SetAutoRefreshDepth(depth);
            if (depth == 0)
            {
                ApplyAllowAutoRefresh();
            }
        }

        private static int GetAutoRefreshDepth() => SessionState.GetInt(SessionKey_AutoRefreshDepth, 0);
        private static void SetAutoRefreshDepth(int value) => SessionState.SetInt(SessionKey_AutoRefreshDepth, Mathf.Max(0, value));

        private static void ApplyDisallowAutoRefresh()
        {
            if (_autoRefreshSuppressedThisLoad) return;
            AssetDatabase.DisallowAutoRefresh();
            _autoRefreshSuppressedThisLoad = true;
        }

        private static void ApplyAllowAutoRefresh()
        {
            if (!_autoRefreshSuppressedThisLoad) return;
            AssetDatabase.AllowAutoRefresh();
            _autoRefreshSuppressedThisLoad = false;
        }

        /// <summary>
        /// After a domain reload the AssetDatabase Disallow/Allow refcount resets to zero (it is
        /// process-local), but our intended depth persists in SessionState. Re-apply one suppression
        /// if the persisted depth says we should still be suppressing; otherwise leave auto-refresh on.
        /// This is the crash/reload rebalance guard — worst case one unbalanced span, never a wedge.
        /// </summary>
        private static void ReconcileAutoRefreshOnLoad()
        {
            int depth = GetAutoRefreshDepth();
            if (depth > 0 && IsDeferActive)
            {
                // Still inside a play/test span: re-establish a single live suppression to match intent.
                ApplyDisallowAutoRefresh();
            }
            else if (depth > 0)
            {
                // Persisted depth but no active blocking state — the span ended across the reload
                // (e.g. crash on play exit). Clear the stale depth so we don't leak suppression.
                SetAutoRefreshDepth(0);
            }
        }

        #endregion

        #region Pending-compile persistence (MCPC-019)

        private static void SetPending(bool pending, string reason)
        {
            SessionState.SetBool(SessionKey_PendingCompile, pending);
            if (pending)
            {
                SessionState.SetString(SessionKey_PendingReason, reason ?? "deferred");
            }
            else
            {
                SessionState.EraseString(SessionKey_PendingReason);
                SessionState.EraseString(SessionKey_PendingImports);
            }
        }

        private static void EnqueuePendingImport(string assetsRelativePath)
        {
            if (string.IsNullOrEmpty(assetsRelativePath)) return;

            var existing = SessionState.GetString(SessionKey_PendingImports, string.Empty);
            foreach (var p in existing.Split('\n'))
            {
                if (string.Equals(p, assetsRelativePath, StringComparison.OrdinalIgnoreCase)) return;
            }

            SessionState.SetString(SessionKey_PendingImports,
                string.IsNullOrEmpty(existing) ? assetsRelativePath : existing + "\n" + assetsRelativePath);
        }

        private static void FlushIfPending(string trigger)
        {
            if (!HasPendingCompile) return;
            if (IsDeferActive) return; // Still blocked; flush will be retried on the next end edge.

            // Import any script edits whose import we held (importing a script is the compile trigger),
            // then issue an explicit compile request to cover non-import-backed pending compiles.
            var pendingImports = SessionState.GetString(SessionKey_PendingImports, string.Empty);
            SetPending(false, null);

            if (!string.IsNullOrEmpty(pendingImports))
            {
                foreach (var p in pendingImports.Split('\n'))
                {
                    if (string.IsNullOrEmpty(p)) continue;
                    try { ImportNow(p, synchronous: true); }
                    catch (Exception e) { McpLog.Warn($"[DeferredCompileService] Flush import failed for '{p}': {e.Message}"); }
                }
            }

            McpLog.Info($"[DeferredCompileService] Flushing deferred compile ({trigger}).");
            CompilationPipeline.RequestScriptCompilation();
        }

        #endregion

        /// <summary>
        /// Public flush entrypoint for service callers (e.g. <see cref="SessionRosterService"/>).
        /// Flushes a held compile when the editor is idle; a no-op when nothing is pending or a
        /// play/test span is still active. Must run on the main thread.
        /// </summary>
        internal static void FlushNow(string trigger)
        {
            FlushIfPending(trigger ?? "flush_now");
        }

        private static void OnPlayModeStateChanged(PlayModeStateChange change)
        {
            switch (change)
            {
                case PlayModeStateChange.ExitingEditMode:
                    // Entering play: suppress auto-refresh for the play span (MCPC-021).
                    BeginAutoRefreshSuppression();
                    break;
                case PlayModeStateChange.EnteredEditMode:
                    // Play exited: lift suppression and flush any held compile (MCPC-019/021).
                    EndAutoRefreshSuppression();
                    FlushIfPending("play_exit");
                    break;
            }
        }

        private sealed class TestCallbacks : ICallbacks
        {
            public void RunStarted(ITestAdaptor testsToRun)
            {
                // Suppress auto-refresh for the test-run span (MCPC-021).
                BeginAutoRefreshSuppression();
            }

            public void RunFinished(ITestResultAdaptor result)
            {
                EndAutoRefreshSuppression();
                // TestRunStatus is cleared by TestRunnerService.RunFinished; defer until the next tick
                // so IsDeferActive reflects the finished state regardless of callback ordering.
                EditorApplication.delayCall += () => FlushIfPending("test_run_end");
            }

            public void TestStarted(ITestAdaptor test) { }
            public void TestFinished(ITestResultAdaptor result) { }
        }
    }
}
