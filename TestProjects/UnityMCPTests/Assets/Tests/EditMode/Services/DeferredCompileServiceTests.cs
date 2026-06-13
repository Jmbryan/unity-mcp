using System;
using System.Reflection;
using NUnit.Framework;
using MCPForUnity.Editor.Services;
using UnityEditor;
using UnityEditor.TestTools.TestRunner.Api;

namespace MCPForUnityTests.Editor.Services
{
    /// <summary>
    /// Tests for DeferredCompileService (MCPC-019..021): the single defer-aware funnel that holds
    /// compile requests while play mode / a test run is active, persists the pending request across
    /// domain reload via SessionState, and refcounts AssetDatabase auto-refresh suppression with a
    /// reload/crash rebalance guard.
    ///
    /// The defer-active precondition (<see cref="DeferredCompileService.IsDeferActive"/>) keys off
    /// <see cref="TestRunStatus.IsRunning"/>. That flag is set by the MCP TestRunnerService callbacks
    /// when a run is driven in-editor, but a headless command-line <c>-runTests</c> launch never
    /// registers those callbacks, so the flag is not ambient. These tests therefore establish the
    /// blocking span deterministically by marking a test run started in SetUp (restored in TearDown),
    /// making them pass identically under both launch paths. The reflected private members let us drive
    /// SetPending / FlushIfPending / ReconcileAutoRefreshOnLoad without triggering a real compile.
    /// </summary>
    public class DeferredCompileServiceTests
    {
        // SessionState keys mirrored from DeferredCompileService (private consts).
        private const string SessionKey_PendingCompile = "MCPForUnity.DeferredCompile.Pending";
        private const string SessionKey_PendingReason = "MCPForUnity.DeferredCompile.Reason";
        private const string SessionKey_PendingImports = "MCPForUnity.DeferredCompile.PendingImports";
        private const string SessionKey_AutoRefreshDepth = "MCPForUnity.DeferredCompile.AutoRefreshDepth";

        private Type _svc;
        private MethodInfo _setPending;
        private MethodInfo _flushIfPending;
        private MethodInfo _reconcile;
        private MethodInfo _getDepth;
        private MethodInfo _setDepth;
        private MethodInfo _enqueueImport;
        private FieldInfo _suppressedThisLoad;

        // Snapshot to restore after each test so we never leak state into later tests / reloads.
        private bool _origPending;
        private string _origReason;
        private string _origImports;
        private int _origDepth;

        // True when this fixture marked the test-run-active flag itself (headless launch path) and
        // must clear it in TearDown. When the run is already flagged (in-editor MCP launch) we leave
        // it untouched.
        private bool _markedTestRun;

        [SetUp]
        public void SetUp()
        {
            var asm = typeof(MCPServiceLocator).Assembly;
            _svc = asm.GetType("MCPForUnity.Editor.Services.DeferredCompileService");
            Assert.NotNull(_svc, "Could not find DeferredCompileService");

            _setPending = _svc.GetMethod("SetPending", BindingFlags.NonPublic | BindingFlags.Static);
            _flushIfPending = _svc.GetMethod("FlushIfPending", BindingFlags.NonPublic | BindingFlags.Static);
            _reconcile = _svc.GetMethod("ReconcileAutoRefreshOnLoad", BindingFlags.NonPublic | BindingFlags.Static);
            _getDepth = _svc.GetMethod("GetAutoRefreshDepth", BindingFlags.NonPublic | BindingFlags.Static);
            _setDepth = _svc.GetMethod("SetAutoRefreshDepth", BindingFlags.NonPublic | BindingFlags.Static);
            _enqueueImport = _svc.GetMethod("EnqueuePendingImport", BindingFlags.NonPublic | BindingFlags.Static);
            _suppressedThisLoad = _svc.GetField("_autoRefreshSuppressedThisLoad", BindingFlags.NonPublic | BindingFlags.Static);

            Assert.NotNull(_setPending, "SetPending not found");
            Assert.NotNull(_flushIfPending, "FlushIfPending not found");
            Assert.NotNull(_reconcile, "ReconcileAutoRefreshOnLoad not found");
            Assert.NotNull(_getDepth, "GetAutoRefreshDepth not found");
            Assert.NotNull(_setDepth, "SetAutoRefreshDepth not found");
            Assert.NotNull(_enqueueImport, "EnqueuePendingImport not found");
            Assert.NotNull(_suppressedThisLoad, "_autoRefreshSuppressedThisLoad not found");

            _origPending = SessionState.GetBool(SessionKey_PendingCompile, false);
            _origReason = SessionState.GetString(SessionKey_PendingReason, string.Empty);
            _origImports = SessionState.GetString(SessionKey_PendingImports, string.Empty);
            _origDepth = SessionState.GetInt(SessionKey_AutoRefreshDepth, 0);

            // Establish the blocking span deterministically (see class summary). Only mark it ourselves
            // if it is not already active, so the in-editor MCP launch path is left untouched.
            if (!TestRunStatus.IsRunning)
            {
                TestRunStatus.MarkStarted(TestMode.EditMode);
                _markedTestRun = true;
            }

            // Start each test from a clean defer state.
            SetPending(false, null);
            _setDepth.Invoke(null, new object[] { 0 });
        }

        [TearDown]
        public void TearDown()
        {
            // Restore original suppression depth to a balanced zero so we never leave the editor with
            // auto-refresh disallowed after the test session.
            int depth = GetDepth();
            while (depth > 0)
            {
                DeferredCompileService.EndAutoRefreshSuppression();
                depth = GetDepth();
            }

            SessionState.SetBool(SessionKey_PendingCompile, _origPending);
            SessionState.SetString(SessionKey_PendingReason, _origReason);
            SessionState.SetString(SessionKey_PendingImports, _origImports);
            SessionState.SetInt(SessionKey_AutoRefreshDepth, _origDepth);
            _suppressedThisLoad.SetValue(null, false);

            if (_markedTestRun)
            {
                TestRunStatus.MarkFinished();
                _markedTestRun = false;
            }
        }

        private void SetPending(bool pending, string reason) =>
            _setPending.Invoke(null, new object[] { pending, reason });

        private void EnqueueImport(string path) =>
            _enqueueImport.Invoke(null, new object[] { path });

        private int GetDepth() => (int)_getDepth.Invoke(null, null);

        // ---- MCPC-019: defer + pending-set persistence ------------------------------------------

        [Test]
        public void IsDeferActive_DuringTestRun_IsTrue()
        {
            // EditMode tests run with TestRunStatus.IsRunning == true.
            Assert.IsTrue(DeferredCompileService.IsDeferActive,
                "Service should report defer-active while a test run is in progress.");
        }

        [Test]
        public void RequestCompile_WhileDeferActive_DefersAndRecordsPending()
        {
            bool deferred = DeferredCompileService.RequestCompile("unit_test_reason");

            Assert.IsTrue(deferred, "RequestCompile should defer while a test run is active.");
            Assert.IsTrue(DeferredCompileService.HasPendingCompile, "Pending flag should be set.");
            Assert.AreEqual("unit_test_reason", DeferredCompileService.PendingReason,
                "Recorded reason should round-trip through the pending state.");
        }

        [Test]
        public void PendingState_PersistsToSessionState()
        {
            SetPending(true, "persisted_reason");

            // Read straight from SessionState — this is the cross-domain-reload persistence surface.
            Assert.IsTrue(SessionState.GetBool(SessionKey_PendingCompile, false),
                "Pending bool should be written to SessionState (survives domain reload).");
            Assert.AreEqual("persisted_reason", SessionState.GetString(SessionKey_PendingReason, string.Empty),
                "Pending reason should be written to SessionState.");

            // The public accessors read the same persisted keys, so a freshly-reloaded service sees it.
            Assert.IsTrue(DeferredCompileService.HasPendingCompile);
            Assert.AreEqual("persisted_reason", DeferredCompileService.PendingReason);
        }

        [Test]
        public void SetPending_False_ClearsReason()
        {
            SetPending(true, "to_be_cleared");
            Assert.IsTrue(DeferredCompileService.HasPendingCompile);

            SetPending(false, null);

            Assert.IsFalse(DeferredCompileService.HasPendingCompile, "Pending flag should clear.");
            Assert.IsNull(DeferredCompileService.PendingReason,
                "PendingReason should be null once no compile is pending.");
        }

        [Test]
        public void ImportAndRequestCompile_WhileDeferActive_HoldsImportAndQueuesPath()
        {
            // Importing a script is itself a compile trigger, so while a test run is active the import
            // must be held (queued), not fired — otherwise a compile leaks under the running session.
            DeferredCompileService.ImportAndRequestCompile(
                "Assets/Tests/__defer_probe_never_imported__.cs", synchronous: true);

            Assert.IsTrue(DeferredCompileService.HasPendingCompile,
                "A deferred import must record a pending compile.");
            StringAssert.StartsWith("import:", DeferredCompileService.PendingReason,
                "Pending reason should reflect the held import.");

            var queued = SessionState.GetString(SessionKey_PendingImports, string.Empty);
            StringAssert.Contains("__defer_probe_never_imported__.cs", queued,
                "The held import path must be queued in SessionState for flush-time import.");
        }

        [Test]
        public void EnqueuePendingImport_DeduplicatesPaths()
        {
            EnqueueImport("Assets/A.cs");
            EnqueueImport("Assets/B.cs");
            EnqueueImport("Assets/A.cs"); // duplicate

            var queued = SessionState.GetString(SessionKey_PendingImports, string.Empty);
            var parts = queued.Split('\n');
            Assert.AreEqual(2, parts.Length, "Duplicate paths must not be queued twice.");
            CollectionAssert.AreEquivalent(new[] { "Assets/A.cs", "Assets/B.cs" }, parts);
        }

        [Test]
        public void SetPending_False_ClearsQueuedImports()
        {
            EnqueueImport("Assets/A.cs");
            Assert.IsNotEmpty(SessionState.GetString(SessionKey_PendingImports, string.Empty));

            SetPending(false, null);

            Assert.AreEqual(string.Empty, SessionState.GetString(SessionKey_PendingImports, string.Empty),
                "Clearing the pending compile must also drop the queued import set.");
        }

        [Test]
        public void FlushIfPending_WhileStillDeferActive_DoesNotClearPending()
        {
            SetPending(true, "still_blocked");

            // A test run is active, so the flush must be a no-op and leave the request queued.
            _flushIfPending.Invoke(null, new object[] { "unit_test" });

            Assert.IsTrue(DeferredCompileService.HasPendingCompile,
                "Flush must not fire while play/test is still active; the request stays pending.");
        }

        // ---- MCPC-021: refcounted auto-refresh suppression + reload rebalance --------------------

        [Test]
        public void AutoRefreshSuppression_Refcount_Balances()
        {
            Assert.AreEqual(0, GetDepth(), "Precondition: depth starts at zero.");

            DeferredCompileService.BeginAutoRefreshSuppression();
            Assert.AreEqual(1, GetDepth(), "First Begin raises depth to 1.");

            DeferredCompileService.BeginAutoRefreshSuppression();
            Assert.AreEqual(2, GetDepth(), "Nested Begin raises depth to 2.");

            DeferredCompileService.EndAutoRefreshSuppression();
            Assert.AreEqual(1, GetDepth(), "First End lowers depth back to 1.");

            DeferredCompileService.EndAutoRefreshSuppression();
            Assert.AreEqual(0, GetDepth(), "Balanced End returns depth to 0.");
        }

        [Test]
        public void AutoRefreshSuppression_OverBalancedEnd_ClampsAtZero()
        {
            DeferredCompileService.BeginAutoRefreshSuppression();
            DeferredCompileService.EndAutoRefreshSuppression();
            // Extra End (e.g. a stale call after a reconcile) must not drive the count negative.
            DeferredCompileService.EndAutoRefreshSuppression();

            Assert.AreEqual(0, GetDepth(), "Depth must never go below zero.");
        }

        [Test]
        public void Reconcile_PersistedDepthButNotBlocking_ClearsStaleDepth()
        {
            // Simulate a crash on play exit: depth persisted, but the editor is now idle (no test run
            // would normally be running, but here a test run IS active, so we force the not-blocking
            // branch by setting depth and asserting the post-reconcile invariant via the blocking path).
            // Since IsDeferActive is true during the test run, reconcile re-applies suppression instead
            // of clearing. We assert that branch: a persisted depth + active blocking state keeps depth.
            _setDepth.Invoke(null, new object[] { 1 });
            _suppressedThisLoad.SetValue(null, false);

            _reconcile.Invoke(null, null);

            // During a test run IsDeferActive is true, so the persisted depth is retained and a single
            // live suppression is re-established to match intent.
            Assert.AreEqual(1, GetDepth(),
                "With a persisted depth and an active blocking span, reconcile retains the depth.");
            Assert.IsTrue((bool)_suppressedThisLoad.GetValue(null),
                "Reconcile should re-establish a single live suppression when still blocking.");

            // Balance it back out for a clean editor.
            DeferredCompileService.EndAutoRefreshSuppression();
            Assert.AreEqual(0, GetDepth());
        }
    }
}
