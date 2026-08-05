using System;
using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using MCPForUnity.Editor.Services;

namespace MCPForUnityTests.Editor.Services
{
    /// <summary>
    /// Tests for the corroboration hardening of <see cref="DeferredCompileService.IsDeferActive"/>:
    /// the test-run flag holds the defer span only while corroborated by a live
    /// <see cref="TestJobManager"/> job or while fresh. A stale, uncorroborated flag (the classic
    /// wedge shape: run flag leaked true with no tracked job) must NOT freeze every bridge-routed
    /// compile for the rest of the editor session.
    ///
    /// These tests run in EditMode so the play-mode terms of IsDeferActive are deterministically
    /// false, leaving the test-run term as the sole input. Ambient TestRunStatus / TestJobManager
    /// state (present when the suite itself runs via MCP) is snapshotted and restored per test.
    /// </summary>
    public class DeferredCompileStalenessTests
    {
        private const string JobId = "defer-staleness-job";

        private FieldInfo _statusIsRunning;
        private FieldInfo _statusMode;
        private FieldInfo _statusStarted;
        private FieldInfo _statusFinished;
        private FieldInfo _currentJobIdField;
        private FieldInfo _jobsField;
        private MethodInfo _persistMethod;

        private object[] _originalStatus;
        private string _originalJobId;

        [SetUp]
        public void SetUp()
        {
            var asm = typeof(MCPServiceLocator).Assembly;

            var statusType = asm.GetType("MCPForUnity.Editor.Services.TestRunStatus");
            Assert.NotNull(statusType, "Could not find TestRunStatus");
            _statusIsRunning = statusType.GetField("_isRunning", BindingFlags.NonPublic | BindingFlags.Static);
            _statusMode = statusType.GetField("_mode", BindingFlags.NonPublic | BindingFlags.Static);
            _statusStarted = statusType.GetField("_startedUnixMs", BindingFlags.NonPublic | BindingFlags.Static);
            _statusFinished = statusType.GetField("_finishedUnixMs", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(_statusIsRunning, "Could not find TestRunStatus._isRunning");
            Assert.NotNull(_statusStarted, "Could not find TestRunStatus._startedUnixMs");

            var managerType = asm.GetType("MCPForUnity.Editor.Services.TestJobManager");
            Assert.NotNull(managerType, "Could not find TestJobManager");
            _currentJobIdField = managerType.GetField("_currentJobId", BindingFlags.NonPublic | BindingFlags.Static);
            _jobsField = managerType.GetField("Jobs", BindingFlags.NonPublic | BindingFlags.Static);
            _persistMethod = managerType.GetMethod("PersistToSessionState", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(_currentJobIdField, "Could not find _currentJobId field");
            Assert.NotNull(_jobsField, "Could not find Jobs field");
            Assert.NotNull(_persistMethod, "Could not find PersistToSessionState method");

            _originalStatus = new[]
            {
                _statusIsRunning.GetValue(null),
                _statusMode.GetValue(null),
                _statusStarted.GetValue(null),
                _statusFinished.GetValue(null)
            };
            _originalJobId = _currentJobIdField.GetValue(null) as string;
        }

        [TearDown]
        public void TearDown()
        {
            _statusIsRunning.SetValue(null, _originalStatus[0]);
            _statusMode.SetValue(null, _originalStatus[1]);
            _statusStarted.SetValue(null, _originalStatus[2]);
            _statusFinished.SetValue(null, _originalStatus[3]);

            _currentJobIdField.SetValue(null, _originalJobId);
            Jobs().Remove(JobId);
            _persistMethod.Invoke(null, new object[] { true });
        }

        private Dictionary<string, TestJob> Jobs() =>
            (Dictionary<string, TestJob>)_jobsField.GetValue(null);

        private void SetRunFlag(long startedMsAgo)
        {
            long now = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            _statusIsRunning.SetValue(null, true);
            _statusStarted.SetValue(null, (long?)(now - startedMsAgo));
            _statusFinished.SetValue(null, null);
        }

        [Test]
        public void StaleUncorroboratedRunFlag_DoesNotHoldDefer()
        {
            // The wedge shape: flag leaked true long ago, no tracked job to corroborate it.
            _currentJobIdField.SetValue(null, null);
            SetRunFlag(startedMsAgo: 120_000);

            Assert.IsFalse(DeferredCompileService.IsDeferActive,
                "A stale run flag with no live TestJobManager job must not hold compiles forever — "
                + "the held compile is the only thing that could reset the statics, so honoring the "
                + "wedged flag would freeze every bridge-routed compile for the editor session.");
        }

        [Test]
        public void FreshUncorroboratedRunFlag_StillHoldsDefer()
        {
            // Right after MarkStarted, job bookkeeping may legitimately lag; freshness bridges it.
            _currentJobIdField.SetValue(null, null);
            SetRunFlag(startedMsAgo: 5_000);

            Assert.IsTrue(DeferredCompileService.IsDeferActive,
                "A freshly set run flag must hold the defer span even before a job corroborates it.");
        }

        [Test]
        public void OldRunFlag_WithLiveJob_HoldsDefer_ForSoakRuns()
        {
            // A genuine long soak run: the flag is hours old but the job is still live-tracked.
            long twoHoursMs = 2L * 60 * 60 * 1000;
            long now = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            Jobs()[JobId] = new TestJob
            {
                JobId = JobId,
                Status = TestJobStatus.Running,
                Mode = "EditMode",
                StartedUnixMs = now - twoHoursMs,
                LastUpdateUnixMs = now,
                TotalTests = 5000,
                FailuresSoFar = new List<TestJobFailure>()
            };
            _currentJobIdField.SetValue(null, JobId);
            SetRunFlag(startedMsAgo: twoHoursMs);

            Assert.IsTrue(DeferredCompileService.IsDeferActive,
                "A live tracked job corroborates the flag regardless of age — a 30+ minute soak run "
                + "must never have its compile fence cut out from under it.");
        }
    }
}
