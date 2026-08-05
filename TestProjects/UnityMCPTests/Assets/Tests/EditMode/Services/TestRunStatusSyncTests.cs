using System;
using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using MCPForUnity.Editor.Services;
using UnityEditor.TestTools.TestRunner.Api;

namespace MCPForUnityTests.Editor.Services
{
    /// <summary>
    /// Tests that TestJobManager never drops a job id while leaving
    /// <see cref="TestRunStatus.IsRunning"/> set. A wedged run flag makes the editor
    /// report a phantom test run forever, which parks every exclusive-class MCP call
    /// (run_tests included) behind a run that will never finish.
    ///
    /// TestRunStatus is ambient when the suite is launched in-editor via MCP, so the
    /// whole struct is snapshotted and restored around each test rather than being
    /// cleared outright.
    /// </summary>
    public class TestRunStatusSyncTests
    {
        private FieldInfo _jobsField;
        private FieldInfo _currentJobIdField;
        private FieldInfo _autoFailedField;
        private MethodInfo _persistMethod;

        private FieldInfo _statusIsRunning;
        private FieldInfo _statusMode;
        private FieldInfo _statusStarted;
        private FieldInfo _statusFinished;

        private string _originalJobId;
        private object[] _originalStatus;

        private const string JobA = "test-run-status-a";
        private const string JobB = "test-run-status-b";

        [SetUp]
        public void SetUp()
        {
            var asm = typeof(MCPServiceLocator).Assembly;

            var managerType = asm.GetType("MCPForUnity.Editor.Services.TestJobManager");
            Assert.NotNull(managerType, "Could not find TestJobManager");
            _jobsField = managerType.GetField("Jobs", BindingFlags.NonPublic | BindingFlags.Static);
            _currentJobIdField = managerType.GetField("_currentJobId", BindingFlags.NonPublic | BindingFlags.Static);
            _autoFailedField = managerType.GetField("_autoFailedInitJobId", BindingFlags.NonPublic | BindingFlags.Static);
            _persistMethod = managerType.GetMethod("PersistToSessionState", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(_jobsField, "Could not find Jobs field");
            Assert.NotNull(_currentJobIdField, "Could not find _currentJobId field");
            Assert.NotNull(_autoFailedField, "Could not find _autoFailedInitJobId field");
            Assert.NotNull(_persistMethod, "Could not find PersistToSessionState method");

            var statusType = asm.GetType("MCPForUnity.Editor.Services.TestRunStatus");
            Assert.NotNull(statusType, "Could not find TestRunStatus");
            _statusIsRunning = statusType.GetField("_isRunning", BindingFlags.NonPublic | BindingFlags.Static);
            _statusMode = statusType.GetField("_mode", BindingFlags.NonPublic | BindingFlags.Static);
            _statusStarted = statusType.GetField("_startedUnixMs", BindingFlags.NonPublic | BindingFlags.Static);
            _statusFinished = statusType.GetField("_finishedUnixMs", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(_statusIsRunning, "Could not find TestRunStatus._isRunning");
            Assert.NotNull(_statusMode, "Could not find TestRunStatus._mode");
            Assert.NotNull(_statusStarted, "Could not find TestRunStatus._startedUnixMs");
            Assert.NotNull(_statusFinished, "Could not find TestRunStatus._finishedUnixMs");

            _originalJobId = _currentJobIdField.GetValue(null) as string;
            _originalStatus = new[]
            {
                _statusIsRunning.GetValue(null),
                _statusMode.GetValue(null),
                _statusStarted.GetValue(null),
                _statusFinished.GetValue(null)
            };
        }

        [TearDown]
        public void TearDown()
        {
            _statusIsRunning.SetValue(null, _originalStatus[0]);
            _statusMode.SetValue(null, _originalStatus[1]);
            _statusStarted.SetValue(null, _originalStatus[2]);
            _statusFinished.SetValue(null, _originalStatus[3]);

            _currentJobIdField.SetValue(null, _originalJobId);
            _autoFailedField.SetValue(null, null); // auto-fail tests leave a tombstone; never leak it
            var jobs = Jobs();
            jobs.Remove(JobA);
            jobs.Remove(JobB);
            // Flush the cleaned state so synthetic jobs do not survive a domain reload.
            _persistMethod.Invoke(null, new object[] { true });
        }

        private Dictionary<string, TestJob> Jobs() =>
            (Dictionary<string, TestJob>)_jobsField.GetValue(null);

        private static TestJob NewJob(string jobId, long startedMsAgo, int? totalTests)
        {
            long now = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            return new TestJob
            {
                JobId = jobId,
                Status = TestJobStatus.Running,
                Mode = "EditMode",
                StartedUnixMs = now - startedMsAgo,
                LastUpdateUnixMs = now - startedMsAgo,
                TotalTests = totalTests,
                InitTimeoutMs = 0,
                FailuresSoFar = new List<TestJobFailure>()
            };
        }

        [Test]
        public void ClearStuckJob_ClearsRunFlag()
        {
            Jobs()[JobA] = NewJob(JobA, startedMsAgo: 1_000, totalTests: 4);
            _currentJobIdField.SetValue(null, JobA);
            TestRunStatus.MarkStarted(TestMode.EditMode);

            bool cleared = TestJobManager.ClearStuckJob();

            Assert.IsTrue(cleared, "A running job should report as cleared");
            Assert.IsNull(_currentJobIdField.GetValue(null), "Current job id should be dropped");
            Assert.IsFalse(TestRunStatus.IsRunning,
                "Clearing a stuck job must clear the run flag, or every exclusive call parks forever");
        }

        [Test]
        public void ClearStuckJob_ClearsWedgedRunFlag_WhenNoJobIsTracked()
        {
            _currentJobIdField.SetValue(null, null);
            TestRunStatus.MarkStarted(TestMode.EditMode);

            bool cleared = TestJobManager.ClearStuckJob();

            Assert.IsFalse(cleared, "No tracked job means nothing was cleared");
            Assert.IsFalse(TestRunStatus.IsRunning,
                "clear_stuck is the manual un-wedge lever and must reconcile a leaked run flag");
        }

        [Test]
        public void ClearStuckJob_LeavesIdleStatusUntouched()
        {
            _currentJobIdField.SetValue(null, null);
            _statusIsRunning.SetValue(null, false);
            _statusFinished.SetValue(null, null);

            Assert.IsFalse(TestJobManager.ClearStuckJob());
            Assert.IsNull(TestRunStatus.FinishedUnixMs,
                "Clearing when nothing ran must not fabricate a last-run timestamp");
        }

        [Test]
        public void GetJob_InitTimeout_ClearsRunFlag_WhenJobIdAlreadyDropped()
        {
            // The wedge shape: the id was dropped by an earlier clear, but the run flag
            // survived, so the id guard on the auto-fail path no longer matches.
            Jobs()[JobA] = NewJob(JobA, startedMsAgo: 70_000, totalTests: null);
            _currentJobIdField.SetValue(null, null);
            TestRunStatus.MarkStarted(TestMode.EditMode);

            var job = TestJobManager.GetJob(JobA);

            Assert.AreEqual(TestJobStatus.Failed, job.Status, "An uninitialized job past the 60s default auto-fails");
            Assert.IsFalse(TestRunStatus.IsRunning,
                "The init-timeout auto-fail must clear the run flag even when the id guard misses");
        }

        [Test]
        public void GetJob_InitTimeout_KeepsRunFlag_WhenNewerJobIsActive()
        {
            Jobs()[JobA] = NewJob(JobA, startedMsAgo: 70_000, totalTests: null);
            Jobs()[JobB] = NewJob(JobB, startedMsAgo: 1_000, totalTests: 4);
            _currentJobIdField.SetValue(null, JobB);
            TestRunStatus.MarkStarted(TestMode.EditMode);

            var job = TestJobManager.GetJob(JobA);

            Assert.AreEqual(TestJobStatus.Failed, job.Status);
            Assert.AreEqual(JobB, _currentJobIdField.GetValue(null), "The newer job stays active");
            Assert.IsTrue(TestRunStatus.IsRunning,
                "A stale job timing out must not clear the flag a newer run owns");
        }
    }
}
