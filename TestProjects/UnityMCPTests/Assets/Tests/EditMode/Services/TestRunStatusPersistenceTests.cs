using System;
using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using MCPForUnity.Editor.Services;
using UnityEditor;
using UnityEditor.TestTools.TestRunner.Api;

namespace MCPForUnityTests.Editor.Services
{
    /// <summary>
    /// Tests for <see cref="TestRunStatus"/> SessionState persistence: a domain reload mid-run
    /// (routine for EditMode suites) must not silently disarm the run flag, but restoration is
    /// reconciled against <see cref="TestJobManager"/> — a persisted snapshot with no live matching
    /// job is erased, never restored, so a wedge cannot survive an editor reload through this path.
    /// </summary>
    public class TestRunStatusPersistenceTests
    {
        private const string JobP = "trs-persist-job";

        // Mirrored private consts from TestRunStatus.
        private const string Key_JobId = "MCPForUnity.TestRunStatus.JobId";
        private const string Key_Mode = "MCPForUnity.TestRunStatus.Mode";
        private const string Key_Started = "MCPForUnity.TestRunStatus.StartedUnixMs";

        private FieldInfo _statusIsRunning;
        private FieldInfo _statusMode;
        private FieldInfo _statusStarted;
        private FieldInfo _statusFinished;
        private MethodInfo _restoreMethod;

        private FieldInfo _jobsField;
        private FieldInfo _currentJobIdField;
        private MethodInfo _persistMethod;

        private object[] _originalStatus;
        private string _originalJobId;
        private string _origKeyJobId;
        private string _origKeyMode;
        private string _origKeyStarted;

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
            _restoreMethod = statusType.GetMethod("TryRestoreAfterReload", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(_statusIsRunning, "Could not find TestRunStatus._isRunning");
            Assert.NotNull(_restoreMethod, "Could not find TestRunStatus.TryRestoreAfterReload");

            var managerType = asm.GetType("MCPForUnity.Editor.Services.TestJobManager");
            Assert.NotNull(managerType, "Could not find TestJobManager");
            _jobsField = managerType.GetField("Jobs", BindingFlags.NonPublic | BindingFlags.Static);
            _currentJobIdField = managerType.GetField("_currentJobId", BindingFlags.NonPublic | BindingFlags.Static);
            _persistMethod = managerType.GetMethod("PersistToSessionState", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(_jobsField, "Could not find Jobs field");
            Assert.NotNull(_currentJobIdField, "Could not find _currentJobId field");
            Assert.NotNull(_persistMethod, "Could not find PersistToSessionState method");

            _originalStatus = new[]
            {
                _statusIsRunning.GetValue(null),
                _statusMode.GetValue(null),
                _statusStarted.GetValue(null),
                _statusFinished.GetValue(null)
            };
            _originalJobId = _currentJobIdField.GetValue(null) as string;
            _origKeyJobId = SessionState.GetString(Key_JobId, string.Empty);
            _origKeyMode = SessionState.GetString(Key_Mode, string.Empty);
            _origKeyStarted = SessionState.GetString(Key_Started, string.Empty);

            // Each test starts with no persisted snapshot.
            SessionState.EraseString(Key_JobId);
            SessionState.EraseString(Key_Mode);
            SessionState.EraseString(Key_Started);
        }

        [TearDown]
        public void TearDown()
        {
            _statusIsRunning.SetValue(null, _originalStatus[0]);
            _statusMode.SetValue(null, _originalStatus[1]);
            _statusStarted.SetValue(null, _originalStatus[2]);
            _statusFinished.SetValue(null, _originalStatus[3]);

            RestoreKey(Key_JobId, _origKeyJobId);
            RestoreKey(Key_Mode, _origKeyMode);
            RestoreKey(Key_Started, _origKeyStarted);

            _currentJobIdField.SetValue(null, _originalJobId);
            Jobs().Remove(JobP);
            _persistMethod.Invoke(null, new object[] { true });
        }

        private static void RestoreKey(string key, string original)
        {
            if (string.IsNullOrEmpty(original))
            {
                SessionState.EraseString(key);
            }
            else
            {
                SessionState.SetString(key, original);
            }
        }

        private Dictionary<string, TestJob> Jobs() =>
            (Dictionary<string, TestJob>)_jobsField.GetValue(null);

        private void InsertRunningJob()
        {
            long now = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            Jobs()[JobP] = new TestJob
            {
                JobId = JobP,
                Status = TestJobStatus.Running,
                Mode = "EditMode",
                StartedUnixMs = now,
                LastUpdateUnixMs = now,
                TotalTests = 4,
                FailuresSoFar = new List<TestJobFailure>()
            };
        }

        private void ForceNotRunning()
        {
            _statusIsRunning.SetValue(null, false);
            _statusMode.SetValue(null, null);
            _statusStarted.SetValue(null, null);
            _statusFinished.SetValue(null, null);
        }

        [Test]
        public void MarkStarted_WithJobId_PersistsSnapshot()
        {
            TestRunStatus.MarkStarted(TestMode.EditMode, JobP);

            Assert.AreEqual(JobP, SessionState.GetString(Key_JobId, string.Empty),
                "The run snapshot must persist keyed to the active job");
            Assert.AreEqual("EditMode", SessionState.GetString(Key_Mode, string.Empty));
            Assert.IsTrue(long.TryParse(SessionState.GetString(Key_Started, string.Empty), out long started) && started > 0,
                "The start time must persist so a restored flag keeps its original clock");

            TestRunStatus.MarkFinished();
            Assert.AreEqual(string.Empty, SessionState.GetString(Key_JobId, string.Empty),
                "MarkFinished must erase the persisted snapshot");
        }

        [Test]
        public void MarkStarted_WithoutJobId_DoesNotPersist()
        {
            TestRunStatus.MarkStarted(TestMode.EditMode);

            Assert.AreEqual(string.Empty, SessionState.GetString(Key_JobId, string.Empty),
                "Without a job to reconcile against there must be no persisted snapshot — it could never be validated on restore");

            TestRunStatus.MarkFinished();
        }

        [Test]
        public void RestoreAfterReload_WithMatchingLiveJob_RearmsFlag()
        {
            InsertRunningJob();
            _currentJobIdField.SetValue(null, JobP);
            SessionState.SetString(Key_JobId, JobP);
            SessionState.SetString(Key_Mode, "EditMode");
            SessionState.SetString(Key_Started, "1234567890123");
            ForceNotRunning(); // the state a domain reload leaves behind

            _restoreMethod.Invoke(null, null);

            Assert.IsTrue(TestRunStatus.IsRunning,
                "A mid-run domain reload must re-arm the flag while the job is still live — "
                + "otherwise the gate and defer funnel are disarmed for the rest of the run");
            Assert.AreEqual(TestMode.EditMode, TestRunStatus.Mode);
            Assert.AreEqual(1234567890123L, TestRunStatus.StartedUnixMs,
                "The restored flag keeps the original start time");
        }

        [Test]
        public void RestoreAfterReload_WithoutMatchingJob_ErasesInsteadOfRestoring()
        {
            _currentJobIdField.SetValue(null, null); // no live job matches the snapshot
            SessionState.SetString(Key_JobId, "some-dead-job");
            SessionState.SetString(Key_Mode, "EditMode");
            SessionState.SetString(Key_Started, "1234567890123");
            ForceNotRunning();

            _restoreMethod.Invoke(null, null);

            Assert.IsFalse(TestRunStatus.IsRunning,
                "A snapshot with no live matching job must NOT be restored — a wedge must never survive a reload via this persistence");
            Assert.AreEqual(string.Empty, SessionState.GetString(Key_JobId, string.Empty),
                "The unmatched snapshot must be erased so it cannot resurrect later");
        }

        [Test]
        public void ClearStuckJob_ErasesPersistedSnapshot()
        {
            InsertRunningJob();
            _currentJobIdField.SetValue(null, JobP);
            TestRunStatus.MarkStarted(TestMode.EditMode, JobP);
            Assert.AreEqual(JobP, SessionState.GetString(Key_JobId, string.Empty), "Precondition: snapshot persisted");

            bool cleared = TestJobManager.ClearStuckJob();

            Assert.IsTrue(cleared, "The running job should report as cleared");
            Assert.IsFalse(TestRunStatus.IsRunning);
            Assert.AreEqual(string.Empty, SessionState.GetString(Key_JobId, string.Empty),
                "clear_stuck must clear the persisted copy too — the un-wedge lever cannot leave a resurrection path");
        }
    }
}
