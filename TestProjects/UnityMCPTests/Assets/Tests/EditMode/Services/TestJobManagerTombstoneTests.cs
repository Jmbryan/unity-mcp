using System;
using System.Collections.Generic;
using System.Reflection;
using NUnit.Framework;
using MCPForUnity.Editor.Services;
using UnityEditor.TestTools.TestRunner.Api;

namespace MCPForUnityTests.Editor.Services
{
    /// <summary>
    /// Tests for the init-timeout tombstone in <see cref="TestJobManager"/>. The auto-fail in
    /// GetJob is a wall-clock guess — Unity may still be building the test tree when it fires —
    /// so the auto-fail leaves a tombstone and a late RunStarted revives-and-adopts the job
    /// (re-arming the run flag), while a late RunFinished adopts the result. Both orderings are
    /// covered, plus the tombstone's expiry and mode guards.
    /// </summary>
    public class TestJobManagerTombstoneTests
    {
        private const string JobT = "tombstone-job";

        // TestRunStatus persisted-snapshot keys (mirrored private consts) — the revive path calls
        // MarkStarted with a job id, which persists; restore/erase these around each test.
        private const string StatusKey_JobId = "MCPForUnity.TestRunStatus.JobId";
        private const string StatusKey_Mode = "MCPForUnity.TestRunStatus.Mode";
        private const string StatusKey_Started = "MCPForUnity.TestRunStatus.StartedUnixMs";

        private FieldInfo _jobsField;
        private FieldInfo _currentJobIdField;
        private FieldInfo _autoFailedField;
        private MethodInfo _persistMethod;

        private FieldInfo _statusIsRunning;
        private FieldInfo _statusMode;
        private FieldInfo _statusStarted;
        private FieldInfo _statusFinished;

        private string _originalJobId;
        private string _originalAutoFailed;
        private object[] _originalStatus;
        private string _origKeyJobId;
        private string _origKeyMode;
        private string _origKeyStarted;

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

            _originalJobId = _currentJobIdField.GetValue(null) as string;
            _originalAutoFailed = _autoFailedField.GetValue(null) as string;
            _originalStatus = new[]
            {
                _statusIsRunning.GetValue(null),
                _statusMode.GetValue(null),
                _statusStarted.GetValue(null),
                _statusFinished.GetValue(null)
            };
            _origKeyJobId = UnityEditor.SessionState.GetString(StatusKey_JobId, string.Empty);
            _origKeyMode = UnityEditor.SessionState.GetString(StatusKey_Mode, string.Empty);
            _origKeyStarted = UnityEditor.SessionState.GetString(StatusKey_Started, string.Empty);
        }

        [TearDown]
        public void TearDown()
        {
            _statusIsRunning.SetValue(null, _originalStatus[0]);
            _statusMode.SetValue(null, _originalStatus[1]);
            _statusStarted.SetValue(null, _originalStatus[2]);
            _statusFinished.SetValue(null, _originalStatus[3]);

            RestoreSessionKey(StatusKey_JobId, _origKeyJobId);
            RestoreSessionKey(StatusKey_Mode, _origKeyMode);
            RestoreSessionKey(StatusKey_Started, _origKeyStarted);

            _currentJobIdField.SetValue(null, _originalJobId);
            _autoFailedField.SetValue(null, _originalAutoFailed);
            Jobs().Remove(JobT);
            _persistMethod.Invoke(null, new object[] { true });
        }

        private static void RestoreSessionKey(string key, string original)
        {
            if (string.IsNullOrEmpty(original))
            {
                UnityEditor.SessionState.EraseString(key);
            }
            else
            {
                UnityEditor.SessionState.SetString(key, original);
            }
        }

        private Dictionary<string, TestJob> Jobs() =>
            (Dictionary<string, TestJob>)_jobsField.GetValue(null);

        private string Tombstone() => _autoFailedField.GetValue(null) as string;

        /// <summary>Inserts an uninitialized running job and drives the GetJob auto-fail over it.</summary>
        private TestJob AutoFailJob(long startedMsAgo)
        {
            long now = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            var job = new TestJob
            {
                JobId = JobT,
                Status = TestJobStatus.Running,
                Mode = "EditMode",
                StartedUnixMs = now - startedMsAgo,
                LastUpdateUnixMs = now - startedMsAgo,
                TotalTests = null, // never reached RunStarted
                InitTimeoutMs = 0, // use the 60s default
                FailuresSoFar = new List<TestJobFailure>()
            };
            Jobs()[JobT] = job;
            _currentJobIdField.SetValue(null, JobT);
            TestRunStatus.MarkStarted(TestMode.EditMode);

            var polled = TestJobManager.GetJob(JobT);
            Assert.AreEqual(TestJobStatus.Failed, polled.Status, "Precondition: the init timeout auto-fails the job");
            Assert.IsNull(_currentJobIdField.GetValue(null), "Precondition: auto-fail drops the current job id");
            Assert.AreEqual(JobT, Tombstone(), "Precondition: auto-fail leaves a tombstone for late-run reconciliation");
            Assert.IsFalse(TestRunStatus.IsRunning, "Precondition: auto-fail clears the run flag");
            return job;
        }

        [Test]
        public void LateRunStarted_RevivesAndAdoptsAutoFailedJob()
        {
            var job = AutoFailJob(startedMsAgo: 70_000);

            // Unity finished building the test tree after the auto-fail: the real run starts.
            TestJobManager.OnRunStarted(42, "EditMode");

            Assert.AreEqual(TestJobStatus.Running, job.Status, "The tombstoned job must be revived, not fought");
            Assert.IsNull(job.Error, "The init-timeout error must be cleared on revive");
            Assert.AreEqual(42, job.TotalTests, "The live run's totals must be adopted into the revived job");
            Assert.AreEqual(JobT, _currentJobIdField.GetValue(null), "The revived job becomes current again");
            Assert.IsNull(Tombstone(), "The tombstone is consumed by the revive");
            Assert.IsTrue(TestRunStatus.IsRunning,
                "Revive must re-arm the run flag the auto-fail cleared, so the gate and defer funnel cover the adopted run");
        }

        [Test]
        public void LateRunFinished_AdoptsResultIntoAutoFailedJob()
        {
            var job = AutoFailJob(startedMsAgo: 70_000);

            // The other ordering: no RunStarted ever fires (run failed before the tree callback),
            // but RunFinished still arrives for the tombstoned job.
            var payload = TestRunResult.Create(null, new List<ITestResultAdaptor>());
            TestJobManager.FinalizeCurrentJobFromRunFinished(payload);

            Assert.AreEqual(TestJobStatus.Succeeded, job.Status,
                "The late result must be adopted into the tombstoned job instead of being discarded");
            Assert.IsNull(job.Error, "The init-timeout error must be replaced by the real outcome");
            Assert.IsNotNull(job.FinishedUnixMs, "Adoption must stamp a finish time");
            Assert.IsNull(_currentJobIdField.GetValue(null), "No job remains current after adoption");
            Assert.IsNull(Tombstone(), "The tombstone is consumed by the adoption");
        }

        [Test]
        public void ExpiredTombstone_DoesNotAdoptLateRun()
        {
            // Job started 11 minutes ago — past the 10-minute init hard cap. A run starting this
            // late cannot be the tombstoned job's init (e.g. it is a human's Test Runner UI run).
            var job = AutoFailJob(startedMsAgo: 660_000);

            TestJobManager.OnRunStarted(5, "EditMode");

            Assert.AreEqual(TestJobStatus.Failed, job.Status, "An expired tombstone must not revive");
            Assert.IsNull(_currentJobIdField.GetValue(null), "No job is adopted");
            Assert.IsNull(Tombstone(), "An expired tombstone is discarded on first consultation");
        }

        [Test]
        public void ModeMismatch_DoesNotAdoptLateRun_AndKeepsTombstone()
        {
            var job = AutoFailJob(startedMsAgo: 70_000);

            // A PlayMode run starting now cannot be the EditMode job's late init.
            TestJobManager.OnRunStarted(5, "PlayMode");

            Assert.AreEqual(TestJobStatus.Failed, job.Status, "A different-mode run must not revive the job");
            Assert.IsNull(_currentJobIdField.GetValue(null), "No job is adopted");
            Assert.AreEqual(JobT, Tombstone(),
                "A mode mismatch leaves the tombstone in place for the run it actually belongs to");
        }
    }

    /// <summary>
    /// Tests for test-job owner attribution: <see cref="TestJobManager.CurrentJobStartedBy"/>
    /// resolves the tracked job's owner label (feeding tests.started_by in the editor-state
    /// snapshot) and the label round-trips SessionState persistence across a domain reload.
    /// </summary>
    public class TestJobStartedByTests
    {
        private const string JobS = "started-by-job";

        private FieldInfo _jobsField;
        private FieldInfo _currentJobIdField;
        private MethodInfo _persistMethod;
        private MethodInfo _restoreMethod;

        private string _originalJobId;

        [SetUp]
        public void SetUp()
        {
            var asm = typeof(MCPServiceLocator).Assembly;
            var managerType = asm.GetType("MCPForUnity.Editor.Services.TestJobManager");
            Assert.NotNull(managerType, "Could not find TestJobManager");
            _jobsField = managerType.GetField("Jobs", BindingFlags.NonPublic | BindingFlags.Static);
            _currentJobIdField = managerType.GetField("_currentJobId", BindingFlags.NonPublic | BindingFlags.Static);
            _persistMethod = managerType.GetMethod("PersistToSessionState", BindingFlags.NonPublic | BindingFlags.Static);
            _restoreMethod = managerType.GetMethod("TryRestoreFromSessionState", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.NotNull(_jobsField, "Could not find Jobs field");
            Assert.NotNull(_currentJobIdField, "Could not find _currentJobId field");
            Assert.NotNull(_persistMethod, "Could not find PersistToSessionState method");
            Assert.NotNull(_restoreMethod, "Could not find TryRestoreFromSessionState method");

            _originalJobId = _currentJobIdField.GetValue(null) as string;
        }

        [TearDown]
        public void TearDown()
        {
            _currentJobIdField.SetValue(null, _originalJobId);
            Jobs().Remove(JobS);
            _persistMethod.Invoke(null, new object[] { true });
        }

        private Dictionary<string, TestJob> Jobs() =>
            (Dictionary<string, TestJob>)_jobsField.GetValue(null);

        private TestJob InsertJob(string startedBy)
        {
            long now = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            var job = new TestJob
            {
                JobId = JobS,
                Status = TestJobStatus.Running,
                Mode = "EditMode",
                StartedUnixMs = now,
                LastUpdateUnixMs = now,
                TotalTests = 4,
                StartedBy = startedBy,
                FailuresSoFar = new List<TestJobFailure>()
            };
            Jobs()[JobS] = job;
            return job;
        }

        [Test]
        public void CurrentJobStartedBy_ReturnsOwnerLabel_OfTrackedJob()
        {
            InsertJob("Aurora");
            _currentJobIdField.SetValue(null, JobS);

            Assert.AreEqual("Aurora", TestJobManager.CurrentJobStartedBy,
                "The snapshot's tests.started_by must name the real owner, not \"unknown\"");
        }

        [Test]
        public void CurrentJobStartedBy_IsNull_WhenNoJobTracked()
        {
            _currentJobIdField.SetValue(null, null);

            Assert.IsNull(TestJobManager.CurrentJobStartedBy,
                "No tracked job (e.g. a Unity Test Runner UI run) has no owner; the snapshot falls back to \"unknown\"");
        }

        [Test]
        public void CurrentJobStartedBy_IsNull_WhenJobHasNoOwnerLabel()
        {
            InsertJob(null);
            _currentJobIdField.SetValue(null, JobS);

            Assert.IsNull(TestJobManager.CurrentJobStartedBy);
        }

        [Test]
        public void StartedBy_SurvivesPersistAndRestore()
        {
            InsertJob("Aurora");
            _currentJobIdField.SetValue(null, JobS);

            _persistMethod.Invoke(null, new object[] { true });
            Jobs().Remove(JobS);
            _currentJobIdField.SetValue(null, null);
            _restoreMethod.Invoke(null, null);

            Assert.IsTrue(Jobs().ContainsKey(JobS), "Job should be restored from SessionState");
            Assert.AreEqual("Aurora", Jobs()[JobS].StartedBy,
                "Owner attribution must survive the domain reload a mid-run compile triggers");
        }
    }
}
