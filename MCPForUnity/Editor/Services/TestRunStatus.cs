using System;
using MCPForUnity.Editor.Helpers;
using UnityEditor;
using UnityEditor.TestTools.TestRunner.Api;

namespace MCPForUnity.Editor.Services
{
    /// <summary>
    /// Thread-safe, minimal shared status for Unity Test Runner execution.
    /// Used by editor readiness snapshots so callers can avoid starting overlapping runs.
    ///
    /// A minimal snapshot (job id, mode, start time) is persisted to <see cref="SessionState"/> so a
    /// domain reload mid-run — routine for EditMode suites — does not silently disarm the busy flag
    /// for the remainder of the run. Restoration is reconciled against <see cref="TestJobManager"/>:
    /// the flag is only re-armed while the persisted job is still the live tracked job, so a wedged
    /// flag can never survive an editor reload via this persistence (anything without a live match is
    /// erased instead of restored).
    /// </summary>
    internal static class TestRunStatus
    {
        private const string SessionKey_JobId = "MCPForUnity.TestRunStatus.JobId";
        private const string SessionKey_Mode = "MCPForUnity.TestRunStatus.Mode";
        private const string SessionKey_StartedUnixMs = "MCPForUnity.TestRunStatus.StartedUnixMs";

        private static readonly object LockObj = new();

        private static bool _isRunning;
        private static TestMode? _mode;
        private static long? _startedUnixMs;
        private static long? _finishedUnixMs;

        static TestRunStatus()
        {
            TryRestoreAfterReload();
        }

        public static bool IsRunning
        {
            get { lock (LockObj) return _isRunning; }
        }

        public static TestMode? Mode
        {
            get { lock (LockObj) return _mode; }
        }

        public static long? StartedUnixMs
        {
            get { lock (LockObj) return _startedUnixMs; }
        }

        public static long? FinishedUnixMs
        {
            get { lock (LockObj) return _finishedUnixMs; }
        }

        /// <summary>
        /// Marks a run started. When <paramref name="jobId"/> identifies a tracked
        /// <see cref="TestJobManager"/> job, the run snapshot is persisted so a mid-run domain
        /// reload can re-arm the flag; without a job id nothing is persisted (there is no record
        /// to reconcile against on restore, so persisting would only manufacture wedges).
        /// </summary>
        public static void MarkStarted(TestMode mode, string jobId = null)
        {
            long started = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            lock (LockObj)
            {
                _isRunning = true;
                _mode = mode;
                _startedUnixMs = started;
                _finishedUnixMs = null;
            }

            if (!string.IsNullOrEmpty(jobId))
            {
                try
                {
                    SessionState.SetString(SessionKey_JobId, jobId);
                    SessionState.SetString(SessionKey_Mode, mode.ToString());
                    SessionState.SetString(SessionKey_StartedUnixMs, started.ToString());
                }
                catch (Exception e)
                {
                    McpLog.Warn($"[TestRunStatus] Failed to persist run snapshot: {e.Message}");
                }
            }
        }

        public static void MarkFinished()
        {
            lock (LockObj)
            {
                _isRunning = false;
                _finishedUnixMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                _mode = null;
            }
            ErasePersisted();
        }

        private static void ErasePersisted()
        {
            try
            {
                SessionState.EraseString(SessionKey_JobId);
                SessionState.EraseString(SessionKey_Mode);
                SessionState.EraseString(SessionKey_StartedUnixMs);
            }
            catch (Exception e)
            {
                McpLog.Warn($"[TestRunStatus] Failed to erase run snapshot: {e.Message}");
            }
        }

        /// <summary>
        /// Restores the run flag after a domain reload when — and only when — the persisted job is
        /// still the job <see cref="TestJobManager"/> tracks as current (its own restore path drops
        /// finished, stale, and orphaned jobs first). Any persisted snapshot without a live match is
        /// erased: this persistence exists to keep a genuine mid-run reload armed, never to let a
        /// wedged flag outlive the run that set it.
        /// </summary>
        private static void TryRestoreAfterReload()
        {
            try
            {
                string jobId = SessionState.GetString(SessionKey_JobId, string.Empty);
                if (string.IsNullOrEmpty(jobId))
                {
                    return;
                }

                if (!string.Equals(TestJobManager.CurrentJobId, jobId, StringComparison.Ordinal))
                {
                    ErasePersisted();
                    return;
                }

                TestMode? mode = null;
                string modeStr = SessionState.GetString(SessionKey_Mode, string.Empty);
                if (Enum.TryParse<TestMode>(modeStr, ignoreCase: true, out var parsedMode))
                {
                    mode = parsedMode;
                }

                long started = 0;
                long.TryParse(SessionState.GetString(SessionKey_StartedUnixMs, string.Empty), out started);
                if (started <= 0)
                {
                    started = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                }

                lock (LockObj)
                {
                    _isRunning = true;
                    _mode = mode;
                    _startedUnixMs = started;
                    _finishedUnixMs = null;
                }
                McpLog.Info($"[TestRunStatus] Restored running flag for live job {jobId} after domain reload.", always: false);
            }
            catch (Exception e)
            {
                McpLog.Warn($"[TestRunStatus] Failed to restore run snapshot: {e.Message}");
            }
        }
    }
}
