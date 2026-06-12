using UnityEditor;

namespace MCPForUnity.Editor.Services
{
    /// <summary>
    /// Records, per editor session, whether the bridge reused an already-running local HTTP
    /// server or started one itself, plus the reused server's reported version and start time.
    /// Backed by SessionState so the facts survive domain reloads but reset on editor restart —
    /// which matches when "did this editor start or reuse the server?" can change (MCPL-013).
    /// The window/toolbar reads these to distinguish "reused" from "started by this editor".
    /// </summary>
    public static class ServerReuseState
    {
        private const string KeyReused = "MCPForUnity.ServerReuse.Reused";
        private const string KeyVersion = "MCPForUnity.ServerReuse.Version";
        private const string KeyStartedAtIso = "MCPForUnity.ServerReuse.StartedAtIso";
        private const string KeyVersionMismatch = "MCPForUnity.ServerReuse.VersionMismatch";

        /// <summary>True when this editor session connected to a server it did not launch.</summary>
        public static bool ReusedExistingServer
        {
            get => SessionState.GetBool(KeyReused, false);
            private set => SessionState.SetBool(KeyReused, value);
        }

        /// <summary>The reused server's reported version, or empty if not reused / unknown.</summary>
        public static string ReusedServerVersion => SessionState.GetString(KeyVersion, string.Empty);

        /// <summary>
        /// ISO-8601 timestamp this editor first observed/reconnected to the reused server, or empty.
        /// The /health endpoint does not report the server's own start time, so this is the
        /// reconnect time — a best-effort stand-in for "running since".
        /// </summary>
        public static string ReusedServerStartedAtIso => SessionState.GetString(KeyStartedAtIso, string.Empty);

        /// <summary>True when a reused server's version did not match this bridge's package version.</summary>
        public static bool HasVersionMismatch => SessionState.GetBool(KeyVersionMismatch, false);

        /// <summary>Record that this editor reused an already-running server.</summary>
        public static void RecordReused(string serverVersion, string startedAtIso, bool versionMismatch)
        {
            ReusedExistingServer = true;
            SessionState.SetString(KeyVersion, serverVersion ?? string.Empty);
            SessionState.SetString(KeyStartedAtIso, startedAtIso ?? string.Empty);
            SessionState.SetBool(KeyVersionMismatch, versionMismatch);
        }

        /// <summary>Record that this editor started the server itself.</summary>
        public static void RecordStarted()
        {
            ReusedExistingServer = false;
            SessionState.SetString(KeyVersion, string.Empty);
            SessionState.SetString(KeyStartedAtIso, string.Empty);
            SessionState.SetBool(KeyVersionMismatch, false);
        }
    }
}
