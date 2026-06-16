using System;
using System.Threading.Tasks;
using MCPForUnity.Editor.Constants;
using MCPForUnity.Editor.Helpers;
using MCPForUnity.Editor.Services.Transport;
using MCPForUnity.Editor.Windows;
using UnityEditor;
using UnityEngine;

namespace MCPForUnity.Editor.Services
{
    /// <summary>
    /// Automatically starts the HTTP MCP bridge on editor load when the user has opted in
    /// via the "Auto-Start on Editor Load" toggle in Advanced Settings.
    /// This complements HttpBridgeReloadHandler (which only resumes after domain reloads).
    /// </summary>
    [InitializeOnLoad]
    internal static class HttpAutoStartHandler
    {
        private const string SessionInitKey = "HttpAutoStartHandler.SessionInitialized";

        static HttpAutoStartHandler()
        {
            // SessionState resets on editor process start but persists across domain reloads.
            // Only run once per session — let HttpBridgeReloadHandler handle reload-resume cases.
            if (SessionState.GetBool(SessionInitKey, false)) return;

            if (Application.isBatchMode &&
                string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("UNITY_MCP_ALLOW_BATCH")))
            {
                return;
            }

            // Only check lightweight EditorPrefs here — services like EditorConfigurationCache
            // and MCPServiceLocator may not be initialized yet on fresh editor launch.
            bool autoStartEnabled = EditorPrefs.GetBool(EditorPrefKeys.AutoStartOnLoad, false);
            if (!autoStartEnabled) return;

            SessionState.SetBool(SessionInitKey, true);

            // Delay to let the editor and services finish initialization.
            EditorApplication.delayCall += OnEditorReady;
        }

        private static void OnEditorReady()
        {
            try
            {
                bool autoStartEnabled = EditorPrefs.GetBool(EditorPrefKeys.AutoStartOnLoad, false);
                if (!autoStartEnabled) return;

                bool useHttp = EditorConfigurationCache.Instance.UseHttpTransport;
                if (!useHttp) return;

                // Don't auto-start if bridge is already running.
                if (MCPServiceLocator.TransportManager.IsRunning(TransportMode.Http)) return;

                _ = AutoStartAsync();
            }
            catch (Exception ex)
            {
                McpLog.Debug($"[HTTP Auto-Start] Deferred check failed: {ex.Message}");
            }
        }

        private static async Task AutoStartAsync()
        {
            try
            {
                bool isLocal = !HttpEndpointUtility.IsRemoteScope();

                if (isLocal)
                {
                    // For HTTP Local: launch the server process first, then connect the bridge.
                    // This mirrors what the UI "Start Server" button does.
                    if (!HttpEndpointUtility.IsHttpLocalUrlAllowedForLaunch(
                            HttpEndpointUtility.GetLocalBaseUrl(), out string policyError))
                    {
                        McpLog.Debug($"[HTTP Auto-Start] Local URL blocked by security policy: {policyError}");
                        return;
                    }

                    // Check if server is already reachable (e.g. it survived a graceful quit, or the
                    // user started it externally). Reuse is the intended path on editor restart.
                    // This fact is only knowable here, before we (possibly) launch one ourselves (MCPL-013).
                    bool reusedExistingServer = MCPServiceLocator.Server.IsLocalHttpServerReachable();
                    if (reusedExistingServer)
                    {
                        // We did not launch this server: query /health and compare versions (MCPL-010/011).
                        await PerformReuseVersionCheckAsync();
                    }
                    else
                    {
                        bool serverStarted = MCPServiceLocator.Server.StartLocalHttpServer(quiet: true);
                        if (!serverStarted)
                        {
                            McpLog.Warn("[HTTP Auto-Start] Failed to start local HTTP server");
                            return;
                        }
                        ServerReuseState.RecordStarted();
                    }

                    // Wait for the server to become reachable, then connect.
                    await WaitForServerAndConnectAsync();
                }
                else
                {
                    // For HTTP Remote: server is external, just connect the bridge.
                    await ConnectBridgeAsync();
                }
            }
            catch (Exception ex)
            {
                McpLog.Warn($"[HTTP Auto-Start] Failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Queries the reused server's /health endpoint and compares its reported version against
        /// this bridge's package version. Warn-only — never kills the server (MCPL-010/011). Records
        /// the reuse fact + version into SessionState for the window/toolbar to surface (MCPL-013).
        /// </summary>
        private static async Task PerformReuseVersionCheckAsync()
        {
            string bridgeVersion = AssetPathUtility.GetPackageVersion();
            string nowIso = DateTime.UtcNow.ToString("o");

            string healthJson = await FetchHealthAsync();
            if (healthJson == null)
            {
                // Reachable on the TCP probe but /health did not respond: treat as an unknown listener.
                McpLog.Warn("[HTTP Auto-Start] Reused a reachable local server but its /health endpoint did not respond — treating as an unknown listener.");
                ServerReuseState.RecordReused(string.Empty, nowIso, versionMismatch: true);
                return;
            }

            var result = ServerHealthCheck.CompareHealthVersion(healthJson, bridgeVersion, out string serverVersion);
            switch (result)
            {
                case ServerVersionCheckResult.Match:
                    McpLog.Info($"[HTTP Auto-Start] Reused running local server (version {serverVersion}).");
                    ServerReuseState.RecordReused(serverVersion, nowIso, versionMismatch: false);
                    break;
                case ServerVersionCheckResult.BridgeVersionUnknown:
                    McpLog.Warn("[HTTP Auto-Start] Reused running local server but bridge package version is unknown — skipping version check.");
                    ServerReuseState.RecordReused(serverVersion, nowIso, versionMismatch: false);
                    break;
                case ServerVersionCheckResult.Unparseable:
                    McpLog.Warn("[HTTP Auto-Start] A process is listening on the local server port but its /health response is not recognizable as an MCP-for-Unity server (unknown listener).");
                    ServerReuseState.RecordReused(string.Empty, nowIso, versionMismatch: true);
                    break;
                default: // Mismatch
                    McpLog.Warn($"[HTTP Auto-Start] Reused running local server version {serverVersion} != bridge version {bridgeVersion} — restart server to update.");
                    ServerReuseState.RecordReused(serverVersion, nowIso, versionMismatch: true);
                    break;
            }
        }

        /// <summary>
        /// Performs a bounded HTTP GET against the local server's /health endpoint.
        /// Returns the raw response body, or null on any failure (timeout, refused, non-success).
        /// </summary>
        private static async Task<string> FetchHealthAsync()
        {
            try
            {
                string baseUrl = HttpEndpointUtility.GetLocalBaseUrl();
                if (string.IsNullOrEmpty(baseUrl))
                {
                    return null;
                }

                string healthEndpoint = $"{baseUrl.TrimEnd('/')}/health";
                using (var client = new System.Net.Http.HttpClient())
                {
                    client.Timeout = TimeSpan.FromSeconds(3);
                    var response = await client.GetAsync(healthEndpoint);
                    if (!response.IsSuccessStatusCode)
                    {
                        return null;
                    }
                    return await response.Content.ReadAsStringAsync();
                }
            }
            catch (Exception ex)
            {
                McpLog.Debug($"[HTTP Auto-Start] /health query failed: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Waits for the local HTTP server to accept connections, then connects the bridge.
        /// Mirrors TryAutoStartSessionAsync in McpConnectionSection: keep polling reachability while
        /// the launched process is alive; declare failure only when it exits without the port coming
        /// up, or a generous hard cap is reached.
        /// </summary>
        private static async Task WaitForServerAndConnectAsync()
        {
            var server = MCPServiceLocator.Server;
            string url = HttpEndpointUtility.GetLocalBaseUrl();
            var pollDelay = TimeSpan.FromMilliseconds(500);
            var hardCap = TimeSpan.FromMinutes(5);
            double startTime = EditorApplication.timeSinceStartup;

            while (true)
            {
                // Abort if user changed settings while we were waiting.
                if (!EditorPrefs.GetBool(EditorPrefKeys.AutoStartOnLoad, false)) return;
                if (!EditorConfigurationCache.Instance.UseHttpTransport) return;
                if (MCPServiceLocator.TransportManager.IsRunning(TransportMode.Http)) return;

                if (server.IsLocalHttpServerReachable())
                {
                    McpLog.Info($"Server ready on {url}");
                    bool started = await MCPServiceLocator.Bridge.StartAsync();
                    if (started)
                    {
                        McpLog.Info("Session connected");
                        MCPForUnityEditorWindow.RequestHealthVerification();
                        return;
                    }
                }

                bool processAlive = server.IsManagedServerLaunchProcessAlive();
                double elapsed = EditorApplication.timeSinceStartup - startTime;

                if ((!processAlive && elapsed > 1.0) || elapsed > hardCap.TotalSeconds)
                {
                    // Last-resort connect attempt in case reachability detection missed a live server.
                    if (await MCPServiceLocator.Bridge.StartAsync())
                    {
                        McpLog.Info("Session connected");
                        MCPForUnityEditorWindow.RequestHealthVerification();
                        return;
                    }

                    server.LogLocalHttpServerLaunchFailure();
                    return;
                }

                try { await Task.Delay(pollDelay); }
                catch { return; }
            }
        }

        /// <summary>
        /// Connects the bridge directly (for remote HTTP where the server is already running).
        /// </summary>
        private static async Task ConnectBridgeAsync()
        {
            string url = HttpEndpointUtility.GetRemoteBaseUrl();
            McpLog.Info($"Connecting to {url}…");
            bool started = await MCPServiceLocator.Bridge.StartAsync();
            if (started)
            {
                McpLog.Info("Connected");
                MCPForUnityEditorWindow.RequestHealthVerification();
            }
            else
            {
                McpLog.Warn("Connection failed: could not connect to remote HTTP server");
            }
        }
    }
}
