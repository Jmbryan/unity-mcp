using System;
using Newtonsoft.Json.Linq;

namespace MCPForUnity.Editor.Services
{
    /// <summary>
    /// Classifies the outcome of comparing a reused server's reported /health version
    /// against this bridge's own package version. Pure logic so it is unit-testable
    /// without a live server; the HTTP fetch lives in the caller.
    /// </summary>
    internal enum ServerVersionCheckResult
    {
        /// <summary>Reported server version matches the bridge package version.</summary>
        Match,

        /// <summary>Reported server version differs from the bridge package version (warn, never kill).</summary>
        Mismatch,

        /// <summary>Bridge package version is "unknown" — the comparison was skipped.</summary>
        BridgeVersionUnknown,

        /// <summary>The /health response could not be parsed as MCP-server health JSON (unknown listener).</summary>
        Unparseable,
    }

    /// <summary>
    /// Pure-logic helper for the reuse-time version/identity check (MCPL-010, MCPL-011).
    /// </summary>
    internal static class ServerHealthCheck
    {
        /// <summary>
        /// Compares a /health JSON body against the bridge package version.
        /// </summary>
        /// <param name="healthJson">Raw body returned by the server's GET /health endpoint.</param>
        /// <param name="bridgeVersion">This bridge's package version (e.g. AssetPathUtility.GetPackageVersion()).</param>
        /// <param name="serverVersion">The version string reported by the server, or null if it could not be read.</param>
        /// <returns>The classified outcome.</returns>
        public static ServerVersionCheckResult CompareHealthVersion(string healthJson, string bridgeVersion, out string serverVersion)
        {
            serverVersion = null;

            if (!TryParseHealthVersion(healthJson, out serverVersion))
            {
                // Body is missing, not JSON, or lacks the MCP health shape: a non-MCP process may be
                // squatting on the port. Surface as an unknown listener rather than silently reusing it.
                return ServerVersionCheckResult.Unparseable;
            }

            // Bridge version unknown (e.g. package.json not found): skip the comparison, warn only.
            if (string.IsNullOrEmpty(bridgeVersion) || string.Equals(bridgeVersion, "unknown", StringComparison.OrdinalIgnoreCase))
            {
                return ServerVersionCheckResult.BridgeVersionUnknown;
            }

            // Server reported "unknown" version — treat as a mismatch we cannot vouch for.
            if (string.IsNullOrEmpty(serverVersion) || string.Equals(serverVersion, "unknown", StringComparison.OrdinalIgnoreCase))
            {
                return ServerVersionCheckResult.Mismatch;
            }

            return string.Equals(serverVersion.Trim(), bridgeVersion.Trim(), StringComparison.OrdinalIgnoreCase)
                ? ServerVersionCheckResult.Match
                : ServerVersionCheckResult.Mismatch;
        }

        /// <summary>
        /// Parses the MCP server's /health body and extracts the reported version.
        /// Returns false when the body is not recognizable MCP-server health JSON
        /// (missing status/version fields, malformed, or empty).
        /// </summary>
        public static bool TryParseHealthVersion(string healthJson, out string serverVersion)
        {
            serverVersion = null;

            if (string.IsNullOrWhiteSpace(healthJson))
            {
                return false;
            }

            JObject obj;
            try
            {
                obj = JObject.Parse(healthJson);
            }
            catch
            {
                return false;
            }

            // The MCP-for-Unity /health endpoint always reports both a status and a version field.
            // A response missing either is not our server — treat as an unknown listener.
            JToken statusToken = obj["status"];
            JToken versionToken = obj["version"];
            if (statusToken == null || versionToken == null)
            {
                return false;
            }

            serverVersion = versionToken.Type == JTokenType.Null ? null : versionToken.ToString();
            return true;
        }
    }
}
