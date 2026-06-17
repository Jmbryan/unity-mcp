using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MCPForUnity.Editor.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace MCPForUnity.Editor.Services
{
    /// <summary>
    /// Public bridge-side surface for the session roster pushed by the MCP server over the
    /// existing plugin WebSocket (schema <c>unity-mcp/session_roster@1</c>). Editor consumers
    /// (dashboards, status overlays) read the last-known roster snapshot for display and invoke
    /// the local cross-session control actions exposed here.
    ///
    /// THREADING: <see cref="IngestRosterMessage"/> is called from the transport's background
    /// receive thread and performs a volatile swap of an immutable snapshot — no UnityEngine APIs
    /// are touched there. Readers (<see cref="TryGetRoster"/> / <see cref="LastSnapshot"/>) get the
    /// most recently published immutable snapshot lock-free. Any consumer that uses the snapshot to
    /// drive UI MUST marshal to the main thread itself; the accessors are thread-safe but the data
    /// they return must still reach the UI on the main thread.
    /// </summary>
    public static class SessionRosterService
    {
        // Volatile reference swap: writers build an immutable snapshot off-thread and publish it
        // with a single reference assignment; readers see either the old or new snapshot, never a
        // torn one. No lock needed because RosterSnapshot and its members are immutable once built.
        private static volatile RosterSnapshot _lastSnapshot;

        private const string SessionRosterMessageType = "session_roster";

        /// <summary>The most recently received roster snapshot, or null if none has arrived yet.</summary>
        public static RosterSnapshot LastSnapshot => _lastSnapshot;

        /// <summary>
        /// Returns the last-known roster snapshot. Returns false (and a null snapshot) when no
        /// roster has been received yet. Safe to call from any thread; callers driving UI must
        /// marshal the returned data to the main thread themselves.
        /// </summary>
        public static bool TryGetRoster(out RosterSnapshot snapshot)
        {
            snapshot = _lastSnapshot;
            return snapshot != null;
        }

        /// <summary>
        /// Parses a <c>session_roster</c> envelope into immutable DTOs and publishes it as the
        /// current snapshot. Called on the transport background receive thread; does not touch any
        /// UnityEngine API. Malformed payloads are ignored (the existing snapshot is left intact)
        /// and never throw — the caller's switch default keeps unknown types backward-safe.
        /// </summary>
        internal static void IngestRosterMessage(JObject payload)
        {
            if (payload == null)
            {
                return;
            }

            try
            {
                RosterSnapshot snapshot = ParseRosterEnvelope(payload);
                if (snapshot != null)
                {
                    // Reject a server-declared failed build (health.ok == false with no
                    // sessions) that would clobber a populated snapshot: the fail-open
                    // envelope carries no authoritative session data, so honoring it here
                    // would replace a good roster with an empty one. A genuine empty roster
                    // (health.ok == true, or no health block) still publishes so real
                    // "all sessions gone" transitions render.
                    bool isFailedEmpty = snapshot.Sessions.Count == 0
                        && snapshot.Health != null
                        && !snapshot.Health.Ok;
                    if (isFailedEmpty && _lastSnapshot != null && _lastSnapshot.Sessions.Count > 0)
                    {
                        McpLog.Debug("[SessionRoster] Ignoring failed-build empty roster; keeping last good snapshot.");
                        return;
                    }

                    _lastSnapshot = snapshot;
                }
            }
            catch (Exception ex)
            {
                // Never let a malformed roster push escape onto the receive loop. Leave the prior
                // snapshot untouched.
                McpLog.Warn($"[SessionRoster] Failed to parse roster push: {ex.Message}");
            }
        }

        /// <summary>
        /// True when this bridge's local <see cref="DeferredCompileService"/> is holding a compile
        /// pending return to idle (play/test span active). Local read; no server round-trip.
        /// </summary>
        public static bool HasPendingCompile => DeferredCompileService.HasPendingCompile;

        /// <summary>
        /// Reason recorded for the locally held compile, or null when nothing is pending. Local read.
        /// </summary>
        public static string PendingCompileReason => DeferredCompileService.PendingReason;

        /// <summary>
        /// Number of script imports the local <see cref="DeferredCompileService"/> is holding pending
        /// flush. Zero when none are queued (a held compile may still exist with no queued import).
        /// Local read.
        /// </summary>
        public static int PendingCompileImportCount => DeferredCompileService.PendingImportCount;

        /// <summary>
        /// Flushes any compile that this bridge's local <see cref="DeferredCompileService"/> is
        /// holding. Local editor operation (a no-op when nothing is pending or a play/test span is
        /// still active) — it acts only on this instance's own held compile, not any peer's.
        /// </summary>
        public static void FlushDeferredCompile()
        {
            DeferredCompileService.FlushNow("session_roster_request");
        }

        /// <summary>
        /// Asks the local MCP server to release the play lease held against this bridge's instance
        /// by POSTing to <c>{baseUrl}/lease/release</c> with body <c>{"instance": "&lt;hash&gt;"}</c>.
        /// Fire-and-forget with a short timeout: failures are logged, never thrown. A server route
        /// performs the actual release.
        /// </summary>
        public static void ForceReleasePlayLease()
        {
            string instance = ProjectIdentityUtility.GetProjectHash();
            string baseUrl = HttpEndpointUtility.GetBaseUrl();
            _ = PostLeaseReleaseAsync(baseUrl, instance);
        }

        private static async Task PostLeaseReleaseAsync(string baseUrl, string instance)
        {
            try
            {
                if (string.IsNullOrEmpty(baseUrl))
                {
                    McpLog.Warn("[SessionRoster] ForceReleasePlayLease skipped: no base URL configured.");
                    return;
                }

                string endpoint = $"{baseUrl.TrimEnd('/')}/lease/release";
                string body = new JObject { ["instance"] = instance ?? string.Empty }.ToString(Formatting.None);

                using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(3) };
                using var content = new StringContent(body, Encoding.UTF8, "application/json");
                HttpResponseMessage response = await client.PostAsync(endpoint, content).ConfigureAwait(false);
                if (!response.IsSuccessStatusCode)
                {
                    McpLog.Warn($"[SessionRoster] Lease release returned {(int)response.StatusCode} ({response.ReasonPhrase}).");
                }
            }
            catch (Exception ex)
            {
                McpLog.Warn($"[SessionRoster] Lease release request failed: {ex.Message}");
            }
        }

        private static RosterSnapshot ParseRosterEnvelope(JObject payload)
        {
            string schema = payload.Value<string>("schema");

            var entries = new List<RosterEntry>();
            if (payload["sessions"] is JArray sessionsArray)
            {
                foreach (JToken token in sessionsArray)
                {
                    if (token is JObject entryObj)
                    {
                        entries.Add(ParseEntry(entryObj));
                    }
                }
            }

            PlayLeaseInfo playLease = null;
            if (payload["play_lease"] is JObject leaseObj)
            {
                playLease = new PlayLeaseInfo(
                    owner: leaseObj.Value<string>("owner"),
                    ownerSessionKey: leaseObj.Value<string>("owner_session_key"),
                    instance: leaseObj.Value<string>("instance"),
                    acquiredAtUnix: leaseObj.Value<double?>("acquired_at_unix") ?? 0d,
                    sinceSeconds: leaseObj.Value<double?>("since_seconds") ?? 0d);
            }

            RosterHealth health = null;
            if (payload["health"] is JObject healthObj)
            {
                health = new RosterHealth(
                    ok: healthObj.Value<bool?>("ok") ?? false,
                    bridgeConfigured: healthObj.Value<bool?>("bridge_configured") ?? false,
                    connectedInstances: healthObj.Value<int?>("connected_instances") ?? 0,
                    sessionCount: healthObj.Value<int?>("session_count") ?? 0);
            }

            return new RosterSnapshot(
                schema: schema,
                generatedAtUnix: payload.Value<double?>("generated_at_unix") ?? 0d,
                receivedAtUtc: DateTime.UtcNow,
                sessions: entries,
                playLease: playLease,
                health: health);
        }

        private static RosterEntry ParseEntry(JObject obj)
        {
            var activityTail = new List<RosterActivity>();
            if (obj["activity_tail"] is JArray tailArray)
            {
                foreach (JToken token in tailArray)
                {
                    if (token is JObject activityObj)
                    {
                        activityTail.Add(new RosterActivity(
                            @event: activityObj.Value<string>("event"),
                            summary: activityObj.Value<string>("summary"),
                            ts: activityObj.Value<double?>("ts") ?? 0d));
                    }
                }
            }

            RosterAttribution attribution = null;
            if (obj["attribution"] is JObject attrObj)
            {
                attribution = new RosterAttribution(
                    holdsPlayLease: attrObj.Value<bool?>("holds_play_lease") ?? false,
                    playLeaseOwner: attrObj.Value<string>("play_lease_owner"),
                    parked: attrObj.Value<bool?>("parked") ?? false,
                    parkedOn: attrObj.Value<string>("parked_on"),
                    parkedOwner: attrObj.Value<string>("parked_owner"));
            }

            var flags = new List<string>();
            if (obj["flags"] is JArray flagsArray)
            {
                foreach (JToken token in flagsArray)
                {
                    string flag = token?.Value<string>();
                    if (!string.IsNullOrEmpty(flag))
                    {
                        flags.Add(flag);
                    }
                }
            }

            return new RosterEntry(
                sessionKey: obj.Value<string>("session_key"),
                name: obj.Value<string>("name"),
                label: obj.Value<string>("label"),
                color: obj.Value<string>("color"),
                state: obj.Value<string>("state"),
                intent: obj.Value<string>("intent"),
                lastActivityUnix: obj.Value<double?>("last_activity_unix"),
                lastActivityAgeSeconds: obj.Value<double?>("last_activity_age_seconds"),
                activityTail: activityTail,
                attribution: attribution,
                flags: flags,
                source: obj.Value<string>("source"));
        }
    }

    /// <summary>Immutable snapshot of a single <c>session_roster</c> push plus its arrival time.</summary>
    public sealed class RosterSnapshot
    {
        public string Schema { get; }
        public double GeneratedAtUnix { get; }

        /// <summary>UTC time this snapshot was received by the bridge — used for staleness display.</summary>
        public DateTime ReceivedAtUtc { get; }
        public IReadOnlyList<RosterEntry> Sessions { get; }

        /// <summary>Current play-lease holder, or null when no lease is held.</summary>
        public PlayLeaseInfo PlayLease { get; }
        public RosterHealth Health { get; }

        public RosterSnapshot(
            string schema,
            double generatedAtUnix,
            DateTime receivedAtUtc,
            IReadOnlyList<RosterEntry> sessions,
            PlayLeaseInfo playLease,
            RosterHealth health)
        {
            Schema = schema;
            GeneratedAtUnix = generatedAtUnix;
            ReceivedAtUtc = receivedAtUtc;
            Sessions = sessions ?? Array.Empty<RosterEntry>();
            PlayLease = playLease;
            Health = health;
        }
    }

    /// <summary>One session in the roster.</summary>
    public sealed class RosterEntry
    {
        public string SessionKey { get; }
        public string Name { get; }
        public string Label { get; }
        public string Color { get; }

        /// <summary>One of: in-play, running-tests, waiting-parked, editing, active, idle, disconnected.</summary>
        public string State { get; }
        public string Intent { get; }
        public double? LastActivityUnix { get; }
        public double? LastActivityAgeSeconds { get; }
        public IReadOnlyList<RosterActivity> ActivityTail { get; }
        public RosterAttribution Attribution { get; }
        public IReadOnlyList<string> Flags { get; }

        /// <summary>"full" or "mcp-only".</summary>
        public string Source { get; }

        public RosterEntry(
            string sessionKey,
            string name,
            string label,
            string color,
            string state,
            string intent,
            double? lastActivityUnix,
            double? lastActivityAgeSeconds,
            IReadOnlyList<RosterActivity> activityTail,
            RosterAttribution attribution,
            IReadOnlyList<string> flags,
            string source)
        {
            SessionKey = sessionKey;
            Name = name;
            Label = label;
            Color = color;
            State = state;
            Intent = intent;
            LastActivityUnix = lastActivityUnix;
            LastActivityAgeSeconds = lastActivityAgeSeconds;
            ActivityTail = activityTail ?? Array.Empty<RosterActivity>();
            Attribution = attribution;
            Flags = flags ?? Array.Empty<string>();
            Source = source;
        }
    }

    /// <summary>A single entry in a session's recent activity tail.</summary>
    public sealed class RosterActivity
    {
        public string Event { get; }
        public string Summary { get; }
        public double Ts { get; }

        public RosterActivity(string @event, string summary, double ts)
        {
            Event = @event;
            Summary = summary;
            Ts = ts;
        }
    }

    /// <summary>Per-entry attribution describing lease ownership / parked relationships.</summary>
    public sealed class RosterAttribution
    {
        public bool HoldsPlayLease { get; }
        public string PlayLeaseOwner { get; }
        public bool Parked { get; }
        public string ParkedOn { get; }
        public string ParkedOwner { get; }

        public RosterAttribution(
            bool holdsPlayLease,
            string playLeaseOwner,
            bool parked,
            string parkedOn,
            string parkedOwner)
        {
            HoldsPlayLease = holdsPlayLease;
            PlayLeaseOwner = playLeaseOwner;
            Parked = parked;
            ParkedOn = parkedOn;
            ParkedOwner = parkedOwner;
        }
    }

    /// <summary>The active play lease, if any.</summary>
    public sealed class PlayLeaseInfo
    {
        public string Owner { get; }
        public string OwnerSessionKey { get; }
        public string Instance { get; }
        public double AcquiredAtUnix { get; }
        public double SinceSeconds { get; }

        public PlayLeaseInfo(
            string owner,
            string ownerSessionKey,
            string instance,
            double acquiredAtUnix,
            double sinceSeconds)
        {
            Owner = owner;
            OwnerSessionKey = ownerSessionKey;
            Instance = instance;
            AcquiredAtUnix = acquiredAtUnix;
            SinceSeconds = sinceSeconds;
        }
    }

    /// <summary>Server-reported roster health block.</summary>
    public sealed class RosterHealth
    {
        public bool Ok { get; }
        public bool BridgeConfigured { get; }
        public int ConnectedInstances { get; }
        public int SessionCount { get; }

        public RosterHealth(bool ok, bool bridgeConfigured, int connectedInstances, int sessionCount)
        {
            Ok = ok;
            BridgeConfigured = bridgeConfigured;
            ConnectedInstances = connectedInstances;
            SessionCount = sessionCount;
        }
    }
}
