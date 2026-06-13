using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using MCPForUnity.Editor.Services;
using MCPForUnity.Editor.Services.Transport.Transports;
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace MCPForUnityTests.Editor.Services
{
    [TestFixture]
    public class WebSocketTransportClientTests
    {
        private const string CandidateBuilderMethodName = "BuildConnectionCandidateUris";
        private const string WebSocketTransportClientTypeName = "MCPForUnity.Editor.Services.Transport.Transports.WebSocketTransportClient";
        private static readonly MethodInfo BuildConnectionCandidateUrisMethod = ResolveCandidateBuilderMethod();

        [Test]
        public void BuildConnectionCandidateUris_NullEndpoint_ReturnsEmptyList()
        {
            // Act
            List<Uri> candidates = InvokeBuildConnectionCandidateUris(null);

            // Assert
            Assert.IsNotNull(candidates);
            Assert.AreEqual(0, candidates.Count);
        }

        [Test]
        public void BuildConnectionCandidateUris_NonLocalhost_ReturnsOriginalOnly()
        {
            // Arrange
            var endpoint = new Uri("ws://127.0.0.1:8080/hub/plugin");

            // Act
            List<Uri> candidates = InvokeBuildConnectionCandidateUris(endpoint);

            // Assert
            Assert.AreEqual(1, candidates.Count);
            Assert.AreEqual(endpoint, candidates[0]);
        }

        [Test]
        public void BuildConnectionCandidateUris_Localhost_AddsIPv4AndIPv6Fallbacks()
        {
            // Arrange
            var endpoint = new Uri("ws://localhost:8080/hub/plugin");

            // Act
            List<Uri> candidates = InvokeBuildConnectionCandidateUris(endpoint);

            // Assert
            Assert.AreEqual(3, candidates.Count);
            CollectionAssert.AreEqual(
                new[] { "localhost", "127.0.0.1", "::1" },
                candidates.Select(uri => NormalizeHostForComparison(uri.Host)).ToArray());

            int uniqueCount = candidates
                .Select(uri => uri.AbsoluteUri)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .Count();
            Assert.AreEqual(candidates.Count, uniqueCount, "Fallback list should not contain duplicate endpoints.");
        }

        [Test]
        public void BuildConnectionCandidateUris_LocalhostFallbacks_PreserveSchemePortPathAndQuery()
        {
            // Arrange
            var endpoint = new Uri("wss://localhost:9443/custom/path?mode=test");

            // Act
            List<Uri> candidates = InvokeBuildConnectionCandidateUris(endpoint);

            // Assert
            Assert.AreEqual(3, candidates.Count);
            foreach (Uri candidate in candidates)
            {
                Assert.AreEqual("wss", candidate.Scheme);
                Assert.AreEqual(9443, candidate.Port);
                Assert.AreEqual("/custom/path", candidate.AbsolutePath);
                Assert.AreEqual("?mode=test", candidate.Query);
            }
        }

        [Test]
        public void BuildToolRegistrationObject_IncludesExplicitConcurrencyClass()
        {
            // Arrange
            var tool = new ToolMetadata
            {
                Name = "sample_tool",
                Description = "desc",
                ConcurrencyClass = "exclusive"
            };

            // Act
            JObject payload = InvokeBuildToolRegistrationObject(tool);

            // Assert
            Assert.AreEqual("exclusive", payload.Value<string>("concurrency_class"));
        }

        [Test]
        public void BuildToolRegistrationObject_DefaultsToMutate_WhenConcurrencyClassMissing()
        {
            // Arrange - empty/whitespace class should fall back to the safe default
            var tool = new ToolMetadata
            {
                Name = "sample_tool",
                Description = "desc",
                ConcurrencyClass = "   "
            };

            // Act
            JObject payload = InvokeBuildToolRegistrationObject(tool);

            // Assert
            Assert.AreEqual("mutate", payload.Value<string>("concurrency_class"));
        }

        [Test]
        public void SessionRoster_WellFormedMessage_PopulatesSnapshot()
        {
            // Arrange — a full envelope with one entry, a play lease, and a health block.
            JObject payload = JObject.Parse(@"{
                ""type"": ""session_roster"",
                ""schema"": ""unity-mcp/session_roster@1"",
                ""generated_at_unix"": 1700000000.5,
                ""sessions"": [
                    {
                        ""session_key"": ""sk-1"",
                        ""name"": ""Alice"",
                        ""label"": ""impl"",
                        ""color"": ""#ff0000"",
                        ""state"": ""in-play"",
                        ""intent"": ""running combat"",
                        ""last_activity_unix"": 1699999990.0,
                        ""last_activity_age_seconds"": 10.5,
                        ""activity_tail"": [
                            { ""event"": ""play_enter"", ""summary"": ""entered play"", ""ts"": 1699999985.0 }
                        ],
                        ""attribution"": {
                            ""holds_play_lease"": true,
                            ""play_lease_owner"": ""Alice""
                        },
                        ""flags"": [ ""looping"", ""long-op"" ],
                        ""source"": ""full""
                    }
                ],
                ""play_lease"": {
                    ""owner"": ""Alice"",
                    ""owner_session_key"": ""sk-1"",
                    ""instance"": ""abc123"",
                    ""acquired_at_unix"": 1699999980.0,
                    ""since_seconds"": 20.0
                },
                ""health"": {
                    ""ok"": true,
                    ""bridge_configured"": true,
                    ""connected_instances"": 2,
                    ""session_count"": 3
                }
            }");

            // Act
            InvokeIngestRosterMessage(payload);

            // Assert — envelope round-trip.
            Assert.IsTrue(TryGetRosterReflective(out object snapshot), "Expected snapshot after ingest.");
            Assert.IsNotNull(snapshot);
            Assert.AreEqual("unity-mcp/session_roster@1", GetProp<string>(snapshot, "Schema"));
            Assert.AreEqual(1700000000.5, GetProp<double>(snapshot, "GeneratedAtUnix"));

            // Entries.
            var sessions = (System.Collections.IEnumerable)GetProp<object>(snapshot, "Sessions");
            object first = null;
            foreach (object s in sessions) { first = s; break; }
            Assert.IsNotNull(first, "Expected one session entry.");
            Assert.AreEqual("sk-1", GetProp<string>(first, "SessionKey"));
            Assert.AreEqual("Alice", GetProp<string>(first, "Name"));
            Assert.AreEqual("in-play", GetProp<string>(first, "State"));
            Assert.AreEqual(10.5, GetProp<double?>(first, "LastActivityAgeSeconds"));
            Assert.AreEqual("full", GetProp<string>(first, "Source"));

            object attribution = GetProp<object>(first, "Attribution");
            Assert.IsNotNull(attribution);
            Assert.IsTrue(GetProp<bool>(attribution, "HoldsPlayLease"));

            // play_lease round-trip.
            object lease = GetProp<object>(snapshot, "PlayLease");
            Assert.IsNotNull(lease, "Expected play_lease parsed.");
            Assert.AreEqual("Alice", GetProp<string>(lease, "Owner"));
            Assert.AreEqual("abc123", GetProp<string>(lease, "Instance"));
            Assert.AreEqual(20.0, GetProp<double>(lease, "SinceSeconds"));

            // health round-trip.
            object health = GetProp<object>(snapshot, "Health");
            Assert.IsNotNull(health);
            Assert.IsTrue(GetProp<bool>(health, "Ok"));
            Assert.AreEqual(2, GetProp<int>(health, "ConnectedInstances"));
            Assert.AreEqual(3, GetProp<int>(health, "SessionCount"));
        }

        [Test]
        public void SessionRoster_EmptySessionsList_HandledGracefully()
        {
            // Arrange
            JObject payload = JObject.Parse(@"{
                ""type"": ""session_roster"",
                ""schema"": ""unity-mcp/session_roster@1"",
                ""generated_at_unix"": 1700000001.0,
                ""sessions"": [],
                ""play_lease"": null,
                ""health"": { ""ok"": false, ""bridge_configured"": false, ""connected_instances"": 0, ""session_count"": 0 }
            }");

            // Act
            InvokeIngestRosterMessage(payload);

            // Assert
            Assert.IsTrue(TryGetRosterReflective(out object snapshot));
            var sessions = (System.Collections.IEnumerable)GetProp<object>(snapshot, "Sessions");
            int count = 0;
            foreach (var _ in sessions) count++;
            Assert.AreEqual(0, count, "Empty sessions list should produce an empty (non-null) collection.");
            Assert.IsNull(GetProp<object>(snapshot, "PlayLease"), "Null play_lease should parse to null.");
        }

        [Test]
        public void SessionRoster_MalformedMessage_IgnoredAndSnapshotUnchanged()
        {
            // Arrange — seed a known-good snapshot first.
            JObject good = JObject.Parse(@"{
                ""type"": ""session_roster"",
                ""schema"": ""unity-mcp/session_roster@1"",
                ""generated_at_unix"": 1700000002.0,
                ""sessions"": [ { ""session_key"": ""keep"", ""name"": ""Keep"", ""color"": ""#fff"", ""state"": ""idle"", ""source"": ""mcp-only"" } ],
                ""play_lease"": null,
                ""health"": null
            }");
            InvokeIngestRosterMessage(good);
            Assert.IsTrue(TryGetRosterReflective(out object before));
            double beforeGenerated = GetProp<double>(before, "GeneratedAtUnix");

            // Act — a payload whose 'sessions' is a non-array scalar. Must not throw, must not replace.
            JObject malformed = JObject.Parse(@"{
                ""type"": ""session_roster"",
                ""schema"": ""unity-mcp/session_roster@1"",
                ""sessions"": ""not-an-array"",
                ""generated_at_unix"": ""also-bad""
            }");
            Assert.DoesNotThrow(() => InvokeIngestRosterMessage(malformed));

            // A null payload must also be a safe no-op.
            Assert.DoesNotThrow(() => InvokeIngestRosterMessage(null));

            // Assert — prior good snapshot is intact (a forgiving parse won't clobber it).
            Assert.IsTrue(TryGetRosterReflective(out object after));
            Assert.AreEqual(beforeGenerated, GetProp<double>(after, "GeneratedAtUnix"),
                "Malformed/null ingests must not replace the prior snapshot's generated timestamp.");
        }

        private static Type ResolveSessionRosterServiceType()
        {
            const string typeName = "MCPForUnity.Editor.Services.SessionRosterService";
            Type direct = Type.GetType(typeName);
            if (direct != null) return direct;
            foreach (Assembly assembly in AppDomain.CurrentDomain.GetAssemblies())
            {
                Type t = assembly.GetType(typeName);
                if (t != null) return t;
            }
            return null;
        }

        private static void InvokeIngestRosterMessage(JObject payload)
        {
            Type type = ResolveSessionRosterServiceType();
            Assert.IsNotNull(type, "Expected SessionRosterService type to exist.");
            MethodInfo method = type.GetMethod("IngestRosterMessage",
                BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Static);
            Assert.IsNotNull(method, "Expected internal static IngestRosterMessage(JObject) to exist.");
            method.Invoke(null, new object[] { payload });
        }

        private static bool TryGetRosterReflective(out object snapshot)
        {
            Type type = ResolveSessionRosterServiceType();
            Assert.IsNotNull(type);
            MethodInfo method = type.GetMethod("TryGetRoster",
                BindingFlags.Public | BindingFlags.Static);
            Assert.IsNotNull(method, "Expected public static TryGetRoster(out RosterSnapshot) to exist.");
            object[] args = { null };
            bool result = (bool)method.Invoke(null, args);
            snapshot = args[0];
            return result;
        }

        private static T GetProp<T>(object instance, string propertyName)
        {
            Assert.IsNotNull(instance, $"Cannot read '{propertyName}' from a null instance.");
            PropertyInfo prop = instance.GetType().GetProperty(propertyName,
                BindingFlags.Public | BindingFlags.Instance);
            Assert.IsNotNull(prop, $"Expected property '{propertyName}' on {instance.GetType().Name}.");
            object value = prop.GetValue(instance);
            if (value == null) return default;
            return (T)value;
        }

        private static JObject InvokeBuildToolRegistrationObject(ToolMetadata tool)
        {
            const BindingFlags flags = BindingFlags.NonPublic | BindingFlags.Static;
            MethodInfo method = typeof(WebSocketTransportClient).GetMethod("BuildToolRegistrationObject", flags);
            Assert.IsNotNull(method, "Expected private static BuildToolRegistrationObject(ToolMetadata) to exist.");
            var result = method.Invoke(null, new object[] { tool });
            Assert.IsInstanceOf<JObject>(result);
            return (JObject)result;
        }

        private static List<Uri> InvokeBuildConnectionCandidateUris(Uri endpoint)
        {
            if (BuildConnectionCandidateUrisMethod == null)
            {
                Assert.Fail(BuildMissingMethodDiagnostic());
            }
            var result = BuildConnectionCandidateUrisMethod.Invoke(null, new object[] { endpoint });
            Assert.IsNotNull(result);
            Assert.IsInstanceOf<List<Uri>>(result);
            return (List<Uri>)result;
        }

        private static MethodInfo ResolveCandidateBuilderMethod()
        {
            MethodInfo direct = GetCandidateBuilderMethod(typeof(WebSocketTransportClient));
            if (direct != null)
            {
                return direct;
            }

            foreach (Assembly assembly in AppDomain.CurrentDomain.GetAssemblies())
            {
                Type candidateType = assembly.GetType(WebSocketTransportClientTypeName);
                if (candidateType == null)
                {
                    continue;
                }

                MethodInfo method = GetCandidateBuilderMethod(candidateType);
                if (method != null)
                {
                    return method;
                }
            }

            return null;
        }

        private static MethodInfo GetCandidateBuilderMethod(Type type)
        {
            const BindingFlags flags = BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Static;
            MethodInfo direct = type.GetMethod(
                CandidateBuilderMethodName,
                flags,
                binder: null,
                types: new[] { typeof(Uri) },
                modifiers: null);
            if (direct != null)
            {
                return direct;
            }

            // Fallback for environments where signature binding can differ between loaded copies.
            return type.GetMethods(flags).FirstOrDefault(method =>
            {
                if (!string.Equals(method.Name, CandidateBuilderMethodName, StringComparison.Ordinal))
                {
                    return false;
                }

                ParameterInfo[] parameters = method.GetParameters();
                return parameters.Length == 1 && parameters[0].ParameterType == typeof(Uri);
            });
        }

        private static string BuildMissingMethodDiagnostic()
        {
            var sb = new StringBuilder();
            sb.Append("Expected private candidate builder method to exist. Searched loaded assemblies for ")
              .Append(WebSocketTransportClientTypeName)
              .Append('.')
              .Append(CandidateBuilderMethodName)
              .Append(". Loaded candidate types:");

            foreach (Assembly assembly in AppDomain.CurrentDomain.GetAssemblies())
            {
                Type candidateType = assembly.GetType(WebSocketTransportClientTypeName);
                if (candidateType == null)
                {
                    continue;
                }

                sb.Append("\n- ")
                  .Append(assembly.FullName)
                  .Append(" @ ")
                  .Append(string.IsNullOrEmpty(assembly.Location) ? "<dynamic>" : assembly.Location);
            }

            return sb.ToString();
        }

        private static string NormalizeHostForComparison(string host)
        {
            if (string.IsNullOrEmpty(host))
            {
                return host;
            }

            return host.Trim('[', ']');
        }
    }
}
