using System;
using System.Reflection;
using System.Threading.Tasks;
using MCPForUnity.Editor.Helpers;
using MCPForUnity.Editor.Services.Transport;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEditor.Compilation;
using UnityEditorInternal;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace MCPForUnity.Editor.Services
{
    /// <summary>
    /// Maintains a cached readiness snapshot (v2) so status reads remain fast even when Unity is busy.
    /// Updated on the main thread via Editor callbacks and periodic update ticks.
    /// </summary>
    [InitializeOnLoad]
    internal static class EditorStateCache
    {
        private static readonly object LockObj = new();
        private static long _sequence;
        private static long _observedUnixMs;

        private static bool _lastIsCompiling;
        private static long? _lastCompileStartedUnixMs;
        private static long? _lastCompileFinishedUnixMs;

        private static bool _domainReloadPending;
        private static long? _domainReloadBeforeUnixMs;
        private static long? _domainReloadAfterUnixMs;

        private static double _lastUpdateTimeSinceStartup;
        private const double MinUpdateIntervalSeconds = 1.0; // Reduced frequency: 1s instead of 0.25s

        // State tracking to detect when snapshot actually changes (checked BEFORE building)
        private static string _lastTrackedScenePath;
        private static string _lastTrackedSceneName;
        private static bool _lastTrackedIsFocused;
        private static bool _lastTrackedIsPlaying;
        private static bool _lastTrackedIsPaused;
        private static bool _lastTrackedIsUpdating;
        private static bool _lastTrackedTestsRunning;
        private static string _lastTrackedActivityPhase;

        private static JObject _cached;

        private sealed class EditorStateSnapshot
        {
            [JsonProperty("schema_version")]
            public string SchemaVersion { get; set; }

            [JsonProperty("observed_at_unix_ms")]
            public long ObservedAtUnixMs { get; set; }

            [JsonProperty("sequence")]
            public long Sequence { get; set; }

            [JsonProperty("unity")]
            public EditorStateUnity Unity { get; set; }

            [JsonProperty("editor")]
            public EditorStateEditor Editor { get; set; }

            [JsonProperty("activity")]
            public EditorStateActivity Activity { get; set; }

            [JsonProperty("compilation")]
            public EditorStateCompilation Compilation { get; set; }

            [JsonProperty("assets")]
            public EditorStateAssets Assets { get; set; }

            [JsonProperty("tests")]
            public EditorStateTests Tests { get; set; }

            [JsonProperty("transport")]
            public EditorStateTransport Transport { get; set; }

            [JsonProperty("settings")]
            public EditorStateSettings Settings { get; set; }
        }

        private sealed class EditorStateUnity
        {
            [JsonProperty("instance_id")]
            public string InstanceId { get; set; }

            [JsonProperty("unity_version")]
            public string UnityVersion { get; set; }

            [JsonProperty("project_id")]
            public string ProjectId { get; set; }

            [JsonProperty("platform")]
            public string Platform { get; set; }

            [JsonProperty("is_batch_mode")]
            public bool? IsBatchMode { get; set; }
        }

        private sealed class EditorStateEditor
        {
            [JsonProperty("is_focused")]
            public bool? IsFocused { get; set; }

            [JsonProperty("play_mode")]
            public EditorStatePlayMode PlayMode { get; set; }

            [JsonProperty("active_scene")]
            public EditorStateActiveScene ActiveScene { get; set; }
        }

        private sealed class EditorStatePlayMode
        {
            [JsonProperty("is_playing")]
            public bool? IsPlaying { get; set; }

            [JsonProperty("is_paused")]
            public bool? IsPaused { get; set; }

            [JsonProperty("is_changing")]
            public bool? IsChanging { get; set; }
        }

        private sealed class EditorStateActiveScene
        {
            [JsonProperty("path")]
            public string Path { get; set; }

            [JsonProperty("guid")]
            public string Guid { get; set; }

            [JsonProperty("name")]
            public string Name { get; set; }
        }

        private sealed class EditorStateActivity
        {
            [JsonProperty("phase")]
            public string Phase { get; set; }

            [JsonProperty("since_unix_ms")]
            public long SinceUnixMs { get; set; }

            [JsonProperty("reasons")]
            public string[] Reasons { get; set; }
        }

        private sealed class EditorStateCompilation
        {
            [JsonProperty("is_compiling")]
            public bool? IsCompiling { get; set; }

            [JsonProperty("is_domain_reload_pending")]
            public bool? IsDomainReloadPending { get; set; }

            [JsonProperty("last_compile_started_unix_ms")]
            public long? LastCompileStartedUnixMs { get; set; }

            [JsonProperty("last_compile_finished_unix_ms")]
            public long? LastCompileFinishedUnixMs { get; set; }

            [JsonProperty("last_domain_reload_before_unix_ms")]
            public long? LastDomainReloadBeforeUnixMs { get; set; }

            [JsonProperty("last_domain_reload_after_unix_ms")]
            public long? LastDomainReloadAfterUnixMs { get; set; }

            // MCPC-020: deferred-compile visibility. True while a compile request is held because play
            // mode or a test run is active; reason is the recorded source of the held request.
            [JsonProperty("deferred_compile_pending")]
            public bool? DeferredCompilePending { get; set; }

            [JsonProperty("deferred_compile_reason")]
            public string DeferredCompileReason { get; set; }
        }

        private sealed class EditorStateAssets
        {
            [JsonProperty("is_updating")]
            public bool? IsUpdating { get; set; }

            [JsonProperty("external_changes_dirty")]
            public bool? ExternalChangesDirty { get; set; }

            [JsonProperty("external_changes_last_seen_unix_ms")]
            public long? ExternalChangesLastSeenUnixMs { get; set; }

            [JsonProperty("external_changes_dirty_since_unix_ms")]
            public long? ExternalChangesDirtySinceUnixMs { get; set; }

            [JsonProperty("external_changes_last_cleared_unix_ms")]
            public long? ExternalChangesLastClearedUnixMs { get; set; }

            [JsonProperty("refresh")]
            public EditorStateRefresh Refresh { get; set; }
        }

        private sealed class EditorStateRefresh
        {
            [JsonProperty("is_refresh_in_progress")]
            public bool? IsRefreshInProgress { get; set; }

            [JsonProperty("last_refresh_requested_unix_ms")]
            public long? LastRefreshRequestedUnixMs { get; set; }

            [JsonProperty("last_refresh_finished_unix_ms")]
            public long? LastRefreshFinishedUnixMs { get; set; }
        }

        private sealed class EditorStateTests
        {
            [JsonProperty("is_running")]
            public bool? IsRunning { get; set; }

            [JsonProperty("mode")]
            public string Mode { get; set; }

            [JsonProperty("current_job_id")]
            public string CurrentJobId { get; set; }

            [JsonProperty("started_unix_ms")]
            public long? StartedUnixMs { get; set; }

            [JsonProperty("started_by")]
            public string StartedBy { get; set; }

            [JsonProperty("last_run")]
            public EditorStateLastRun LastRun { get; set; }
        }

        private sealed class EditorStateLastRun
        {
            [JsonProperty("finished_unix_ms")]
            public long? FinishedUnixMs { get; set; }

            [JsonProperty("result")]
            public string Result { get; set; }

            [JsonProperty("counts")]
            public object Counts { get; set; }
        }

        private sealed class EditorStateTransport
        {
            [JsonProperty("unity_bridge_connected")]
            public bool? UnityBridgeConnected { get; set; }

            [JsonProperty("last_message_unix_ms")]
            public long? LastMessageUnixMs { get; set; }
        }

        private sealed class EditorStateSettings
        {
            [JsonProperty("batch_execute_max_commands")]
            public int BatchExecuteMaxCommands { get; set; }
        }

        static EditorStateCache()
        {
            try
            {
                _sequence = 0;
                _observedUnixMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                _cached = BuildSnapshot("init");

                EditorApplication.update += OnUpdate;
                EditorApplication.playModeStateChanged += OnPlayModeStateChanged;

                CompilationPipeline.compilationStarted += _ => EmitEditorEdgeEvent("compile_started");
                CompilationPipeline.compilationFinished += _ => EmitEditorEdgeEvent("compile_finished");

                // Tracks whether an assembly compilation is actually running, for
                // GetActualIsCompiling. Statics reset on domain reload and this
                // [InitializeOnLoad] ctor re-subscribes, so the flag is per-domain.
                UnityEditor.Compilation.CompilationPipeline.compilationStarted += _ => _pipelineCompilationRunning = true;
                UnityEditor.Compilation.CompilationPipeline.compilationFinished += _ => _pipelineCompilationRunning = false;

                AssemblyReloadEvents.beforeAssemblyReload += () =>
                {
                    _domainReloadPending = true;
                    _domainReloadBeforeUnixMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                    ForceUpdate("before_domain_reload");
                };
                AssemblyReloadEvents.afterAssemblyReload += () =>
                {
                    _domainReloadPending = false;
                    _domainReloadAfterUnixMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                    ForceUpdate("after_domain_reload");
                    EmitEditorEdgeEvent("domain_reload_done");
                };
            }
            catch (Exception ex)
            {
                McpLog.Error($"[EditorStateCache] Failed to initialise: {ex.Message}\n{ex.StackTrace}");
            }
        }

        private static void OnUpdate()
        {
            // Throttle to reduce overhead while keeping the snapshot fresh enough for polling clients.
            double now = EditorApplication.timeSinceStartup;
            // Use GetActualIsCompiling() to avoid isCompiling false positives (issues #549, #1276)
            bool isCompiling = GetActualIsCompiling();

            // Check for compilation edge transitions (always update on these)
            bool compilationEdge = isCompiling != _lastIsCompiling;

            if (!compilationEdge && now - _lastUpdateTimeSinceStartup < MinUpdateIntervalSeconds)
            {
                return;
            }

            // Fast state-change detection BEFORE building snapshot.
            // This avoids the expensive BuildSnapshot() call entirely when nothing changed.
            // These checks are much cheaper than building a full JSON snapshot.
            var scene = EditorSceneManager.GetActiveScene();
            string scenePath = string.IsNullOrEmpty(scene.path) ? null : scene.path;
            string sceneName = scene.name ?? string.Empty;
            bool isFocused = InternalEditorUtility.isApplicationActive;
            bool isPlaying = EditorApplication.isPlaying;
            bool isPaused = EditorApplication.isPaused;
            bool isUpdating = EditorApplication.isUpdating;
            bool testsRunning = TestRunStatus.IsRunning;

            var activityPhase = "idle";
            if (testsRunning)
            {
                activityPhase = "running_tests";
            }
            else if (isCompiling)
            {
                activityPhase = "compiling";
            }
            else if (_domainReloadPending)
            {
                activityPhase = "domain_reload";
            }
            else if (isUpdating)
            {
                activityPhase = "asset_import";
            }
            else if (ComputeIsChangingPlaymode(isPlaying, EditorApplication.isPlayingOrWillChangePlaymode))
            {
                activityPhase = "playmode_transition";
            }

            bool hasChanges = compilationEdge
                || _lastTrackedScenePath != scenePath
                || _lastTrackedSceneName != sceneName
                || _lastTrackedIsFocused != isFocused
                || _lastTrackedIsPlaying != isPlaying
                || _lastTrackedIsPaused != isPaused
                || _lastTrackedIsUpdating != isUpdating
                || _lastTrackedTestsRunning != testsRunning
                || _lastTrackedActivityPhase != activityPhase;

            if (!hasChanges)
            {
                // No state change - skip the expensive BuildSnapshot entirely.
                // This is the key optimization that prevents the 28ms GC spikes.
                return;
            }

            // Update tracked state
            _lastTrackedScenePath = scenePath;
            _lastTrackedSceneName = sceneName;
            _lastTrackedIsFocused = isFocused;
            _lastTrackedIsPlaying = isPlaying;
            _lastTrackedIsPaused = isPaused;
            _lastTrackedIsUpdating = isUpdating;
            _lastTrackedTestsRunning = testsRunning;
            _lastTrackedActivityPhase = activityPhase;

            _lastUpdateTimeSinceStartup = now;
            ForceUpdate("tick");
        }

        private static void ForceUpdate(string reason)
        {
            lock (LockObj)
            {
                _cached = BuildSnapshot(reason);
            }
        }

        private static void OnPlayModeStateChanged(PlayModeStateChange change)
        {
            ForceUpdate("playmode");

            // Map the two stable edges to the server's play events. EnteredPlayMode/EnteredEditMode
            // (not the Exiting* edges) are the settled states the server keys gate releases off.
            switch (change)
            {
                case PlayModeStateChange.EnteredPlayMode:
                    EmitEditorEdgeEvent("entered_play");
                    break;
                case PlayModeStateChange.EnteredEditMode:
                    EmitEditorEdgeEvent("exited_play");
                    break;
            }
        }

        /// <summary>
        /// Pushes a lightweight editor-edge event to the server via the active transport so parked
        /// gate calls can release event-driven. Strictly best-effort and fire-and-forget: any send
        /// is bounded by a short timeout inside the transport and never blocks this callback — which
        /// is critical because some edges fire on or adjacent to the domain-reload path.
        /// </summary>
        private static void EmitEditorEdgeEvent(string eventName)
        {
            try
            {
                IMcpTransportClient client = MCPServiceLocator.TransportManager.GetClient(TransportMode.Http);
                if (client == null || !client.IsConnected)
                {
                    return;
                }

                _ = Task.Run(async () =>
                {
                    try
                    {
                        await client.PushEventAsync(eventName).ConfigureAwait(false);
                    }
                    catch (Exception ex)
                    {
                        McpLog.Debug($"[EditorStateCache] Event push '{eventName}' failed: {ex.Message}");
                    }
                });
            }
            catch (Exception ex)
            {
                McpLog.Debug($"[EditorStateCache] Failed to emit edge event '{eventName}': {ex.Message}");
            }
        }

        private static JObject BuildSnapshot(string reason)
        {
            _sequence++;
            _observedUnixMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();

            bool isCompiling = GetActualIsCompiling();
            if (isCompiling && !_lastIsCompiling)
            {
                _lastCompileStartedUnixMs = _observedUnixMs;
            }
            else if (!isCompiling && _lastIsCompiling)
            {
                _lastCompileFinishedUnixMs = _observedUnixMs;
            }
            _lastIsCompiling = isCompiling;

            var scene = EditorSceneManager.GetActiveScene();
            string scenePath = string.IsNullOrEmpty(scene.path) ? null : scene.path;
            string sceneGuid = !string.IsNullOrEmpty(scenePath) ? AssetDatabase.AssetPathToGUID(scenePath) : null;

            bool testsRunning = TestRunStatus.IsRunning;
            var testsMode = TestRunStatus.Mode?.ToString();
            string currentJobId = TestJobManager.CurrentJobId;
            string testsStartedBy = TestJobManager.CurrentJobStartedBy;
            bool isFocused = InternalEditorUtility.isApplicationActive;

            // Reconcile the play-mode transition from current EditorApplication facts rather than
            // trusting playModeStateChanged event continuity. The play-entry domain reload destroys
            // and rebuilds this static cache, so the EnteredPlayMode event that would have closed a
            // transition is never observed by the rebuilt instance. Deriving "is changing" from the
            // live facts makes every snapshot self-correcting across that reload (and the exit-play
            // reload) without depending on event delivery.
            bool isPlaying = EditorApplication.isPlaying;
            bool isChangingPlaymode = ComputeIsChangingPlaymode(isPlaying, EditorApplication.isPlayingOrWillChangePlaymode);

            var activityPhase = "idle";
            if (testsRunning)
            {
                activityPhase = "running_tests";
            }
            else if (isCompiling)
            {
                activityPhase = "compiling";
            }
            else if (_domainReloadPending)
            {
                activityPhase = "domain_reload";
            }
            else if (EditorApplication.isUpdating)
            {
                activityPhase = "asset_import";
            }
            else if (isChangingPlaymode)
            {
                activityPhase = "playmode_transition";
            }

            var snapshot = new EditorStateSnapshot
            {
                SchemaVersion = "unity-mcp/editor_state@2",
                ObservedAtUnixMs = _observedUnixMs,
                Sequence = _sequence,
                Unity = new EditorStateUnity
                {
                    InstanceId = null,
                    UnityVersion = Application.unityVersion,
                    ProjectId = null,
                    Platform = Application.platform.ToString(),
                    IsBatchMode = Application.isBatchMode
                },
                Editor = new EditorStateEditor
                {
                    IsFocused = isFocused,
                    PlayMode = new EditorStatePlayMode
                    {
                        IsPlaying = isPlaying,
                        IsPaused = EditorApplication.isPaused,
                        IsChanging = isChangingPlaymode
                    },
                    ActiveScene = new EditorStateActiveScene
                    {
                        Path = scenePath,
                        Guid = sceneGuid,
                        Name = scene.name ?? string.Empty
                    }
                },
                Activity = new EditorStateActivity
                {
                    Phase = activityPhase,
                    SinceUnixMs = _observedUnixMs,
                    Reasons = new[] { reason }
                },
                Compilation = new EditorStateCompilation
                {
                    IsCompiling = isCompiling,
                    IsDomainReloadPending = _domainReloadPending,
                    LastCompileStartedUnixMs = _lastCompileStartedUnixMs,
                    LastCompileFinishedUnixMs = _lastCompileFinishedUnixMs,
                    LastDomainReloadBeforeUnixMs = _domainReloadBeforeUnixMs,
                    LastDomainReloadAfterUnixMs = _domainReloadAfterUnixMs,
                    DeferredCompilePending = DeferredCompileService.HasPendingCompile,
                    DeferredCompileReason = DeferredCompileService.PendingReason
                },
                Assets = new EditorStateAssets
                {
                    IsUpdating = EditorApplication.isUpdating,
                    ExternalChangesDirty = false,
                    ExternalChangesLastSeenUnixMs = null,
                    ExternalChangesDirtySinceUnixMs = null,
                    ExternalChangesLastClearedUnixMs = null,
                    Refresh = new EditorStateRefresh
                    {
                        IsRefreshInProgress = false,
                        LastRefreshRequestedUnixMs = null,
                        LastRefreshFinishedUnixMs = null
                    }
                },
                Tests = new EditorStateTests
                {
                    IsRunning = testsRunning,
                    Mode = testsMode,
                    CurrentJobId = string.IsNullOrEmpty(currentJobId) ? null : currentJobId,
                    StartedUnixMs = TestRunStatus.StartedUnixMs,
                    // Real owner when the tracked job carries one; "unknown" only when there is
                    // genuinely no owner (e.g. a run started from the Unity Test Runner UI).
                    StartedBy = string.IsNullOrEmpty(testsStartedBy) ? "unknown" : testsStartedBy,
                    LastRun = TestRunStatus.FinishedUnixMs.HasValue
                        ? new EditorStateLastRun
                        {
                            FinishedUnixMs = TestRunStatus.FinishedUnixMs,
                            Result = "unknown",
                            Counts = null
                        }
                        : null
                },
                Transport = new EditorStateTransport
                {
                    UnityBridgeConnected = null,
                    LastMessageUnixMs = null
                },
                Settings = new EditorStateSettings
                {
                    BatchExecuteMaxCommands = Tools.BatchExecute.GetMaxCommandsPerBatch()
                }
            };

            return JObject.FromObject(snapshot);
        }

        public static JObject GetSnapshot()
        {
            lock (LockObj)
            {
                // Defensive: if something went wrong early, rebuild once.
                if (_cached == null)
                {
                    _cached = BuildSnapshot("rebuild");
                }

                // Always return a fresh clone to prevent mutation bugs.
                // The main GC optimization comes from state-change detection (OnUpdate)
                // which prevents unnecessary _cached rebuilds, not from caching the clone.
                var clone = (JObject)_cached.DeepClone();

                // When Unity is backgrounded, OnUpdate is throttled and the
                // cached timestamp grows stale even though the data is current.
                // Re-stamp only in that case so the server-side staleness check
                // still fires for genuinely unresponsive editors when focused.
                if (!InternalEditorUtility.isApplicationActive)
                {
                    clone["observed_at_unix_ms"] = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                }

                return clone;
            }
        }

        // Set/cleared by the CompilationPipeline.compilationStarted/Finished events
        // subscribed in the static ctor. NOTE: CompilationPipeline.isCompiling does not
        // exist on the supported Unity range (verified by reflection probe on 2021.3 and
        // 6000.4 — neither public nor non-public), so the reflection this replaced never
        // resolved and always fell back to the raw signal.
        private static bool _pipelineCompilationRunning;

        /// <summary>
        /// Determines whether the editor is mid play-mode transition purely from the two
        /// EditorApplication facts, with no dependence on playModeStateChanged event continuity.
        /// </summary>
        /// <remarks>
        /// <see cref="EditorApplication.isPlayingOrWillChangePlaymode"/> is true in BOTH stable play
        /// and during a transition, so it cannot be used directly as an "is changing" signal. It only
        /// diverges from <see cref="EditorApplication.isPlaying"/> on the transition edges:
        /// entering play (isPlaying=false, willChange=true) and exiting play (isPlaying=true,
        /// willChange=false). In stable edit (false/false) and stable play (true/true) they match.
        /// This makes the function self-reconciling after the play-entry/exit domain reloads that
        /// destroy the cache and drop the closing playModeStateChanged event.
        /// </remarks>
        internal static bool ComputeIsChangingPlaymode(bool isPlaying, bool isPlayingOrWillChangePlaymode)
        {
            return isPlaying != isPlayingOrWillChangePlaymode;
        }

        /// <summary>
        /// Returns the actual compilation state, working around known Unity quirks where
        /// EditorApplication.isCompiling reports false positives while no compilation is
        /// running: a recompile deferred by Recompile-After-Finished-Playing keeps it true
        /// for the whole play session (issue #549), and a project holding
        /// EditorApplication.LockReloadAssemblies keeps it true until the lock is released
        /// (issue #1276). In both cases the event-tracked pipeline flag is authoritative.
        /// </summary>
        internal static bool GetActualIsCompiling()
        {
            // If EditorApplication.isCompiling is false, Unity is definitely not compiling
            if (!EditorApplication.isCompiling)
            {
                return false;
            }

            // Otherwise trust the event-tracked pipeline state: isCompiling stays true for as
            // long as an assembly reload is deferred, with no compilation actually running.
            return _pipelineCompilationRunning;
        }
    }
}


