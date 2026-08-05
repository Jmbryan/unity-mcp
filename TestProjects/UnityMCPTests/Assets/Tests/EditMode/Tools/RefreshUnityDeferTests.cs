using System.Reflection;
using NUnit.Framework;
using MCPForUnity.Editor.Helpers;
using MCPForUnity.Editor.Services;
using MCPForUnity.Editor.Tools;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEditor.TestTools.TestRunner.Api;

namespace MCPForUnityTests.Editor.Tools
{
    /// <summary>
    /// Tests that refresh_unity routes its explicit <c>AssetDatabase.Refresh</c> through the
    /// DeferredCompileService funnel. DisallowAutoRefresh only suppresses AUTOMATIC refreshes, so
    /// an explicit refresh mid play/test span would import every held-back script write and start
    /// a compile under the running session — the funnel holds it and replays on return to idle.
    /// </summary>
    public class RefreshUnityDeferTests
    {
        private const string SessionKey_PendingRefresh = "MCPForUnity.DeferredCompile.PendingRefresh";
        private const string SessionKey_PendingRefreshOptions = "MCPForUnity.DeferredCompile.PendingRefreshOptions";
        private const string SessionKey_PendingCompile = "MCPForUnity.DeferredCompile.Pending";
        private const string SessionKey_PendingReason = "MCPForUnity.DeferredCompile.Reason";

        private bool _origPendingRefresh;
        private int _origPendingRefreshOptions;
        private bool _origPendingCompile;
        private string _origPendingReason;
        private bool _markedTestRun;

        [SetUp]
        public void SetUp()
        {
            _origPendingRefresh = SessionState.GetBool(SessionKey_PendingRefresh, false);
            _origPendingRefreshOptions = SessionState.GetInt(SessionKey_PendingRefreshOptions, 0);
            _origPendingCompile = SessionState.GetBool(SessionKey_PendingCompile, false);
            _origPendingReason = SessionState.GetString(SessionKey_PendingReason, string.Empty);

            // Establish the blocking span deterministically (in-editor MCP launches already have the
            // flag ambient; headless -runTests launches do not).
            if (!TestRunStatus.IsRunning)
            {
                TestRunStatus.MarkStarted(TestMode.EditMode);
                _markedTestRun = true;
            }

            SessionState.SetBool(SessionKey_PendingRefresh, false);
            SessionState.EraseInt(SessionKey_PendingRefreshOptions);
            SessionState.SetBool(SessionKey_PendingCompile, false);
            SessionState.EraseString(SessionKey_PendingReason);
        }

        [TearDown]
        public void TearDown()
        {
            SessionState.SetBool(SessionKey_PendingRefresh, _origPendingRefresh);
            SessionState.SetInt(SessionKey_PendingRefreshOptions, _origPendingRefreshOptions);
            SessionState.SetBool(SessionKey_PendingCompile, _origPendingCompile);
            if (string.IsNullOrEmpty(_origPendingReason))
            {
                SessionState.EraseString(SessionKey_PendingReason);
            }
            else
            {
                SessionState.SetString(SessionKey_PendingReason, _origPendingReason);
            }

            if (_markedTestRun)
            {
                TestRunStatus.MarkFinished();
                _markedTestRun = false;
            }
        }

        private static object DataProp(object response, string name)
        {
            var data = ((SuccessResponse)response).Data;
            Assert.NotNull(data, "Response data payload expected");
            PropertyInfo prop = data.GetType().GetProperty(name);
            Assert.NotNull(prop, $"Response data should carry '{name}'");
            return prop.GetValue(data);
        }

        [Test]
        public void ForceRefresh_WhileDeferActive_IsDeferredNotExecuted()
        {
            var @params = new JObject
            {
                ["mode"] = "force",
                ["scope"] = "all",
                ["compile"] = "none",
                ["wait_for_ready"] = false
            };

            object response = RefreshUnity.HandleCommand(@params).GetAwaiter().GetResult();

            Assert.IsInstanceOf<SuccessResponse>(response, "Deferral is a success, not an error");
            Assert.IsTrue((bool)DataProp(response, "refresh_deferred"),
                "The explicit refresh must be held while a test run is active");
            Assert.IsFalse((bool)DataProp(response, "refresh_triggered"),
                "No AssetDatabase.Refresh may fire into the running span");
            Assert.IsTrue(DeferredCompileService.HasPendingRefresh,
                "The held refresh must be recorded for replay on return to idle");
        }

        [Test]
        public void ScriptsScope_WithRequestCompile_WhileDeferActive_DefersCompileWithoutRefresh()
        {
            var @params = new JObject
            {
                ["mode"] = "force",
                ["scope"] = "scripts",
                ["compile"] = "request",
                ["wait_for_ready"] = false
            };

            object response = RefreshUnity.HandleCommand(@params).GetAwaiter().GetResult();

            Assert.IsInstanceOf<SuccessResponse>(response);
            Assert.IsTrue((bool)DataProp(response, "compile_deferred"),
                "The compile request routes through the funnel and is held");
            Assert.IsFalse((bool)DataProp(response, "refresh_triggered"),
                "The scripts scope performs no full refresh");
            Assert.IsFalse(DeferredCompileService.HasPendingRefresh,
                "No refresh was requested, so none may be recorded");
        }
    }
}
