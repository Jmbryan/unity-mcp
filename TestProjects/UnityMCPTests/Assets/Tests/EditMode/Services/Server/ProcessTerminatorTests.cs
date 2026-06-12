using NUnit.Framework;
using MCPForUnity.Editor.Services;
using MCPForUnity.Editor.Services.Server;
using MCPForUnity.Editor.Constants;
using UnityEditor;

namespace MCPForUnityTests.Editor.Services.Server
{
    /// <summary>
    /// Tests for the quit-time shutdown gating (MCPL-002/003/004). The transport "goodbye" step
    /// is unconditional in both toggle states; only the server-stop step is gated by the toggle.
    /// These are non-Explicit (no process termination) so they run in the default EditMode suite.
    /// </summary>
    [TestFixture]
    public class ShutdownCleanupGatingTests
    {
        private bool _savedStopOnQuit;
        private bool _hadStopOnQuit;

        [SetUp]
        public void SetUp()
        {
            _hadStopOnQuit = EditorPrefs.HasKey(EditorPrefKeys.StopServerOnEditorQuit);
            _savedStopOnQuit = EditorPrefs.GetBool(EditorPrefKeys.StopServerOnEditorQuit, false);
        }

        [TearDown]
        public void TearDown()
        {
            if (_hadStopOnQuit)
            {
                EditorPrefs.SetBool(EditorPrefKeys.StopServerOnEditorQuit, _savedStopOnQuit);
            }
            else
            {
                EditorPrefs.DeleteKey(EditorPrefKeys.StopServerOnEditorQuit);
            }
        }

        [Test]
        public void ShouldStopServerOnQuit_DefaultsOff_ServerSurvivesGracefulQuit()
        {
            // Arrange — fresh install: pref never set.
            EditorPrefs.DeleteKey(EditorPrefKeys.StopServerOnEditorQuit);

            // Act / Assert — the server-stop step is skipped, but the goodbye (step 1) still runs
            // because it is outside this gate.
            Assert.IsFalse(McpEditorShutdownCleanup.ShouldStopServerOnQuit(),
                "Default OFF: graceful quit leaves the server running so live sessions survive.");
        }

        [Test]
        public void ShouldStopServerOnQuit_ToggleOn_StopsServer()
        {
            // Arrange
            EditorPrefs.SetBool(EditorPrefKeys.StopServerOnEditorQuit, true);

            // Act / Assert
            Assert.IsTrue(McpEditorShutdownCleanup.ShouldStopServerOnQuit(),
                "Toggle ON restores legacy behavior: the server is stopped on graceful quit.");
        }

        [Test]
        public void ShouldStopServerOnQuit_ToggleOff_SkipsStop()
        {
            // Arrange
            EditorPrefs.SetBool(EditorPrefKeys.StopServerOnEditorQuit, false);

            // Act / Assert
            Assert.IsFalse(McpEditorShutdownCleanup.ShouldStopServerOnQuit(),
                "Toggle OFF: the server-stop step is skipped.");
        }
    }

    /// <summary>
    /// Unit tests for ProcessTerminator component.
    /// Note: Most tests avoid actually terminating processes to prevent test instability.
    /// Uses ProcessDetector which executes subprocess commands (ps, tasklist, etc.), so marked as [Explicit].
    /// </summary>
    [TestFixture]
    [Explicit]
    public class ProcessTerminatorTests
    {
        private ProcessTerminator _terminator;
        private ProcessDetector _detector;

        [SetUp]
        public void SetUp()
        {
            _detector = new ProcessDetector();
            _terminator = new ProcessTerminator(_detector);
        }

        #region Constructor Tests

        [Test]
        public void Constructor_NullDetector_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<System.ArgumentNullException>(() =>
            {
                new ProcessTerminator(null);
            });
        }

        [Test]
        public void Constructor_ValidDetector_Succeeds()
        {
            // Act & Assert
            Assert.DoesNotThrow(() =>
            {
                new ProcessTerminator(_detector);
            });
        }

        #endregion

        #region Terminate Tests

        [Test]
        public void Terminate_InvalidPid_ReturnsFalse()
        {
            // Act
            bool result = _terminator.Terminate(-1);

            // Assert
            Assert.IsFalse(result, "Invalid PID should fail to terminate");
        }

        [Test]
        public void Terminate_ZeroPid_ReturnsFalse()
        {
            // Act
            bool result = _terminator.Terminate(0);

            // Assert
            Assert.IsFalse(result, "Zero PID should fail to terminate");
        }

        [Test]
        public void Terminate_Pid1_ReturnsFalse()
        {
            // PID 1 is init/launchd and must never be killed
            // Act
            bool result = _terminator.Terminate(1);

            // Assert
            Assert.IsFalse(result, "PID 1 (init/launchd) should never be terminated");
        }

        [Test]
        public void Terminate_CurrentProcessPid_ReturnsFalse()
        {
            // Should never kill the Unity Editor process
            // Act
            int currentPid = _detector.GetCurrentProcessId();
            bool result = _terminator.Terminate(currentPid);

            // Assert
            Assert.IsFalse(result, "Current process PID should never be terminated");
        }

        [Test]
        public void Terminate_NonExistentPid_ReturnsFalseOrHandlesGracefully()
        {
            // Act - Use a very high PID unlikely to exist
            bool result = _terminator.Terminate(9999999);

            // Assert - Should not terminate non-existent PID
            Assert.IsFalse(result, $"Terminate returned {result} for non-existent PID");
        }

        [Test]
        public void Terminate_DoesNotThrow()
        {
            // Act & Assert - Should handle any PID gracefully
            Assert.DoesNotThrow(() =>
            {
                _terminator.Terminate(int.MaxValue);
            });
        }

        #endregion

        #region Interface Implementation Tests

        [Test]
        public void ProcessTerminator_ImplementsIProcessTerminator()
        {
            // Assert
            Assert.IsInstanceOf<IProcessTerminator>(_terminator);
        }

        [Test]
        public void ProcessTerminator_CanBeUsedViaInterface()
        {
            // Arrange
            IProcessTerminator terminator = new ProcessTerminator(_detector);

            // Act & Assert - Should be callable via interface
            Assert.DoesNotThrow(() =>
            {
                // Don't actually terminate anything
                terminator.Terminate(-1);
            });
        }

        #endregion

        #region Integration Tests (with real detector)

        [Test]
        public void Terminate_WithRealDetector_HandlesMissingProcess()
        {
            // Arrange
            var realDetector = new ProcessDetector();
            var terminator = new ProcessTerminator(realDetector);

            // Act - Try to terminate a PID that definitely doesn't exist
            bool result = terminator.Terminate(int.MaxValue);

            // Assert - Should return false without throwing
            Assert.IsFalse(result, "Terminating non-existent process should return false");
        }

        #endregion
    }
}
