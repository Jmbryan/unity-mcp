using System.Threading.Tasks;

namespace MCPForUnity.Editor.Services.Transport
{
    /// <summary>
    /// Abstraction for MCP transport implementations (e.g. WebSocket push, stdio).
    /// </summary>
    public interface IMcpTransportClient
    {
        bool IsConnected { get; }
        string TransportName { get; }
        TransportState State { get; }

        Task<bool> StartAsync();
        Task StopAsync();
        Task<bool> VerifyAsync();
        Task ReregisterToolsAsync();

        /// <summary>
        /// Pushes a lightweight editor-edge event to the server so it can release parked gate
        /// calls event-driven instead of waiting out their bounded poll. Best-effort and
        /// fire-and-forget: a short timeout bounds the send and failures are swallowed, so this
        /// never blocks the caller (notably the domain-reload path) and never throws.
        /// </summary>
        /// <param name="eventName">
        /// One of the editor-edge event names the server accepts (e.g. entered_play, exited_play,
        /// compile_started, compile_finished, domain_reload_done).
        /// </param>
        Task PushEventAsync(string eventName);
    }
}
