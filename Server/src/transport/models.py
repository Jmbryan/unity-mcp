from typing import Any
from pydantic import BaseModel, Field
from models.models import ToolDefinitionModel

# Outgoing (Server -> Plugin)


class WelcomeMessage(BaseModel):
    type: str = "welcome"
    serverTimeout: int
    keepAliveInterval: int


class RegisteredMessage(BaseModel):
    type: str = "registered"
    session_id: str


class ExecuteCommandMessage(BaseModel):
    type: str = "execute"
    id: str
    name: str
    params: dict[str, Any]
    timeout: float


class PingMessage(BaseModel):
    """Server-initiated ping to detect dead connections."""
    type: str = "ping"

# Incoming (Plugin -> Server)


class RegisterMessage(BaseModel):
    type: str = "register"
    project_name: str = "Unknown Project"
    project_hash: str
    unity_version: str = "Unknown"
    project_path: str | None = None  # Full path to project root (for focus nudging)


class RegisterToolsMessage(BaseModel):
    type: str = "register_tools"
    tools: list[ToolDefinitionModel]


class PongMessage(BaseModel):
    type: str = "pong"
    session_id: str | None = None


class CommandResultMessage(BaseModel):
    type: str = "command_result"
    id: str
    result: dict[str, Any] = Field(default_factory=dict)


# Editor-edge event names the bridge pushes so parked gate calls release
# event-driven instead of waiting out their bounded poll. Best-effort: the
# bounded-wait deadline remains the backstop, so an unknown or dropped event
# never wedges a call.
EVENT_ENTERED_PLAY = "entered_play"
EVENT_EXITED_PLAY = "exited_play"
EVENT_COMPILE_STARTED = "compile_started"
EVENT_COMPILE_FINISHED = "compile_finished"
EVENT_DOMAIN_RELOAD_DONE = "domain_reload_done"

EDITOR_EDGE_EVENTS = frozenset(
    {
        EVENT_ENTERED_PLAY,
        EVENT_EXITED_PLAY,
        EVENT_COMPILE_STARTED,
        EVENT_COMPILE_FINISHED,
        EVENT_DOMAIN_RELOAD_DONE,
    }
)


class EventMessage(BaseModel):
    """Lightweight editor-edge event pushed by the bridge (MCPC-030).

    Every field other than ``event`` is optional so an older bridge (or a
    truncated push) parses without error. ``instance`` / ``project_hash``
    identify which Unity instance the edge belongs to; either may be absent,
    in which case the default instance key is used. ``ts`` is the bridge's
    unix timestamp (seconds); ``payload`` carries optional event-specific
    detail and is ignored when absent.
    """

    type: str = "event"
    event: str
    instance: str | None = None
    project_hash: str | None = None
    ts: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

# Session Info (API response)


class SessionDetails(BaseModel):
    project: str
    hash: str
    unity_version: str
    connected_at: str


class SessionList(BaseModel):
    sessions: dict[str, SessionDetails]
