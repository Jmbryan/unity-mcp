from typing import Any

from fastmcp import Context
from mcp.types import ToolAnnotations
from models.models import MCPResponse

from services.custom_tool_service import (
    CustomToolService,
    get_user_id_from_context,
    resolve_project_id_for_unity_instance,
)
from services.registry import mcp_for_unity_tool
from services.state.operation_gate import gate_for_class, normalize_class
from services.tools import get_unity_instance_from_context


@mcp_for_unity_tool(
    name="execute_custom_tool",
    unity_target=None,
    group=None,
    description="Execute a project-scoped custom tool registered by Unity.",
    annotations=ToolAnnotations(
        title="Execute Custom Tool",
        destructiveHint=True,
    ),
    concurrency_class="wrapper",
)
async def execute_custom_tool(ctx: Context, tool_name: str, parameters: dict[str, Any] | None = None) -> MCPResponse:
    unity_instance = await get_unity_instance_from_context(ctx)
    if not unity_instance:
        return MCPResponse(
            success=False,
            message="No active Unity instance. Call set_active_instance with Name@hash from mcpforunity://instances.",
        )

    project_id = resolve_project_id_for_unity_instance(unity_instance)
    if project_id is None:
        return MCPResponse(
            success=False,
            message=f"Could not resolve project id for {unity_instance}. Ensure Unity is running and reachable.",
        )

    # The signature accepts None (parameter-less custom tools). Treat it as an empty
    # dict rather than rejecting — the previous behavior contradicted the optional type.
    if parameters is None:
        parameters = {}
    elif not isinstance(parameters, dict):
        return MCPResponse(
            success=False,
            message="parameters must be an object/dictionary",
        )

    service = CustomToolService.get_instance()
    user_id = await get_user_id_from_context(ctx)

    # Unwrap for the operation-class gate: the dispatched tool bypasses the
    # per-tool middleware, so classify it here from its registered definition
    # (unknown/missing class gates as mutate).
    definition = await service.get_tool_definition(project_id, tool_name, user_id=user_id)
    if definition is not None:
        inner_class = normalize_class(getattr(definition, "concurrency_class", None))
        busy = await gate_for_class(ctx, inner_class, tool_name, unity_instance)
        if busy is not None:
            return MCPResponse(**busy)

    return await service.execute_tool(
        project_id,
        tool_name,
        unity_instance,
        parameters,
        user_id=user_id,
    )
