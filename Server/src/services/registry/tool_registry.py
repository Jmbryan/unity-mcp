"""
Tool registry for auto-discovery of MCP tools.

Tools can be assigned to *groups* via the ``group`` parameter.  Groups map to
FastMCP tags (``"group:<name>"``) which drive the per-session visibility
system exposed through the ``manage_tools`` meta-tool.

The special group value ``None`` means the tool is *always visible* and
cannot be disabled by the group system (used for server meta-tools like
``set_active_instance`` and ``manage_tools``).
"""
from typing import Callable, Any

# Global registry to collect decorated tools
_tool_registry: list[dict[str, Any]] = []

# Valid group names. ``None`` is also accepted (always-visible meta-tools).
TOOL_GROUPS: dict[str, str] = {
    "core": "Essential scene, script, asset & editor tools (always on by default)",
    "docs": "Unity API reflection and documentation lookup",
    "vfx": "Visual effects – VFX Graph, shaders, procedural textures",
    "animation": "Animator control & AnimationClip creation",
    "ui": "UI Toolkit (UXML, USS, UIDocument)",
    "scripting_ext": "ScriptableObject management",
    "testing": "Test runner & async test jobs",
    "probuilder": "ProBuilder 3D modeling – requires com.unity.probuilder package",
    "profiling": "Unity Profiler session control, counters, memory snapshots & Frame Debugger",
}

DEFAULT_ENABLED_GROUPS: set[str] = {"core"}

# Concurrency classes for the operation-class gate. ``wrapper`` is
# server-internal: the tool's effective class is computed inside its own body
# (batch/custom-tool dispatch), so the middleware gate passes it through.
CONCURRENCY_CLASSES: set[str] = {
    "read", "mutate", "exclusive", "play-scoped", "wrapper",
}
DEFAULT_CONCURRENCY_CLASS = "mutate"


def mcp_for_unity_tool(
    name: str | None = None,
    description: str | None = None,
    unity_target: str | None = "self",
    group: str | None = "core",
    concurrency_class: str | None = None,
    **kwargs
) -> Callable:
    """
    Decorator for registering MCP tools in the server's tools directory.

    Tools are registered in the global tool registry.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description
        unity_target: Visibility target used by middleware filtering.
            - "self" (default): tool follows its own enabled state.
            - None: server-only tool, always visible in tool listing.
            - "<tool_name>": alias tool that follows another Unity tool state.
        group: Tool group for dynamic visibility.
            - A group name string (e.g. "core", "vfx") assigns the tool to
              that group and adds a ``tags={"group:<name>"}`` entry.
            - None: the tool is *always visible* (server meta-tools).
        concurrency_class: Declared concurrency class for the operation gate.
            One of "read" (always passes), "mutate" (default; parks while the
            editor is busy), "exclusive" (compile/test/package-ish; also gated
            during test runs and by the compile fence), "play-scoped"
            (mutates a running play session), or "wrapper" (effective class is
            computed inside the tool body). None means the default ("mutate").
        **kwargs: Additional arguments passed to @mcp.tool()

    Example:
        @mcp_for_unity_tool(description="Does something cool")
        async def my_custom_tool(ctx: Context, ...):
            pass
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name if name is not None else func.__name__
        # Safety guard: unity_target is internal metadata and must never leak into mcp.tool kwargs.
        tool_kwargs = dict(kwargs)  # Create a copy to avoid side effects
        if "unity_target" in tool_kwargs:
            del tool_kwargs["unity_target"]
        if "group" in tool_kwargs:
            del tool_kwargs["group"]
        if "concurrency_class" in tool_kwargs:
            del tool_kwargs["concurrency_class"]

        # Validate and normalize concurrency class
        resolved_class = (
            DEFAULT_CONCURRENCY_CLASS
            if concurrency_class is None
            else concurrency_class
        )
        if resolved_class not in CONCURRENCY_CLASSES:
            raise ValueError(
                f"Unknown concurrency_class '{concurrency_class}' for tool '{tool_name}'. "
                f"Valid classes: {', '.join(sorted(CONCURRENCY_CLASSES))}."
            )

        # Validate and normalize group
        resolved_group: str | None = None
        if group is not None:
            if group not in TOOL_GROUPS:
                raise ValueError(
                    f"Unknown group '{group}' for tool '{tool_name}'. "
                    f"Valid groups: {', '.join(sorted(TOOL_GROUPS))}."
                )
            resolved_group = group
            # Merge the group tag into any existing tags the caller provided
            existing_tags: set[str] = set(tool_kwargs.get("tags") or set())
            existing_tags.add(f"group:{group}")
            tool_kwargs["tags"] = existing_tags

        if unity_target is None:
            normalized_unity_target: str | None = None
        elif isinstance(unity_target, str) and unity_target.strip():
            normalized_unity_target = (
                tool_name if unity_target == "self" else unity_target.strip()
            )
        else:
            raise ValueError(
                f"Invalid unity_target for tool '{tool_name}': {unity_target!r}. "
                "Expected None or a non-empty string."
            )

        _tool_registry.append({
            'func': func,
            'name': tool_name,
            'description': description,
            'unity_target': normalized_unity_target,
            'group': resolved_group,
            'concurrency_class': resolved_class,
            'kwargs': tool_kwargs,
        })

        return func

    return decorator


def get_registered_tools() -> list[dict[str, Any]]:
    """Get all registered tools"""
    return _tool_registry.copy()


def get_group_tool_names() -> dict[str, list[str]]:
    """Return a mapping of group name -> list of tool names in that group."""
    result: dict[str, list[str]] = {g: [] for g in TOOL_GROUPS}
    for tool in _tool_registry:
        g = tool.get("group")
        if g and g in result:
            result[g].append(tool["name"])
    return result


def clear_tool_registry():
    """Clear the tool registry (useful for testing)"""
    _tool_registry.clear()
