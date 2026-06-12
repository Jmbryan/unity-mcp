"""Server-wide protocol constants."""

# HTTP header name for API key authentication
API_KEY_HEADER = "X-API-Key"

# HTTP header name for the optional human-meaningful agent label supplied at
# harness launch (e.g. expanded from an environment variable in the MCP client
# config). Links a session's MCP traffic to the same harness's file edits.
AGENT_LABEL_HEADER = "X-Agent-Label"
