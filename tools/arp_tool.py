"""ARP Client Tools — Agent Registry Protocol v0.3 client.

Allows Hermes to interact with an ARP server to discover, message, and
manage A2A agents across workspaces. Covers both the HTTP proxy/REST
endpoints and the MCP lifecycle tools (via JSON-RPC over HTTP).

Tools:
    arp_list_agents    — List all ready A2A agents via GET /a2a/agents
    arp_get_agent_card — Get enriched agent card via GET /a2a/agents/{id}/.well-known/agent-card.json
    arp_send_message   — Send a message to a specific agent via POST /a2a/agents/{id}/message:send
    arp_route_message  — Route a message by skill tags via POST /a2a/route/message:send
    arp_list_workspaces — List workspaces via GET /api/workspaces
    arp_manage         — Invoke any ARP MCP tool (project/workspace/agent lifecycle)
"""

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from tools.registry import registry

logger = logging.getLogger(__name__)

# Allowed MCP tool names for the arp_manage tool (whitelist)
_ALLOWED_MCP_TOOLS = frozenset({
    "project/list", "project/register", "project/unregister",
    "workspace/create", "workspace/list", "workspace/get", "workspace/destroy",
    "agent/spawn", "agent/list", "agent/status", "agent/stop", "agent/restart",
    "agent/message", "agent/task", "agent/task_status",
    "token/create",
})


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class ARPError(Exception):
    """Error from an ARP HTTP call or MCP tool invocation."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"ARP error {status_code}: {message}")


# ---------------------------------------------------------------------------
# Async bridge
# ---------------------------------------------------------------------------

def _run_arp_async(coro):
    """Run an async coroutine from sync context."""
    from model_tools import _run_async
    return _run_async(coro)


# ---------------------------------------------------------------------------
# ARPClient
# ---------------------------------------------------------------------------

class ARPClient:
    """Async HTTP client for the ARP v0.3 proxy/REST and MCP endpoints."""

    def __init__(self, base_url: str, token: Optional[str] = None, timeout: float = 120.0):
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _headers(self) -> Dict[str, str]:
        """Build request headers, including auth if token is set."""
        h: Dict[str, str] = {}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    # -- HTTP helpers -------------------------------------------------------

    async def _get(self, path: str) -> Any:
        """GET request returning parsed JSON."""
        url = f"{self._base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=self._headers())
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            raise ARPError(e.response.status_code, f"HTTP {e.response.status_code} from {url}")
        except httpx.TimeoutException as e:
            raise ARPError(-1, f"Timeout: {e}")
        except httpx.ConnectError as e:
            raise ARPError(-1, f"Connection error: {e}")
        except Exception as e:
            raise ARPError(-1, f"Request failed: {e}")

    async def _post(self, path: str, body: Dict[str, Any]) -> Any:
        """POST request returning parsed JSON."""
        url = f"{self._base_url}{path}"
        headers = {**self._headers(), "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=body, headers=headers)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            raise ARPError(e.response.status_code, f"HTTP {e.response.status_code} from {url}")
        except httpx.TimeoutException as e:
            raise ARPError(-1, f"Timeout: {e}")
        except httpx.ConnectError as e:
            raise ARPError(-1, f"Connection error: {e}")
        except Exception as e:
            raise ARPError(-1, f"Request failed: {e}")

    # -- MCP tool_call (JSON-RPC over HTTP) ---------------------------------

    async def mcp_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Invoke an ARP MCP tool via JSON-RPC 2.0 POST to /mcp.

        Sends a ``tools/call`` request and returns the result.
        Raises ARPError on JSON-RPC errors or transport failures.
        """
        req_id = self._next_id()
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        url = f"{self._base_url}/mcp"
        headers = {**self._headers(), "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ARPError(e.response.status_code, f"HTTP {e.response.status_code} from {url}")
        except httpx.TimeoutException as e:
            raise ARPError(-1, f"Timeout calling {tool_name}: {e}")
        except httpx.ConnectError as e:
            raise ARPError(-1, f"Connection error calling {tool_name}: {e}")
        except Exception as e:
            raise ARPError(-1, f"Transport error calling {tool_name}: {e}")

        data = resp.json()
        if "error" in data:
            err = data["error"]
            raise ARPError(err.get("code", -1), err.get("message", "Unknown error"))
        return data.get("result")

    # -- A2A Proxy endpoints ------------------------------------------------

    async def list_agents(self) -> List[Dict[str, Any]]:
        """GET /a2a/agents — list all ready agent cards."""
        return await self._get("/a2a/agents")

    async def discover(self) -> Dict[str, Any]:
        """GET /a2a/discover — agent discovery endpoint."""
        return await self._get("/a2a/discover")

    async def get_agent_card(self, agent_id: str) -> Dict[str, Any]:
        """GET /a2a/agents/{id}/.well-known/agent-card.json — enriched agent card."""
        return await self._get(f"/a2a/agents/{agent_id}/.well-known/agent-card.json")

    async def send_message(
        self, agent_id: str, text: str, context_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /a2a/agents/{id}/message:send — proxy A2A SendMessage."""
        message: Dict[str, Any] = {"role": "ROLE_USER", "parts": [{"text": text}]}
        if context_id:
            message["contextId"] = context_id
        return await self._post(f"/a2a/agents/{agent_id}/message:send", {"message": message})

    async def route_message(
        self, text: str, tags: Optional[List[str]] = None, context_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /a2a/route/message:send — skill-based routing."""
        message: Dict[str, Any] = {"role": "ROLE_USER", "parts": [{"text": text}]}
        if context_id:
            message["contextId"] = context_id
        body: Dict[str, Any] = {"message": message}
        if tags:
            body["routing"] = {"tags": tags}
        return await self._post("/a2a/route/message:send", body)

    # -- REST API endpoints -------------------------------------------------

    async def list_workspaces(self) -> List[Dict[str, Any]]:
        """GET /api/workspaces — list all workspaces."""
        return await self._get("/api/workspaces")

    async def get_workspace(self, name: str) -> Dict[str, Any]:
        """GET /api/workspaces/{name} — get single workspace."""
        return await self._get(f"/api/workspaces/{name}")

    async def list_projects(self) -> List[Dict[str, Any]]:
        """GET /api/projects — list all projects."""
        return await self._get("/api/projects")

    # -- MCP lifecycle convenience methods ----------------------------------

    async def project_list(self) -> Any:
        return await self.mcp_tool_call("project/list", {})

    async def project_register(
        self, name: str, repo: str, agents: Optional[List[Dict]] = None,
    ) -> Any:
        args: Dict[str, Any] = {"name": name, "repo": repo}
        if agents:
            args["agents"] = agents
        return await self.mcp_tool_call("project/register", args)

    async def project_unregister(self, name: str) -> Any:
        return await self.mcp_tool_call("project/unregister", {"name": name})

    async def workspace_create(
        self, name: str, project: str, auto_agents: Optional[List[str]] = None,
    ) -> Any:
        args: Dict[str, Any] = {"name": name, "project": project}
        if auto_agents:
            args["auto_agents"] = auto_agents
        return await self.mcp_tool_call("workspace/create", args)

    async def workspace_get(self, name: str) -> Any:
        return await self.mcp_tool_call("workspace/get", {"name": name})

    async def workspace_destroy(self, name: str, keep_worktree: bool = False) -> Any:
        args: Dict[str, Any] = {"name": name}
        if keep_worktree:
            args["keep_worktree"] = True
        return await self.mcp_tool_call("workspace/destroy", args)

    async def agent_spawn(
        self, workspace: str, template: str, *,
        name: Optional[str] = None, prompt: Optional[str] = None,
        scope: Optional[List[str]] = None, permission: Optional[str] = None,
    ) -> Any:
        args: Dict[str, Any] = {"workspace": workspace, "template": template}
        if name:
            args["name"] = name
        if prompt:
            args["prompt"] = prompt
        if scope:
            args["scope"] = scope
        if permission:
            args["permission"] = permission
        return await self.mcp_tool_call("agent/spawn", args)

    async def agent_status(self, agent_id: str) -> Any:
        return await self.mcp_tool_call("agent/status", {"agent_id": agent_id})

    async def agent_stop(self, agent_id: str, grace_period_ms: Optional[int] = None) -> Any:
        args: Dict[str, Any] = {"agent_id": agent_id}
        if grace_period_ms is not None:
            args["grace_period_ms"] = grace_period_ms
        return await self.mcp_tool_call("agent/stop", args)

    async def agent_restart(self, agent_id: str) -> Any:
        return await self.mcp_tool_call("agent/restart", {"agent_id": agent_id})

    async def agent_message(
        self, agent_id: str, message: str, *,
        context_id: Optional[str] = None, blocking: Optional[bool] = None,
    ) -> Any:
        args: Dict[str, Any] = {"agent_id": agent_id, "message": message}
        if context_id:
            args["context_id"] = context_id
        if blocking is not None:
            args["blocking"] = blocking
        return await self.mcp_tool_call("agent/message", args)

    async def agent_task(
        self, agent_id: str, message: str, context_id: Optional[str] = None,
    ) -> Any:
        args: Dict[str, Any] = {"agent_id": agent_id, "message": message}
        if context_id:
            args["context_id"] = context_id
        return await self.mcp_tool_call("agent/task", args)

    async def agent_task_status(
        self, agent_id: str, task_id: str, history_length: Optional[int] = None,
    ) -> Any:
        args: Dict[str, Any] = {"agent_id": agent_id, "task_id": task_id}
        if history_length is not None:
            args["history_length"] = history_length
        return await self.mcp_tool_call("agent/task_status", args)


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _make_client(args: dict) -> ARPClient:
    """Extract url/token from args and build an ARPClient."""
    return ARPClient(
        args.get("url", "").strip(),
        token=args.get("token", "").strip() or None,
    )


def _handler_boilerplate(args: dict, label: str, coro_fn):
    """Shared error-handling wrapper for all ARP tool handlers."""
    url = args.get("url", "").strip()
    if not url:
        return json.dumps({"error": f"Required parameter 'url' is missing. Provide the ARP server URL."})
    try:
        result = _run_arp_async(coro_fn())
        return json.dumps(result)
    except ARPError as e:
        return json.dumps({"error": f"ARP {label} failed ({e.status_code}): {e.message}"})
    except Exception as e:
        return json.dumps({"error": f"ARP {label} failed: {e}"})


def arp_list_agents_handler(args: dict, **kw) -> str:
    return _handler_boilerplate(args, "list agents", lambda: _make_client(args).list_agents())


def arp_get_agent_card_handler(args: dict, **kw) -> str:
    agent_id = args.get("agent_id", "").strip()
    if not agent_id:
        return json.dumps({"error": "Required parameter 'agent_id' is missing."})
    return _handler_boilerplate(args, "get agent card", lambda: _make_client(args).get_agent_card(agent_id))


def arp_send_message_handler(args: dict, **kw) -> str:
    agent_id = args.get("agent_id", "").strip()
    message = args.get("message", "").strip()
    if not agent_id:
        return json.dumps({"error": "Required parameter 'agent_id' is missing."})
    if not message:
        return json.dumps({"error": "Required parameter 'message' is missing."})
    context_id = args.get("context_id", "").strip() or None
    return _handler_boilerplate(
        args, "send message",
        lambda: _make_client(args).send_message(agent_id, message, context_id=context_id),
    )


def arp_route_message_handler(args: dict, **kw) -> str:
    message = args.get("message", "").strip()
    if not message:
        return json.dumps({"error": "Required parameter 'message' is missing."})
    tags = args.get("tags", [])
    context_id = args.get("context_id", "").strip() or None
    return _handler_boilerplate(
        args, "route message",
        lambda: _make_client(args).route_message(message, tags=tags or None, context_id=context_id),
    )


def arp_list_workspaces_handler(args: dict, **kw) -> str:
    return _handler_boilerplate(args, "list workspaces", lambda: _make_client(args).list_workspaces())


def arp_manage_handler(args: dict, **kw) -> str:
    """Generic handler for ARP MCP tool invocations (lifecycle operations)."""
    url = args.get("url", "").strip()
    tool = args.get("tool", "").strip()
    arguments = args.get("arguments", {})
    if not url:
        return json.dumps({"error": "Required parameter 'url' is missing."})
    if not tool:
        return json.dumps({"error": "Required parameter 'tool' is missing. Specify the ARP MCP tool name (e.g. 'project/register', 'agent/spawn')."})
    if tool not in _ALLOWED_MCP_TOOLS:
        return json.dumps({"error": f"Tool '{tool}' is not a recognized ARP MCP tool. Allowed: {', '.join(sorted(_ALLOWED_MCP_TOOLS))}"})
    client = _make_client(args)
    try:
        result = _run_arp_async(client.mcp_tool_call(tool, arguments or {}))
        return json.dumps(result)
    except ARPError as e:
        return json.dumps({"error": f"ARP {tool} failed ({e.status_code}): {e.message}"})
    except Exception as e:
        return json.dumps({"error": f"ARP {tool} failed: {e}"})


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

ARP_LIST_AGENTS_SCHEMA = {
    "name": "arp_list_agents",
    "description": (
        "List all ready A2A agents managed by an ARP server. "
        "Returns an array of enriched AgentCard objects with name, skills, "
        "capabilities, and ARP metadata (agent_id, workspace, project, status, URLs)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Base URL of the ARP server (e.g. http://localhost:9099)"},
            "token": {"type": "string", "description": "Optional ARP bearer token for authentication"},
        },
        "required": ["url"],
    },
}

ARP_GET_AGENT_CARD_SCHEMA = {
    "name": "arp_get_agent_card",
    "description": (
        "Get the enriched A2A agent card for a specific agent managed by ARP. "
        "Returns the full AgentCard with metadata.arp (agent_id, workspace, project, "
        "template, status, direct_url, started_at)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Base URL of the ARP server"},
            "agent_id": {"type": "string", "description": "Agent ID, name, or workspace/name composite"},
            "token": {"type": "string", "description": "Optional ARP bearer token"},
        },
        "required": ["url", "agent_id"],
    },
}

ARP_SEND_MESSAGE_SCHEMA = {
    "name": "arp_send_message",
    "description": (
        "Send a text message to a specific A2A agent via the ARP proxy. "
        "Returns the agent's response (Task or Message). Use context_id "
        "from a previous response to continue a multi-turn conversation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Base URL of the ARP server"},
            "agent_id": {"type": "string", "description": "Agent ID, name, or workspace/name composite"},
            "message": {"type": "string", "description": "Text message to send"},
            "context_id": {"type": "string", "description": "Optional context ID for multi-turn conversations"},
            "token": {"type": "string", "description": "Optional ARP bearer token"},
        },
        "required": ["url", "agent_id", "message"],
    },
}

ARP_ROUTE_MESSAGE_SCHEMA = {
    "name": "arp_route_message",
    "description": (
        "Route a message to an A2A agent by skill tags via the ARP proxy. "
        "ARP selects the best matching agent based on AgentCard.skills[].tags, "
        "preferring 'ready' agents over 'busy'. Use when you don't know the "
        "specific agent_id but know what skill you need."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Base URL of the ARP server"},
            "message": {"type": "string", "description": "Text message to send"},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Skill tags to match (e.g. ['coding', 'python'])"},
            "context_id": {"type": "string", "description": "Optional context ID for multi-turn conversations"},
            "token": {"type": "string", "description": "Optional ARP bearer token"},
        },
        "required": ["url", "message"],
    },
}

ARP_LIST_WORKSPACES_SCHEMA = {
    "name": "arp_list_workspaces",
    "description": (
        "List all workspaces managed by an ARP server. "
        "Returns workspace name, project, active status, and directory path."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Base URL of the ARP server"},
            "token": {"type": "string", "description": "Optional ARP bearer token"},
        },
        "required": ["url"],
    },
}

ARP_MANAGE_SCHEMA = {
    "name": "arp_manage",
    "description": (
        "Invoke an ARP MCP lifecycle tool to manage projects, workspaces, and agents. "
        "This is the primary interface for creating/destroying infrastructure and "
        "controlling agent lifecycle.\n\n"
        "Available tools:\n"
        "  project/list — List registered projects\n"
        "  project/register — Register a project (name, repo, optional agents templates)\n"
        "  project/unregister — Remove a project\n"
        "  workspace/create — Create a workspace (name, project, optional auto_agents)\n"
        "  workspace/list — List workspaces (optional project/status filter)\n"
        "  workspace/get — Get full workspace details\n"
        "  workspace/destroy — Destroy workspace, stop agents, remove worktree\n"
        "  agent/spawn — Spawn an agent from a template (workspace, template, optional name/prompt/scope/permission)\n"
        "  agent/list — List agent instances (optional workspace/status/template filter)\n"
        "  agent/status — Get full agent status with AgentCard\n"
        "  agent/stop — Stop an agent (optional grace_period_ms)\n"
        "  agent/restart — Restart an agent, preserving template and workspace\n"
        "  agent/message — Send A2A message to agent (agent_id, message, optional context_id/blocking)\n"
        "  agent/task — Send message expecting Task response (agent_id, message)\n"
        "  agent/task_status — Get task status (agent_id, task_id, optional history_length)\n"
        "  token/create — Create auth token (requires admin, subject/scope/permission)"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Base URL of the ARP server"},
            "tool": {
                "type": "string",
                "description": "ARP MCP tool name (e.g. 'project/register', 'agent/spawn', 'workspace/create')",
            },
            "arguments": {
                "type": "object",
                "description": "Arguments to pass to the MCP tool (varies per tool)",
            },
            "token": {"type": "string", "description": "Optional ARP bearer token"},
        },
        "required": ["url", "tool"],
    },
}


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------

for _name, _schema, _handler, _desc, _emoji in [
    ("arp_list_agents", ARP_LIST_AGENTS_SCHEMA, arp_list_agents_handler, "List all ready A2A agents on an ARP server", "📋"),
    ("arp_get_agent_card", ARP_GET_AGENT_CARD_SCHEMA, arp_get_agent_card_handler, "Get enriched agent card from an ARP server", "🪪"),
    ("arp_send_message", ARP_SEND_MESSAGE_SCHEMA, arp_send_message_handler, "Send a message to an ARP-managed A2A agent", "📡"),
    ("arp_route_message", ARP_ROUTE_MESSAGE_SCHEMA, arp_route_message_handler, "Route a message to an agent by skill tags via ARP", "🔀"),
    ("arp_list_workspaces", ARP_LIST_WORKSPACES_SCHEMA, arp_list_workspaces_handler, "List workspaces on an ARP server", "🏗️"),
    ("arp_manage", ARP_MANAGE_SCHEMA, arp_manage_handler, "Manage ARP projects, workspaces, and agents", "⚙️"),
]:
    registry.register(
        name=_name, toolset="arp", schema=_schema, handler=_handler,
        is_async=False, description=_desc, emoji=_emoji,
    )
