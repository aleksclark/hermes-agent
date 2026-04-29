"""ARP Client Tools — Agent Registry Protocol v0.3 HTTP client.

Allows Hermes to interact with an ARP server to discover, message, and
manage A2A agents across workspaces. Uses the ARP HTTP proxy/REST endpoints.

Tools:
    arp_list_agents    — List all ready A2A agents via GET /a2a/agents
    arp_get_agent_card — Get enriched agent card via GET /a2a/agents/{id}/.well-known/agent-card.json
    arp_send_message   — Send a message to a specific agent via POST /a2a/agents/{id}/message:send
    arp_route_message  — Route a message by skill tags via POST /a2a/route/message:send
    arp_list_workspaces — List workspaces via GET /api/workspaces
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import httpx

from tools.registry import registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class ARPError(Exception):
    """Error from an ARP HTTP call."""

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
    """Async HTTP client for the ARP v0.3 proxy/REST endpoints."""

    def __init__(self, base_url: str, token: Optional[str] = None, timeout: float = 120.0):
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout

    def _headers(self) -> Dict[str, str]:
        """Build request headers, including auth if token is set."""
        h: Dict[str, str] = {}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    # -- Helpers ------------------------------------------------------------

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
        self,
        agent_id: str,
        text: str,
        context_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /a2a/agents/{id}/message:send — proxy A2A SendMessage."""
        message: Dict[str, Any] = {
            "role": "ROLE_USER",
            "parts": [{"text": text}],
        }
        if context_id:
            message["contextId"] = context_id
        return await self._post(
            f"/a2a/agents/{agent_id}/message:send",
            {"message": message},
        )

    async def route_message(
        self,
        text: str,
        tags: Optional[List[str]] = None,
        context_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /a2a/route/message:send — skill-based routing."""
        message: Dict[str, Any] = {
            "role": "ROLE_USER",
            "parts": [{"text": text}],
        }
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


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def arp_list_agents_handler(args: dict, **kw) -> str:
    """Tool handler for arp_list_agents."""
    url = args.get("url", "").strip()
    if not url:
        return json.dumps({"error": "Required parameter 'url' is missing. Provide the ARP server URL."})
    token = args.get("token", "").strip() or None
    try:
        client = ARPClient(url, token=token)
        result = _run_arp_async(client.list_agents())
        return json.dumps(result)
    except ARPError as e:
        return json.dumps({"error": f"ARP list agents failed ({e.status_code}): {e.message}"})
    except Exception as e:
        return json.dumps({"error": f"ARP list agents failed: {e}"})


def arp_get_agent_card_handler(args: dict, **kw) -> str:
    """Tool handler for arp_get_agent_card."""
    url = args.get("url", "").strip()
    agent_id = args.get("agent_id", "").strip()
    if not url:
        return json.dumps({"error": "Required parameter 'url' is missing."})
    if not agent_id:
        return json.dumps({"error": "Required parameter 'agent_id' is missing."})
    token = args.get("token", "").strip() or None
    try:
        client = ARPClient(url, token=token)
        result = _run_arp_async(client.get_agent_card(agent_id))
        return json.dumps(result)
    except ARPError as e:
        return json.dumps({"error": f"ARP get agent card failed ({e.status_code}): {e.message}"})
    except Exception as e:
        return json.dumps({"error": f"ARP get agent card failed: {e}"})


def arp_send_message_handler(args: dict, **kw) -> str:
    """Tool handler for arp_send_message."""
    url = args.get("url", "").strip()
    agent_id = args.get("agent_id", "").strip()
    message = args.get("message", "").strip()
    context_id = args.get("context_id", "").strip() or None
    if not url:
        return json.dumps({"error": "Required parameter 'url' is missing."})
    if not agent_id:
        return json.dumps({"error": "Required parameter 'agent_id' is missing."})
    if not message:
        return json.dumps({"error": "Required parameter 'message' is missing."})
    token = args.get("token", "").strip() or None
    try:
        client = ARPClient(url, token=token)
        result = _run_arp_async(client.send_message(agent_id, message, context_id=context_id))
        return json.dumps(result)
    except ARPError as e:
        return json.dumps({"error": f"ARP send message failed ({e.status_code}): {e.message}"})
    except Exception as e:
        return json.dumps({"error": f"ARP send message failed: {e}"})


def arp_route_message_handler(args: dict, **kw) -> str:
    """Tool handler for arp_route_message."""
    url = args.get("url", "").strip()
    message = args.get("message", "").strip()
    tags = args.get("tags", [])
    context_id = args.get("context_id", "").strip() or None
    if not url:
        return json.dumps({"error": "Required parameter 'url' is missing."})
    if not message:
        return json.dumps({"error": "Required parameter 'message' is missing."})
    token = args.get("token", "").strip() or None
    try:
        client = ARPClient(url, token=token)
        result = _run_arp_async(client.route_message(message, tags=tags or None, context_id=context_id))
        return json.dumps(result)
    except ARPError as e:
        return json.dumps({"error": f"ARP route message failed ({e.status_code}): {e.message}"})
    except Exception as e:
        return json.dumps({"error": f"ARP route message failed: {e}"})


def arp_list_workspaces_handler(args: dict, **kw) -> str:
    """Tool handler for arp_list_workspaces."""
    url = args.get("url", "").strip()
    if not url:
        return json.dumps({"error": "Required parameter 'url' is missing."})
    token = args.get("token", "").strip() or None
    try:
        client = ARPClient(url, token=token)
        result = _run_arp_async(client.list_workspaces())
        return json.dumps(result)
    except ARPError as e:
        return json.dumps({"error": f"ARP list workspaces failed ({e.status_code}): {e.message}"})
    except Exception as e:
        return json.dumps({"error": f"ARP list workspaces failed: {e}"})


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

ARP_LIST_AGENTS_SCHEMA = {
    "name": "arp_list_agents",
    "description": (
        "List all ready A2A agents managed by an ARP server. "
        "Returns an array of enriched AgentCard objects with name, skills, "
        "capabilities, and ARP metadata (agent_id, workspace, project, status, URLs). "
        "Use this to discover which agents are available before sending messages."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Base URL of the ARP server (e.g. http://localhost:9099)",
            },
            "token": {
                "type": "string",
                "description": "Optional ARP bearer token for authentication",
            },
        },
        "required": ["url"],
    },
}

ARP_GET_AGENT_CARD_SCHEMA = {
    "name": "arp_get_agent_card",
    "description": (
        "Get the enriched A2A agent card for a specific agent managed by ARP. "
        "Returns the full AgentCard with metadata.arp containing agent_id, workspace, "
        "project, template, status, direct_url, and started_at."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Base URL of the ARP server",
            },
            "agent_id": {
                "type": "string",
                "description": "The agent ID, name, or workspace/name composite to look up",
            },
            "token": {
                "type": "string",
                "description": "Optional ARP bearer token for authentication",
            },
        },
        "required": ["url", "agent_id"],
    },
}

ARP_SEND_MESSAGE_SCHEMA = {
    "name": "arp_send_message",
    "description": (
        "Send a text message to a specific A2A agent via the ARP proxy. "
        "The ARP server proxies the request to the agent's A2A endpoint. "
        "Returns the agent's response (a Task or Message). Use context_id "
        "from a previous response to continue a multi-turn conversation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Base URL of the ARP server",
            },
            "agent_id": {
                "type": "string",
                "description": "The agent ID, name, or workspace/name composite to message",
            },
            "message": {
                "type": "string",
                "description": "The text message to send to the agent",
            },
            "context_id": {
                "type": "string",
                "description": "Optional context ID for multi-turn conversations",
            },
            "token": {
                "type": "string",
                "description": "Optional ARP bearer token for authentication",
            },
        },
        "required": ["url", "agent_id", "message"],
    },
}

ARP_ROUTE_MESSAGE_SCHEMA = {
    "name": "arp_route_message",
    "description": (
        "Route a message to an A2A agent by skill tags via the ARP proxy. "
        "The ARP server selects the best matching agent based on "
        "AgentCard.skills[].tags, preferring agents with 'ready' status. "
        "Use this when you don't know the specific agent_id but know what "
        "skill you need (e.g. tags=['coding'], tags=['summarization'])."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Base URL of the ARP server",
            },
            "message": {
                "type": "string",
                "description": "The text message to send",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Skill tags to match agents on (e.g. ['coding', 'python'])",
            },
            "context_id": {
                "type": "string",
                "description": "Optional context ID for multi-turn conversations",
            },
            "token": {
                "type": "string",
                "description": "Optional ARP bearer token for authentication",
            },
        },
        "required": ["url", "message"],
    },
}

ARP_LIST_WORKSPACES_SCHEMA = {
    "name": "arp_list_workspaces",
    "description": (
        "List all workspaces managed by an ARP server. "
        "Returns workspace name, project, active status, and directory path. "
        "Workspaces contain agents and are associated with a project."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Base URL of the ARP server",
            },
            "token": {
                "type": "string",
                "description": "Optional ARP bearer token for authentication",
            },
        },
        "required": ["url"],
    },
}


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------

registry.register(
    name="arp_list_agents",
    toolset="arp",
    schema=ARP_LIST_AGENTS_SCHEMA,
    handler=arp_list_agents_handler,
    is_async=False,
    description="List all ready A2A agents on an ARP server",
    emoji="📋",
)

registry.register(
    name="arp_get_agent_card",
    toolset="arp",
    schema=ARP_GET_AGENT_CARD_SCHEMA,
    handler=arp_get_agent_card_handler,
    is_async=False,
    description="Get enriched agent card from an ARP server",
    emoji="🪪",
)

registry.register(
    name="arp_send_message",
    toolset="arp",
    schema=ARP_SEND_MESSAGE_SCHEMA,
    handler=arp_send_message_handler,
    is_async=False,
    description="Send a message to an ARP-managed A2A agent",
    emoji="📡",
)

registry.register(
    name="arp_route_message",
    toolset="arp",
    schema=ARP_ROUTE_MESSAGE_SCHEMA,
    handler=arp_route_message_handler,
    is_async=False,
    description="Route a message to an agent by skill tags via ARP",
    emoji="🔀",
)

registry.register(
    name="arp_list_workspaces",
    toolset="arp",
    schema=ARP_LIST_WORKSPACES_SCHEMA,
    handler=arp_list_workspaces_handler,
    is_async=False,
    description="List workspaces on an ARP server",
    emoji="🏗️",
)
