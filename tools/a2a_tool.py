"""A2A Client Tools — Agent-to-Agent protocol v1.0 client.

Allows Hermes to discover and communicate with remote A2A-compliant agents
via JSON-RPC 2.0 over HTTP. Implements the client side of the A2A v1.0 spec.

Tools:
    a2a_discover  — Fetch an agent's card from /.well-known/agent-card.json
    a2a_send      — Send a message to an A2A agent (JSON-RPC SendMessage)
    a2a_get_task  — Retrieve task status (JSON-RPC GetTask)
    a2a_cancel_task — Cancel a running task (JSON-RPC CancelTask)
"""

import json
import logging
import uuid
from typing import Any, Dict, Optional

import httpx

from tools.registry import registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class A2AError(Exception):
    """Error from an A2A JSON-RPC call or transport failure."""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"A2A error {code}: {message}")


# ---------------------------------------------------------------------------
# Async bridge (reuses model_tools pattern)
# ---------------------------------------------------------------------------

def _run_a2a_async(coro):
    """Run an async coroutine from sync context.

    Delegates to model_tools._run_async for consistent event-loop management.
    """
    from model_tools import _run_async
    return _run_async(coro)


# ---------------------------------------------------------------------------
# A2AClient
# ---------------------------------------------------------------------------

class A2AClient:
    """Async client for the A2A v1.0 protocol (JSON-RPC over HTTP)."""

    def __init__(self, base_url: str, timeout: float = 120.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    # -- Discovery ----------------------------------------------------------

    async def discover(self) -> Dict[str, Any]:
        """GET /.well-known/agent-card.json and return the parsed AgentCard."""
        url = f"{self._base_url}/.well-known/agent-card.json"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                try:
                    return resp.json()
                except Exception:
                    raise A2AError(-1, f"Agent card response is not valid JSON: {resp.text[:200]}")
        except A2AError:
            raise
        except httpx.HTTPStatusError as e:
            raise A2AError(e.response.status_code, f"HTTP {e.response.status_code} from {url}")
        except httpx.TimeoutException as e:
            raise A2AError(-1, f"Timeout discovering agent at {url}: {e}")
        except httpx.ConnectError as e:
            raise A2AError(-1, f"Connection error discovering agent at {url}: {e}")
        except Exception as e:
            raise A2AError(-1, f"Failed to discover agent at {url}: {e}")

    # -- JSON-RPC helpers ---------------------------------------------------

    async def _rpc(self, method: str, params: Dict[str, Any]) -> Any:
        """Send a JSON-RPC 2.0 request and return the result.

        Raises A2AError on JSON-RPC errors or transport failures.
        """
        req_id = self._next_id()
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    self._base_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise A2AError(
                e.response.status_code,
                f"HTTP {e.response.status_code}: {e.response.text[:300]}",
            )
        except httpx.TimeoutException as e:
            raise A2AError(-1, f"Timeout calling {method}: {e}")
        except httpx.ConnectError as e:
            raise A2AError(-1, f"Connection error calling {method}: {e}")
        except Exception as e:
            raise A2AError(-1, f"Transport error calling {method}: {e}")

        data = resp.json()
        if "error" in data:
            err = data["error"]
            raise A2AError(err.get("code", -1), err.get("message", "Unknown error"))
        return data.get("result")

    # -- SendMessage --------------------------------------------------------

    async def send_message(
        self,
        text: str,
        context_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a text message to the remote agent (JSON-RPC SendMessage).

        Returns the result — either a Task dict or a Message dict.
        """
        message: Dict[str, Any] = {
            "messageId": str(uuid.uuid4()),
            "role": "ROLE_USER",
            "parts": [{"text": text}],
        }
        if context_id:
            message["contextId"] = context_id
        if task_id:
            message["taskId"] = task_id

        return await self._rpc("SendMessage", {"message": message})

    # -- GetTask ------------------------------------------------------------

    async def get_task(self, task_id: str) -> Dict[str, Any]:
        """Retrieve a task by ID (JSON-RPC GetTask)."""
        return await self._rpc("GetTask", {"id": task_id})

    # -- CancelTask ---------------------------------------------------------

    async def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """Cancel a task by ID (JSON-RPC CancelTask)."""
        return await self._rpc("CancelTask", {"id": task_id})


# ---------------------------------------------------------------------------
# Tool handler functions
# ---------------------------------------------------------------------------

def a2a_discover_handler(args: dict, **kw) -> str:
    """Tool handler for a2a_discover."""
    url = args.get("url", "").strip()
    if not url:
        return json.dumps({"error": "Required parameter 'url' is missing. Provide the base URL of the A2A agent."})

    try:
        client = A2AClient(url)
        result = _run_a2a_async(client.discover())
        return json.dumps(result)
    except A2AError as e:
        return json.dumps({"error": f"A2A discovery failed ({e.code}): {e.message}"})
    except Exception as e:
        return json.dumps({"error": f"A2A discovery failed: {e}"})


def a2a_send_handler(args: dict, **kw) -> str:
    """Tool handler for a2a_send."""
    url = args.get("url", "").strip()
    message = args.get("message", "").strip()
    context_id = args.get("context_id", "").strip() or None
    a2a_task_id = args.get("a2a_task_id", "").strip() or None

    if not url:
        return json.dumps({"error": "Required parameter 'url' is missing."})
    if not message:
        return json.dumps({"error": "Required parameter 'message' is missing."})

    try:
        client = A2AClient(url)
        result = _run_a2a_async(client.send_message(message, context_id=context_id, task_id=a2a_task_id))
        return json.dumps(result)
    except A2AError as e:
        return json.dumps({"error": f"A2A SendMessage failed ({e.code}): {e.message}"})
    except Exception as e:
        return json.dumps({"error": f"A2A SendMessage failed: {e}"})


def a2a_get_task_handler(args: dict, **kw) -> str:
    """Tool handler for a2a_get_task."""
    url = args.get("url", "").strip()
    task_id = args.get("task_id", "").strip()

    if not url:
        return json.dumps({"error": "Required parameter 'url' is missing."})
    if not task_id:
        return json.dumps({"error": "Required parameter 'task_id' is missing."})

    try:
        client = A2AClient(url)
        result = _run_a2a_async(client.get_task(task_id))
        return json.dumps(result)
    except A2AError as e:
        return json.dumps({"error": f"A2A GetTask failed ({e.code}): {e.message}"})
    except Exception as e:
        return json.dumps({"error": f"A2A GetTask failed: {e}"})


def a2a_cancel_task_handler(args: dict, **kw) -> str:
    """Tool handler for a2a_cancel_task."""
    url = args.get("url", "").strip()
    task_id = args.get("task_id", "").strip()

    if not url:
        return json.dumps({"error": "Required parameter 'url' is missing."})
    if not task_id:
        return json.dumps({"error": "Required parameter 'task_id' is missing."})

    try:
        client = A2AClient(url)
        result = _run_a2a_async(client.cancel_task(task_id))
        return json.dumps(result)
    except A2AError as e:
        return json.dumps({"error": f"A2A CancelTask failed ({e.code}): {e.message}"})
    except Exception as e:
        return json.dumps({"error": f"A2A CancelTask failed: {e}"})


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

A2A_DISCOVER_SCHEMA = {
    "name": "a2a_discover",
    "description": (
        "Discover a remote A2A agent by fetching its agent card. "
        "Returns the agent's name, description, skills, capabilities, "
        "and supported interfaces. Use this before sending messages to "
        "learn what the agent can do."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Base URL of the A2A agent (e.g. https://agent.example.com). The agent card will be fetched from /.well-known/agent-card.json",
            },
        },
        "required": ["url"],
    },
}

A2A_SEND_SCHEMA = {
    "name": "a2a_send",
    "description": (
        "Send a text message to a remote A2A agent via JSON-RPC SendMessage. "
        "The response is either a Task (with id, status, artifacts) or a "
        "Message (with messageId, role, parts). Use context_id to continue "
        "a multi-turn conversation with the same agent."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Base URL of the A2A agent's JSON-RPC endpoint",
            },
            "message": {
                "type": "string",
                "description": "The text message to send to the agent",
            },
            "context_id": {
                "type": "string",
                "description": "Optional context ID for multi-turn conversations. Pass the contextId from a previous response to continue the conversation.",
            },
            "a2a_task_id": {
                "type": "string",
                "description": "Optional task ID to associate this message with an existing task.",
            },
        },
        "required": ["url", "message"],
    },
}

A2A_GET_TASK_SCHEMA = {
    "name": "a2a_get_task",
    "description": (
        "Retrieve the current state of a task from a remote A2A agent. "
        "Returns the task's id, status (state + message), artifacts, and history. "
        "Use the task_id from a previous a2a_send response."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Base URL of the A2A agent's JSON-RPC endpoint",
            },
            "task_id": {
                "type": "string",
                "description": "The task ID to retrieve (from a previous SendMessage response)",
            },
        },
        "required": ["url", "task_id"],
    },
}

A2A_CANCEL_TASK_SCHEMA = {
    "name": "a2a_cancel_task",
    "description": (
        "Cancel a running task on a remote A2A agent. "
        "The task will transition to CANCELED state if the agent supports cancellation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Base URL of the A2A agent's JSON-RPC endpoint",
            },
            "task_id": {
                "type": "string",
                "description": "The task ID to cancel",
            },
        },
        "required": ["url", "task_id"],
    },
}


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------

registry.register(
    name="a2a_discover",
    toolset="a2a",
    schema=A2A_DISCOVER_SCHEMA,
    handler=a2a_discover_handler,
    is_async=False,
    description="Discover a remote A2A agent's capabilities",
    emoji="🔍",
)

registry.register(
    name="a2a_send",
    toolset="a2a",
    schema=A2A_SEND_SCHEMA,
    handler=a2a_send_handler,
    is_async=False,
    description="Send a message to a remote A2A agent",
    emoji="📡",
)

registry.register(
    name="a2a_get_task",
    toolset="a2a",
    schema=A2A_GET_TASK_SCHEMA,
    handler=a2a_get_task_handler,
    is_async=False,
    description="Get task status from a remote A2A agent",
    emoji="📋",
)

registry.register(
    name="a2a_cancel_task",
    toolset="a2a",
    schema=A2A_CANCEL_TASK_SCHEMA,
    handler=a2a_cancel_task_handler,
    is_async=False,
    description="Cancel a task on a remote A2A agent",
    emoji="🛑",
)
