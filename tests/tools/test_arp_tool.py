"""Tests for tools/arp_tool.py — ARP (Agent Registry Protocol) client tools.

TDD: tests written before implementation.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_AGENT_CARD = {
    "name": "CodeAgent",
    "version": "1.0.0",
    "supportedInterfaces": [
        {"url": "http://localhost:9099/a2a/agents/abc-123", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}
    ],
    "skills": [{"id": "code", "name": "Code", "description": "Writes code", "tags": ["coding"]}],
    "metadata": {
        "arp": {
            "agent_id": "abc-123",
            "workspace": "my-ws",
            "project": "my-project",
            "template": "code-agent",
            "status": "ready",
            "direct_url": "http://localhost:12345",
            "started_at": "2025-01-01T00:00:00Z",
        }
    },
}

FAKE_AGENTS_LIST = [FAKE_AGENT_CARD]

FAKE_WORKSPACE = {
    "name": "my-ws",
    "project": "my-project",
    "active": True,
    "dir": "/tmp/workspaces/my-ws",
}

FAKE_WORKSPACES = [FAKE_WORKSPACE]

FAKE_PROJECT = {"name": "my-project", "repo": "/tmp/repos/my-project", "branch": "main"}

FAKE_PROJECTS = [FAKE_PROJECT]

FAKE_MESSAGE_RESULT = {
    "messageId": "msg-001",
    "role": "ROLE_AGENT",
    "parts": [{"text": "Hello, I'm the code agent."}],
    "contextId": "ctx-001",
}

FAKE_AGENT_INSTANCE = {
    "id": "agent-001",
    "template": "code-agent",
    "workspace": "my-ws",
    "status": "ready",
    "port": 12345,
    "direct_url": "http://localhost:12345",
    "proxy_url": "http://localhost:9099/a2a/agents/agent-001",
}

FAKE_TASK = {
    "id": "task-001",
    "contextId": "ctx-001",
    "status": {"state": "TASK_STATE_COMPLETED"},
}


def _mock_httpx_response(status_code=200, json_data=None, headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = headers or {"content-type": "application/json"}
    resp.text = json.dumps(json_data or {})
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import httpx
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


def _jsonrpc_ok(result, req_id=1):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _jsonrpc_error(code, message, req_id=1):
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


# ---------------------------------------------------------------------------
# ARPClient — HTTP proxy endpoint tests (existing)
# ---------------------------------------------------------------------------

class TestARPClientProxy:
    """Tests for the ARPClient HTTP proxy methods."""

    def test_list_agents(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, FAKE_AGENTS_LIST)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            result = asyncio.run(ARPClient("http://localhost:9099").list_agents())
        assert len(result) == 1
        assert result[0]["name"] == "CodeAgent"

    def test_get_agent_card(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, FAKE_AGENT_CARD)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            result = asyncio.run(ARPClient("http://localhost:9099").get_agent_card("abc-123"))
        assert result["metadata"]["arp"]["agent_id"] == "abc-123"
        assert "/a2a/agents/abc-123/.well-known/agent-card.json" in instance.get.call_args[0][0]

    def test_send_message_to_agent(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, FAKE_MESSAGE_RESULT)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            result = asyncio.run(ARPClient("http://localhost:9099").send_message("abc-123", "Hello!"))
        assert result["role"] == "ROLE_AGENT"
        body = instance.post.call_args[1]["json"]
        assert body["message"]["parts"][0]["text"] == "Hello!"

    def test_send_message_with_context_id(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, FAKE_MESSAGE_RESULT)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            asyncio.run(ARPClient("http://localhost:9099").send_message("abc-123", "Follow up", context_id="ctx-001"))
        assert instance.post.call_args[1]["json"]["message"]["contextId"] == "ctx-001"

    def test_route_message_by_tags(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, FAKE_MESSAGE_RESULT)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            result = asyncio.run(ARPClient("http://localhost:9099").route_message("Write a function", tags=["coding"]))
        assert result["role"] == "ROLE_AGENT"
        assert instance.post.call_args[1]["json"]["routing"]["tags"] == ["coding"]

    def test_list_workspaces(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, FAKE_WORKSPACES)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            result = asyncio.run(ARPClient("http://localhost:9099").list_workspaces())
        assert result[0]["name"] == "my-ws"

    def test_list_projects(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, FAKE_PROJECTS)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            result = asyncio.run(ARPClient("http://localhost:9099").list_projects())
        assert result[0]["name"] == "my-project"

    def test_get_workspace(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, FAKE_WORKSPACE)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            result = asyncio.run(ARPClient("http://localhost:9099").get_workspace("my-ws"))
        assert result["name"] == "my-ws"

    def test_bearer_token_sent(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, FAKE_AGENTS_LIST)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            asyncio.run(ARPClient("http://localhost:9099", token="my-secret-token").list_agents())
        assert instance.get.call_args[1]["headers"]["Authorization"] == "Bearer my-secret-token"

    def test_no_token_no_auth_header(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, FAKE_AGENTS_LIST)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            asyncio.run(ARPClient("http://localhost:9099").list_agents())
        assert "Authorization" not in instance.get.call_args[1].get("headers", {})

    def test_404_raises_arp_error(self):
        from tools.arp_tool import ARPClient, ARPError
        mock_resp = _mock_httpx_response(404)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            with pytest.raises(ARPError) as exc_info:
                asyncio.run(ARPClient("http://localhost:9099").get_agent_card("nonexistent"))
            assert exc_info.value.status_code == 404

    def test_connection_error_raises_arp_error(self):
        import httpx
        from tools.arp_tool import ARPClient, ARPError
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.side_effect = httpx.ConnectError("connection refused")
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            with pytest.raises(ARPError):
                asyncio.run(ARPClient("http://localhost:9099").list_agents())

    def test_discover(self):
        from tools.arp_tool import ARPClient
        discovery = {"agents": FAKE_AGENTS_LIST}
        mock_resp = _mock_httpx_response(200, discovery)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            result = asyncio.run(ARPClient("http://localhost:9099").discover())
        assert "agents" in result


# ---------------------------------------------------------------------------
# ARPClient — MCP tool_call tests (lifecycle operations)
# ---------------------------------------------------------------------------

class TestARPClientMCP:
    """Tests for ARPClient MCP tool_call methods."""

    def test_mcp_tool_call_sends_jsonrpc(self):
        """mcp_tool_call sends a JSON-RPC 2.0 tools/call request."""
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, _jsonrpc_ok(FAKE_PROJECT))
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            result = asyncio.run(ARPClient("http://localhost:9099").mcp_tool_call(
                "project/list", {}
            ))
        body = instance.post.call_args[1]["json"]
        assert body["jsonrpc"] == "2.0"
        assert body["method"] == "tools/call"
        assert body["params"]["name"] == "project/list"
        assert body["params"]["arguments"] == {}

    def test_mcp_tool_call_jsonrpc_error(self):
        """mcp_tool_call with JSON-RPC error raises ARPError."""
        from tools.arp_tool import ARPClient, ARPError
        mock_resp = _mock_httpx_response(200, _jsonrpc_error(-32602, "Invalid params"))
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            with pytest.raises(ARPError) as exc_info:
                asyncio.run(ARPClient("http://localhost:9099").mcp_tool_call(
                    "project/register", {"name": "x"}
                ))
        assert "Invalid params" in str(exc_info.value)

    def test_mcp_tool_call_uses_mcp_path(self):
        """mcp_tool_call POSTs to /mcp endpoint."""
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, _jsonrpc_ok([]))
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            asyncio.run(ARPClient("http://localhost:9099").mcp_tool_call("project/list", {}))
        url = instance.post.call_args[0][0]
        assert url.endswith("/mcp")

    def test_project_register(self):
        """project_register calls mcp_tool_call with correct params."""
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, _jsonrpc_ok(FAKE_PROJECT))
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            result = asyncio.run(ARPClient("http://localhost:9099").project_register(
                "my-project", "/tmp/repos/my-project"
            ))
        assert result["name"] == "my-project"
        body = instance.post.call_args[1]["json"]
        assert body["params"]["name"] == "project/register"
        assert body["params"]["arguments"]["name"] == "my-project"
        assert body["params"]["arguments"]["repo"] == "/tmp/repos/my-project"

    def test_project_register_with_agents(self):
        """project_register passes agent templates."""
        from tools.arp_tool import ARPClient
        agents = [{"name": "coder", "command": "echo serve", "port_env": "A2A_PORT"}]
        mock_resp = _mock_httpx_response(200, _jsonrpc_ok({**FAKE_PROJECT, "agents": agents}))
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            asyncio.run(ARPClient("http://localhost:9099").project_register(
                "my-project", "/tmp/repo", agents=agents
            ))
        body = instance.post.call_args[1]["json"]
        assert body["params"]["arguments"]["agents"] == agents

    def test_project_unregister(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, _jsonrpc_ok({"success": True}))
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            asyncio.run(ARPClient("http://localhost:9099").project_unregister("my-project"))
        body = instance.post.call_args[1]["json"]
        assert body["params"]["name"] == "project/unregister"
        assert body["params"]["arguments"]["name"] == "my-project"

    def test_project_list(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, _jsonrpc_ok(FAKE_PROJECTS))
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            result = asyncio.run(ARPClient("http://localhost:9099").project_list())
        assert result == FAKE_PROJECTS

    def test_workspace_create(self):
        from tools.arp_tool import ARPClient
        ws = {**FAKE_WORKSPACE, "status": "active", "agents": []}
        mock_resp = _mock_httpx_response(200, _jsonrpc_ok(ws))
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            result = asyncio.run(ARPClient("http://localhost:9099").workspace_create(
                "my-ws", "my-project"
            ))
        body = instance.post.call_args[1]["json"]
        assert body["params"]["name"] == "workspace/create"
        assert body["params"]["arguments"]["name"] == "my-ws"
        assert body["params"]["arguments"]["project"] == "my-project"

    def test_workspace_create_with_auto_agents(self):
        from tools.arp_tool import ARPClient
        ws = {**FAKE_WORKSPACE, "status": "active", "agents": [FAKE_AGENT_INSTANCE]}
        mock_resp = _mock_httpx_response(200, _jsonrpc_ok(ws))
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            asyncio.run(ARPClient("http://localhost:9099").workspace_create(
                "my-ws", "my-project", auto_agents=["code-agent"]
            ))
        body = instance.post.call_args[1]["json"]
        assert body["params"]["arguments"]["auto_agents"] == ["code-agent"]

    def test_workspace_destroy(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, _jsonrpc_ok({"success": True}))
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            asyncio.run(ARPClient("http://localhost:9099").workspace_destroy("my-ws"))
        body = instance.post.call_args[1]["json"]
        assert body["params"]["name"] == "workspace/destroy"

    def test_agent_spawn(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, _jsonrpc_ok(FAKE_AGENT_INSTANCE))
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            result = asyncio.run(ARPClient("http://localhost:9099").agent_spawn(
                "my-ws", "code-agent"
            ))
        assert result["id"] == "agent-001"
        body = instance.post.call_args[1]["json"]
        assert body["params"]["name"] == "agent/spawn"
        assert body["params"]["arguments"]["workspace"] == "my-ws"
        assert body["params"]["arguments"]["template"] == "code-agent"

    def test_agent_spawn_with_prompt(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, _jsonrpc_ok(FAKE_AGENT_INSTANCE))
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            asyncio.run(ARPClient("http://localhost:9099").agent_spawn(
                "my-ws", "code-agent", prompt="Initialize yourself"
            ))
        body = instance.post.call_args[1]["json"]
        assert body["params"]["arguments"]["prompt"] == "Initialize yourself"

    def test_agent_stop(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, _jsonrpc_ok({"status": "stopped"}))
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            asyncio.run(ARPClient("http://localhost:9099").agent_stop("agent-001"))
        body = instance.post.call_args[1]["json"]
        assert body["params"]["name"] == "agent/stop"
        assert body["params"]["arguments"]["agent_id"] == "agent-001"

    def test_agent_restart(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, _jsonrpc_ok(FAKE_AGENT_INSTANCE))
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            result = asyncio.run(ARPClient("http://localhost:9099").agent_restart("agent-001"))
        assert result["id"] == "agent-001"
        body = instance.post.call_args[1]["json"]
        assert body["params"]["name"] == "agent/restart"

    def test_agent_status(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, _jsonrpc_ok(FAKE_AGENT_INSTANCE))
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            result = asyncio.run(ARPClient("http://localhost:9099").agent_status("agent-001"))
        assert result["status"] == "ready"
        body = instance.post.call_args[1]["json"]
        assert body["params"]["name"] == "agent/status"

    def test_agent_message(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, _jsonrpc_ok(FAKE_MESSAGE_RESULT))
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            result = asyncio.run(ARPClient("http://localhost:9099").agent_message(
                "agent-001", "Hello agent"
            ))
        assert result["role"] == "ROLE_AGENT"
        body = instance.post.call_args[1]["json"]
        assert body["params"]["name"] == "agent/message"
        assert body["params"]["arguments"]["message"] == "Hello agent"

    def test_agent_task(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, _jsonrpc_ok(FAKE_TASK))
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            result = asyncio.run(ARPClient("http://localhost:9099").agent_task(
                "agent-001", "Do a long task"
            ))
        assert result["id"] == "task-001"
        body = instance.post.call_args[1]["json"]
        assert body["params"]["name"] == "agent/task"

    def test_agent_task_status(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, _jsonrpc_ok(FAKE_TASK))
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            result = asyncio.run(ARPClient("http://localhost:9099").agent_task_status(
                "agent-001", "task-001"
            ))
        body = instance.post.call_args[1]["json"]
        assert body["params"]["name"] == "agent/task_status"
        assert body["params"]["arguments"]["task_id"] == "task-001"

    def test_workspace_get(self):
        from tools.arp_tool import ARPClient
        ws = {**FAKE_WORKSPACE, "status": "active", "agents": [], "created_at": "2025-01-01"}
        mock_resp = _mock_httpx_response(200, _jsonrpc_ok(ws))
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            result = asyncio.run(ARPClient("http://localhost:9099").workspace_get("my-ws"))
        body = instance.post.call_args[1]["json"]
        assert body["params"]["name"] == "workspace/get"
        assert result["name"] == "my-ws"

    def test_request_ids_increment(self):
        from tools.arp_tool import ARPClient
        mock_resp = _mock_httpx_response(200, _jsonrpc_ok([]))
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance
            c = ARPClient("http://localhost:9099")
            asyncio.run(c.mcp_tool_call("project/list", {}))
            asyncio.run(c.mcp_tool_call("project/list", {}))
        id1 = instance.post.call_args_list[0][1]["json"]["id"]
        id2 = instance.post.call_args_list[1][1]["json"]["id"]
        assert id2 > id1


# ---------------------------------------------------------------------------
# Tool handler tests — proxy tools (existing)
# ---------------------------------------------------------------------------

class TestARPProxyHandlers:
    """Tests for the HTTP proxy tool handlers."""

    def test_arp_list_agents_success(self):
        from tools.arp_tool import arp_list_agents_handler
        with patch("tools.arp_tool._run_arp_async") as m:
            m.return_value = FAKE_AGENTS_LIST
            result = json.loads(arp_list_agents_handler({"url": "http://localhost:9099"}, task_id="t1"))
        assert result[0]["name"] == "CodeAgent"

    def test_arp_list_agents_missing_url(self):
        from tools.arp_tool import arp_list_agents_handler
        result = json.loads(arp_list_agents_handler({}, task_id="t1"))
        assert "error" in result

    def test_arp_send_message_success(self):
        from tools.arp_tool import arp_send_message_handler
        with patch("tools.arp_tool._run_arp_async") as m:
            m.return_value = FAKE_MESSAGE_RESULT
            result = json.loads(arp_send_message_handler(
                {"url": "http://localhost:9099", "agent_id": "abc-123", "message": "Hi"}, task_id="t1"
            ))
        assert result["role"] == "ROLE_AGENT"

    def test_arp_send_message_missing_params(self):
        from tools.arp_tool import arp_send_message_handler
        for args in [
            {"url": "http://localhost:9099", "message": "Hi"},
            {"url": "http://localhost:9099", "agent_id": "x"},
        ]:
            result = json.loads(arp_send_message_handler(args, task_id="t1"))
            assert "error" in result

    def test_arp_route_message_success(self):
        from tools.arp_tool import arp_route_message_handler
        with patch("tools.arp_tool._run_arp_async") as m:
            m.return_value = FAKE_MESSAGE_RESULT
            result = json.loads(arp_route_message_handler(
                {"url": "http://localhost:9099", "message": "Code", "tags": ["coding"]}, task_id="t1"
            ))
        assert result["role"] == "ROLE_AGENT"

    def test_arp_get_agent_card_success(self):
        from tools.arp_tool import arp_get_agent_card_handler
        with patch("tools.arp_tool._run_arp_async") as m:
            m.return_value = FAKE_AGENT_CARD
            result = json.loads(arp_get_agent_card_handler(
                {"url": "http://localhost:9099", "agent_id": "abc-123"}, task_id="t1"
            ))
        assert result["metadata"]["arp"]["status"] == "ready"

    def test_arp_list_workspaces_success(self):
        from tools.arp_tool import arp_list_workspaces_handler
        with patch("tools.arp_tool._run_arp_async") as m:
            m.return_value = FAKE_WORKSPACES
            result = json.loads(arp_list_workspaces_handler({"url": "http://localhost:9099"}, task_id="t1"))
        assert result[0]["name"] == "my-ws"

    def test_arp_error_returns_json(self):
        from tools.arp_tool import arp_list_agents_handler, ARPError
        with patch("tools.arp_tool._run_arp_async") as m:
            m.side_effect = ARPError(404, "Not found")
            result = json.loads(arp_list_agents_handler({"url": "http://localhost:9099"}, task_id="t1"))
        assert "error" in result

    def test_generic_exception_returns_json(self):
        from tools.arp_tool import arp_list_agents_handler
        with patch("tools.arp_tool._run_arp_async") as m:
            m.side_effect = RuntimeError("boom")
            result = json.loads(arp_list_agents_handler({"url": "http://localhost:9099"}, task_id="t1"))
        assert "boom" in result["error"]


# ---------------------------------------------------------------------------
# Tool handler tests — MCP lifecycle tools (new)
# ---------------------------------------------------------------------------

class TestARPLifecycleHandlers:
    """Tests for the MCP lifecycle tool handlers."""

    def test_arp_manage_project_register(self):
        from tools.arp_tool import arp_manage_handler
        with patch("tools.arp_tool._run_arp_async") as m:
            m.return_value = FAKE_PROJECT
            result = json.loads(arp_manage_handler({
                "url": "http://localhost:9099",
                "tool": "project/register",
                "arguments": {"name": "proj", "repo": "/tmp/r"},
            }, task_id="t1"))
        assert result["name"] == "my-project"

    def test_arp_manage_agent_spawn(self):
        from tools.arp_tool import arp_manage_handler
        with patch("tools.arp_tool._run_arp_async") as m:
            m.return_value = FAKE_AGENT_INSTANCE
            result = json.loads(arp_manage_handler({
                "url": "http://localhost:9099",
                "tool": "agent/spawn",
                "arguments": {"workspace": "ws", "template": "coder"},
            }, task_id="t1"))
        assert result["id"] == "agent-001"

    def test_arp_manage_missing_tool(self):
        from tools.arp_tool import arp_manage_handler
        result = json.loads(arp_manage_handler({
            "url": "http://localhost:9099",
            "arguments": {"name": "x"},
        }, task_id="t1"))
        assert "error" in result

    def test_arp_manage_missing_url(self):
        from tools.arp_tool import arp_manage_handler
        result = json.loads(arp_manage_handler({
            "tool": "project/list",
        }, task_id="t1"))
        assert "error" in result

    def test_arp_manage_invalid_tool_rejected(self):
        """Only allowed ARP MCP tools can be called."""
        from tools.arp_tool import arp_manage_handler
        result = json.loads(arp_manage_handler({
            "url": "http://localhost:9099",
            "tool": "dangerous/hack",
            "arguments": {},
        }, task_id="t1"))
        assert "error" in result

    def test_arp_manage_defaults_to_empty_arguments(self):
        from tools.arp_tool import arp_manage_handler
        with patch("tools.arp_tool._run_arp_async") as m:
            m.return_value = FAKE_PROJECTS
            result = json.loads(arp_manage_handler({
                "url": "http://localhost:9099",
                "tool": "project/list",
            }, task_id="t1"))
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tool registration tests
# ---------------------------------------------------------------------------

class TestARPToolRegistration:
    """Tests that all ARP tools are properly registered."""

    def test_tools_registered(self):
        import importlib
        try:
            importlib.import_module("tools.arp_tool")
        except Exception:
            pytest.skip("arp_tool not yet implemented")

        from tools.registry import registry
        arp_tools = [
            "arp_list_agents", "arp_get_agent_card",
            "arp_send_message", "arp_route_message",
            "arp_list_workspaces", "arp_manage",
        ]
        for tool_name in arp_tools:
            entry = registry._tools.get(tool_name)
            assert entry is not None, f"Tool {tool_name} not registered"
            assert entry.toolset == "arp"

    def test_tool_schemas_valid(self):
        import importlib
        try:
            importlib.import_module("tools.arp_tool")
        except Exception:
            pytest.skip("arp_tool not yet implemented")

        from tools.registry import registry
        for tool_name in [
            "arp_list_agents", "arp_get_agent_card", "arp_send_message",
            "arp_route_message", "arp_list_workspaces", "arp_manage",
        ]:
            entry = registry._tools.get(tool_name)
            assert entry is not None
            assert "description" in entry.schema
            assert "parameters" in entry.schema

    def test_no_env_requirements(self):
        import importlib
        try:
            importlib.import_module("tools.arp_tool")
        except Exception:
            pytest.skip("arp_tool not yet implemented")

        from tools.registry import registry
        for tool_name in [
            "arp_list_agents", "arp_get_agent_card", "arp_send_message",
            "arp_route_message", "arp_list_workspaces", "arp_manage",
        ]:
            entry = registry._tools.get(tool_name)
            assert entry is not None
            assert entry.requires_env == []
