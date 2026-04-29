"""Tests for tools/arp_tool.py — ARP (Agent Registry Protocol) client tools.

TDD: tests written before implementation.
"""

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


# ---------------------------------------------------------------------------
# ARPClient unit tests
# ---------------------------------------------------------------------------

class TestARPClient:
    """Tests for the ARPClient class."""

    def test_list_agents(self):
        """GET /a2a/agents returns list of agent cards."""
        import asyncio
        from tools.arp_tool import ARPClient

        mock_resp = _mock_httpx_response(200, FAKE_AGENTS_LIST)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = ARPClient("http://localhost:9099")
            result = asyncio.run(client.list_agents())

        assert len(result) == 1
        assert result[0]["name"] == "CodeAgent"
        call_url = instance.get.call_args[0][0]
        assert "/a2a/agents" in call_url

    def test_get_agent_card(self):
        """GET /a2a/agents/{id}/.well-known/agent-card.json returns enriched card."""
        import asyncio
        from tools.arp_tool import ARPClient

        mock_resp = _mock_httpx_response(200, FAKE_AGENT_CARD)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = ARPClient("http://localhost:9099")
            result = asyncio.run(client.get_agent_card("abc-123"))

        assert result["metadata"]["arp"]["agent_id"] == "abc-123"
        call_url = instance.get.call_args[0][0]
        assert "/a2a/agents/abc-123/.well-known/agent-card.json" in call_url

    def test_send_message_to_agent(self):
        """POST /a2a/agents/{id}/message:send proxies message."""
        import asyncio
        from tools.arp_tool import ARPClient

        mock_resp = _mock_httpx_response(200, FAKE_MESSAGE_RESULT)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = ARPClient("http://localhost:9099")
            result = asyncio.run(client.send_message("abc-123", "Hello!"))

        assert result["role"] == "ROLE_AGENT"
        call_url = instance.post.call_args[0][0]
        assert "/a2a/agents/abc-123/message:send" in call_url
        body = instance.post.call_args[1]["json"]
        assert body["message"]["parts"][0]["text"] == "Hello!"
        assert body["message"]["role"] == "ROLE_USER"

    def test_send_message_with_context_id(self):
        """POST /a2a/agents/{id}/message:send passes context_id."""
        import asyncio
        from tools.arp_tool import ARPClient

        mock_resp = _mock_httpx_response(200, FAKE_MESSAGE_RESULT)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = ARPClient("http://localhost:9099")
            asyncio.run(client.send_message("abc-123", "Follow up", context_id="ctx-001"))

        body = instance.post.call_args[1]["json"]
        assert body["message"]["contextId"] == "ctx-001"

    def test_route_message_by_tags(self):
        """POST /a2a/route/message:send routes by skill tags."""
        import asyncio
        from tools.arp_tool import ARPClient

        mock_resp = _mock_httpx_response(200, FAKE_MESSAGE_RESULT)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = ARPClient("http://localhost:9099")
            result = asyncio.run(client.route_message("Write a function", tags=["coding"]))

        assert result["role"] == "ROLE_AGENT"
        call_url = instance.post.call_args[0][0]
        assert "/a2a/route/message:send" in call_url
        body = instance.post.call_args[1]["json"]
        assert body["routing"]["tags"] == ["coding"]

    def test_list_workspaces(self):
        """GET /api/workspaces returns workspace list."""
        import asyncio
        from tools.arp_tool import ARPClient

        mock_resp = _mock_httpx_response(200, FAKE_WORKSPACES)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = ARPClient("http://localhost:9099")
            result = asyncio.run(client.list_workspaces())

        assert len(result) == 1
        assert result[0]["name"] == "my-ws"

    def test_list_projects(self):
        """GET /api/projects returns project list."""
        import asyncio
        from tools.arp_tool import ARPClient

        mock_resp = _mock_httpx_response(200, FAKE_PROJECTS)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = ARPClient("http://localhost:9099")
            result = asyncio.run(client.list_projects())

        assert len(result) == 1
        assert result[0]["name"] == "my-project"

    def test_get_workspace(self):
        """GET /api/workspaces/{name} returns single workspace."""
        import asyncio
        from tools.arp_tool import ARPClient

        mock_resp = _mock_httpx_response(200, FAKE_WORKSPACE)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = ARPClient("http://localhost:9099")
            result = asyncio.run(client.get_workspace("my-ws"))

        assert result["name"] == "my-ws"
        call_url = instance.get.call_args[0][0]
        assert "/api/workspaces/my-ws" in call_url

    def test_bearer_token_sent(self):
        """When token is provided, requests include Authorization header."""
        import asyncio
        from tools.arp_tool import ARPClient

        mock_resp = _mock_httpx_response(200, FAKE_AGENTS_LIST)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = ARPClient("http://localhost:9099", token="my-secret-token")
            asyncio.run(client.list_agents())

        headers = instance.get.call_args[1].get("headers", {})
        assert headers.get("Authorization") == "Bearer my-secret-token"

    def test_no_token_no_auth_header(self):
        """When no token is provided, no Authorization header is sent."""
        import asyncio
        from tools.arp_tool import ARPClient

        mock_resp = _mock_httpx_response(200, FAKE_AGENTS_LIST)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = ARPClient("http://localhost:9099")
            asyncio.run(client.list_agents())

        headers = instance.get.call_args[1].get("headers", {})
        assert "Authorization" not in headers

    def test_404_raises_arp_error(self):
        """HTTP 404 raises ARPError."""
        import asyncio
        from tools.arp_tool import ARPClient, ARPError

        mock_resp = _mock_httpx_response(404)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = ARPClient("http://localhost:9099")
            with pytest.raises(ARPError) as exc_info:
                asyncio.run(client.get_agent_card("nonexistent"))
            assert exc_info.value.status_code == 404

    def test_connection_error_raises_arp_error(self):
        """Connection failure raises ARPError."""
        import asyncio
        import httpx
        from tools.arp_tool import ARPClient, ARPError

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.side_effect = httpx.ConnectError("connection refused")
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = ARPClient("http://localhost:9099")
            with pytest.raises(ARPError):
                asyncio.run(client.list_agents())

    def test_discover(self):
        """GET /a2a/discover returns discovery info."""
        import asyncio
        from tools.arp_tool import ARPClient

        discovery = {"agents": FAKE_AGENTS_LIST}
        mock_resp = _mock_httpx_response(200, discovery)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = ARPClient("http://localhost:9099")
            result = asyncio.run(client.discover())

        assert "agents" in result
        call_url = instance.get.call_args[0][0]
        assert "/a2a/discover" in call_url


# ---------------------------------------------------------------------------
# Tool handler tests
# ---------------------------------------------------------------------------

class TestARPToolHandlers:
    """Tests for the tool handler functions exposed to the agent."""

    def test_arp_list_agents_success(self):
        """arp_list_agents returns agent list."""
        from tools.arp_tool import arp_list_agents_handler

        with patch("tools.arp_tool._run_arp_async") as mock_run:
            mock_run.return_value = FAKE_AGENTS_LIST
            result = json.loads(arp_list_agents_handler(
                {"url": "http://localhost:9099"}, task_id="t1"
            ))

        assert len(result) == 1
        assert result[0]["name"] == "CodeAgent"

    def test_arp_list_agents_missing_url(self):
        """arp_list_agents without url returns error."""
        from tools.arp_tool import arp_list_agents_handler

        result = json.loads(arp_list_agents_handler({}, task_id="t1"))
        assert "error" in result

    def test_arp_send_message_success(self):
        """arp_send_message sends to a specific agent and returns result."""
        from tools.arp_tool import arp_send_message_handler

        with patch("tools.arp_tool._run_arp_async") as mock_run:
            mock_run.return_value = FAKE_MESSAGE_RESULT
            result = json.loads(arp_send_message_handler(
                {"url": "http://localhost:9099", "agent_id": "abc-123", "message": "Hello!"},
                task_id="t1",
            ))

        assert result["role"] == "ROLE_AGENT"

    def test_arp_send_message_missing_agent_id(self):
        """arp_send_message without agent_id returns error."""
        from tools.arp_tool import arp_send_message_handler

        result = json.loads(arp_send_message_handler(
            {"url": "http://localhost:9099", "message": "Hello!"},
            task_id="t1",
        ))
        assert "error" in result

    def test_arp_send_message_missing_message(self):
        """arp_send_message without message returns error."""
        from tools.arp_tool import arp_send_message_handler

        result = json.loads(arp_send_message_handler(
            {"url": "http://localhost:9099", "agent_id": "abc-123"},
            task_id="t1",
        ))
        assert "error" in result

    def test_arp_route_message_success(self):
        """arp_route_message routes by tags."""
        from tools.arp_tool import arp_route_message_handler

        with patch("tools.arp_tool._run_arp_async") as mock_run:
            mock_run.return_value = FAKE_MESSAGE_RESULT
            result = json.loads(arp_route_message_handler(
                {"url": "http://localhost:9099", "message": "Write code", "tags": ["coding"]},
                task_id="t1",
            ))

        assert result["role"] == "ROLE_AGENT"

    def test_arp_route_message_missing_message(self):
        """arp_route_message without message returns error."""
        from tools.arp_tool import arp_route_message_handler

        result = json.loads(arp_route_message_handler(
            {"url": "http://localhost:9099", "tags": ["coding"]},
            task_id="t1",
        ))
        assert "error" in result

    def test_arp_get_agent_card_success(self):
        """arp_get_agent_card returns enriched agent card."""
        from tools.arp_tool import arp_get_agent_card_handler

        with patch("tools.arp_tool._run_arp_async") as mock_run:
            mock_run.return_value = FAKE_AGENT_CARD
            result = json.loads(arp_get_agent_card_handler(
                {"url": "http://localhost:9099", "agent_id": "abc-123"},
                task_id="t1",
            ))

        assert result["metadata"]["arp"]["status"] == "ready"

    def test_arp_error_returns_json(self):
        """ARPError is caught and returned as JSON error."""
        from tools.arp_tool import arp_list_agents_handler, ARPError

        with patch("tools.arp_tool._run_arp_async") as mock_run:
            mock_run.side_effect = ARPError(404, "Not found")
            result = json.loads(arp_list_agents_handler(
                {"url": "http://localhost:9099"}, task_id="t1"
            ))

        assert "error" in result
        assert "404" in result["error"] or "Not found" in result["error"]

    def test_generic_exception_returns_json(self):
        """Unexpected exceptions are caught and returned as JSON error."""
        from tools.arp_tool import arp_list_agents_handler

        with patch("tools.arp_tool._run_arp_async") as mock_run:
            mock_run.side_effect = RuntimeError("boom")
            result = json.loads(arp_list_agents_handler(
                {"url": "http://localhost:9099"}, task_id="t1"
            ))

        assert "error" in result
        assert "boom" in result["error"]

    def test_arp_list_workspaces_success(self):
        """arp_list_workspaces returns workspace list."""
        from tools.arp_tool import arp_list_workspaces_handler

        with patch("tools.arp_tool._run_arp_async") as mock_run:
            mock_run.return_value = FAKE_WORKSPACES
            result = json.loads(arp_list_workspaces_handler(
                {"url": "http://localhost:9099"}, task_id="t1"
            ))

        assert len(result) == 1
        assert result[0]["name"] == "my-ws"


# ---------------------------------------------------------------------------
# Tool registration tests
# ---------------------------------------------------------------------------

class TestARPToolRegistration:
    """Tests that ARP tools are properly registered."""

    def test_tools_registered(self):
        """All ARP tools appear in the registry."""
        import importlib
        try:
            importlib.import_module("tools.arp_tool")
        except Exception:
            pytest.skip("arp_tool not yet implemented")

        from tools.registry import registry

        arp_tools = [
            "arp_list_agents", "arp_get_agent_card",
            "arp_send_message", "arp_route_message",
            "arp_list_workspaces",
        ]
        for tool_name in arp_tools:
            entry = registry._tools.get(tool_name)
            assert entry is not None, f"Tool {tool_name} not registered"
            assert entry.toolset == "arp"

    def test_tool_schemas_valid(self):
        """Each tool schema has name, description, and parameters."""
        import importlib
        try:
            importlib.import_module("tools.arp_tool")
        except Exception:
            pytest.skip("arp_tool not yet implemented")

        from tools.registry import registry

        for tool_name in ["arp_list_agents", "arp_get_agent_card", "arp_send_message", "arp_route_message", "arp_list_workspaces"]:
            entry = registry._tools.get(tool_name)
            assert entry is not None
            schema = entry.schema
            assert "description" in schema
            assert "parameters" in schema

    def test_no_env_requirements(self):
        """ARP tools have no env requirements (pure HTTP client)."""
        import importlib
        try:
            importlib.import_module("tools.arp_tool")
        except Exception:
            pytest.skip("arp_tool not yet implemented")

        from tools.registry import registry

        for tool_name in ["arp_list_agents", "arp_get_agent_card", "arp_send_message", "arp_route_message", "arp_list_workspaces"]:
            entry = registry._tools.get(tool_name)
            assert entry is not None
            assert entry.requires_env == []
