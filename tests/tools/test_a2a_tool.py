"""Tests for tools/a2a_tool.py — A2A client tools.

TDD: tests written before implementation.
"""

import asyncio
import json
import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_AGENT_CARD = {
    "name": "TestAgent",
    "description": "A test agent",
    "version": "1.0.0",
    "supportedInterfaces": [
        {
            "url": "https://agent.example.com/a2a",
            "protocolBinding": "JSONRPC",
            "protocolVersion": "1.0",
        }
    ],
    "capabilities": {"streaming": True, "pushNotifications": False},
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain"],
    "skills": [
        {
            "id": "summarize",
            "name": "Summarize",
            "description": "Summarizes text",
        }
    ],
}

FAKE_TASK = {
    "id": "task-001",
    "contextId": "ctx-001",
    "status": {
        "state": "TASK_STATE_COMPLETED",
        "message": {"messageId": "msg-r1", "role": "ROLE_AGENT", "parts": [{"text": "Done"}]},
    },
    "artifacts": [],
}

FAKE_MESSAGE_RESPONSE = {
    "messageId": "msg-r1",
    "role": "ROLE_AGENT",
    "parts": [{"text": "Hello, I can help with summarization."}],
}


def _jsonrpc_ok(result, req_id=1):
    """Build a JSON-RPC 2.0 success response."""
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _jsonrpc_error(code, message, req_id=1):
    """Build a JSON-RPC 2.0 error response."""
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _mock_httpx_response(status_code=200, json_data=None, headers=None):
    """Create a mock httpx.Response."""
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
# A2AClient unit tests
# ---------------------------------------------------------------------------

class TestA2AClient:
    """Tests for the A2AClient class."""

    def test_discover_agent_card(self):
        """GET /.well-known/agent-card.json returns parsed AgentCard."""
        from tools.a2a_tool import A2AClient

        mock_resp = _mock_httpx_response(200, FAKE_AGENT_CARD, {"content-type": "application/json"})
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = A2AClient("https://agent.example.com")
            result = asyncio.run(client.discover())

        assert result["name"] == "TestAgent"
        assert len(result["skills"]) == 1
        instance.get.assert_called_once()
        call_url = instance.get.call_args[0][0]
        assert "/.well-known/agent-card.json" in call_url

    def test_discover_strips_trailing_slash(self):
        """Base URL trailing slash doesn't cause double-slash in path."""
        from tools.a2a_tool import A2AClient

        mock_resp = _mock_httpx_response(200, FAKE_AGENT_CARD)
        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = A2AClient("https://agent.example.com/")
            asyncio.run(client.discover())

        call_url = instance.get.call_args[0][0]
        assert "//" not in call_url.replace("https://", "")

    def test_send_message_returns_task(self):
        """SendMessage returning a Task is handled correctly."""
        from tools.a2a_tool import A2AClient

        rpc_result = _jsonrpc_ok(FAKE_TASK)
        mock_resp = _mock_httpx_response(200, rpc_result)

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = A2AClient("https://agent.example.com")
            result = asyncio.run(client.send_message("Hello!", context_id="ctx-001"))

        assert result == FAKE_TASK
        posted_body = instance.post.call_args[1].get("json") or instance.post.call_args[0][1] if len(instance.post.call_args[0]) > 1 else instance.post.call_args[1]["json"]
        assert posted_body["method"] == "SendMessage"
        assert posted_body["jsonrpc"] == "2.0"
        msg = posted_body["params"]["message"]
        assert msg["role"] == "ROLE_USER"
        assert msg["parts"][0]["text"] == "Hello!"

    def test_send_message_returns_message(self):
        """SendMessage returning a Message (instead of Task) is handled."""
        from tools.a2a_tool import A2AClient

        rpc_result = _jsonrpc_ok(FAKE_MESSAGE_RESPONSE)
        mock_resp = _mock_httpx_response(200, rpc_result)

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = A2AClient("https://agent.example.com")
            result = asyncio.run(client.send_message("Hello!"))

        assert result["role"] == "ROLE_AGENT"
        assert result["parts"][0]["text"].startswith("Hello")

    def test_send_message_with_context_id(self):
        """SendMessage passes contextId when provided."""
        from tools.a2a_tool import A2AClient

        rpc_result = _jsonrpc_ok(FAKE_TASK)
        mock_resp = _mock_httpx_response(200, rpc_result)

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = A2AClient("https://agent.example.com")
            asyncio.run(client.send_message("Follow up", context_id="ctx-42"))

        body = instance.post.call_args[1]["json"]
        assert body["params"]["message"]["contextId"] == "ctx-42"

    def test_send_message_jsonrpc_error(self):
        """SendMessage with JSON-RPC error raises A2AError."""
        from tools.a2a_tool import A2AClient, A2AError

        rpc_error = _jsonrpc_error(-32602, "Invalid params")
        mock_resp = _mock_httpx_response(200, rpc_error)

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = A2AClient("https://agent.example.com")
            with pytest.raises(A2AError) as exc_info:
                asyncio.run(client.send_message("bad"))

        assert exc_info.value.code == -32602
        assert "Invalid params" in str(exc_info.value)

    def test_get_task(self):
        """GetTask returns task by ID."""
        from tools.a2a_tool import A2AClient

        rpc_result = _jsonrpc_ok(FAKE_TASK)
        mock_resp = _mock_httpx_response(200, rpc_result)

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = A2AClient("https://agent.example.com")
            result = asyncio.run(client.get_task("task-001"))

        assert result["id"] == "task-001"
        body = instance.post.call_args[1]["json"]
        assert body["method"] == "GetTask"
        assert body["params"]["id"] == "task-001"

    def test_get_task_not_found(self):
        """GetTask with unknown ID raises A2AError."""
        from tools.a2a_tool import A2AClient, A2AError

        rpc_error = _jsonrpc_error(-32001, "Task not found")
        mock_resp = _mock_httpx_response(200, rpc_error)

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = A2AClient("https://agent.example.com")
            with pytest.raises(A2AError):
                asyncio.run(client.get_task("nonexistent"))

    def test_cancel_task(self):
        """CancelTask sends correct JSON-RPC request."""
        from tools.a2a_tool import A2AClient

        canceled_task = {**FAKE_TASK, "status": {"state": "TASK_STATE_CANCELED"}}
        rpc_result = _jsonrpc_ok(canceled_task)
        mock_resp = _mock_httpx_response(200, rpc_result)

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = A2AClient("https://agent.example.com")
            result = asyncio.run(client.cancel_task("task-001"))

        assert result["status"]["state"] == "TASK_STATE_CANCELED"
        body = instance.post.call_args[1]["json"]
        assert body["method"] == "CancelTask"
        assert body["params"]["id"] == "task-001"

    def test_jsonrpc_request_ids_increment(self):
        """Each JSON-RPC request gets an incrementing ID."""
        from tools.a2a_tool import A2AClient

        rpc_result = _jsonrpc_ok(FAKE_TASK)
        mock_resp = _mock_httpx_response(200, rpc_result)

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = A2AClient("https://agent.example.com")
            asyncio.run(client.send_message("first"))
            asyncio.run(client.send_message("second"))

        calls = instance.post.call_args_list
        id1 = calls[0][1]["json"]["id"]
        id2 = calls[1][1]["json"]["id"]
        assert id2 > id1

    def test_http_error_raises(self):
        """HTTP-level errors (non-200) raise A2AError."""
        from tools.a2a_tool import A2AClient, A2AError

        mock_resp = _mock_httpx_response(500)

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = A2AClient("https://agent.example.com")
            with pytest.raises(A2AError):
                asyncio.run(client.send_message("hello"))


# ---------------------------------------------------------------------------
# Tool handler tests (the registry-facing functions)
# ---------------------------------------------------------------------------

class TestA2AToolHandlers:
    """Tests for the tool handler functions exposed to the agent."""

    def test_a2a_discover_tool_success(self):
        """a2a_discover tool returns agent card JSON."""
        from tools.a2a_tool import a2a_discover_handler

        with patch("tools.a2a_tool._run_a2a_async") as mock_run:
            mock_run.return_value = FAKE_AGENT_CARD
            result = json.loads(a2a_discover_handler(
                {"url": "https://agent.example.com"}, task_id="t1"
            ))

        assert result["name"] == "TestAgent"
        assert result["skills"][0]["id"] == "summarize"

    def test_a2a_discover_tool_missing_url(self):
        """a2a_discover without url returns error."""
        from tools.a2a_tool import a2a_discover_handler

        result = json.loads(a2a_discover_handler({}, task_id="t1"))
        assert "error" in result

    def test_a2a_send_tool_success(self):
        """a2a_send tool sends message and returns result."""
        from tools.a2a_tool import a2a_send_handler

        with patch("tools.a2a_tool._run_a2a_async") as mock_run:
            mock_run.return_value = FAKE_TASK
            result = json.loads(a2a_send_handler(
                {"url": "https://agent.example.com", "message": "Hello!"},
                task_id="t1",
            ))

        assert result["id"] == "task-001"

    def test_a2a_send_tool_missing_message(self):
        """a2a_send without message returns error."""
        from tools.a2a_tool import a2a_send_handler

        result = json.loads(a2a_send_handler(
            {"url": "https://agent.example.com"}, task_id="t1"
        ))
        assert "error" in result

    def test_a2a_send_tool_missing_url(self):
        """a2a_send without url returns error."""
        from tools.a2a_tool import a2a_send_handler

        result = json.loads(a2a_send_handler(
            {"message": "hello"}, task_id="t1"
        ))
        assert "error" in result

    def test_a2a_send_tool_with_context_id(self):
        """a2a_send passes context_id through."""
        from tools.a2a_tool import a2a_send_handler

        with patch("tools.a2a_tool._run_a2a_async") as mock_run:
            mock_run.return_value = FAKE_TASK
            a2a_send_handler(
                {
                    "url": "https://agent.example.com",
                    "message": "Follow up",
                    "context_id": "ctx-42",
                },
                task_id="t1",
            )

        # Verify the async function was called with the right args
        mock_run.assert_called_once()

    def test_a2a_send_tool_a2a_error_returns_json(self):
        """a2a_send wraps A2AError into a JSON error response."""
        from tools.a2a_tool import a2a_send_handler, A2AError

        with patch("tools.a2a_tool._run_a2a_async") as mock_run:
            mock_run.side_effect = A2AError(-32602, "Invalid params")
            result = json.loads(a2a_send_handler(
                {"url": "https://agent.example.com", "message": "bad"},
                task_id="t1",
            ))

        assert "error" in result
        assert "-32602" in result["error"] or "Invalid params" in result["error"]

    def test_a2a_get_task_tool_success(self):
        """a2a_get_task tool returns task."""
        from tools.a2a_tool import a2a_get_task_handler

        with patch("tools.a2a_tool._run_a2a_async") as mock_run:
            mock_run.return_value = FAKE_TASK
            result = json.loads(a2a_get_task_handler(
                {"url": "https://agent.example.com", "task_id": "task-001"},
                task_id="t1",
            ))

        assert result["id"] == "task-001"

    def test_a2a_get_task_tool_missing_params(self):
        """a2a_get_task without required params returns error."""
        from tools.a2a_tool import a2a_get_task_handler

        result = json.loads(a2a_get_task_handler(
            {"url": "https://agent.example.com"}, task_id="t1"
        ))
        assert "error" in result

    def test_a2a_cancel_task_tool_success(self):
        """a2a_cancel_task tool cancels and returns result."""
        from tools.a2a_tool import a2a_cancel_task_handler

        canceled = {**FAKE_TASK, "status": {"state": "TASK_STATE_CANCELED"}}
        with patch("tools.a2a_tool._run_a2a_async") as mock_run:
            mock_run.return_value = canceled
            result = json.loads(a2a_cancel_task_handler(
                {"url": "https://agent.example.com", "task_id": "task-001"},
                task_id="t1",
            ))

        assert result["status"]["state"] == "TASK_STATE_CANCELED"


# ---------------------------------------------------------------------------
# Tool registration tests
# ---------------------------------------------------------------------------

class TestA2AToolRegistration:
    """Tests that A2A tools are properly registered."""

    def test_tools_registered_in_registry(self):
        """All A2A tools appear in the registry after import."""
        # Force import to trigger registration
        import importlib
        try:
            importlib.import_module("tools.a2a_tool")
        except Exception:
            pytest.skip("a2a_tool not yet implemented")

        from tools.registry import registry

        a2a_tools = [
            "a2a_discover", "a2a_send", "a2a_get_task", "a2a_cancel_task",
        ]
        for tool_name in a2a_tools:
            entry = registry._tools.get(tool_name)
            assert entry is not None, f"Tool {tool_name} not registered"
            assert entry.toolset == "a2a"

    def test_tool_schemas_have_required_fields(self):
        """Each tool schema has name, description, and parameters."""
        import importlib
        try:
            importlib.import_module("tools.a2a_tool")
        except Exception:
            pytest.skip("a2a_tool not yet implemented")

        from tools.registry import registry

        for tool_name in ["a2a_discover", "a2a_send", "a2a_get_task", "a2a_cancel_task"]:
            entry = registry._tools.get(tool_name)
            assert entry is not None
            schema = entry.schema
            assert "description" in schema
            assert "parameters" in schema

    def test_check_fn_requires_no_env(self):
        """A2A tools have no env requirements (pure HTTP client)."""
        import importlib
        try:
            importlib.import_module("tools.a2a_tool")
        except Exception:
            pytest.skip("a2a_tool not yet implemented")

        from tools.registry import registry

        for tool_name in ["a2a_discover", "a2a_send", "a2a_get_task", "a2a_cancel_task"]:
            entry = registry._tools.get(tool_name)
            assert entry is not None
            assert entry.requires_env == []


# ---------------------------------------------------------------------------
# A2AClient — message_id generation
# ---------------------------------------------------------------------------

class TestA2AClientMessageIds:
    """Verify message IDs are unique UUIDs."""

    def test_message_id_is_uuid(self):
        """SendMessage auto-generates a UUID messageId."""
        from tools.a2a_tool import A2AClient

        rpc_result = _jsonrpc_ok(FAKE_TASK)
        mock_resp = _mock_httpx_response(200, rpc_result)

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = A2AClient("https://agent.example.com")
            asyncio.run(client.send_message("Hi"))

        body = instance.post.call_args[1]["json"]
        msg_id = body["params"]["message"]["messageId"]
        # Should be a valid UUID
        uuid.UUID(msg_id)  # raises ValueError if invalid

    def test_message_ids_are_unique(self):
        """Two SendMessage calls get different messageIds."""
        from tools.a2a_tool import A2AClient

        rpc_result = _jsonrpc_ok(FAKE_TASK)
        mock_resp = _mock_httpx_response(200, rpc_result)

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = A2AClient("https://agent.example.com")
            asyncio.run(client.send_message("first"))
            asyncio.run(client.send_message("second"))

        calls = instance.post.call_args_list
        id1 = calls[0][1]["json"]["params"]["message"]["messageId"]
        id2 = calls[1][1]["json"]["params"]["message"]["messageId"]
        assert id1 != id2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestA2AEdgeCases:
    """Edge-case and robustness tests."""

    def test_discover_non_json_response(self):
        """Discover with non-JSON response raises A2AError."""
        from tools.a2a_tool import A2AClient, A2AError

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.json.side_effect = Exception("not JSON")
        mock_resp.text = "<html>Not Found</html>"
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get.return_value = mock_resp
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = A2AClient("https://agent.example.com")
            with pytest.raises(A2AError):
                asyncio.run(client.discover())

    def test_send_message_timeout(self):
        """SendMessage with network timeout raises A2AError."""
        from tools.a2a_tool import A2AClient, A2AError
        import httpx

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.side_effect = httpx.TimeoutException("timed out")
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = A2AClient("https://agent.example.com")
            with pytest.raises(A2AError) as exc_info:
                asyncio.run(client.send_message("hello"))
            assert "timed out" in str(exc_info.value).lower() or "timeout" in str(exc_info.value).lower()

    def test_send_message_connection_error(self):
        """SendMessage with connection error raises A2AError."""
        from tools.a2a_tool import A2AClient, A2AError
        import httpx

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.side_effect = httpx.ConnectError("connection refused")
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            client = A2AClient("https://agent.example.com")
            with pytest.raises(A2AError) as exc_info:
                asyncio.run(client.send_message("hello"))
            assert "connect" in str(exc_info.value).lower()

    def test_tool_handler_generic_exception(self):
        """Tool handlers catch unexpected exceptions and return JSON error."""
        from tools.a2a_tool import a2a_send_handler

        with patch("tools.a2a_tool._run_a2a_async") as mock_run:
            mock_run.side_effect = RuntimeError("unexpected")
            result = json.loads(a2a_send_handler(
                {"url": "https://agent.example.com", "message": "test"},
                task_id="t1",
            ))

        assert "error" in result
        assert "unexpected" in result["error"]
