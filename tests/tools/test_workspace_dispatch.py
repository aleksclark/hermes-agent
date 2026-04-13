"""Tests for workspace_dispatch tool (tools/workspace_dispatch.py)."""

import json
import os
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helpers — mock the httpx responses
# ---------------------------------------------------------------------------

def _mock_response(status=200, json_data=None, text=""):
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.text = text or json.dumps(json_data or {})
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("no json")
    return resp


def _make_workspace_info(name="test-ws", project="curri", active=True,
                         acp_port=9106, acp_status="running"):
    return {
        "name": name,
        "project": project,
        "active": active,
        "tag_index": 5,
        "dir": f"/home/aleks/worktrees/{project}/{name}",
        "acp_port": acp_port,
        "acp_url": None,
        "acp_session_id": None,
        "acp_status": acp_status,
    }


# ---------------------------------------------------------------------------
# Import with daemon check mocked out
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Prevent real HTTP calls and isolate the dispatch tracker."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    # Reset the in-memory dispatch tracker
    import tools.workspace_dispatch as wd
    wd._dispatches.clear()
    yield


# ---------------------------------------------------------------------------
# Tests: _api helper
# ---------------------------------------------------------------------------

class TestApiHelper:
    def test_connect_error(self):
        from tools.workspace_dispatch import _api, _DAEMON_URL
        import tools.workspace_dispatch as wd
        # Point at a port that isn't listening
        original = wd._DAEMON_URL
        wd._DAEMON_URL = "http://127.0.0.1:19999"
        try:
            result = _api("GET", "/api/projects", timeout=1)
            assert "error" in result
        finally:
            wd._DAEMON_URL = original

    @patch("tools.workspace_dispatch.httpx.Client")
    def test_get_success(self, mock_client_cls):
        from tools.workspace_dispatch import _api
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_response(200, json_data={"ok": True})

        result = _api("GET", "/api/projects")
        assert result == {"ok": True}

    @patch("tools.workspace_dispatch.httpx.Client")
    def test_post_error(self, mock_client_cls):
        from tools.workspace_dispatch import _api
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _mock_response(
            404, text='{"error":"not found"}'
        )
        mock_client.post.return_value.json.return_value = {"error": "not found"}

        result = _api("POST", "/api/workspaces", json_body={"name": "x"})
        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: workspace_dispatch
# ---------------------------------------------------------------------------

class TestWorkspaceDispatch:
    @patch("tools.workspace_dispatch._send_task_fire_and_forget")
    @patch("tools.workspace_dispatch._wait_for_health")
    @patch("tools.workspace_dispatch._api")
    def test_dispatch_existing_workspace(self, mock_api, mock_health, mock_send):
        """Dispatch to an existing workspace — no creation needed."""
        from tools.workspace_dispatch import workspace_dispatch

        ws_info = _make_workspace_info("fix-bug")
        mock_api.return_value = ws_info
        mock_health.return_value = True
        mock_send.return_value = {"status": "dispatched"}

        result = json.loads(workspace_dispatch([{
            "workspace": "fix-bug",
            "message": "Fix the bug in foo.ts",
        }]))

        assert result["status"] == "dispatched"
        assert result["workspace"] == "fix-bug"
        assert "created" not in result

    @patch("tools.workspace_dispatch._send_task_fire_and_forget")
    @patch("tools.workspace_dispatch._wait_for_health")
    @patch("tools.workspace_dispatch._api")
    def test_dispatch_creates_workspace(self, mock_api, mock_health, mock_send):
        """Dispatch with project creates a new workspace."""
        from tools.workspace_dispatch import workspace_dispatch

        ws_info = _make_workspace_info("new-branch")
        # First call: GET workspace — not found
        # Second call: POST create workspace — success
        mock_api.side_effect = [
            {"error": "HTTP 404: not found"},
            ws_info,
        ]
        mock_health.return_value = True
        mock_send.return_value = {"status": "dispatched"}

        result = json.loads(workspace_dispatch([{
            "workspace": "new-branch",
            "project": "curri",
            "message": "Implement feature X",
        }]))

        assert result["status"] == "dispatched"
        assert result["created"] is True

    @patch("tools.workspace_dispatch._api")
    def test_dispatch_missing_workspace_no_project(self, mock_api):
        """Dispatch to missing workspace without project gives helpful error."""
        from tools.workspace_dispatch import workspace_dispatch

        mock_api.return_value = {"error": "HTTP 404: not found"}

        result = json.loads(workspace_dispatch([{
            "workspace": "nonexistent",
            "message": "Do stuff",
        }]))

        assert "error" in result
        assert "project" in result["error"].lower()

    @patch("tools.workspace_dispatch._send_task")
    @patch("tools.workspace_dispatch._wait_for_health")
    @patch("tools.workspace_dispatch._api")
    def test_dispatch_wait_mode(self, mock_api, mock_health, mock_send):
        """Dispatch with wait=True blocks and returns the response."""
        from tools.workspace_dispatch import workspace_dispatch

        mock_api.return_value = _make_workspace_info("sync-ws")
        mock_health.return_value = True
        mock_send.return_value = {
            "session_id": "sess-123",
            "text": "Fixed the bug, all tests pass.",
            "status": "completed",
        }

        result = json.loads(workspace_dispatch([{
            "workspace": "sync-ws",
            "message": "Fix it",
            "wait": True,
        }]))

        assert result["status"] == "completed"
        assert "Fixed the bug" in result["response"]

    @patch("tools.workspace_dispatch._send_task_fire_and_forget")
    @patch("tools.workspace_dispatch._wait_for_health")
    @patch("tools.workspace_dispatch._api")
    def test_parallel_dispatch(self, mock_api, mock_health, mock_send):
        """Multiple tasks dispatch in parallel."""
        from tools.workspace_dispatch import workspace_dispatch

        mock_api.return_value = _make_workspace_info("ws")
        mock_health.return_value = True
        mock_send.return_value = {"status": "dispatched"}

        result = json.loads(workspace_dispatch([
            {"workspace": "ws-1", "message": "Task 1"},
            {"workspace": "ws-2", "message": "Task 2"},
            {"workspace": "ws-3", "message": "Task 3"},
        ]))

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(r["status"] == "dispatched" for r in result)

    def test_dispatch_empty_tasks(self):
        from tools.workspace_dispatch import workspace_dispatch
        result = json.loads(workspace_dispatch([]))
        assert "error" in result

    def test_dispatch_too_many_tasks(self):
        from tools.workspace_dispatch import workspace_dispatch
        tasks = [{"workspace": f"ws-{i}", "message": "x"} for i in range(6)]
        result = json.loads(workspace_dispatch(tasks))
        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: workspace_poll
# ---------------------------------------------------------------------------

class TestWorkspacePoll:
    @patch("tools.workspace_dispatch._get_history")
    @patch("tools.workspace_dispatch._api")
    def test_poll_workspace(self, mock_api, mock_history):
        """Poll returns status and recent messages."""
        from tools.workspace_dispatch import workspace_poll

        mock_api.return_value = {"status": "ok"}
        mock_history.return_value = {
            "messages": [
                {"role": "user", "content": "Fix the bug"},
                {"role": "agent", "content": "Looking at the code now..."},
                {"role": "agent", "content": "Found the issue, fixing it."},
                {"role": "agent", "content": "All tests pass. Committed and pushed."},
            ]
        }

        result = json.loads(workspace_poll(["my-ws"]))
        assert len(result) == 1
        ws = result[0]
        assert ws["workspace"] == "my-ws"
        assert ws["healthy"] is True
        assert ws["agent_messages"] == 3
        assert ws["appears_done"] is True

    @patch("tools.workspace_dispatch._get_history")
    @patch("tools.workspace_dispatch._api")
    def test_poll_in_progress(self, mock_api, mock_history):
        """Poll shows agent is still working."""
        from tools.workspace_dispatch import workspace_poll

        mock_api.return_value = {"status": "ok"}
        mock_history.return_value = {
            "messages": [
                {"role": "user", "content": "Refactor the module"},
                {"role": "agent", "content": "Starting to read the codebase..."},
            ]
        }

        result = json.loads(workspace_poll(["ws-1"]))
        ws = result[0]
        assert ws["appears_done"] is False
        assert ws["agent_messages"] == 1

    def test_poll_no_workspaces_no_dispatches(self):
        """Poll with no args and no dispatches returns helpful message."""
        from tools.workspace_dispatch import workspace_poll
        result = json.loads(workspace_poll([]))
        assert "message" in result

    @patch("tools.workspace_dispatch._get_history")
    @patch("tools.workspace_dispatch._api")
    def test_poll_tracks_dispatches(self, mock_api, mock_history):
        """After dispatch, poll with no args shows tracked dispatches."""
        from tools.workspace_dispatch import workspace_dispatch, workspace_poll, _dispatches

        _dispatches["ws-test"] = {
            "dispatched_at": time.time(),
            "message": "Fix the thing",
            "status": "dispatched",
        }

        result = json.loads(workspace_poll([]))
        assert isinstance(result, list)
        assert any(r["workspace"] == "ws-test" for r in result)


# ---------------------------------------------------------------------------
# Tests: registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_dispatch_registered(self):
        from tools.registry import registry
        assert "workspace_dispatch" in registry.get_all_tool_names()

    def test_poll_registered(self):
        from tools.registry import registry
        assert "workspace_poll" in registry.get_all_tool_names()

    def test_dispatch_in_acp_toolset(self):
        from tools.registry import registry
        toolset = registry.get_toolset_for_tool("workspace_dispatch")
        assert toolset == "acp"
